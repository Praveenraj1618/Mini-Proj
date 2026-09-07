import os
import json
import difflib
import re
import time
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from config import get_llm, get_llm_candidates, RETRIEVAL_TOP_K, check_api_key_configured
from core.document_loader import LoadedDocument, ExtractedImage
from core.vector_store import LegalVectorStore
from core.hybrid_retriever import LegalHybridRetriever

LEGAL_SYSTEM_PROMPT = """You are a senior legal document intelligence assistant.
Your job is to analyze legal documents, extract critical facts, identify risks, and answer questions accurately.

CRITICAL RULES:
1. ALWAYS base your answers strictly on the provided document excerpts.
2. ALWAYS cite the exact Page number(s) (e.g. "[Page 2]") and specific clause names/numbers when referencing facts.
3. If the provided excerpts do not contain the answer or the clause is absent, state clearly: "The provided document excerpts do not specify [missing item]." Never guess or hallucinate terms.
4. Highlight legal implications, potential ambiguities, or missing standard protections where relevant.
5. Use clear, professional, and structured formatting (bullet points, bold highlights).
"""


def parse_llm_json(raw_content: Any) -> Any:
    """Extract the first valid JSON object or array from an LLM response."""
    if not isinstance(raw_content, str):
        raise ValueError("LLM response content was not text.")
    text = raw_content.strip()
    if not text:
        raise ValueError("LLM returned an empty response.")

    # Fast path for a clean JSON response.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Models may wrap JSON in Markdown or place reasoning before it. Try each
    # object/array boundary and accept the first complete JSON value.
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            return value
        except json.JSONDecodeError:
            continue
    raise ValueError("LLM response did not contain valid JSON.")


def parse_llm_json_object(raw_content: Any) -> Dict[str, Any]:
    value = parse_llm_json(raw_content)
    if not isinstance(value, dict):
        raise ValueError("LLM response JSON was not an object.")
    return value


def parse_llm_json_array(raw_content: Any) -> List[Any]:
    value = parse_llm_json(raw_content)
    if not isinstance(value, list):
        raise ValueError("LLM response JSON was not an array.")
    return value

class LegalReviewer:
    """
    Core intelligence engine for legal first-pass review, risk auditing, visual inspection, and grounded Q&A.
    """

    def __init__(
        self,
        document: LoadedDocument,
        vector_store: Optional[LegalVectorStore],
        hybrid_retriever: Optional[LegalHybridRetriever] = None,
    ):
        self.document = document
        self.vector_store = vector_store
        if hybrid_retriever is not None:
            self.hybrid_retriever = hybrid_retriever
        elif vector_store is not None and vector_store.chunks:
            self.hybrid_retriever = LegalHybridRetriever(vector_store, vector_store.chunks)
        else:
            self.hybrid_retriever = None
        self._llm = None
        self._vision_llm = None
        self._llm_candidates = None
        self._vision_llm_candidates = None
        self._last_llm_diagnostic = None
        self._cached_summary = None
        self._cached_risk_analysis = None
        self._cached_heatmap = None

    @property
    def llm(self):
        if self._llm is not None:
            return self._llm
        candidates = self._candidate_clients(vision=False)
        return candidates[0][1] if candidates else get_llm(vision=False)

    @property
    def vision_llm(self):
        if self._vision_llm is not None:
            return self._vision_llm
        candidates = self._candidate_clients(vision=True)
        return candidates[0][1] if candidates else get_llm(vision=True)

    @property
    def last_llm_diagnostic(self) -> Optional[str]:
        """A credential-free description of the most recent provider result."""
        return self._last_llm_diagnostic

    def _candidate_clients(self, vision: bool = False):
        injected_client = self._vision_llm if vision else self._llm
        if injected_client is not None:
            return [("configured client", injected_client)]

        cache_name = "_vision_llm_candidates" if vision else "_llm_candidates"
        candidates = getattr(self, cache_name)
        if candidates is None:
            candidates = get_llm_candidates(vision=vision)
            setattr(self, cache_name, candidates)
        return candidates

    @staticmethod
    def _message_text(message: Any) -> str:
        """Normalize text responses returned by OpenAI-compatible providers."""
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "\n".join(parts).strip()
        return ""

    def _format_context(self, docs) -> str:
        formatted = []
        for d in docs:
            page_num = d.metadata.get("page_number", "Unknown")
            formatted.append(f"--- [Page {page_num}] ---\n{d.page_content}")
        return "\n\n".join(formatted)

    def _retrieve(self, query: str, top_k: int = RETRIEVAL_TOP_K):
        """Use hybrid retrieval in production, with dense search as a fallback."""
        if self.hybrid_retriever is not None:
            return self.hybrid_retriever.hybrid_search(query, top_k=top_k)
        if self.vector_store is None:
            return []
        return self.vector_store.similarity_search(query, k=top_k)

    def _invoke_llm(self, messages, max_tokens: int = 700, vision: bool = False) -> str:
        """Invoke configured providers with bounded output, retry, and failover."""
        failures = []
        candidates = self._candidate_clients(vision=vision)
        if not candidates:
            raise RuntimeError("No supported LLM provider is configured.")

        for provider, base_client in candidates:
            client = base_client.bind(max_tokens=max_tokens)
            for attempt in range(2):
                try:
                    response = self._message_text(client.invoke(messages))
                    if not response:
                        raise ValueError("LLM returned an empty response.")
                    if failures:
                        prior = ", ".join(
                            f"{name} ({error_type})" for name, error_type in failures
                        )
                        self._last_llm_diagnostic = (
                            f"Fallback provider {provider} succeeded after {prior}."
                        )
                    else:
                        self._last_llm_diagnostic = f"Provider {provider} succeeded."
                    return response
                except Exception as error:
                    message = str(error)
                    is_rate_limit = "429" in message or "rate_limit" in message.lower()
                    if is_rate_limit and attempt == 0:
                        match = re.search(r"try again in\s+([\d.]+)s", message, re.IGNORECASE)
                        wait_seconds = float(match.group(1)) if match else 2.0
                        time.sleep(min(max(wait_seconds + 0.5, 1.0), 5.0))
                        continue
                    failures.append((provider, type(error).__name__))
                    break

        diagnostic = ", ".join(
            f"{provider} ({error_type})" for provider, error_type in failures
        )
        self._last_llm_diagnostic = f"All configured providers failed: {diagnostic}."
        raise RuntimeError(self._last_llm_diagnostic)

    def generate_executive_summary(self) -> str:
        """
        Extracts high-level document metadata, parties, effective dates, governing law, and key terms.
        """
        if self._cached_summary:
            return self._cached_summary

        if not check_api_key_configured():
            return (
                "⚠️ LLM API Key is not configured. Please set XAI_API_KEY, OPENAI_API_KEY, GROQ_API_KEY, or GEMINI_API_KEY in your .env file.\n\n"
                f"Document Statistics:\n"
                f"- File Name: {self.document.file_name}\n"
                f"- Total Pages: {self.document.total_pages}\n"
                f"- Total Words: {self.document.total_words}\n"
                f"- Embedded Images/Tables: {len(self.document.images)}"
            )

        # Retrieve representative chunks from the beginning, middle, and end or top search
        initial_pages_text = "\n\n".join(
            [f"--- [Page {p.page_num}] ---\n{p.text}" for p in self.document.pages[:min(4, len(self.document.pages))]]
        )

        prompt = f"""
Analyze the following legal document (first several pages provided below) and produce a structured Executive Summary.

Document Excerpts:
{initial_pages_text}

Provide the summary using this exact structure:
1. **Document Title & Type**: (e.g., Master Services Agreement, Non-Disclosure Agreement, Lease, Court Pleading)
2. **Parties Involved**: (Disclosing Party / Receiving Party / Client / Contractor, include entity types if mentioned)
3. **Effective Date & Term**: (Start date, duration, expiration, renewal terms)
4. **Governing Law & Jurisdiction**: (State/Country and court venue)
5. **Key Commercial / Financial Terms**: (Payment terms, consideration, fees, or N/A)
6. **Core Purpose & Scope**: (2-3 sentence overview of what this agreement accomplishes)
7. **Key Milestones / Notice Periods**: (e.g., 30-day written notice for termination)

Include page citations [Page X] for every extracted fact.
"""
        messages = [
            SystemMessage(content=LEGAL_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = self._invoke_llm(messages, max_tokens=1400)
            self._cached_summary = response
            return response
        except Exception as e:
            return (
                "⚠️ LLM analysis could not be completed.\n\n"
                "Tip: If using xAI/OpenAI, ensure your account has active credits or billing configured.\n\n"
                f"Document Statistics:\n"
                f"- File Name: {self.document.file_name}\n"
                f"- Total Pages: {self.document.total_pages}\n"
                f"- Total Words: {self.document.total_words:,}\n\n"
                f"Initial Document Excerpt:\n{initial_pages_text[:800]}..."
            )

    def run_risk_analysis(self) -> str:
        """
        Performs a comprehensive First-Pass Risk & Red-Flag Audit for common high-risk legal clauses.
        """
        if self._cached_risk_analysis:
            return self._cached_risk_analysis

        if not check_api_key_configured():
            return "⚠️ LLM API Key is not configured. Please set your API key in .env to perform Risk & Red-Flag Analysis."

        # Search for key risk topics in the vector store
        risk_topics = [
            "limitation of liability indemnification indemnity",
            "termination for convenience default notice period",
            "confidentiality non-disclosure non-compete exclusivity non-solicit",
            "intellectual property ownership assignment work for hire",
            "governing law dispute resolution arbitration jurisdiction class action waiver",
            "warranties representations disclaimers liquidated damages penalties",
        ]

        retrieved_risk_chunks = []
        seen_chunk_ids = set()

        for topic in risk_topics:
            docs = self._retrieve(topic, top_k=2)
            for d in docs:
                chunk_id = d.metadata.get("chunk_id")
                if chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    retrieved_risk_chunks.append(d)

        risk_context = self._format_context(retrieved_risk_chunks)

        prompt = f"""
Perform a First-Pass "Red Flag" and Legal Risk Audit on the following document excerpts.

Excerpts:
{risk_context}

Analyze and grade the risk for each of the following areas:
1. **Liability & Indemnification**: (Is liability uncapped? Are indemnities unilateral or broad?)
2. **Termination Rights**: (Can either party terminate for convenience? What are the cure periods?)
3. **Restrictive Covenants**: (Non-compete, non-solicit, exclusivity clauses?)
4. **Intellectual Property**: (Who owns IP created during the term? Any unintended transfer?)
5. **Dispute Resolution & Jurisdiction**: (Mandatory arbitration, fee-shifting, foreign venue?)
6. **Warranties & Penalties**: (Liquidated damages, onerous guarantees, disclaimers?)

For each area, provide:
- **Risk Level**: [🟢 LOW RISK / 🟡 MODERATE RISK / 🔴 HIGH RISK / ⚪ NOT FOUND / SILENT]
- **Clause Finding & Citation**: Quote relevant sentence with [Page X] citation.
- **Reviewer Recommendation**: Actionable advice on what to negotiate or review.
"""
        messages = [
            SystemMessage(content=LEGAL_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        
        try:
            response = self._invoke_llm(messages, max_tokens=1800)
            self._cached_risk_analysis = response
            return response
        except Exception as e:
            return (
                "⚠️ LLM risk analysis could not be completed.\n\n"
                "Relevant Risk Clauses Retrieved from Document:\n"
                f"{risk_context[:1200]}..."
            )

    def answer_query(self, question: str, top_k: int = RETRIEVAL_TOP_K) -> Dict[str, Any]:
        """
        Answers a user's question grounded strictly on retrieved document excerpts.
        """
        docs = self._retrieve(question, top_k=min(top_k, 3))
        context = self._format_context(docs)

        sources = []
        citations = set()
        for doc in docs:
            page = doc.metadata.get("page_number")
            citations.add(page)
            score = doc.metadata.get("reranker_score", doc.metadata.get("rrf_score"))
            sources.append({
                "page": page,
                "text": doc.page_content,
                "score": score,
                "score_type": "cross_encoder" if "reranker_score" in doc.metadata else "rrf",
            })

        if not check_api_key_configured():
            return {
                "answer": "⚠️ LLM API Key is not configured. Displaying relevant raw context excerpts below:",
                "citations": sorted(list(citations)),
                "sources": sources,
            }

        prompt = f"""
Answer the following legal query using ONLY the provided document excerpts.

Context:
{context}

Question:
{question}

Instructions:
- Provide a direct, well-reasoned answer.
- Always include exact citations (e.g. "[Page 3]") for all statements.
- If the excerpts do not address the question, explicitly state that the document does not contain this information.
- Keep the answer under 250 words and finish every sentence.
"""
        messages = [
            SystemMessage(content=LEGAL_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
        
        try:
            answer = self._invoke_llm(messages, max_tokens=1200)
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                answer = (
                    "The AI service is temporarily rate-limited. Please wait about 20 seconds and try again. "
                    "The retrieved source excerpts below are still available."
                )
            else:
                answer = (
                    "The AI service could not synthesize an answer. "
                    "Please inspect the retrieved source excerpts below and try again."
                )

        return {
            "answer": answer,
            "citations": sorted(list(citations)),
            "sources": sources,
        }

    def search_clauses(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Hybrid search for specific clauses or topics.
        """
        results = self._retrieve(query, top_k=top_k)
        return [
            {
                "page": doc.metadata.get("page_number"),
                "chunk_id": doc.metadata.get("chunk_id"),
                "text": doc.page_content,
                "score": doc.metadata.get("reranker_score", doc.metadata.get("rrf_score")),
            }
            for doc in results
        ]

    def analyze_visuals(self) -> List[Dict[str, Any]]:
        """
        Inspects extracted document images, tables, stamps, and signatures using multimodal vision.
        """
        if not self.document.images:
            return [{"status": "No embedded images, tables, or visual figures detected in this document."}]

        if not check_api_key_configured():
            return [{
                "status": f"Found {len(self.document.images)} visual element(s).",
                "details": [
                    f"Page {img.page_num} (Dimensions: {img.width}x{img.height}, Type: {img.mime_type})"
                    for img in self.document.images
                ],
                "note": "Set LLM API Key in .env to run AI visual analysis on these elements."
            }]

        results = []
        for idx, img in enumerate(self.document.images[:5]):  # Process up to 5 main visuals
            image_url_payload = f"data:{img.mime_type};base64,{img.base64_data}"
            
            prompt_content = [
                {
                    "type": "text",
                    "text": (
                        f"Analyze this visual element extracted from Page {img.page_num} of a legal document. "
                        "Determine if this is a:\n"
                        "- Signature block / Signed execution page\n"
                        "- Official seal / Notary stamp / Watermark\n"
                        "- Data Table / Schedule / Exhibit\n"
                        "- Organizational Chart / Flowchart\n"
                        "Provide a concise breakdown of any names, dates, amounts, or legibility issues."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url_payload},
                },
            ]

            try:
                msg = HumanMessage(content=prompt_content)
                response = self._invoke_llm([msg], max_tokens=700, vision=True)
                results.append({
                    "page": img.page_num,
                    "image_index": img.image_index,
                    "analysis": response,
                })
            except Exception as e:
                results.append({
                    "page": img.page_num,
                    "image_index": img.image_index,
                    "analysis": f"Error analyzing visual: {str(e)}",
                })

        return results

    def export_report(self, output_path: str) -> str:
        """
        Generates and saves a complete First-Review Audit Report in Markdown.
        """
        summary = self.generate_executive_summary()
        risks = self.run_risk_analysis()

        report_content = f"""# Legal First-Pass Review Audit Report

**Document Name:** {self.document.file_name}  
**Total Pages:** {self.document.total_pages}  
**Total Words:** {self.document.total_words:,}  

---

## 1. Executive Summary & Key Terms

{summary}

---

## 2. Risk & Red-Flag Audit

{risks}

---
*Report generated by Legal Document Intelligence System*
"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        return output_path

    def _classify_document_heuristically(self) -> Dict[str, Any]:
        """Reliable local classification used when LLM classification is unavailable."""
        text_sample = self.document.full_text[:3000].lower()
        if "master services agreement" in text_sample or "services agreement" in text_sample:
            doc_type = "Master Services Agreement (MSA)"
            focus = ["Scope of Work", "SLA / Payment", "Liability Cap", "Indemnification"]
        elif "non-disclosure agreement" in text_sample or "confidentiality agreement" in text_sample:
            doc_type = "Non-Disclosure Agreement (NDA)"
            focus = ["Confidentiality", "Term", "Return of Materials", "Exclusions"]
        elif "lease" in text_sample or "tenant" in text_sample or "landlord" in text_sample:
            doc_type = "Commercial / Residential Lease Agreement"
            focus = ["Rent Payment", "Security Deposit", "Maintenance", "Termination"]
        elif "employment" in text_sample or "employee" in text_sample or "employer" in text_sample:
            doc_type = "Employment Agreement"
            focus = ["Compensation", "IP Ownership", "Non-Compete", "Termination"]
        else:
            doc_type = "General Commercial Contract"
            focus = ["Governing Law", "Liability", "Termination", "Payment Terms"]

        return {
            "doc_type": doc_type,
            "confidence": 0.85,
            "target_risk_focus": focus,
            "source": "Rule-based Heuristic",
        }

    def classify_document(self) -> Dict[str, Any]:
        """
        Automatically classifies the legal document type (NDA, MSA, Lease, Employment, Vendor, Litigation)
        and highlights key risk focus areas for that document class.
        """
        if not check_api_key_configured():
            return self._classify_document_heuristically()

        sample_text = "\n\n".join([f"[Page {p.page_num}]\n{p.text}" for p in self.document.pages[:3]])
        prompt = f"""
Analyze the first few pages of this legal document and classify its type.

Document Sample:
{sample_text}

Respond ONLY with a valid JSON object with the following structure:
{{
  "doc_type": "Name of Document Category (e.g., Non-Disclosure Agreement, Commercial Lease, Master Services Agreement, Employment Contract, Patent License)",
  "confidence": 0.95,
  "target_risk_focus": ["Area 1", "Area 2", "Area 3", "Area 4"],
  "summary": "1-sentence summary of agreement purpose"
}}
"""
        try:
            res = self._invoke_llm(
                [SystemMessage(content="You are a legal document classification system."), HumanMessage(content=prompt)],
                max_tokens=250,
            )
            # Clean JSON formatting
            return parse_llm_json_object(res)
        except Exception:
            return self._classify_document_heuristically()

    def generate_clause_risk_heatmap(self) -> List[Dict[str, Any]]:
        """
        Generates a clause-by-clause structural risk heatmap (🟢 LOW, 🟡 MODERATE, 🔴 HIGH).
        Provides clause name, page citation, risk level, why it is risky, and evidence.
        """
        if self._cached_heatmap is not None:
            return self._cached_heatmap

        risk_categories = [
            ("1. Definitions & Scope", "definition scope parameters exclusions"),
            ("2. Payment & Commercial Terms", "payment consideration fees penalty late fee due date"),
            ("3. Liability & Limitation of Liability", "limitation of liability indemnification indemnity maximum exposure uncapped"),
            ("4. Intellectual Property Rights", "intellectual property ip ownership work for hire license patent copyright"),
            ("5. Confidentiality & Non-Disclosure", "confidentiality non-disclosure term exceptions trade secret"),
            ("6. Termination & Remedies", "termination for convenience default breach cure period notice"),
            ("7. Restrictive Covenants", "non-compete non-solicit exclusivity geographical restriction"),
            ("8. Dispute Resolution & Governing Law", "governing law jurisdiction venue arbitration fee-shifting class action waiver"),
        ]

        evidence_by_category = []
        for cat_name, query_terms in risk_categories:
            top_docs = self._retrieve(query_terms, top_k=2)
            if not top_docs:
                continue
            ctx = "\n\n".join([f"[Page {d.metadata.get('page_number')}]\n{d.page_content}" for d in top_docs])
            evidence_by_category.append((cat_name, top_docs, ctx))

        if not check_api_key_configured():
            heatmap_results = []
            for cat_name, top_docs, _ in evidence_by_category:
                heatmap_results.append({
                    "category": cat_name,
                    "risk_level": "REVIEW REQUIRED",
                    "page": top_docs[0].metadata.get("page_number", 1),
                    "clause_title": cat_name,
                    "excerpt": top_docs[0].page_content[:150] + "...",
                    "why_risky": "No automated risk rating was produced because an LLM is not configured.",
                    "recommendation": "Review the cited excerpt with qualified legal counsel.",
                })
            self._cached_heatmap = heatmap_results
            return heatmap_results

        combined_context = "\n\n".join(
            f"### {cat_name}\n{ctx}"
            for cat_name, _, ctx in evidence_by_category
        )
        category_names = [cat_name for cat_name, _, _ in evidence_by_category]
        prompt = f"""
Perform one clause-level legal risk review for every category below.

Document type: {self._classify_document_heuristically()['doc_type']}

Category evidence:
{combined_context}

Return ONLY one valid JSON array, with exactly one object for each category and
in this exact order: {json.dumps(category_names)}

Each object must use this schema:
{{
  "clause_title": "exact category name",
  "risk_level": "LOW" or "MODERATE" or "HIGH",
  "page": page_number_int,
  "excerpt": "exact 1-2 sentence quote from the supplied evidence",
  "why_risky": "concise explanation of legal risk or exposure",
  "recommendation": "actionable negotiation tip"
}}
Do not omit a category. If a protection is absent or unclear, explain that fact
using the most relevant cited excerpt instead of inventing contract language.
"""

        try:
            response = self._invoke_llm(
                [SystemMessage(content=LEGAL_SYSTEM_PROMPT), HumanMessage(content=prompt)],
                max_tokens=2600,
            )
            generated_items = parse_llm_json_array(response)
            by_title = {
                str(item.get("clause_title", "")).strip(): item
                for item in generated_items
                if isinstance(item, dict)
            }

            heatmap_results = []
            for index, (cat_name, top_docs, _) in enumerate(evidence_by_category):
                item = by_title.get(cat_name)
                if item is None and index < len(generated_items):
                    candidate = generated_items[index]
                    item = candidate if isinstance(candidate, dict) else None
                if item is None:
                    raise ValueError(f"Model omitted heatmap category: {cat_name}")

                risk_level = str(item.get("risk_level", "")).upper()
                if not any(level in risk_level for level in ("LOW", "MODERATE", "HIGH")):
                    raise ValueError(f"Model returned an invalid risk level for: {cat_name}")

                allowed_pages = {doc.metadata.get("page_number") for doc in top_docs}
                page = item.get("page")
                if page not in allowed_pages:
                    page = top_docs[0].metadata.get("page_number", 1)

                heatmap_results.append({
                    "category": cat_name,
                    "clause_title": cat_name,
                    "risk_level": risk_level,
                    "page": page,
                    "excerpt": str(item.get("excerpt") or top_docs[0].page_content[:150] + "..."),
                    "why_risky": str(item.get("why_risky") or "Human review is required."),
                    "recommendation": str(item.get("recommendation") or "Review with qualified legal counsel."),
                })
        except Exception as error:
            diagnostic = self.last_llm_diagnostic or f"Model response failed ({type(error).__name__})."
            if "succeeded" in diagnostic.lower():
                diagnostic = f"Provider response could not be used ({type(error).__name__})."
            heatmap_results = []
            for cat_name, top_docs, _ in evidence_by_category:
                heatmap_results.append({
                    "category": cat_name,
                    "risk_level": "ANALYSIS FAILED",
                    "page": top_docs[0].metadata.get("page_number"),
                    "clause_title": cat_name,
                    "excerpt": top_docs[0].page_content[:150] + "...",
                    "why_risky": "The LLM could not analyze this category.",
                    "recommendation": "Check provider availability or use the cited excerpt for manual review.",
                    "error_type": type(error).__name__,
                    "diagnostic": diagnostic,
                })

        if all(item.get("risk_level") != "ANALYSIS FAILED" for item in heatmap_results):
            self._cached_heatmap = heatmap_results
        return heatmap_results

    def generate_negotiation_strategy(self, clause_text: str) -> Dict[str, Any]:
        """
        Contract Negotiation Assistant: Evaluates if a specific clause is unfavorable
        and drafts a safer, balanced alternative clause.
        """
        disclaimer = "⚠️ Disclaimer: AI-generated negotiation suggestion for human/legal review. Not formal legal advice."

        if not check_api_key_configured():
            return {
                "status": "llm_unavailable",
                "risk_level": "NOT ASSESSED",
                "clause_evaluated": clause_text,
                "assessment": "An LLM API key is required to assess this clause.",
                "potential_impact": "Not assessed.",
                "negotiation_tactic": "Not generated.",
                "safer_alternative": "Not generated. Configure an LLM and obtain human legal review.",
                "legal_disclaimer": disclaimer,
            }

        prompt = f"""
Act as a senior contract negotiation attorney. Evaluate the following contract clause for fairness and risk.

Clause Text to Evaluate:
"{clause_text}"

Produce a structured negotiation analysis JSON with this structure:
{{
  "risk_level": "🟢 LOW" or "🟡 MODERATE" or "🔴 HIGH",
  "assessment": "Detailed evaluation of why this clause is favorable or unfavorable",
  "potential_impact": "What could happen legally or financially to the client",
  "negotiation_tactic": "Specific argument to present to the counterparty",
  "safer_alternative": "A complete, professionally drafted alternative clause that balances risk fairly",
  "legal_disclaimer": "{disclaimer}"
}}
"""
        try:
            res = self._invoke_llm(
                [SystemMessage(content=LEGAL_SYSTEM_PROMPT), HumanMessage(content=prompt)],
                max_tokens=1200,
            )
            item = parse_llm_json_object(res)
            item["legal_disclaimer"] = disclaimer
            return item
        except Exception as e:
            return {
                "status": "analysis_failed",
                "risk_level": "ANALYSIS FAILED",
                "clause_evaluated": clause_text,
                "assessment": "The AI service could not complete the negotiation analysis.",
                "potential_impact": "Not assessed.",
                "negotiation_tactic": "Not generated.",
                "safer_alternative": "Not generated because analysis failed.",
                "legal_disclaimer": disclaimer,
            }


    def extract_obligations(self) -> List[Dict[str, Any]]:
        """
        Extracts structured legal obligations (Party, Action, Deadline/Frequency, Penalty, Page Citation).
        """
        docs = self._retrieve("shall must agree obligation payment notice fee deliver terminate", top_k=6)
        ctx = self._format_context(docs)

        if not check_api_key_configured():
            return self._extract_obligations_deterministically(docs)

        prompt = f"""
Extract all structured legal obligations and deadlines from the following document excerpts.

Excerpts:
{ctx}

Respond strictly with a JSON list of objects matching this format:
[
  {{
    "party": "Party Name (e.g. Receiving Party, Disclosing Party, Contractor, Client, Tenant)",
    "obligation": "Specific duty or action required",
    "deadline_frequency": "Due date, notice period, or frequency (e.g. Monthly, 30 days prior)",
    "consequence": "Penalty, late fee, or remedy upon failure",
    "page": page_number_int
  }}
]
"""
        try:
            res = self._invoke_llm(
                [SystemMessage(content=LEGAL_SYSTEM_PROMPT), HumanMessage(content=prompt)],
                max_tokens=1200,
            )
            return parse_llm_json_array(res)
        except Exception:
            return self._extract_obligations_deterministically(docs)

    @staticmethod
    def _extract_obligations_deterministically(docs) -> List[Dict[str, Any]]:
        """Extract cited obligation sentences without inventing missing details."""
        obligations = []
        seen = set()
        deadline_pattern = re.compile(
            r"\b(?:within\s+\d+\s+days?|\d+\s+days?\s+(?:after|before|prior)|"
            r"monthly|quarterly|annually|upon\s+[^,.;]+)",
            re.IGNORECASE,
        )
        for doc in docs:
            for sentence in re.split(r"(?<=[.;])\s+", doc.page_content):
                if not re.search(r"\b(?:shall|must|required to)\b", sentence, re.IGNORECASE):
                    continue
                normalized = sentence.strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                deadline = deadline_pattern.search(normalized)
                obligations.append({
                    "party": "Party not deterministically identified",
                    "obligation": normalized,
                    "deadline_frequency": deadline.group(0) if deadline else "Not specified in extracted sentence",
                    "consequence": "Not specified in extracted sentence",
                    "page": doc.metadata.get("page_number"),
                })
        return obligations[:20]

    @staticmethod
    def compare_contracts(reviewer_v1: 'LegalReviewer', reviewer_v2: 'LegalReviewer') -> Dict[str, Any]:
        """
        Contract Version Comparison Engine: Compares two contract versions (e.g., v1.pdf vs v2.pdf)
        and extracts clause changes, additions, deletions, and legal impact ratings.
        """
        doc1 = reviewer_v1.document
        doc2 = reviewer_v2.document

        if not check_api_key_configured():
            matcher = difflib.SequenceMatcher(None, doc1.full_text, doc2.full_text)
            changes = []
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag == "equal":
                    continue
                before = doc1.full_text[i1:i2].strip()
                after = doc2.full_text[j1:j2].strip()
                changes.append({
                    "clause": "Document text change",
                    "v1_text": before[:500] or "[No text]",
                    "v2_text": after[:500] or "[No text]",
                    "impact": "REVIEW REQUIRED",
                    "analysis": "Textual difference detected; legal impact was not assessed because an LLM is not configured.",
                })
            return {
                "doc1_name": doc1.file_name,
                "doc2_name": doc2.file_name,
                "total_changes_detected": len(changes),
                "comparison_table": changes[:50],
            }

        prompt = f"""
Compare these two legal document versions and highlight key differences, modified terms, and legal impact.

Version 1 ({doc1.file_name}):
{doc1.full_text[:2500]}

Version 2 ({doc2.file_name}):
{doc2.full_text[:2500]}

Respond strictly with a JSON object in this format:
{{
  "doc1_name": "{doc1.file_name}",
  "doc2_name": "{doc2.file_name}",
  "total_changes_detected": int,
  "comparison_table": [
    {{
      "clause": "Clause Name (e.g., Termination, Liability, Payment)",
      "v1_text": "Text or summary in Version 1",
      "v2_text": "Text or summary in Version 2",
      "impact": "🔴 HIGH" or "🟡 MODERATE" or "🟢 LOW",
      "analysis": "Explanation of how the change affects legal risk or obligations"
    }}
  ]
}}
"""
        try:
            res = reviewer_v1._invoke_llm(
                [SystemMessage(content=LEGAL_SYSTEM_PROMPT), HumanMessage(content=prompt)],
                max_tokens=1600,
            )
            return parse_llm_json_object(res)
        except Exception as e:
            return {
                "doc1_name": doc1.file_name,
                "doc2_name": doc2.file_name,
                "total_changes_detected": 1,
                "comparison_table": [
                    {
                        "clause": "General Terms Comparison",
                        "v1_text": f"Pages: {doc1.total_pages}, Words: {doc1.total_words}",
                        "v2_text": f"Pages: {doc2.total_pages}, Words: {doc2.total_words}",
                        "impact": "🟡 MODERATE",
                        "analysis": "The AI service could not complete the legal-impact comparison.",
                    }
                ],
            }
