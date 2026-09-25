import os
import sys
import json
import threading
import urllib.error
import urllib.request
from types import SimpleNamespace
from http.cookiejar import CookieJar
from http.server import ThreadingHTTPServer
from email.message import Message
from io import BytesIO
from pathlib import Path

import pytest
from langchain_core.documents import Document

os.environ["LEGAL_AI_OFFLINE"] = "true"
BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from app_api import LegalAPIRequestHandler, RequestError, SESSION_DATA
from core.chunker import DocumentChunk, LegalChunker
from core.document_loader import LegalDocumentLoader, LoadedDocument, PageData
from core.hybrid_retriever import BM25Retriever, LegalHybridRetriever, expand_legal_query, is_question_list_chunk
from evaluator import unique_ranked_pages
from core.legal_reviewer import (
    LegalReviewer,
    parse_llm_json,
    parse_llm_json_array,
    parse_llm_json_object,
)
import core.legal_reviewer as legal_reviewer_module


class FakeVectorStore:
    def __init__(self, documents):
        self.documents = documents

    def similarity_search(self, query, k=5):
        return self.documents[:k]


class FakeCrossEncoder:
    def __init__(self):
        self.pairs = None

    def predict(self, pairs, show_progress_bar=False):
        self.pairs = pairs
        return [0.2 + index for index, _ in enumerate(pairs)]


class FakeHybridRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.last_query = None

    def hybrid_search(self, query, top_k=5):
        self.last_query = query
        return self.documents[:top_k]


class RateLimitedOnceClient:
    def __init__(self):
        self.calls = 0
        self.bound_max_tokens = None

    def bind(self, max_tokens):
        self.bound_max_tokens = max_tokens
        return self

    def invoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("Error code: 429 rate_limit_exceeded; try again in 0.01s")
        return SimpleNamespace(content='{"status": "ok"}')


class StaticResponseClient:
    def __init__(self, content="", error=None):
        self.content = content
        self.error = error
        self.calls = 0
        self.bound_max_tokens = None

    def bind(self, max_tokens):
        self.bound_max_tokens = max_tokens
        return self

    def invoke(self, messages):
        self.calls += 1
        if self.error:
            raise self.error
        return SimpleNamespace(content=self.content)


def make_document(name="contract.pdf", text="The Buyer shall pay within 30 days."):
    return LoadedDocument(
        file_path="<memory>",
        file_name=name,
        total_pages=1,
        pages=[PageData(page_num=1, text=text, char_count=len(text), image_count=0)],
        full_text=text,
        total_words=len(text.split()),
        total_chars=len(text),
    )


def test_pdf_loader_accepts_bytes_and_preserves_name():
    pdf_bytes = (PROJECT_DIR / "Mutual-NDA-public-2page.pdf").read_bytes()
    document = LegalDocumentLoader().load_pdf(pdf_bytes, file_name="uploaded.pdf")
    assert document.file_name == "uploaded.pdf"
    assert document.file_path == "<memory>"
    assert document.total_pages > 0
    assert document.total_words > 0


def test_chunker_keeps_page_citations():
    chunks = LegalChunker(chunk_size=80, chunk_overlap=10).chunk_document(make_document())
    assert chunks
    assert all(chunk.page_number == 1 for chunk in chunks)
    assert all("contract.pdf_p1" in chunk.chunk_id for chunk in chunks)


def test_chunker_excludes_post_signature_evaluation_notes():
    contract_text = "MASTER SERVICES AGREEMENT\nCustomer shall pay within 30 days."
    artifact_text = "EVALUATION DESIGN NOTES — DO NOT USE AS CONTRACT TERMS\nKnown evaluation targets"
    document = LoadedDocument(
        file_path="<memory>",
        file_name="evaluation.pdf",
        total_pages=2,
        pages=[
            PageData(page_num=1, text=contract_text, char_count=len(contract_text), image_count=0),
            PageData(page_num=2, text=artifact_text, char_count=len(artifact_text), image_count=0),
        ],
        full_text=contract_text + "\n" + artifact_text,
        total_words=len((contract_text + " " + artifact_text).split()),
        total_chars=len(contract_text) + len(artifact_text),
    )

    chunks = LegalChunker(chunk_size=200, chunk_overlap=20).chunk_document(document)

    assert chunks
    assert {chunk.page_number for chunk in chunks} == {1}


def test_legal_query_expansion_retrieves_named_contracting_entities():
    cover = DocumentChunk(
        "cover", 1,
        "Customer: Meridian Retail Technologies Pvt. Ltd. Service Provider: Northstar Cloud Systems Pvt. Ltd.",
        "contract.pdf", 100,
    )
    signature = DocumentChunk(
        "signature", 9,
        "The Parties have caused this Agreement to be executed by authorized representatives.",
        "contract.pdf", 85,
    )
    unrelated = [
        DocumentChunk(
            f"other-{index}", index + 2,
            "Technical security controls and operational service requirements.",
            "contract.pdf", 65,
        )
        for index in range(5)
    ]

    results = BM25Retriever([signature, cover, *unrelated]).search(
        "Who are the named parties entering into this agreement?",
        top_k=2,
    )

    assert "customer service provider" in expand_legal_query("Who are the named parties?").lower()
    assert results[0][0].page_number == 1


def test_page_level_rankings_are_deduplicated_in_first_seen_order():
    assert unique_ranked_pages([4, 4, None, 2, 4, 7, 2]) == [4, 2, 7]


def test_benchmark_separates_contract_relevance_from_absence_queries():
    dataset = json.loads((BACKEND_DIR / "benchmark_dataset.json").read_text(encoding="utf-8"))
    answerable = [item for item in dataset if item.get("answerable", True)]
    unanswerable = [item for item in dataset if not item.get("answerable", True)]

    assert len(dataset) == 50
    assert len(answerable) == 42
    assert len(unanswerable) == 8
    assert all(item["expected_pages"] for item in answerable)
    assert all(not item["expected_pages"] for item in unanswerable)
    assert all(1 <= page <= 9 for item in answerable for page in item["expected_pages"])


@pytest.mark.parametrize(
    "response",
    [
        '{"status": "ok"}',
        '```json\n{"status": "ok"}\n```',
        'Analysis follows.\n{"status": "ok"}\nEnd of response.',
        '<think>internal reasoning</think>\n{"status": "ok"}',
    ],
)
def test_llm_json_parser_accepts_wrapped_json(response):
    assert parse_llm_json(response) == {"status": "ok"}


def test_llm_json_shape_validation():
    assert parse_llm_json_object('{"status": "ok"}') == {"status": "ok"}
    assert parse_llm_json_array('[{"status": "ok"}]') == [{"status": "ok"}]
    with pytest.raises(ValueError, match="not an object"):
        parse_llm_json_object("[]")
    with pytest.raises(ValueError, match="not an array"):
        parse_llm_json_array("{}")


def test_llm_invocation_retries_one_rate_limit(monkeypatch):
    client = RateLimitedOnceClient()
    reviewer = LegalReviewer(make_document(), None)
    reviewer._llm = client
    monkeypatch.setattr(legal_reviewer_module.time, "sleep", lambda _: None)

    response = reviewer._invoke_llm([], max_tokens=321)

    assert response == '{"status": "ok"}'
    assert client.calls == 2
    assert client.bound_max_tokens == 321


def test_llm_invocation_falls_back_after_empty_provider_response():
    groq = StaticResponseClient(content="")
    gemini = StaticResponseClient(content='{"status": "ok"}')
    reviewer = LegalReviewer(make_document(), None)
    reviewer._llm_candidates = [("Groq", groq), ("Gemini", gemini)]

    response = reviewer._invoke_llm([], max_tokens=500)

    assert response == '{"status": "ok"}'
    assert groq.calls == 1
    assert gemini.calls == 1
    assert reviewer.last_llm_diagnostic == "Fallback provider Gemini succeeded after Groq (ValueError)."


def test_summary_generation_uses_provider_response(monkeypatch):
    summary = "1. **Document Title & Type**: Test Agreement [Page 1]"
    client = StaticResponseClient(content=summary)
    reviewer = LegalReviewer(make_document(), None)
    reviewer._llm = client
    monkeypatch.setattr(legal_reviewer_module, "check_api_key_configured", lambda: True)

    assert reviewer.generate_executive_summary() == summary
    assert client.calls == 1
    assert client.bound_max_tokens == 1400


def test_heatmap_uses_one_structured_llm_request(monkeypatch):
    source = Document(
        page_content="The agreement limits liability and requires thirty days notice.",
        metadata={"page_number": 1, "rrf_score": 0.031},
    )
    items = [
        {
            "clause_title": title,
            "risk_level": "MODERATE",
            "page": 1,
            "excerpt": source.page_content,
            "why_risky": "The clause requires human review.",
            "recommendation": "Confirm the allocation with counsel.",
        }
        for title in (
            "Parties purpose and scope",
            "Rights duties payment and deadlines",
            "Exceptions liability and remedies",
            "Term termination and dispute clauses",
        )
    ]
    client = StaticResponseClient(content=json.dumps(items))
    reviewer = LegalReviewer(
        make_document(text=source.page_content),
        None,
        hybrid_retriever=FakeHybridRetriever([source]),
    )
    reviewer._llm = client
    monkeypatch.setattr(legal_reviewer_module, "check_api_key_configured", lambda: True)

    heatmap = reviewer.generate_clause_risk_heatmap()

    assert len(heatmap) == 4
    assert client.calls == 1
    assert client.bound_max_tokens == 2600
    assert all(item["risk_level"] == "MODERATE" for item in heatmap)


def test_hybrid_retriever_calls_cross_encoder_and_exposes_scores():
    chunks = [
        DocumentChunk("a", 1, "payment is due in thirty days", "contract.pdf", 29),
        DocumentChunk("b", 2, "termination requires written notice", "contract.pdf", 35),
    ]
    dense_documents = [
        Document(page_content=chunk.text, metadata={"chunk_id": chunk.chunk_id, "page_number": chunk.page_number})
        for chunk in chunks
    ]
    retriever = LegalHybridRetriever(FakeVectorStore(dense_documents), chunks)
    fake_cross_encoder = FakeCrossEncoder()
    retriever.reranker._cross_encoder = fake_cross_encoder

    results = retriever.hybrid_search("written termination notice", top_k=2)

    assert fake_cross_encoder.pairs
    assert len(results) == 2
    assert "reranker_score" in results[0].metadata
    assert "rrf_score" in results[0].metadata


def test_question_list_appendix_is_not_treated_as_answer_evidence():
    appendix = "• What are the audit rights?\n• Who owns the data?\n• Where are disputes heard?"
    clause = "Customer may conduct one audit per year on thirty days written notice."
    assert is_question_list_chunk(appendix)
    assert not is_question_list_chunk(clause)

    chunks = [
        DocumentChunk("appendix", 11, appendix, "contract.pdf", len(appendix)),
        DocumentChunk("audit", 7, clause, "contract.pdf", len(clause)),
    ]
    dense_documents = [
        Document(page_content=chunk.text, metadata={"chunk_id": chunk.chunk_id, "page_number": chunk.page_number})
        for chunk in chunks
    ]
    retriever = LegalHybridRetriever(FakeVectorStore(dense_documents), chunks)
    retriever.reranker._cross_encoder = FakeCrossEncoder()

    results = retriever.hybrid_search("What are the audit rights?", top_k=2)

    assert results
    assert all(result.metadata["page_number"] != 11 for result in results)


def test_reviewer_qa_uses_hybrid_results(monkeypatch):
    source = Document(
        page_content="The Buyer shall pay within 30 days.",
        metadata={"page_number": 1, "rrf_score": 0.031},
    )
    hybrid = FakeHybridRetriever([source])
    reviewer = LegalReviewer(make_document(), None, hybrid_retriever=hybrid)

    response = reviewer.answer_query("When is payment due?")

    assert hybrid.last_query == "When is payment due?"
    assert response["citations"] == [1]
    assert response["sources"][0]["score_type"] == "rrf"


def test_offline_comparison_reports_real_text_changes_only():
    first = LegalReviewer(make_document("v1.pdf", "Payment is due in 30 days."), None)
    second = LegalReviewer(make_document("v2.pdf", "Payment is due in 15 days."), None)

    result = LegalReviewer.compare_contracts(first, second)

    assert result["doc1_name"] == "v1.pdf"
    assert result["doc2_name"] == "v2.pdf"
    assert result["total_changes_detected"] > 0
    combined = " ".join(
        item["v1_text"] + " " + item["v2_text"] for item in result["comparison_table"]
    )
    assert "30" in combined and "15" in combined
    assert all(item["impact"] == "REVIEW REQUIRED" for item in result["comparison_table"])


def test_offline_classifier_prefers_document_title_over_incidental_clauses():
    msa_text = "MASTER SERVICES AGREEMENT\nThe parties shall protect confidential information."
    nda_text = "MUTUAL NON-DISCLOSURE AGREEMENT\nConfidentiality obligations apply."

    assert LegalReviewer(make_document(text=msa_text), None).classify_document()["doc_type"] == "Master Services Agreement (MSA)"
    assert LegalReviewer(make_document(text=nda_text), None).classify_document()["doc_type"] == "Non-Disclosure Agreement (NDA)"


def test_raw_upload_rejects_non_pdf_content():
    handler = object.__new__(LegalAPIRequestHandler)
    handler.headers = Message()
    handler.headers["Content-Type"] = "application/pdf"
    handler.headers["Content-Length"] = "12"
    handler.rfile = BytesIO(b"not a pdf!!!")

    with pytest.raises(RequestError, match="valid PDF"):
        handler._read_uploaded_pdf()


def test_api_sessions_are_isolated_and_compare_uses_second_upload(monkeypatch):
    def fake_build_reviewer(file_data, filename, source_language="auto"):
        text = file_data.decode("latin-1")
        return LegalReviewer(make_document(filename, text), None), 1

    monkeypatch.setattr(
        LegalAPIRequestHandler,
        "_build_reviewer",
        staticmethod(fake_build_reviewer),
    )
    SESSION_DATA.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), LegalAPIRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    first_browser = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
    second_browser = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))

    try:
        upload = urllib.request.Request(
            f"{base_url}/api/upload",
            data=b"%PDF-version-one",
            headers={"Content-Type": "application/pdf", "X-Upload-Filename": "v1.pdf"},
            method="POST",
        )
        assert json.load(first_browser.open(upload))["file_name"] == "v1.pdf"

        with pytest.raises(urllib.error.HTTPError) as isolated_error:
            second_browser.open(f"{base_url}/api/obligations")
        assert isolated_error.value.code == 400

        compare = urllib.request.Request(
            f"{base_url}/api/compare",
            data=b"%PDF-version-two",
            headers={"Content-Type": "application/pdf", "X-Upload-Filename": "v2.pdf"},
            method="POST",
        )
        comparison = json.load(first_browser.open(compare))
        assert comparison["doc1_name"] == "v1.pdf"
        assert comparison["doc2_name"] == "v2.pdf"
        assert comparison["total_changes_detected"] > 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
