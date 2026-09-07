import os
import sys
import time
import json
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Ensure UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Setup backend package path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None

from langchain_core.documents import Document
from core.document_loader import LegalDocumentLoader
from core.chunker import LegalChunker
from core.vector_store import LegalVectorStore
from core.hybrid_retriever import LegalHybridRetriever, BM25Retriever


def calculate_dcg(relevance_scores: List[int], k: int) -> float:
    """Calculates Discounted Cumulative Gain (DCG@K)."""
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k], start=1):
        dcg += (2**rel - 1) / math.log2(i + 1)
    return dcg

def calculate_ndcg(retrieved_pages: List[int], expected_pages: List[int], k: int) -> float:
    """Calculates Normalized Discounted Cumulative Gain (nDCG@K)."""
    rel = [1 if page in expected_pages else 0 for page in retrieved_pages[:k]]
    dcg = calculate_dcg(rel, k)
    ideal_rel = [1] * min(len(set(expected_pages)), k)
    idcg = calculate_dcg(ideal_rel, k)
    return round(dcg / idcg, 4) if idcg > 0 else 0.0


def unique_ranked_pages(pages: List[int]) -> List[int]:
    """Deduplicate chunk results for page-level relevance evaluation."""
    unique = []
    seen = set()
    for page in pages:
        if page is None or page in seen:
            continue
        seen.add(page)
        unique.append(page)
    return unique

class RigorousLegalEvaluator:
    """
    Research-Grade Evaluation Framework & Ablation Study Suite.
    Runs a 5-Stage retrieval ablation over the answerable benchmark queries:
      - Stage A: Dense Vector Only (ChromaDB)
      - Stage B: BM25 Lexical Only
      - Stage C: Hybrid (Dense + BM25)
      - Stage D: Hybrid + RRF (Reciprocal Rank Fusion)
      - Stage E: Hybrid + RRF + Cross-Encoder Reranker
    """

    def __init__(self, sample_pdf_path: str, dataset_path: str):
        self.sample_pdf_path = sample_pdf_path
        self.dataset_path = dataset_path

    def run_ablation_study(self) -> Dict[str, Any]:
        if HAS_RICH:
            console.print(Panel("[bold cyan]📊 RESEARCH-GRADE LEGAL RAG EVALUATION & ABLATION STUDY (50 QUERIES)[/bold cyan]", border_style="cyan"))
        else:
            print("\n" + "=" * 70)
            print("  RESEARCH-GRADE LEGAL RAG EVALUATION & ABLATION STUDY (50 QUERIES)")
            print("=" * 70)

        # Load Document & Build Indexes
        t_start = time.time()
        loader = LegalDocumentLoader()
        doc = loader.load_pdf(self.sample_pdf_path)
        ingest_time = time.time() - t_start

        t_start = time.time()
        chunker = LegalChunker(chunk_size=1000, chunk_overlap=200)
        chunks = chunker.chunk_document(doc)
        chunk_time = time.time() - t_start

        t_start = time.time()
        vector_store = LegalVectorStore()
        vector_store.build_index(chunks)
        hybrid_retriever = LegalHybridRetriever(vector_store=vector_store, chunks=chunks)
        index_time = time.time() - t_start

        # Load the full benchmark, then keep absence queries separate from IR metrics.
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        evaluation_items = [
            item for item in dataset
            if item.get("answerable", True) and item.get("expected_pages")
        ]
        unanswerable_count = len(dataset) - len(evaluation_items)

        stages = ["Dense Only", "BM25 Only", "Hybrid", "Hybrid + RRF", "Hybrid + Reranker"]
        metrics_by_stage = {
            s: {
                "r1": [], "r3": [], "r5": [], "r10": [],
                "mrr": [], "ndcg5": [], "latency": []
            } for s in stages
        }
        misses_at_5 = {stage: [] for stage in stages}

        for item in evaluation_items:
            query = item["query"]
            expected_pages = item.get("expected_pages", [])

            # --- Stage A: Dense Only ---
            t0 = time.time()
            dense_docs = vector_store.similarity_search(query, k=10)
            t_dense = (time.time() - t0) * 1000
            dense_pages = unique_ranked_pages([d.metadata.get("page_number") for d in dense_docs])

            # --- Stage B: BM25 Only ---
            t0 = time.time()
            bm25_scored = hybrid_retriever.bm25_retriever.search(query, top_k=10)
            t_bm25 = (time.time() - t0) * 1000
            bm25_pages = unique_ranked_pages([c.page_number for c, _ in bm25_scored])

            # --- Stage C: Hybrid candidate union without rank fusion ---
            t0 = time.time()
            hybrid_dense = vector_store.similarity_search(query, k=5)
            hybrid_bm25 = hybrid_retriever.bm25_retriever.search(query, top_k=5)
            combined = hybrid_dense + [
                Document(
                    page_content=c.text,
                    metadata={"page_number": c.page_number, "chunk_id": c.chunk_id},
                )
                for c, _ in hybrid_bm25
            ]
            seen_chunk_ids = set()
            hybrid_raw = []
            for candidate in combined:
                chunk_id = candidate.metadata.get("chunk_id", candidate.page_content)
                if chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    hybrid_raw.append(candidate)
                if len(hybrid_raw) == 10:
                    break
            t_hybrid = (time.time() - t0) * 1000
            hybrid_pages = unique_ranked_pages([d.metadata.get("page_number") for d in hybrid_raw])

            # --- Stage D: Hybrid + RRF ---
            t0 = time.time()
            rrf_docs = hybrid_retriever.hybrid_search(query, top_k=10, use_reranker=False)
            t_rrf = (time.time() - t0) * 1000
            rrf_pages = unique_ranked_pages([d.metadata.get("page_number") for d in rrf_docs])

            # --- Stage E: Hybrid + RRF + Cross-Encoder Reranker ---
            t0 = time.time()
            reranked_docs = hybrid_retriever.hybrid_search(query, top_k=10, use_reranker=True)
            t_rerank = (time.time() - t0) * 1000
            reranked_pages = unique_ranked_pages([d.metadata.get("page_number") for d in reranked_docs])

            stage_pages_map = {
                "Dense Only": (dense_pages, t_dense),
                "BM25 Only": (bm25_pages, t_bm25),
                "Hybrid": (hybrid_pages, t_hybrid),
                "Hybrid + RRF": (rrf_pages, t_rrf),
                "Hybrid + Reranker": (reranked_pages, t_rerank),
            }

            for s_name, (pages, lat) in stage_pages_map.items():
                m = metrics_by_stage[s_name]
                expected_set = set(expected_pages)
                denominator = max(1, len(expected_set))
                m["r1"].append(len(set(pages[:1]) & expected_set) / denominator)
                m["r3"].append(len(set(pages[:3]) & expected_set) / denominator)
                m["r5"].append(len(set(pages[:5]) & expected_set) / denominator)
                m["r10"].append(len(set(pages[:10]) & expected_set) / denominator)
                
                # MRR
                mrr_val = 0.0
                for rank_idx, p in enumerate(pages, start=1):
                    if p in expected_pages:
                        mrr_val = 1.0 / rank_idx
                        break
                m["mrr"].append(mrr_val)
                m["ndcg5"].append(calculate_ndcg(pages, expected_pages, 5))
                m["latency"].append(lat)
                missing_pages = sorted(expected_set - set(pages[:5]))
                if missing_pages:
                    misses_at_5[s_name].append({
                        "id": item.get("id"),
                        "query": query,
                        "expected_pages": expected_pages,
                        "missing_pages": missing_pages,
                        "ranked_pages": pages[:10],
                    })

        # Aggregate Averages
        summary_results = []
        for s_name in stages:
            m = metrics_by_stage[s_name]
            n = len(evaluation_items)
            summary_results.append({
                "stage": s_name,
                "recall1": f"{round((sum(m['r1']) / n) * 100, 1)}%",
                "recall3": f"{round((sum(m['r3']) / n) * 100, 1)}%",
                "recall5": f"{round((sum(m['r5']) / n) * 100, 1)}%",
                "recall10": f"{round((sum(m['r10']) / n) * 100, 1)}%",
                "mrr": round(sum(m["mrr"]) / n, 3),
                "ndcg5": round(sum(m["ndcg5"]) / n, 3),
                "latency_ms": f"{round(sum(m['latency']) / n, 2)} ms",
            })
        stage_summary = {result["stage"]: result for result in summary_results}

        # Display Ablation Summary Table
        if HAS_RICH:
            table = Table(
                title=f"🔬 5-Stage Retrieval Ablation Study ({len(evaluation_items)} Answerable Queries)",
                header_style="bold cyan",
            )
            table.add_column("Retrieval Pipeline Stage", style="white")
            table.add_column("Recall@1", justify="center", style="yellow")
            table.add_column("Recall@3", justify="center", style="yellow")
            table.add_column("Recall@5", justify="center", style="green")
            table.add_column("Recall@10", justify="center", style="green")
            table.add_column("MRR", justify="center", style="magenta")
            table.add_column("nDCG@5", justify="center", style="cyan")
            table.add_column("Mean Latency", justify="right", style="blue")

            for sr in summary_results:
                table.add_row(
                    sr["stage"],
                    sr["recall1"],
                    sr["recall3"],
                    sr["recall5"],
                    sr["recall10"],
                    str(sr["mrr"]),
                    str(sr["ndcg5"]),
                    sr["latency_ms"],
                )
            console.print(table)
        else:
            print(f"\n--- 5-Stage Retrieval Ablation Study ({len(evaluation_items)} Answerable Queries) ---")
            for sr in summary_results:
                print(f"Stage: {sr['stage']} | Recall@5: {sr['recall5']} | MRR: {sr['mrr']} | nDCG@5: {sr['ndcg5']} | Latency: {sr['latency_ms']}")

        best_stage_misses = misses_at_5["Hybrid + Reranker"]
        if best_stage_misses:
            print("\nHybrid + Reranker misses at page-level Recall@5:")
            for miss in best_stage_misses:
                print(
                    f"- Query {miss['id']}: expected {miss['expected_pages']}, "
                    f"ranked {miss['ranked_pages']} — {miss['query']}"
                )

        # Export Markdown Report to docs/ablation_study.md
        docs_dir = BASE_DIR.parent / "docs"
        docs_dir.mkdir(exist_ok=True)
        report_path = docs_dir / "ablation_study.md"

        ablation_md = f"""# 🔬 Research-Grade 5-Stage Retrieval Ablation Study

> **Evaluation Dataset:** {len(dataset)} Ground-Truth Legal Queries ([backend/benchmark_dataset.json](../backend/benchmark_dataset.json)); {len(evaluation_items)} answerable queries used for page-retrieval metrics and {unanswerable_count} negative/absence queries reserved for abstention evaluation.  
> **Document Evaluated:** `{doc.file_name}` ({doc.total_words:,} Words, {len(chunks)} Semantic Chunks)  
> **Evaluation Metric Standards:** Page-deduplicated Recall@K (K=1,3,5,10), MRR (Mean Reciprocal Rank), nDCG@5, and Mean Latency.

---

## 📌 Executive Summary

This empirical study compares 5 distinct retrieval architectures to demonstrate the contribution of **Dense Vector Embeddings**, **BM25 Lexical Matching**, **Reciprocal Rank Fusion (RRF)**, and **Cross-Encoder Reranking** in legal contract question answering.

---

## 📊 Empirical Ablation Results ({len(evaluation_items)} Answerable Queries)

| Retrieval Pipeline Stage | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | nDCG@5 | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
        for sr in summary_results:
            ablation_md += f"| **{sr['stage']}** | {sr['recall1']} | {sr['recall3']} | **{sr['recall5']}** | {sr['recall10']} | **{sr['mrr']}** | **{sr['ndcg5']}** | `{sr['latency_ms']}` |\n"

        miss_lines = "\n".join(
            f"- Query {miss['id']}: expected pages {miss['expected_pages']}; "
            f"ranked pages {miss['ranked_pages']} — {miss['query']}"
            for miss in best_stage_misses
        ) or "- No page-level Recall@5 misses."

        ablation_md += f"""
---

## Failure Audit for Hybrid + Reranker

{miss_lines}

---

## 💡 Key Architectural Insights & Findings

1. **Leakage prevention:** Post-signature benchmark notes and question lists are excluded before indexing, so retrieval is measured only against contract language.
2. **Recall-first profile:** Hybrid + RRF reaches **{stage_summary['Hybrid + RRF']['recall5']} Recall@5** and **{stage_summary['Hybrid + RRF']['recall10']} Recall@10** at {stage_summary['Hybrid + RRF']['latency_ms']} mean latency.
3. **Precision-first profile:** Cross-encoder reranking reaches **{stage_summary['Hybrid + Reranker']['recall1']} Recall@1**, **{stage_summary['Hybrid + Reranker']['mrr']} MRR**, and **{stage_summary['Hybrid + Reranker']['ndcg5']} nDCG@5**, with the documented latency/recall tradeoff.
4. **Lexical coverage:** Conservative legal query expansion improves BM25 candidate generation for clause terminology and contract metadata without changing the source document.
5. **Metric scope:** Recall@K is the fraction of unique expected pages retrieved. The {unanswerable_count} negative queries are kept out of IR recall and reserved for a separate abstention evaluation.
"""

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(ablation_md)

        print(f"\n✅ Ablation Study Report successfully exported to: {report_path}")
        return {
            "summary": summary_results,
            "misses_at_5": misses_at_5,
            "report_path": str(report_path),
        }


if __name__ == "__main__":
    possible_msas = [
        BASE_DIR.parent / "legal_ai_evaluation_msa.pdf",
        BASE_DIR / "legal_ai_evaluation_msa.pdf",
        BASE_DIR.parent / "Mutual-NDA-public-2page.pdf",
        BASE_DIR / "Mutual-NDA-public-2page.pdf"
    ]
    pdf_path = None
    for p in possible_msas:
        if p.exists():
            pdf_path = p
            break

    json_path = BASE_DIR / "benchmark_dataset.json"

    if pdf_path and pdf_path.exists() and json_path.exists():
        evaluator = RigorousLegalEvaluator(str(pdf_path), str(json_path))
        evaluator.run_ablation_study()
    else:
        print(f"❌ Prerequisites missing. PDF: {pdf_path}, Dataset: {json_path.exists()}")
