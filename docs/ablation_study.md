# 🔬 Research-Grade 5-Stage Retrieval Ablation Study

> **Evaluation Dataset:** 50 Ground-Truth Legal Queries ([backend/benchmark_dataset.json](../backend/benchmark_dataset.json)); 42 answerable queries used for page-retrieval metrics and 8 negative/absence queries reserved for abstention evaluation.  
> **Document Evaluated:** `legal_ai_evaluation_msa.pdf` (3,320 Words, 28 Semantic Chunks)  
> **Evaluation Metric Standards:** Page-deduplicated Recall@K (K=1,3,5,10), MRR (Mean Reciprocal Rank), nDCG@5, and Mean Latency.

---

## 📌 Executive Summary

This empirical study compares 5 distinct retrieval architectures to demonstrate the contribution of **Dense Vector Embeddings**, **BM25 Lexical Matching**, **Reciprocal Rank Fusion (RRF)**, and **Cross-Encoder Reranking** in legal contract question answering.

---

## 📊 Empirical Ablation Results (42 Answerable Queries)

| Retrieval Pipeline Stage | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | nDCG@5 | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Only** | 69.0% | 91.7% | **95.2%** | 95.2% | **0.806** | **0.844** | `15.62 ms` |
| **BM25 Only** | 78.6% | 94.0% | **97.6%** | 97.6% | **0.875** | **0.901** | `0.29 ms` |
| **Hybrid** | 69.0% | 91.7% | **100.0%** | 100.0% | **0.817** | **0.864** | `15.33 ms` |
| **Hybrid + RRF** | 81.0% | 94.0% | **100.0%** | 100.0% | **0.884** | **0.913** | `16.48 ms` |
| **Hybrid + Reranker** | 84.5% | 94.0% | **95.2%** | 95.2% | **0.905** | **0.914** | `477.28 ms` |

---

## Failure Audit for Hybrid + Reranker

- Query 21: expected pages [1]; ranked pages [9, 5, 7, 8, 4] — Who are the named parties entering into this agreement?
- Query 22: expected pages [1]; ranked pages [9, 4, 2, 5] — What entity type or state of incorporation is listed for the parties?

---

## 💡 Key Architectural Insights & Findings

1. **Leakage prevention:** Post-signature benchmark notes and question lists are excluded before indexing, so retrieval is measured only against contract language.
2. **Recall-first profile:** Hybrid + RRF reaches **100.0% Recall@5** and **100.0% Recall@10** at 16.48 ms mean latency.
3. **Precision-first profile:** Cross-encoder reranking reaches **84.5% Recall@1**, **0.905 MRR**, and **0.914 nDCG@5**, with the documented latency/recall tradeoff.
4. **Lexical coverage:** Conservative legal query expansion improves BM25 candidate generation for clause terminology and contract metadata without changing the source document.
5. **Metric scope:** Recall@K is the fraction of unique expected pages retrieved. The 8 negative queries are kept out of IR recall and reserved for a separate abstention evaluation.
