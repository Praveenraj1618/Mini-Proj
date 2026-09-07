# Retrieval Metrics Improvement Report

> **Document:** `legal_ai_evaluation_msa.pdf`  
> **Benchmark:** 50 legal questions: 42 answerable retrieval queries and 8 negative/absence queries  
> **Generated:** August 30, 2026  
> **Detailed current benchmark:** [ablation_study.md](ablation_study.md)

## Executive Summary

The retrieval evaluation was repaired and the retrieval pipeline was improved. The current recall-first configuration, **Hybrid + Reciprocal Rank Fusion (RRF)**, reaches **100.0% Recall@5 and Recall@10**. The precision-first cross-encoder configuration reaches **84.5% Recall@1, 0.905 MRR, and 0.914 nDCG@5**.

The before-and-after figures are not a pure model-speed comparison. The previous benchmark contained incorrect page labels and allowed post-signature evaluation notes into the search index. The current benchmark uses corrected contract-page ground truth, excludes those artifacts, deduplicates pages, and separates absence questions from retrieval recall. The new figures are therefore both higher and more trustworthy.

## Previous Documented Metrics

These were the metrics recorded before the benchmark and retrieval corrections.

| Retrieval Stage | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | nDCG@5 | Mean Latency |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense Only | 4.0% | 4.0% | 6.0% | 19.0% | 0.086 | 0.061 | 13.63 ms |
| BM25 Only | 4.0% | 8.0% | 22.0% | 52.0% | 0.168 | 0.154 | 0.10 ms |
| Hybrid candidate union | 4.0% | 4.0% | 6.0% | 24.0% | 0.103 | 0.061 | 11.89 ms |
| Hybrid + RRF | 4.0% | 9.0% | 14.0% | 29.0% | 0.134 | 0.105 | 12.03 ms |
| Hybrid + Cross-Encoder | 4.0% | 9.0% | 14.0% | 29.0% | 0.134 | 0.105 | 15.77 ms |

## Current Metrics

The current retrieval metrics use 42 answerable queries. The 8 absence queries are reserved for a separate abstention evaluation because they have no relevant page to retrieve.

| Retrieval Stage | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | nDCG@5 | Mean Latency |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense Only | 69.0% | 91.7% | 95.2% | 95.2% | 0.806 | 0.844 | 15.62 ms |
| BM25 Only | 78.6% | 94.0% | 97.6% | 97.6% | 0.875 | 0.901 | 0.29 ms |
| Hybrid candidate union | 69.0% | 91.7% | **100.0%** | **100.0%** | 0.817 | 0.864 | 15.33 ms |
| Hybrid + RRF | 81.0% | 94.0% | **100.0%** | **100.0%** | 0.884 | 0.913 | 16.48 ms |
| Hybrid + Cross-Encoder | **84.5%** | 94.0% | 95.2% | 95.2% | **0.905** | **0.914** | 477.28 ms |

## Before-versus-Now Improvement

Values for Recall are absolute percentage-point changes. MRR and nDCG changes are absolute score changes.

| Retrieval Stage | Recall@1 Change | Recall@5 Change | Recall@10 Change | MRR Change | nDCG@5 Change |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Dense Only | +65.0 pp | +89.2 pp | +76.2 pp | +0.720 | +0.783 |
| BM25 Only | +74.6 pp | +75.6 pp | +45.6 pp | +0.707 | +0.747 |
| Hybrid candidate union | +65.0 pp | +94.0 pp | +76.0 pp | +0.714 | +0.803 |
| Hybrid + RRF | +77.0 pp | +86.0 pp | +71.0 pp | +0.750 | +0.808 |
| Hybrid + Cross-Encoder | +80.5 pp | +81.2 pp | +66.2 pp | +0.771 | +0.809 |

## Improvements Made

1. **Corrected benchmark ground truth**
   - The previous dataset pointed most expected answers to pages 1 and 2 even though relevant clauses occur across contract pages 1–9.
   - Each answerable query now references its actual contract page or pages.

2. **Removed evaluation leakage**
   - Pages 10 and 11 contain evaluation notes, suggested questions, and answer hints rather than contract terms.
   - These pages are detected and excluded before chunk indexing and retrieval.
   - The index now contains 28 contract chunks instead of 32 mixed contract-and-evaluation chunks.

3. **Added conservative legal query expansion**
   - BM25 candidate generation now recognizes closely related contract language for parties, payment, termination, dispute resolution, intellectual property, confidentiality, liability, and notices.
   - Expansion affects search terms only; it does not alter source documents or generated answers.

4. **Corrected page-level metric calculation**
   - Multiple chunks from the same page are deduplicated in their original rank order.
   - Recall@K now measures the fraction of unique relevant pages retrieved instead of a mislabeled binary hit score.

5. **Separated answerability from retrieval**
   - The 42 answerable questions are used for Recall, MRR, and nDCG.
   - The 8 questions about absent clauses are explicitly labeled unanswerable and reserved for abstention/hallucination testing.

6. **Added failure auditing and regression protection**
   - The evaluator reports the exact questions and expected pages missed at Recall@5.
   - Automated tests verify artifact filtering, query expansion, page deduplication, benchmark counts, and valid contract-page labels.
   - Current validation result: **22 tests passed**.

7. **Evaluated and rejected an ineffective blend**
   - A weighted RRF/cross-encoder score blend was tested.
   - It reduced early-ranking quality without fixing the remaining cover-page misses, so it was removed rather than included merely to claim another improvement.

## Recommended Runtime Profiles

### Recall-first: Hybrid + RRF

Use this profile when missing a relevant contract page is the larger risk. It provides **100.0% Recall@5/10** with approximately **16.48 ms** mean retrieval latency.

### Precision-first: Hybrid + Cross-Encoder

Use this profile when the first result matters most. It provides the best **Recall@1, MRR, and nDCG@5**, but its approximately **477.28 ms** latency is much higher and it misses two cover-page metadata queries within the first five unique pages.

## Remaining Work

- Add an abstention benchmark for the 8 negative queries, measuring false-answer rate and correct refusal rate.
- Evaluate a legal-domain reranker using a separate validation set before changing the production reranking model.
- Expand the benchmark to multiple contract types so results are not dependent on one MSA.

## Related Files

- [Benchmark dataset](../backend/benchmark_dataset.json)
- [Evaluator](../backend/evaluator.py)
- [Hybrid retriever](../backend/core/hybrid_retriever.py)
- [Artifact filtering](../backend/core/chunker.py)
- [Regression tests](../backend/tests/test_core.py)
- [Current ablation report](ablation_study.md)
