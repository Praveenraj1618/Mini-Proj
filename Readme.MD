# ⚖️ Legal AI Intelligence & Contract Negotiation Platform

> **Domain:** Artificial Intelligence / Machine Learning – Natural Language Processing (NLP) & LegalTech  
> **Subtitle:** *A multimodal, retrieval-augmented system for contract analysis, risk detection, evidence-grounded question answering, obligation extraction, and negotiation assistance.*  
> **Target Document Evaluated:** `legal_ai_evaluation_msa.pdf` (11 Pages | 3,320 Words | 28 indexed contract chunks; post-signature evaluation notes excluded)  
> **Architecture:** Python JSON API backend with an independent HTML5 frontend.

---

## 📌 Executive Summary & Architecture

This project delivers a **Research-Grade Legal AI & Contract Negotiation Platform** featuring:
1. **Multi-Stage Hybrid Retrieval Engine**: Dense vector search (ChromaDB + `all-MiniLM-L6-v2`) and BM25 lexical search, fused with Reciprocal Rank Fusion (RRF) and reranked by `cross-encoder/ms-marco-MiniLM-L-6-v2` when the model is available locally or can be downloaded.
2. **Automated Document Classification**: Identifies contract types (NDA, Lease, MSA, Employment, Litigation) and tailors target risk focus rulesets.
3. **Clause-Level Risk Heatmaps**: 8-category section breakdown with risk ratings (`🔴 HIGH`, `🟡 MODERATE`, `🟢 LOW`), why it is risky, and actionable recommendations.
4. **Contract Negotiation Assistant**: Evaluates clause fairness, identifies potential client exposure, and drafts **safer, balanced replacement clause drafts**.
5. **Contract Version Comparison Engine ("What's Changed?")**: Multi-document diff engine comparing 2 contract versions (`v1.pdf` vs `v2.pdf`) with legal impact scoring.
6. **Structured Obligation Extractor**: Parses duties, deadlines, payment schedules, and penalties into structured JSON schemas.
7. **Empirical 5-Stage Ablation Study**: Tested on `legal_ai_evaluation_msa.pdf` across 42 answerable retrieval queries, with 8 absence queries reserved for abstention evaluation.
8. **Decoupled Frontend Layer**: REST API backend exposing clean JSON endpoints for **React.js**, **Spring Boot**, **Vue.js**, or static HTML5 Web UI.

---

## 🔬 Research-Grade 5-Stage Retrieval Ablation Study (`legal_ai_evaluation_msa.pdf`)

| Retrieval Pipeline Stage | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | nDCG@5 | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage A: Dense Only** | 69.0% | 91.7% | 95.2% | 95.2% | 0.806 | 0.844 | `15.62 ms` |
| **Stage B: BM25 Only** | 78.6% | 94.0% | 97.6% | 97.6% | 0.875 | 0.901 | `0.29 ms` |
| **Stage C: Hybrid candidate union** | 69.0% | 91.7% | **100.0%** | **100.0%** | 0.817 | 0.864 | `15.33 ms` |
| **Stage D: Hybrid + RRF** | 81.0% | 94.0% | **100.0%** | **100.0%** | 0.884 | 0.913 | `16.48 ms` |
| **Stage E: Hybrid + Cross-Encoder** | **84.5%** | 94.0% | 95.2% | 95.2% | **0.905** | **0.914** | `477.28 ms` |

*The evaluation dataset is available at [backend/benchmark_dataset.json](backend/benchmark_dataset.json), with detailed results in [docs/ablation_study.md](docs/ablation_study.md).*

### 💡 Empirical Insights & Observations
1. **Leakage-resistant evaluation:** Pages containing benchmark hints and question lists are excluded before indexing, and relevance labels point only to the nine contract pages.
2. **Recall-first profile:** Hybrid + RRF retrieves every expected page by rank 5 at about 16 ms mean latency.
3. **Precision-first profile:** The cross-encoder improves Recall@1, MRR, and nDCG@5, but costs substantially more latency and misses two cover-page metadata queries at Recall@5.
4. **Correct metric scope:** Recall@K uses unique expected pages. The 8 absence queries are not incorrectly scored as retrieval failures; they are reserved for abstention evaluation.

---

## 📂 Project Structure

```
Mini-Proj-main/
├── legal_ai_evaluation_msa.pdf    # Target Evaluation Contract (11 Pages, 3,320 Words)
├── backend/                       # 🧠 Python Backend REST API Engine
│   ├── core/                      # RAG Core Modules (Loader, Chunker, ChromaDB, BM25, Reviewer)
│   ├── benchmark_dataset.json     # 50-Query Ground-Truth Evaluation Dataset
│   ├── app_api.py                 # Decoupled REST API Server (http://localhost:8080)
│   ├── config.py                  # Model & Hybrid Retrieval configuration
│   ├── evaluator.py               # 5-Stage Ablation Benchmark Evaluator
│   └── test_core.py               # End-to-end pipeline verification test suite
│
├── frontend/                      # 🎨 Standalone Frontend Layer
│   ├── index.html                 # Web Dashboard UI with Legal Disclaimer & Risk Cards
│   └── README_FRONTEND.md         # React & Spring Boot REST API integration guide
│
├── docs/                          # 📚 Benchmark Reports & Architecture Flowcharts
│   ├── ablation_study.md          # 5-Stage Ablation Study Report (50 Queries)
│   ├── workflow_diagram.md        # Mermaid system architecture diagrams
│   ├── workflow_diagram.html      # Visual HTML workflow diagram
│   ├── p1.md                      # Initial Phase 1 notes
│   └── p2_benchmark.md            # Phase 2 hybrid benchmark report
│
├── .env.example                   # Template environment configuration
├── requirements.txt               # Backend dependencies
└── Readme.MD                      # Master project overview
```

---

## ⚡ Quick Start Guide

### 1. Environment Setup
Activate the Conda environment (`sde`):
```bash
conda activate sde
pip install -r requirements.txt
```

### 2. Start Python REST API Backend
```bash
cd backend
python app_api.py
```
The REST API server will run live on **`http://localhost:8080`**.

### 3. Open Frontend / Connect React / Spring Boot
- **Static Web UI**: Start the backend and open `http://localhost:8080`; the backend serves `frontend/index.html`.
- **React / Spring Boot**: See [frontend/README_FRONTEND.md](frontend/README_FRONTEND.md) for API client integration examples.

### 4. Run 50-Query Ablation Study & Test Suite
```bash
cd backend

# Run 50-query 5-stage retrieval ablation study on legal_ai_evaluation_msa.pdf
python evaluator.py

# Run end-to-end pipeline verification on legal_ai_evaluation_msa.pdf
python test_core.py

# Run the focused automated tests
python -m pytest tests -q
```

Never commit real credentials. Copy `.env.example` to `.env`, add your keys only to `.env`, and rotate any key that has previously appeared in version control.

The backend supports ordered provider failover through `LLM_PROVIDER_ORDER`. The recommended free-development setup is Groq first (`qwen/qwen3.8-27b`) with Google Gemini (`gemini-3.7-flash`) as a fallback. Set `GROQ_API_KEY` and/or `GEMINI_API_KEY` in the root `.env`; provider errors never expose credential values to the frontend.
