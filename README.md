# Legal AI Intelligence and Negotiation Platform

A PDF-based legal document review prototype with English, Tamil and Hindi workflows. Upload a document to inspect its category, summary, risk areas, cited Q&A, obligations, and visual pages. Contract-like categories also support negotiation drafts and version comparison. **This is a review aid, not legal advice or a verified translation.** Check every result against the cited source and have a qualified person review consequential decisions.

> **Branch note:** Multilingual support and the document-family expansion are on `feat/multilingual-document-review` (draft PR #3), not necessarily on `main`. Use that branch for the instructions below.

## What you need

- Python and pip (a virtual environment or Conda is recommended), Git, and internet access for the initial model downloads and hosted LLM calls. The frontend is plain HTML/CSS/JavaScript; there is no npm build step or separate frontend server.
- At least one hosted LLM API key for generated summaries, Q&A synthesis, structured extraction, and negotiation. Supported providers: xAI, OpenAI, Groq and Gemini. Retrieval and PDF extraction can run without a hosted LLM, but AI analysis will be unavailable or fall back to limited rule-based output.
- For scanned PDFs, Tesseract OCR with the `eng`, `tam`, and `hin` traineddata files. Native-text PDFs may work without OCR. A vision-capable provider model is additionally required for AI visual analysis; page previews do not need one.
- Memory and disk space for the multilingual embedding model and optional reranker downloaded on first use. The default maximum upload is 20 MB.

## Setup (Windows PowerShell)

Run these commands from a terminal. If you already have a checkout, enter it and switch to the feature branch; do not overwrite your private `.env`.

```powershell
git clone https://github.com/Praveenraj1618/Mini-Proj.git
cd Mini-Proj
git switch --track origin/feat/multilingual-document-review
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

If you use the existing Conda environment instead: `conda activate sde`, then install `requirements.txt`. If PowerShell blocks activation, use a Conda prompt or run `.venv\Scripts\python.exe` in place of `python`.

Edit **root** `.env` and add at least one real key. For example:

```dotenv
GROQ_API_KEY=your_private_key
GROQ_MODEL=qwen/qwen3.8-27b
LLM_PROVIDER_ORDER=groq,gemini
EMBEDDING_MODEL=intfloat/multilingual-e5-small
OCR_ENABLED=true
OCR_LANGUAGE=eng+tam+hin
OCR_TESSDATA=C:/Program Files/Tesseract-OCR/tessdata
```

`GEMINI_API_KEY` can be added as a second provider. Use a model ID available to your account and edit `LLM_PROVIDER_ORDER` to match the keys you configured. You may use just one provider. Never commit `.env` or share API keys. `HF_TOKEN` is optional and only helps Hugging Face downloads; it is not an LLM key. `LEGAL_AI_OFFLINE=true` explicitly disables hosted LLM calls.

For OCR, install Tesseract and place `eng.traineddata`, `tam.traineddata`, and `hin.traineddata` in the directory named by `OCR_TESSDATA`. Confirm with `tesseract --list-langs` (or inspect that directory if the executable is not on PATH). Select the actual PDF/OCR language in the UI before upload; Auto/mixed requires all configured packs. Missing packs produce an extraction warning, not an English-only substitute. See [the multilingual setup notes](docs/multilingual_document_review.md) for source-language and font details.

### Start the whole application

From the repository root, with your environment activated:

```powershell
cd backend
python app_api.py
```

Open **http://localhost:8080** in a browser. This one process serves both `frontend/index.html` at `/` and the JSON API at `/api/*`. Leave the terminal open; stop with Ctrl+C. A healthy server responds at **http://localhost:8080/api/status**. Upload one of the sample PDFs from the repository root, then inspect the displayed page count, OCR warnings, and source citations. The first upload may pause while the embedding model downloads. After changing `.env` model or OCR settings, restart the server and re-upload the PDF to rebuild its index.

### macOS/Linux variation

Use `python3 -m venv .venv`, `source .venv/bin/activate`, `python -m pip install -r requirements.txt`, and `cp .env.example .env`. Install Tesseract plus its English/Tamil/Hindi language data using your OS package manager, and set `OCR_TESSDATA` to the actual traineddata directory. Run `cd backend` and `python app_api.py` as above.

## First walkthrough

1. At the top of the page, choose **Response language** (English, Tamil, Hindi) and **PDF language / OCR** (the source language) before upload.
2. Upload a PDF, confirm the detected **Document category**, and override it if incorrect. Check page count and extraction warnings. The application handles PDF only.
3. Open **Summary** and **Risk Heatmap**. Ask a specific question in **Grounded Legal Q&A** and verify its cited page and original source excerpt. The answer can fail even when retrieval succeeds, for example when the provider returns HTTP 429.
4. Inspect **Obligations** for candidate duties, dates, and their source evidence. Use **Visuals & OCR** for page previews; AI image inspection needs a separately configured vision model.
5. For a contract-like document, use **Negotiation Assistant** and upload a second PDF in **Contract Comparison**. Negotiation is disabled for non-contract families such as judgments and legislation. Comparison is not a certified legal redline.
6. Use **Check AI connection** before a live demo. `/api/status` only reports that a key is configured; it does not verify quota or model access.

Changing response language or category clears prior analysis panels. Changing OCR/source language requires re-upload. Original quotations remain in the source language.

## How it is implemented

```text
PDF upload -> native text / OCR per page -> legal chunks
                                      -> multilingual embeddings -> Chroma dense index
                                      -> BM25 keyword index
question -> dense + BM25 candidates -> RRF rank fusion -> optional reranker
         -> cited source context -> configured LLM -> grounded answer + evidence
```

| Part | Code | Responsibility |
| --- | --- | --- |
| HTTP API and sessions | `backend/app_api.py` | Serves UI/API, validates PDF uploads, isolates sessions and routes requests. |
| Configuration/providers | `backend/config.py`, `.env.example` | Reads settings, builds LLM clients and tries configured providers in order. |
| PDF/OCR/chunking | `backend/core/document_loader.py`, `chunker.py` | Extracts page text or local OCR, carries page references and splits readable text. |
| Retrieval | `backend/core/retrieval_embeddings.py`, `vector_store.py`, `hybrid_retriever.py` | Multilingual E5 embeddings, per-document Chroma index, BM25, reciprocal rank fusion and optional cross-encoder. |
| Review logic | `backend/core/legal_reviewer.py`, `review_workflow.py`, `review_profiles.py` | Category routing, summary, heatmap, Q&A, obligations, negotiation and comparisons. |
| Dashboard | `frontend/index.html` | Single-page UI, language/category controls, loading and result panels. |

Dense retrieval matches meaning (including cross-language queries); BM25 matches exact words and clause identifiers. RRF combines their ranked candidates. The English cross-encoder reranker is used for English-only content when enabled; for Tamil/Hindi or mixed content it is skipped unless `MULTILINGUAL_RERANKER_MODEL` is explicitly set. `RERANKER_ENABLED=false` disables reranking. The LLM operates **after** retrieval to synthesize answers; it is also used for summaries, category suggestions, obligation extraction, risk analysis and drafting. Tesseract handles OCR locally; it is not the LLM.

Supported broad categories include NDA, services, employment, lease, property, judgments, pleadings, affidavits, legislation, corporate records, wills, powers of attorney, notices, policies, IP licences and other contracts, with an unknown/mixed fallback. This is category routing, **not proof of accuracy for every subtype or jurisdiction**.

Main API routes: `GET /api/status`, `GET /api/summary`, `GET /api/heatmap`, `GET /api/obligations?include_status=1`, `GET /api/visuals`; `POST /api/upload`, `POST /api/qa`, `POST /api/negotiate`, `POST /api/compare`, `POST /api/visuals/analyze`, `POST /api/llm/check`. Upload and analysis use a browser session cookie; use the served UI for the simplest end-to-end workflow. The [frontend integration guide](frontend/README_FRONTEND.md) has API examples.

## Tests and evaluation

From the repository root:

```powershell
python -m pytest backend/tests -q
node --test frontend/tests/state.test.cjs
cd backend
python test_core.py
python evaluator.py
```

Node.js is needed for the frontend test command, **not** to run the UI. The evaluator's historical 50-query ablation describes an older English-only benchmark; its reported numbers must not be presented as fresh results after changing the embedding model. The feature branch previously passed 100 backend tests, 12 frontend tests, 9 synthetic multilingual retrieval cases, and 6 native/scanned OCR examples; these are engineering checks, not legal-answer accuracy or translation quality. See [multilingual validation notes](docs/multilingual_document_review.md) and [ablation study](docs/ablation_study.md). Rerun tests and benchmark on your own environment before quoting results.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `/api/status` works but Summary/Q&A fails | Click **Check AI connection**. Verify key, model ID, quota and network. HTTP 429 means provider rate/quota limit; wait, switch to another configured provider, or reduce requests. HTTP 200 from this API does not guarantee successful AI analysis; read the result diagnostic. |
| HF Hub unauthenticated warning | Optional `HF_TOKEN` improves model-download limits. It does not resolve an LLM provider's HTTP 429. |
| Tamil/Hindi scan has missing text | Verify `OCR_TESSDATA`, installed `tam`/`hin` packs, and selected PDF language; restart and re-upload. |
| No reranker on non-English text | Expected without `MULTILINGUAL_RERANKER_MODEL`. RRF still runs. |
| Existing document results look stale | Refresh, re-upload and confirm the current document/category; older indexes need rebuilding after changing `EMBEDDING_MODEL`. |
| Port 8080 busy | Stop the other process; `run_server(port=...)` in `backend/app_api.py` can be invoked with a different port. |

Only upload documents you are allowed to process. Hosted LLM analysis transmits selected document content to the configured provider. Do not use this prototype for privileged or sensitive records without an approved privacy/security review.
