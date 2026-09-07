# Second-review fixes and setup

This update addresses the eight issues identified in the code review. Historical benchmark numbers remain unchanged; the retrieval models were not retrained and no new accuracy gain is claimed.

## Retrieval methods

| Method | What it does | What it adds |
|---|---|---|
| Dense only | Embeds the question and matches it against independently embedded chunks | Semantic similarity and paraphrases |
| BM25 | Scores token overlap, rarity and length-normalized term frequency | Explicit terminology; this repository also uses rule-based legal query expansion |
| Candidate union | Combines dense and BM25 results and removes duplicate chunk IDs | Broader evidence; the benchmark places dense candidates first |
| Hybrid + RRF | Adds `1 / (RRF_K + rank)` for each candidate's position in each ranking | Fuses rankings without adding incompatible raw score scales |
| Hybrid + cross-encoder | Jointly scores the question with each fused candidate | A more expensive relevance ordering before generation |

Retrieval locates evidence; the generative LLM writes the answer. Retrieval scores are not calibrated answer-confidence probabilities. The historical ablation uses page-deduplicated rankings from up to ten chunks, while live QA uses `RETRIEVAL_TOP_K` chunks. The candidate-union stage uses five candidates from each retriever and RRF uses ten, so that particular comparison changes two factors.

## Changes

| Finding | Implementation |
|---|---|
| Four-page summary cutoff | `document_analysis.evidence_windows()` covers all readable contract pages and every long-page segment. Multiple windows are summarized into cited notes and consolidated recursively. A failed intermediate call returns an explicit incomplete-analysis message and is not cached. |
| 2,500-character comparison cutoff | Deterministic sentence/paragraph differences span all readable contract pages. Long changes are segmented without dropping text. Every change is retained; small batches are sent for AI impact assessment. Original text and page references cannot be replaced by model output. |
| Fake moderate change on failure | Identical documents bypass the LLM and return `no_changes`. Actual differences remain `REVIEW REQUIRED` on failure, with `analysis_failed` or `partial_analysis`. Counts represent text changes, not legal-clause counts. |
| Fixed 85% confidence | Heuristic and LLM classification return `confidence: null`, a method/source, and a calibration note. UI displays the method. |
| Vector isolation | Every store receives a random collection name and explicit chunk IDs. Rebuild replaces its own collection; session replacement/eviction deletes old collections. Temporary comparison collections are closed after use. |
| Stale dashboard | New uploads clear all derived panels, abort old requests, and advance a document generation. Per-endpoint request tokens prevent older questions overwriting newer answers. `X-Document-Id` rejects requests for replaced documents. Per-session locking prevents cleanup during active analysis. |
| Ignored configuration | Chunking defaults and API calls use configured sizes. Hybrid retrieval uses configured candidate counts, RRF constant and reranking switch. Hybrid can be disabled; QA honors configured top-k. Active settings appear in `/api/status`. |
| Visual/OCR support not exposed | Scanned or image-dominated pages use local OCR when enabled. Visual pages are rendered as PNG previews, including vector marks. The UI exposes previews and an explicit AI-analysis action with dedicated vision-model configuration. |

## OCR setup

PDF text extraction works without OCR dependencies. To read scans, install Tesseract and the language data for `OCR_LANGUAGE`.

- Ubuntu/Debian: `sudo apt-get install tesseract-ocr tesseract-ocr-eng`
- macOS with Homebrew: `brew install tesseract`
- Windows: install Tesseract, add its installation directory to PATH, and set `OCR_TESSDATA` to its `tessdata` folder if automatic discovery fails.

Example `.env` settings:

```dotenv
OCR_ENABLED=true
OCR_LANGUAGE=eng
OCR_DPI=200
# OCR_TESSDATA=C:/Program Files/Tesseract-OCR/tessdata
MAX_VISUAL_PAGES=5
SUMMARY_CONTEXT_CHARS=12000
```

OCR runs on pages likely to contain scanned content, including image-dominated pages with a native header/footer. Native text is retained on mixed pages. Extraction reports OCR page numbers and warnings; missing language data does not silently turn a partially read PDF into a fully covered contract. Image-only uploads without readable OCR text return a useful 422 error. OCR is not a guarantee of correct transcription, handwriting recognition, or table structure extraction.

Visual previews are limited to `MAX_VISUAL_PAGES`; the interface reports preview coverage. This limit does not cap text/OCR processing. PNG previews capture page appearance, including raster and vector marks. PyMuPDF operations are serialized because its document operations are not thread-safe.

The OCR implementation uses [PyMuPDF's page OCR API](https://pymupdf.readthedocs.io/en/latest/page.html#Page.get_textpage_ocr).

## Vision setup

Text-generation models are not automatically treated as vision-capable. Configure the matching provider API key and explicitly select its supported vision model:

```dotenv
# Set the real key locally; never commit it.
OPENAI_API_KEY=
OPENAI_VISION_MODEL=gpt-4o
```

Equivalent options are `XAI_VISION_MODEL`, `GROQ_VISION_MODEL`, and `GEMINI_VISION_MODEL`. Model availability depends on the provider/account. The normal provider order applies only to eligible vision clients. Leave unused model settings unset. `LEGAL_AI_OFFLINE=true` disables hosted inference for both text and visuals; local OCR and previews remain available.

The Visuals & OCR tab first fetches previews. Clicking **Analyze Visual Pages** sends those previews to configured vision providers. Without a configured vision provider, the interface reports `vision_unavailable` and retains previews. Descriptions can identify visible signature marks or stamps; they do not authenticate them.

## Retrieval configuration

```dotenv
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_TOP_K=3
TOP_K_DENSE=10
TOP_K_BM25=10
RRF_K=60
HYBRID_RETRIEVAL_ENABLED=true
RERANKER_ENABLED=true
```

Restart the backend after editing `.env`. Set `RERANKER_ENABLED=false` for RRF-only hybrid retrieval. Set `HYBRID_RETRIEVAL_ENABLED=false` for dense-only retrieval. Positive counts and valid overlap are checked at startup. The historical evaluator retains explicit stage-specific candidate budgets for its experiment; runtime settings are not a promise that its old table will be reproduced under different configurations.

## API additions and client compatibility

- `GET /api/visuals`: document ID, preview images, total visual-page count, extraction warnings.
- `POST /api/visuals/analyze`: per-page visual analysis or explicit unavailable/failure statuses.
- `POST /api/upload`: additionally returns `document_id`, `ocr_pages`, and `warnings`.
- `/api/status`: additionally returns active retrieval settings, OCR enablement, and configured vision providers.
- Comparison additionally returns status, original page references, extraction warnings, failed-assessment counts when applicable, and an explicit comparison scope.

Browser requests send `X-Document-Id` for document operations. Existing API clients may omit the header for compatibility, but should send it to detect replaced documents. A mismatch returns HTTP 409. Session cookies remain necessary to select the browser's document.

## Validation

Run from the repository root:

```bash
python -m pytest backend/tests -q
node --test frontend/tests/state.test.cjs
```

Backend regressions exercise full-document summary windows, later-page changes, identical-input comparison, outage/malformed-output handling, more than 50 changes, classification labeling, live configuration, actual Chroma collection isolation, scanned-PDF OCR, and HTTP session isolation/visual routes. Chroma tests use deterministic local embeddings to test the real database without downloading neural models. OCR integration tests require installed Tesseract language data and report a skip when it is unavailable. LLM calls use controlled responses/failures; passing tests do not measure hosted model quality.

Frontend tests execute the actual inline JavaScript with a small DOM harness, checking panel clearing, document IDs, and stale success/failure/JSON-decoding responses. They are not a substitute for a visual browser check.

## Remaining limitations

- Summary consolidation processes every readable source window but generated notes can still omit or misinterpret details. Exact factual coverage is not guaranteed by complete input coverage.
- Text differences are not semantic clause alignment. Layout or sentence-boundary changes can produce noisy differences, and impact assessment may lack surrounding contract context.
- LLM answers, risk labels, and citations still need evaluation and human verification.
- Scans with poor print quality, handwriting, rotation, complex tables, or missing language packs may need manual correction.
- State is temporary. Unique local collections fix isolation; durable history, user authentication, and cloud storage are separate work.
- Larger documents and many changes require more hosted-model calls; this improves coverage at the cost of latency and usage.
