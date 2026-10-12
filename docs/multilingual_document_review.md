# English, Tamil and Hindi document review

This update expands the **PDF review workflow**, not a guarantee that every legal document will be interpreted correctly. Document classification is a suggestion you can override. OCR, retrieval, translation and legal reasoning each have separate failure modes.

## Run the updated version on Windows

From the repository root in your existing `sde` environment:

```powershell
git fetch origin
git switch --track origin/feat/multilingual-document-review
pip install -r requirements.txt
```

If that branch already exists locally, switch to it and pull instead. If you originally downloaded a ZIP, download the feature branch ZIP from GitHub and copy your private `.env` into that checkout. Do not commit API keys.

Update the existing `.env` (an older explicit value overrides the new defaults):

```dotenv
EMBEDDING_MODEL=intfloat/multilingual-e5-small
OCR_ENABLED=true
OCR_LANGUAGE=eng+tam+hin
OCR_TESSDATA=C:/Program Files/Tesseract-OCR/tessdata
MULTILINGUAL_RERANKER_MODEL=
```

Install Tesseract language data for **English (`eng`), Tamil (`tam`), and Hindi (`hin`)**. The directory above must contain `eng.traineddata`, `tam.traineddata`, and `hin.traineddata`. Obtain the packs from [Tesseract tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast). Adjust the path to your installation; do not copy this example unchanged if your path differs. `tesseract --list-langs` should show all three when the CLI uses that directory.

Then restart and **re-upload** the document so its index uses the new embedding model:

```powershell
cd backend
python app_api.py
```

The first upload downloads the multilingual model and can take longer. An optional Hugging Face token helps downloads; it is separate from an LLM API key. Summary, answer generation and other AI analyses still need a working configured provider. Groq/Gemini/OpenAI rate limits remain applicable; this change does not increase provider quotas. Use **Check AI connection** before your demonstration.

## Use the controls

1. Select **Response language**: English, தமிழ், or हिन्दी. This requests explanations in that language and changes the principal interface controls and common application messages.
2. Select **PDF language / OCR** before uploading. Choose the actual source language; choose Auto / mixed for mixed-language scans. Auto requires all configured packs. Explicit English only needs `eng`; Tamil needs `eng+tam`; Hindi needs `eng+hin`.
3. Upload a PDF and confirm **Document category**. Override it if detection is wrong. Script detection is only a hint: Devanagari is not unique to Hindi.
4. Generate a summary, ask a question in any of the three languages, and compare the answer with its original page evidence. Source quotations are kept in their original language; explanations may use a different language.
5. Changing response language or category clears old result panels and cached analyses. It keeps the current document. Click the required analysis again. Changing OCR language requires re-uploading.

Some technical diagnostics, review-topic names, less common static messages and original evidence remain in English or the source language. This is not a complete translation of every interface string, a full-document translation feature, or certified translation.

## Document families

| Family | Review focus | Negotiation workflow |
|---|---|---|
| NDA | Confidential information, disclosure, duration, remedies | Available |
| Services / vendor | Deliverables, fees, liability, termination | Available |
| Employment | Role, compensation, restrictions, notice | Available |
| Lease / rental | Property, rent, deposit, maintenance, renewal | Available |
| Property transfer / mortgage | Description, consideration, encumbrance statements, execution | Disabled |
| Judgment / court order | History, issues, reasoning, operative directions | Disabled |
| Petition / pleading | Allegations, grounds, relief sought, procedural dates | Disabled |
| Affidavit / declaration | Declarant, knowledge basis, exhibits, attestation statements | Disabled |
| Act / rules / regulation | Scope, commencement, provisions, exceptions, penalties | Disabled |
| Corporate / governance | Authority, resolutions, approvals, delegated powers | Disabled |
| Will / trust instrument | Beneficiaries, assets, conditions, executor/trustee powers | Disabled |
| Power of attorney | Principal, agent, powers, exclusions, revocation | Disabled |
| Legal notice | Claims, demands, service statements, response dates | Disabled |
| Privacy policy / terms | Data uses, retention, consent, rights, disputes | Available |
| IP / licence agreement | Rights, territory, permitted use, royalties, termination | Available |
| Other contract | Parties, duties, remedies, termination | Available |
| Unknown / mixed | Facts, ambiguity and manual scope confirmation | Disabled |

These are **16 broad families plus an unknown/mixed fallback**, not 17 independently validated specialist models. Subtypes may require a more precise profile. Negotiation availability means the workflow is enabled; it does not establish that a particular provision is legally negotiable. A court order is not rewritten as if it were a private contract. A deed alone cannot establish ownership, an affidavit cannot establish the truth of assertions, and a PDF statute cannot establish current law or amendments.

## What changed internally

- `review_profiles.py` routes documents to four relevant review areas per family, with a user override and uncalibrated classification labels.
- Summaries cover all readable evidence windows using the existing bounded map/reduce process. Obligations now also process all readable pages, rather than six retrieved chunks. Long documents can therefore use more LLM calls/tokens.
- Prompts request English/Tamil/Hindi explanations while preserving JSON keys, page markers, original quotes, names, dates, amounts, negation and exceptions. This is a prompt instruction, not a proof of faithful translation.
- Generated obligation and heatmap quotations must match text on the cited source page. Invalid rows trigger a labelled failure/fallback instead of invented source citations. This verifies quotation grounding, not the correctness or completeness of the interpretation.
- `intfloat/multilingual-e5-small` supplies multilingual dense embeddings. Queries use `query: ` and documents use `passage: `. Long token sequences are divided into windows and their vectors averaged/normalized so Indic text tails are not silently discarded. This pooling adaptation needs corpus evaluation.
- BM25 preserves Unicode letters, combining marks and numbers. A small function-word list and matching-token filter prevent irrelevant lexical votes from overriding cross-language dense matches. It uses whitespace tokens, not a Tamil/Hindi morphological analyzer. RRF combines dense and lexical ranks.
- The existing English cross-encoder is skipped for non-English/mixed-script documents or queries. Without an explicitly configured and evaluated multilingual cross-encoder, hybrid RRF order is returned and reported. English document + English query can still use the English reranker. Unknown scripts are handled conservatively.
- OCR requests explicit language packs. Missing packs generate an extraction warning; they do not silently fall back to English-only OCR. Suspicious PDF text encodings trigger OCR recovery. Selected Tamil/Hindi can also trigger recovery on English-only pages in a mixed document.
- Comparisons of differing detected scripts/languages return text differences with a language-mismatch warning and do not assign AI legal-impact ratings. Same-script translations may escape this heuristic; compare versions in the same language.

## Validation and review demonstration

Automated tests cover document routing, Unicode preservation, preference validation/cache invalidation, stale frontend responses, quote/page validation, optional reranking, OCR failure reporting, and existing isolation/provider behavior. These are engineering tests, not an accuracy dataset.

```powershell
python -m pytest backend/tests -q
node --test frontend/tests/state.test.cjs
```

`scripts/verify_multilingual.py --assets PATH` is an optional **synthetic pipeline smoke test**. Assets are the three language packs plus regular Noto Sans Tamil and Noto Sans Devanagari fonts named `NotoSansTamil.ttf` and `NotoSansDevanagari.ttf`. It checks numeric retention and script extraction in native/scanned PDFs, then asks one rent question in each language against three small documents. It uses the actual E5 model and local Chroma; it does not call an LLM. A passing result must not be presented as a legal accuracy percentage or compared to the old English-only 50-query ablation results. Those retrieval results need rerunning with the new model.

For the academic review, demonstrate one English agreement, one Tamil rental PDF, one Hindi notice, and a judgment whose negotiation feature is disabled. Show a late-page fact with its citation and a deliberately unavailable-provider fallback. Review translations with fluent readers before presenting them as correct.

Before claiming “most document types supported correctly,” build a held-out, permission-cleared corpus across languages and families. Have bilingual legal reviewers annotate document types, relevant pages, duties, deadlines, exceptions and reference answers. Report per-family/per-language classification F1, retrieval recall@k/MRR, obligation field precision/recall, citation correctness, OCR character error rate, and human-rated translation/factual accuracy. Include latency, failures and token usage. Do not invent improvement percentages.

Limits: PDF input only; handwriting, poor scans, mixed layouts, transliterated Tamil/Hindi, regional legal terminology and long multipart documents are not validated. Live provider output quality and broad legal correctness require separate evaluation. Native text can contain undetected encoding errors. No jurisdiction-specific legal validity, signature authentication, title verification or current-law database lookup is provided.

Implementation references: [E5 model card](https://huggingface.co/intfloat/multilingual-e5-small), [PyMuPDF OCR](https://pymupdf.readthedocs.io/en/latest/page.html#Page.get_textpage_ocr), [Tesseract language data](https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html).

### Engineering validation on this change

- Backend regression suite: **100 passed**.
- Frontend state/API suite: **12 passed**.
- DOM execution: English → Tamil → Hindi → English controls, category warnings and unchanged source evidence passed.
- Actual E5 + Chroma retrieval: all **9** language combinations retrieved the expected rent page in three synthetic documents. Native/scanned extraction: **6** examples retained the tested amount and script; Indic font encoding recovery used OCR. See [raw smoke results](multilingual_smoke_results.json). This is not a representative benchmark.
- Browser-rendered visual inspection was not completed in this environment.
- Live hosted LLM output was not tested; no claim of translation quality or legal correctness is made.
