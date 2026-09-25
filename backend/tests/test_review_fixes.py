"""Regression coverage for document-wide analysis, isolation, OCR and configuration."""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import pymupdf as fitz
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from core.chunker import DocumentChunk, LegalChunker
from core.document_loader import LegalDocumentLoader, LoadedDocument, PageData
from core.document_analysis import evidence_windows, contract_changes
from core.vector_store import LegalVectorStore
from core.legal_reviewer import LegalReviewer
import core.legal_reviewer as reviewer_module
import config


def document(texts, name="contract.pdf"):
    return LoadedDocument(
        file_path="<memory>", file_name=name, total_pages=len(texts),
        pages=[PageData(i, text, len(text), 0) for i, text in enumerate(texts, 1)],
        full_text="\n\n".join(texts), total_words=sum(len(text.split()) for text in texts),
    )


class LocalTestEmbeddings(Embeddings):
    """Small deterministic vectors: tests real Chroma, without downloading a model."""
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        return [1.0, float('alpha' in text.lower()), float('beta' in text.lower())]


def test_real_chroma_collections_are_isolated_and_rebuild_replaces():
    first, second = LegalVectorStore(), LegalVectorStore()
    first._embeddings = second._embeddings = LocalTestEmbeddings()
    try:
        first.build_index([DocumentChunk('same-id', 1, 'alpha secret payment', 'same.pdf', 20)])
        second.build_index([DocumentChunk('same-id', 1, 'beta private notice', 'same.pdf', 20)])
        assert first.collection_name != second.collection_name
        assert [d.page_content for d in first.similarity_search('beta', 10)] == ['alpha secret payment']
        assert [d.page_content for d in second.similarity_search('alpha', 10)] == ['beta private notice']
        first.build_index([DocumentChunk('replacement', 2, 'alpha replacement', 'same.pdf', 17)])
        assert [d.page_content for d in first.similarity_search('payment', 10)] == ['alpha replacement']
        first.close()
        assert second.similarity_search('alpha', 10)[0].page_content == 'beta private notice'
    finally:
        first.close()
        second.close()


def test_summary_evidence_includes_every_page_and_tail_of_long_pages(monkeypatch):
    texts = [('Common contract terms. ' * 180) + f' UNIQUE_PAGE_{i}' for i in range(1, 9)]
    texts[-1] += ' Governing law: Singapore. Notice: 75 days.'
    doc = document(texts)
    reviewer = LegalReviewer(doc, None)
    prompts = []
    def respond(messages, **kwargs):
        prompt = messages[-1].content
        prompts.append(prompt)
        return 'Contract facts [Page 8]: Governing law Singapore; notice 75 days.'
    monkeypatch.setattr(reviewer_module, 'check_api_key_configured', lambda: True)
    monkeypatch.setattr(reviewer, '_invoke_llm', respond)
    assert 'Singapore' in reviewer.generate_executive_summary()
    evidence_prompts = [p for p in prompts if 'evidence window' in p]
    assert len(evidence_prompts) > 1
    assert all(f'UNIQUE_PAGE_{i}' in '\n'.join(evidence_prompts) for i in range(1, 9))
    assert 'Notice: 75 days.' in '\n'.join(evidence_prompts)
    assert all('[Page ' in p for p in evidence_prompts)
    count = len(prompts)
    reviewer.generate_executive_summary()
    assert len(prompts) == count


def test_summary_does_not_cache_partial_success(monkeypatch):
    reviewer = LegalReviewer(document(['Terms. ' * 4000, 'Later important provision.']), None)
    monkeypatch.setattr(reviewer_module, 'check_api_key_configured', lambda: True)
    calls = []
    def fail_later(*args, **kwargs):
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError('offline')
        return 'First-window notes.'
    monkeypatch.setattr(reviewer, '_invoke_llm', fail_later)
    assert 'failed' in reviewer.generate_executive_summary().lower()
    assert reviewer._cached_summary is None


def test_evidence_windows_do_not_drop_long_unbroken_text_or_include_artifacts():
    text = 'X' * 9000 + ' FINAL_TERM'
    doc = document([text, 'EVALUATION DESIGN NOTES — DO NOT USE AS CONTRACT TERMS'])
    windows = evidence_windows(doc, 2000)
    assert all(len(w) <= 2000 for w in windows)
    assert ''.join(w.split('\n', 1)[1] for w in windows) == text


def test_comparison_detects_late_changes_and_preserves_source_evidence(monkeypatch):
    common = ['Common terms. ' * 200] * 7
    left = LegalReviewer(document(common + ['The Buyer shall pay within 30 days.'], 'v1.pdf'), None)
    right = LegalReviewer(document(common + ['The Buyer shall pay within 75 days.'], 'v2.pdf'), None)
    prompts = []
    def assess(messages, **kwargs):
        prompts.append(messages[-1].content)
        return '[{"change_id":1,"impact":"MODERATE","analysis":"The payment period changed.","v1_text":"invented"}]'
    monkeypatch.setattr(reviewer_module, 'check_api_key_configured', lambda: True)
    monkeypatch.setattr(left, '_invoke_llm', assess)
    result = LegalReviewer.compare_contracts(left, right)
    assert result['status'] == 'analyzed'
    assert result['total_changes_detected'] == 1
    assert '75 days' in prompts[0]
    row = result['comparison_table'][0]
    assert row['v1_page'] == row['v2_page'] == 8
    assert row['v1_text'] == 'The Buyer shall pay within 30 days.'


def test_comparison_identical_documents_never_call_llm(monkeypatch):
    first = LegalReviewer(document(['Exactly the same terms.']), None)
    monkeypatch.setattr(reviewer_module, 'check_api_key_configured', lambda: True)
    monkeypatch.setattr(first, '_invoke_llm', lambda *a, **k: pytest.fail('No LLM call for identical documents'))
    result = LegalReviewer.compare_contracts(first, first)
    assert result['status'] == 'no_changes'
    assert result['total_changes_detected'] == 0
    assert result['comparison_table'] == []


@pytest.mark.parametrize('response', [None, 'not JSON', '[{"change_id":99,"impact":"HIGH","analysis":"Wrong ID"}]'])
def test_failed_comparison_preserves_actual_changes_without_risk_claims(monkeypatch, response):
    first = LegalReviewer(document(['Payment due in 30 days.']), None)
    second = LegalReviewer(document(['Payment due in 90 days.']), None)
    def fail(*args, **kwargs):
        if response is None:
            raise RuntimeError('Provider unavailable')
        return response
    monkeypatch.setattr(reviewer_module, 'check_api_key_configured', lambda: True)
    monkeypatch.setattr(first, '_invoke_llm', fail)
    result = LegalReviewer.compare_contracts(first, second)
    assert result['status'] == 'analysis_failed'
    assert result['total_changes_detected'] == 1
    assert result['comparison_table'][0]['impact'] == 'REVIEW REQUIRED'
    assert '90 days' in result['comparison_table'][0]['v2_text']


def test_large_comparison_retains_all_changes_in_bounded_requests(monkeypatch):
    first = LegalReviewer(document([f'Old obligation {i}.' for i in range(60)]), None)
    second = LegalReviewer(document([f'New obligation {i}.' for i in range(60)]), None)
    prompts = []
    def assess(messages, **kwargs):
        prompt = messages[-1].content
        prompts.append(prompt)
        batch = json.loads(prompt.split('\n\n', 1)[1])
        return json.dumps([dict(change_id=x['change_id'], impact='REVIEW REQUIRED', analysis='Needs context.') for x in batch])
    monkeypatch.setattr(reviewer_module, 'check_api_key_configured', lambda: True)
    monkeypatch.setattr(first, '_invoke_llm', assess)
    result = LegalReviewer.compare_contracts(first, second)
    assert len(result['comparison_table']) == result['total_changes_detected'] == 60
    assert len(prompts) == 30
    assert 'New obligation 59' in prompts[-1]


def test_classifier_never_exposes_uncalibrated_numeric_confidence(monkeypatch):
    reviewer = LegalReviewer(document(['MASTER SERVICES AGREEMENT']), None)
    assert reviewer.classify_document()['confidence'] is None
    monkeypatch.setattr(reviewer_module, 'check_api_key_configured', lambda: True)
    monkeypatch.setattr(reviewer, '_invoke_llm', lambda *a, **k: '{"doc_type":"MSA","confidence":0.99}')
    classified = reviewer.classify_document()
    assert classified['source'] == 'LLM'
    assert classified['confidence'] is None


def test_qa_honors_evidence_count_greater_than_three():
    docs = [Document(page_content=f'Clause {i}', metadata={'page_number':i}) for i in range(1, 7)]
    retriever = SimpleNamespace(hybrid_search=lambda query, top_k: docs[:top_k])
    reviewer = LegalReviewer(document(['Text']), None, hybrid_retriever=retriever)
    assert len(reviewer.answer_query('question', top_k=5)['sources']) == 5


def test_disabled_hybrid_uses_dense_retrieval(monkeypatch):
    monkeypatch.setattr(reviewer_module, 'HYBRID_RETRIEVAL_ENABLED', False)
    docs = [Document(page_content='Dense evidence', metadata={'page_number':1})]
    store = SimpleNamespace(chunks=[1], similarity_search=lambda q, k: docs)
    reviewer = LegalReviewer(document(['Text']), store)
    assert reviewer.hybrid_retriever is None
    assert reviewer.answer_query('question')['sources'][0]['text'] == 'Dense evidence'


def scan_bytes():
    with fitz.open() as source:
        page = source.new_page()
        page.insert_text((60, 100), 'The Buyer shall pay within 45 days.', fontsize=20)
        png = page.get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes('png')
    with fitz.open() as scanned:
        page = scanned.new_page()
        page.insert_image(page.rect, stream=png)
        return scanned.tobytes()


def test_real_ocr_extracts_scanned_payment_and_preserves_page():
    data = scan_bytes()
    loaded = LegalDocumentLoader(ocr_enabled=True, source_language='en').load_pdf(data, 'scan.pdf')
    if loaded.warnings and not loaded.ocr_pages:
        pytest.skip('Tesseract language data unavailable; run in the OCR-enabled test environment')
    assert loaded.ocr_pages == [1]
    assert '45 days' in loaded.full_text
    assert LegalChunker().chunk_document(loaded)[0].page_number == 1
    assert loaded.images[0].mime_type == 'image/png'


def test_ocr_failure_is_reported_and_native_text_is_preserved(monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError('no tessdata')
    monkeypatch.setattr(fitz.Page, 'get_textpage_ocr', unavailable)
    with fitz.open(stream=scan_bytes(), filetype='pdf') as mixed:
        mixed[0].insert_text((50, 30), 'Native header')
        data = mixed.tobytes()
    loaded = LegalDocumentLoader().load_pdf(data, 'scan.pdf')
    assert 'Native header' in loaded.full_text
    assert not loaded.ocr_pages
    assert any('OCR failed' in w for w in loaded.warnings)
    assert loaded.images


def test_explicit_vision_models_are_used_and_text_only_providers_are_excluded(monkeypatch):
    monkeypatch.setenv('LEGAL_AI_OFFLINE', 'false')
    monkeypatch.setenv('LLM_PROVIDER_ORDER', 'groq,openai')
    monkeypatch.setenv('GROQ_API_KEY', 'test-key')
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    monkeypatch.delenv('GROQ_VISION_MODEL', raising=False)
    monkeypatch.setenv('OPENAI_VISION_MODEL', 'test-vision-model')
    monkeypatch.setattr(config, 'ChatOpenAI', lambda **kwargs: kwargs)
    candidates = config.get_llm_candidates(vision=True)
    assert len(candidates) == 1
    assert candidates[0][1]['model'] == 'test-vision-model'


def test_environment_controls_actual_chunking_and_retrieval_defaults():
    backend = Path(__file__).resolve().parents[1]
    env = dict(os.environ, CHUNK_SIZE='120', CHUNK_OVERLAP='15', RETRIEVAL_TOP_K='5',
               TOP_K_DENSE='7', TOP_K_BM25='8', RRF_K='31', RERANKER_ENABLED='false',
               HYBRID_RETRIEVAL_ENABLED='true', LEGAL_AI_OFFLINE='true')
    code = '''
from core.chunker import LegalChunker, DocumentChunk
from core.document_loader import LoadedDocument, PageData
from core.hybrid_retriever import LegalHybridRetriever
from langchain_core.documents import Document
text = "Payment and notice clauses. " * 30
doc = LoadedDocument("<memory>", "test.pdf", 1, pages=[PageData(1, text, len(text), 0)])
chunks = LegalChunker().chunk_document(doc)
assert max(len(c.text) for c in chunks) <= 120
class Store:
    def similarity_search(self, query, k):
        assert k == 7
        return []
r = LegalHybridRetriever(Store(), chunks)
assert r.rrf_k == 31
original_search = r.bm25_retriever.search
def search(query, top_k):
    assert top_k == 8
    return original_search(query, top_k)
r.bm25_retriever.search = search
r.reranker.rerank = lambda *a, **k: (_ for _ in ()).throw(AssertionError("Reranker must be disabled"))
results = r.hybrid_search("payment")
assert results and all("reranker_error" not in d.metadata for d in results)
'''
    subprocess.run([sys.executable, '-c', code], cwd=backend, env=env, check=True, capture_output=True, text=True)


def test_real_api_isolates_uploads_rejects_stale_ids_and_serves_visuals(monkeypatch):
    import threading
    import urllib.request
    import urllib.error
    from http.cookiejar import CookieJar
    from http.server import ThreadingHTTPServer
    from app_api import LegalAPIRequestHandler, SESSION_DATA
    from core.hybrid_retriever import LegalCrossEncoderReranker

    monkeypatch.setattr(LegalVectorStore, 'embeddings', property(lambda self: LocalTestEmbeddings()))
    monkeypatch.setattr(LegalCrossEncoderReranker, 'rerank',
                        lambda self, query, docs, top_k: [(d, 0.0) for d in docs[:top_k]])
    def pdf(text):
        with fitz.open() as doc:
            page = doc.new_page()
            page.insert_text((50, 80), text)
            page.draw_rect(fitz.Rect(50, 100, 100, 150))
            return doc.tobytes()
    first = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
    second = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(CookieJar()))
    SESSION_DATA.clear()
    server = ThreadingHTTPServer(('127.0.0.1', 0), LegalAPIRequestHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    def request(browser, path, data=None, doc_id=None):
        headers = {}
        if data is not None:
            headers['Content-Type'] = 'application/pdf' if isinstance(data, bytes) else 'application/json'
            if not isinstance(data, bytes):
                data = json.dumps(data).encode()
        if doc_id:
            headers['X-Document-Id'] = doc_id
        return json.load(browser.open(urllib.request.Request(base + path, data=data, headers=headers)))
    try:
        a = request(first, '/api/upload', pdf('Alpha shall pay 30 coins.'))
        b = request(second, '/api/upload', pdf('Beta shall pay 90 coins.'))
        assert a['document_id'] != b['document_id']
        sources = request(first, '/api/qa', {'question':'Beta payment'}, a['document_id'])['sources']
        assert sources and all('Alpha' in x['text'] and 'Beta' not in x['text'] for x in sources)
        previews = request(first, '/api/visuals', doc_id=a['document_id'])
        assert previews['previews'][0]['image_url'].startswith('data:image/png;base64,')
        analysis = request(first, '/api/visuals/analyze', {}, a['document_id'])
        assert analysis['results'][0]['status'] == 'vision_unavailable'
        comparison = request(first, '/api/compare', pdf('Alpha shall pay 30 coins.'), a['document_id'])
        assert comparison['total_changes_detected'] == 0
        old = next(s['doc1_reviewer'] for s in SESSION_DATA.values() if s['doc1_reviewer'].document_id == a['document_id'])
        replacement = request(first, '/api/upload', pdf('Gamma shall pay 45 coins.'))
        assert old.vector_store.total_chunks == 0
        with pytest.raises(urllib.error.HTTPError) as stale:
            request(first, '/api/qa', {'question':'payment'}, a['document_id'])
        assert stale.value.code == 409
        sources = request(first, '/api/qa', {'question':'Alpha'}, replacement['document_id'])['sources']
        assert sources and all('Gamma' in x['text'] and 'Alpha' not in x['text'] for x in sources)
        other_sources = request(second, '/api/qa', {'question':'payment'}, b['document_id'])['sources']
        assert other_sources and all('Beta' in x['text'] for x in other_sources)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        for session in SESSION_DATA.values():
            LegalAPIRequestHandler._close_reviewers(session)
        SESSION_DATA.clear()


def test_scanned_pdf_can_build_api_reviewer(monkeypatch):
    from app_api import LegalAPIRequestHandler
    monkeypatch.setattr(LegalVectorStore, 'embeddings', property(lambda self: LocalTestEmbeddings()))
    reviewer, count = LegalAPIRequestHandler._build_reviewer(scan_bytes(), 'scan.pdf', 'en')
    try:
        assert count > 0
        assert reviewer.document.ocr_pages == [1]
        assert '45 days' in reviewer.document.full_text
    finally:
        reviewer.vector_store.close()
