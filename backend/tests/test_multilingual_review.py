"""Behavior and evidence checks; these tests do not measure legal/translation accuracy."""
import json
from types import SimpleNamespace
import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from core.review_profiles import infer_profile, detect_languages, PROFILES
from core.legal_reviewer import LegalReviewer
from core.document_loader import LoadedDocument, PageData, LegalDocumentLoader
from core.chunker import DocumentChunk
from core.hybrid_retriever import tokenize_legal_text, BM25Retriever, LegalHybridRetriever
import core.legal_reviewer as module
from app_api import LegalAPIRequestHandler, RequestError


def reviewer(texts):
    return LegalReviewer(LoadedDocument('<memory>','sample.pdf',len(texts),
        pages=[PageData(i,t,len(t),0) for i,t in enumerate(texts,1)], full_text='\n'.join(texts)),None)

@pytest.mark.parametrize('text,key',[
    ('Master Services Agreement. A court order may apply.','services'),
    ('Non-disclosure agreement','nda'), ('Employment contract','employment'),
    ('வாடகை ஒப்பந்தம்','lease'), ('விற்பனைப் பத்திரம்','property'),
    ('நீதிமன்ற ஆணை','judgment'), ('रिट याचिका','pleading'), ('शपथपत्र','affidavit'),
    ('அறக்கட்டளைப் பத்திரம்','will'), ('அதிகாரப் பத்திரம்','power_of_attorney'),
    ('कानूनी नोटिस','notice'),('अधिनियम','legislation'), ('Board resolution','corporate'),
    ('Privacy policy','policy'),('लाइसेंस समझौता','ip_license'),('Other agreement','general_contract'),
    ('Unidentified material','unknown')])
def test_document_families(text,key):
    assert infer_profile(text) == key

@pytest.mark.parametrize('text',['வாடகை செலுத்த வேண்டும்','किराया देना होगा'])
def test_lexical_matching_preserves_combining_marks(text):
    assert tokenize_legal_text(text) == text.split()
    assert BM25Retriever([DocumentChunk('a',1,'!!!','x',3)]).search(text) == []

def test_mixed_language_hints_are_not_english_only():
    assert set(detect_languages('வாடகை rent किराया')) == {'ta','hi','en'}

@pytest.mark.parametrize('query,source',[('கடமை என்ன','Payment terms'),('payment','வாடகை செலுத்த வேண்டும்')])
def test_english_cross_encoder_is_not_used_for_indic(query,source):
    chunk=DocumentChunk('a',1,source,'x',len(source))
    doc=Document(page_content=source,metadata={'chunk_id':'a','page_number':1})
    r=LegalHybridRetriever(SimpleNamespace(similarity_search=lambda *a,**k:[doc]),[chunk])
    r.multilingual_reranker=None
    r.reranker=SimpleNamespace(rerank=lambda *a,**k:pytest.fail('English reranker was called'))
    assert 'reranker_skipped' in r.hybrid_search(query)[0].metadata


def test_language_preferences_invalidate_all_outputs_and_gate_judgments():
    r=reviewer(['Judgment: the respondent shall pay costs.'])
    r._cached_summary='old';r._cached_heatmap=['old'];r._cached_obligations=['old']
    r.set_preferences('ta','judgment')
    assert r._cached_summary is None and r._cached_heatmap is None and r._cached_obligations is None
    result=r.generate_negotiation_strategy('Rewrite the order')
    assert result['status']=='not_applicable' and 'ஆவண' in result['assessment']
    with pytest.raises(ValueError):r.set_preferences('xx','judgment')


def test_language_policy_precedes_original_evidence():
    r=reviewer(['வாடகை ஒப்பந்தம்']);r.set_preferences('hi')
    captured=[]
    class Client:
        def bind(self,**kw):return self
        def invoke(self,messages):captured.extend(messages);return SimpleNamespace(content='उत्तर')
    r._llm=Client();source='வாடகை 10000. [Page 1]'
    assert r._invoke_llm([HumanMessage(content=source)])=='उत्तर'
    assert 'Hindi' in captured[0].content and 'verbatim' in captured[0].content
    assert captured[-1].content==source


def test_judgment_summary_has_appropriate_topics(monkeypatch):
    r=reviewer(['Court order: relief granted.'])
    monkeypatch.setattr(module,'check_api_key_configured',lambda:True)
    prompts=[]
    def respond(messages,**kw):prompts.append(messages[-1].content);return 'Summary [Page 1]'
    monkeypatch.setattr(r,'_invoke_llm',respond)
    r.generate_executive_summary()
    assert 'Operative directions' in prompts[0] and 'Commercial/payment terms' not in prompts[0]


def test_full_page_obligations_are_cited_and_cached(monkeypatch):
    r=reviewer(['Background only.']*8+['Tenant must pay 10000 monthly.'])
    monkeypatch.setattr(module,'check_api_key_configured',lambda:True)
    calls=[]
    def respond(messages,**kw):
        calls.append(messages[-1].content)
        return json.dumps([dict(party='Tenant',obligation='Pay rent',deadline_frequency='Monthly',consequence='Unknown',page=9,evidence='Tenant must pay 10000 monthly.')])
    monkeypatch.setattr(r,'_invoke_llm',respond)
    assert r.extract_obligations()[0]['page']==9
    assert '[Page 9]' in calls[0] and r.obligation_status['mode']=='ai'
    r.extract_obligations();assert len(calls)==1

@pytest.mark.parametrize('page,quote',[(2,'Tenant must pay rent.'),(1,'Invented payment.')])
def test_uncited_obligations_fall_back_honestly(monkeypatch,page,quote):
    r=reviewer(['Tenant must pay rent.'])
    monkeypatch.setattr(module,'check_api_key_configured',lambda:True)
    monkeypatch.setattr(r,'_invoke_llm',lambda *a,**k:json.dumps([dict(party='Tenant',obligation='Pay',deadline_frequency='Unknown',consequence='Unknown',page=page,evidence=quote)]))
    result=r.extract_obligations()
    assert r.obligation_status['mode']=='rule_based' and result[0]['page']==1
    assert r._cached_obligations is None


def test_empty_document_does_not_claim_ai_success(monkeypatch):
    r=reviewer([])
    monkeypatch.setattr(r,'_invoke_llm',lambda *a,**k:pytest.fail('No evidence'))
    assert r.extract_obligations()==[] and r.obligation_status['mode']=='rule_based'


def test_cross_language_comparison_does_not_rate_translation_as_legal_change():
    a=reviewer(['Rent is payable monthly.']);b=reviewer(['வாடகை மாதந்தோறும் செலுத்த வேண்டும்.'])
    result=LegalReviewer.compare_contracts(a,b)
    assert result['status']=='language_mismatch' and result['warnings']

@pytest.mark.parametrize('headers',[{'X-Response-Language':'fr'},{'X-Document-Category':'fiction'}])
def test_api_rejects_invalid_preferences(headers):
    handler=object.__new__(LegalAPIRequestHandler);handler.headers=headers
    with pytest.raises(RequestError):handler._preferences()

@pytest.mark.parametrize('language,expected',[('en','eng'),('ta','eng+tam'),('hi','eng+hin')])
def test_explicit_ocr_languages(language,expected):
    assert LegalDocumentLoader(source_language=language).ocr_language==expected


def test_token_windows_include_indic_tail_and_prefix():
    from core.retrieval_embeddings import RetrievalEmbeddings
    calls=[]
    tokenizer=SimpleNamespace(encode=lambda text,**kw:list(range(80)),decode=lambda ids,**kw:' '.join(map(str,ids)))
    client=SimpleNamespace(_client=SimpleNamespace(tokenizer=tokenizer,max_seq_length=48),
        embed_documents=lambda texts:(calls.extend(texts) or [[1.,0.] for t in texts]))
    r=object.__new__(RetrievalEmbeddings);r.e5=True;r.client=client
    assert len(r.embed_documents(['Tamil text'])[0])==2
    assert len(calls)==3 and calls[-1].endswith('79') and calls[0].startswith('passage: ')
    calls.clear();r.embed_query('Hindi query');assert calls[0].startswith('query: ')


def test_localized_failure_never_translates_source_quote():
    from core.review_localization import localize_payload
    text='Not specified in extracted sentence'
    result=localize_payload(dict(consequence=text,evidence=text,sources=[dict(text=text)]),'ta')
    assert result['consequence']!=text and result['evidence']==text and result['sources'][0]['text']==text


def test_missing_ocr_pack_reports_failure_without_silently_using_english(monkeypatch,tmp_path):
    import core.document_loader as loader
    from tests.test_review_fixes import scan_bytes
    monkeypatch.setattr(loader,'OCR_TESSDATA',str(tmp_path))
    monkeypatch.setattr(loader.fitz.Page,'get_textpage_ocr',lambda *a,**k:pytest.fail('Missing pack should be checked first'))
    result=LegalDocumentLoader(source_language='ta').load_pdf(scan_bytes(),'scan.pdf')
    assert not result.ocr_pages and any('eng+tam' in w for w in result.warnings)


def test_failed_heatmap_can_retry_without_returning_cached_failure(monkeypatch):
    r=reviewer(['Other agreement. Tenant must pay rent.'])
    doc=Document(page_content='Tenant must pay rent.',metadata={'page_number':1})
    monkeypatch.setattr(module,'check_api_key_configured',lambda:True)
    monkeypatch.setattr(r,'_retrieve',lambda *a,**k:[doc])
    calls=[]
    def answer(*a,**k):
        calls.append(1)
        if len(calls)==1:return 'invalid'
        return json.dumps([dict(clause_title=t,risk_level='REVIEW REQUIRED',page=1,excerpt=doc.page_content,why_risky='Review',recommendation='Check') for t in r.profile['topics']])
    monkeypatch.setattr(r,'_invoke_llm',answer)
    assert all(i['risk_level']=='ANALYSIS FAILED' for i in r.generate_clause_risk_heatmap())
    assert all(i['risk_level']=='REVIEW REQUIRED' for i in r.generate_clause_risk_heatmap())
    assert len(calls)==2


def test_unrelated_function_words_do_not_override_cross_language_dense_match():
    chunks=[DocumentChunk('rent',1,'வாடகையாளர் 10000 வாடகை செலுத்த வேண்டும்.','x',40),
            DocumentChunk('other',2,'The parties shall keep information confidential.','x',48),
            DocumentChunk('court',3,'The court has jurisdiction over disputes.','x',40)]
    docs=[Document(page_content=c.text,metadata={'chunk_id':c.chunk_id,'page_number':c.page_number}) for c in chunks]
    r=LegalHybridRetriever(SimpleNamespace(similarity_search=lambda *a,**k:docs),chunks)
    assert r.bm25_retriever.search('How much rent must the tenant pay?')==[]
    assert r.hybrid_search('How much rent must the tenant pay?',use_reranker=False)[0].metadata['page_number']==1
