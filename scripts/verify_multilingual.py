"""Optional synthetic OCR/retrieval smoke test, NOT a legal accuracy benchmark.
Run with --assets DIR containing eng/tam/hin.traineddata and the regular
NotoSansTamil.ttf and NotoSansDevanagari.ttf font files. Requires model download.
"""
import argparse,json,os,sys
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--assets',required=True,type=Path)
args=parser.parse_args()
os.environ['OCR_TESSDATA']=str(args.assets.resolve())
os.environ['LEGAL_AI_OFFLINE']='true'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
import torch
torch.set_num_threads(2)
import pymupdf as fitz
from core.document_loader import LegalDocumentLoader
from core.chunker import LegalChunker
from core.vector_store import LegalVectorStore
from core.hybrid_retriever import LegalHybridRetriever
from core.review_profiles import detect_languages
texts={
'en':'Rental agreement. Tenant must pay rent of 10000 every month by the fifth day.',
'ta':'வாடகை ஒப்பந்தம். வாடகையாளர் ஒவ்வொரு மாதமும் ஐந்தாம் தேதிக்குள் 10000 வாடகை செலுத்த வேண்டும்.',
'hi':'किरायानामा। किरायेदार को हर महीने की पाँचवीं तारीख तक 10000 किराया देना होगा।'}
queries={'en':'How much rent must the tenant pay?', 'ta':'வாடகையாளர் எவ்வளவு வாடகை செலுத்த வேண்டும்?', 'hi':'किरायेदार को कितना किराया देना होगा?'}
results={'ocr':[],'retrieval':[],'scope':'Synthetic pipeline checks only; no legal accuracy or live LLM evaluation.'}
for language,text in texts.items():
    doc=fitz.open();page=doc.new_page()
    css='body {font-size:22pt;}'
    if language!='en':
        font='NotoSansTamil' if language=='ta' else 'NotoSansDevanagari'
        css+=f"@font-face {{font-family:review;src:url({font}.ttf);}} body {{font-family:review;}}"
    page.insert_htmlbox(fitz.Rect(45,50,550,450),text,css=css,archive=fitz.Archive(str(args.assets)))
    native=doc.tobytes();pix=page.get_pixmap(matrix=fitz.Matrix(2,2))
    scan=fitz.open();sp=scan.new_page();sp.insert_image(sp.rect,stream=pix.tobytes('png'))
    for kind,data in [('native',native),('scan',scan.tobytes())]:
        loaded=LegalDocumentLoader(source_language=language).load_pdf(data,f'{language}-{kind}.pdf')
        assert '10000' in loaded.full_text,(language,kind,loaded.warnings,loaded.full_text)
        assert language in detect_languages(loaded.full_text),(language,kind,loaded.full_text)
        if kind=='scan':assert loaded.ocr_pages==[1]
        results['ocr'].append(dict(language=language,input=kind,method=loaded.pages[0].extraction_method,warnings=loaded.warnings))
    for other in ['The parties shall keep business information confidential.','The court has jurisdiction over all disputes.']:
        doc.new_page().insert_text((50,60),other)
    loaded=LegalDocumentLoader(source_language=language).load_pdf(doc.tobytes(),f'{language}.pdf')
    store=LegalVectorStore(model_name='intfloat/multilingual-e5-small')
    try:
        chunks=LegalChunker().chunk_document(loaded);store.build_index(chunks)
        retriever=LegalHybridRetriever(store,chunks)
        for query_language,query in queries.items():
            hits=retriever.hybrid_search(query,top_k=1,use_reranker=False)
            page=hits[0].metadata['page_number'];assert page==1,(language,query_language,page)
            results['retrieval'].append(dict(document=language,query=query_language,top_page=page))
    finally:store.close();doc.close();scan.close()
print(json.dumps(results,ensure_ascii=False,indent=2))
