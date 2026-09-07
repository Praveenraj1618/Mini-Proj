import os
import sys

# Ensure UTF-8 output encoding for Windows PowerShell/CMD
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def test_document_pipeline():
    print("========================================")
    print("Testing Legal Document Intelligence Core")
    print("========================================")

    from core.document_loader import LegalDocumentLoader
    from core.chunker import LegalChunker
    from core.vector_store import LegalVectorStore
    from core.hybrid_retriever import LegalHybridRetriever
    from core.legal_reviewer import LegalReviewer

    from pathlib import Path
    pdf_files = list(Path(".").glob("*.pdf")) + list(Path("..").glob("*.pdf"))
    possible_paths = [
        "legal_ai_evaluation_msa.pdf",
        "../legal_ai_evaluation_msa.pdf",
        "Mutual-NDA-public-2page.pdf",
        "../Mutual-NDA-public-2page.pdf"
    ]
    sample_pdf = None
    for p in possible_paths:
        if os.path.exists(p):
            sample_pdf = str(Path(p).resolve())
            break
    if not sample_pdf and pdf_files:
        sample_pdf = str(pdf_files[0].resolve())


    if not sample_pdf or not os.path.exists(sample_pdf):
        print("❌ No PDF document found to test.")
        return


    print(f"1. Loading document: {sample_pdf}")
    loader = LegalDocumentLoader()
    doc = loader.load_pdf(sample_pdf)
    print(f"   -> Loaded {doc.total_pages} pages, {doc.total_words} words, {len(doc.images)} visual assets.")
    assert doc.total_pages > 0, "Expected at least 1 page."

    print("2. Testing Chunker...")
    chunker = LegalChunker(chunk_size=500, chunk_overlap=100)
    chunks = chunker.chunk_document(doc)
    print(f"   -> Generated {len(chunks)} chunks.")
    assert len(chunks) > 0, "Expected at least 1 chunk."
    assert hasattr(chunks[0], "page_number"), "Chunk must have page_number."
    assert hasattr(chunks[0], "chunk_id"), "Chunk must have chunk_id."
    print(f"   -> Sample chunk ID: {chunks[0].chunk_id}, Page: {chunks[0].page_number}")

    print("3. Testing Vector Store (ChromaDB + HuggingFace Embeddings)...")
    vector_store = LegalVectorStore()
    vector_store.build_index(chunks)
    print(f"   -> Indexed {vector_store.total_chunks} chunks.")
    assert vector_store.total_chunks == len(chunks), "Total chunks mismatch."

    print("4. Testing Hybrid Retriever (Dense Vectors + BM25 Lexical + RRF Reranker)...")
    hybrid_retriever = LegalHybridRetriever(vector_store=vector_store, chunks=chunks)
    hybrid_results = hybrid_retriever.hybrid_search("confidentiality disclosure termination", top_k=3)
    print(f"   -> Hybrid Search returned {len(hybrid_results)} fused results.")
    assert len(hybrid_results) > 0, "Expected hybrid retrieval results."

    print("5. Testing Document Classifier & Intelligence Core...")
    reviewer = LegalReviewer(document=doc, vector_store=vector_store)
    doc_class = reviewer.classify_document()
    print(f"   -> Classified Document Type: {doc_class.get('doc_type')} (Confidence: {doc_class.get('confidence')})")
    assert "doc_type" in doc_class, "Expected doc_type field."

    print("6. Testing Clause Risk Heatmap...")
    heatmap = reviewer.generate_clause_risk_heatmap()
    print(f"   -> Generated {len(heatmap)} risk heatmap sections.")
    assert len(heatmap) > 0, "Expected heatmap section results."

    print("7. Testing Contract Negotiation Assistant...")
    sample_clause = "Either party may terminate this agreement at any time without cause upon 3 days written notice."
    neg_strategy = reviewer.generate_negotiation_strategy(sample_clause)
    print(f"   -> Negotiation Risk Rating: {neg_strategy.get('risk_level')}")
    print(f"   -> Safer Alternative Draft: {neg_strategy.get('safer_alternative')[:70]}...")
    assert "safer_alternative" in neg_strategy, "Expected safer_alternative field."

    print("8. Testing Obligation Extractor...")
    obligations = reviewer.extract_obligations()
    print(f"   -> Extracted {len(obligations)} structured obligations.")
    assert len(obligations) > 0, "Expected extracted obligations."

    print("9. Testing Contract Version Comparison Diff Engine...")
    diff_res = LegalReviewer.compare_contracts(reviewer, reviewer)
    print(f"   -> Comparison completed between {diff_res.get('doc1_name')} and {diff_res.get('doc2_name')}")
    assert "comparison_table" in diff_res, "Expected comparison_table field."

    report_path = "test_review_report.md"
    reviewer.export_report(report_path)
    assert os.path.exists(report_path), "Expected report file to be created."
    print(f"   -> Report exported to {report_path}")

    # Clean up test report
    if os.path.exists(report_path):
        os.remove(report_path)

    print("\n✅ ALL ADVANCED LEGAL INTELLIGENCE CORE TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_document_pipeline()
