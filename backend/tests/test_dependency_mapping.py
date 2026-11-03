import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from core.document_loader import LoadedDocument, PageData
from core.dependency_mapping import build_clause_relationships


def make_document(text):
    return LoadedDocument(
        file_path="<memory>", file_name="agreement.pdf", total_pages=2,
        pages=[
            PageData(page_num=1, text=text, char_count=len(text), image_count=0),
            PageData(page_num=2, text="4. GENERAL\nThis agreement is governed by its written terms.", char_count=57, image_count=0),
        ],
    )


def test_explicit_reference_is_resolved_and_page_grounded():
    result = build_clause_relationships(make_document(
        "1. PAYMENT\nFees are due within 30 days. Subject to Clause 2, overdue amounts accrue interest.\n"
        "2. LATE PAYMENT\nLate payments accrue interest after the due date."
    ))
    relation = next(item for item in result["relationships"] if item["source_type"] == "explicit")
    assert relation["type"] == "depends_on"
    assert relation["pages"] == [1]
    assert relation["target_pages"] == [1]
    assert "Clause 2" in relation["evidence"]


def test_unresolved_reference_is_not_invented_as_a_node():
    result = build_clause_relationships(make_document(
        "1. PAYMENT\nSee Clause 99 for details. Payment is due within 30 days."
    ))
    assert not any(item["source_type"] == "explicit" for item in result["relationships"])


def test_inferred_relationships_are_clearly_marked_as_candidates():
    result = build_clause_relationships(make_document(
        "1. PAYMENT\nThe customer shall pay each invoice within 30 days.\n"
        "2. LATE PAYMENT\nOverdue payment may accrue interest after a delay."
    ))
    inferred = [item for item in result["relationships"] if item["source_type"] == "inferred"]
    assert inferred
    assert all(item["type"] == "related_to" for item in inferred)
    assert all(item["confidence"] == "heuristic_candidate" for item in inferred)
    assert all(item["evidence"] is None for item in inferred)


def test_graph_contains_clause_nodes_and_relationship_edges():
    result = build_clause_relationships(make_document(
        "1. PAYMENT\nSubject to Clause 2, fees are due monthly.\n"
        "2. LATE PAYMENT\nOverdue amounts accrue interest."
    ))
    assert len(result["graph"]["nodes"]) >= 2
    assert result["graph"]["edges"]
    assert result["impact_analysis"]


def test_llm_relationship_requires_valid_ids_type_page_and_verbatim_evidence():
    document = make_document(
        "1. PAYMENT\nThe customer shall pay each invoice within 30 days.\n"
        "2. LATE PAYMENT\nOverdue payments may accrue interest after a delay."
    )
    clauses = __import__("core.legal_intelligence", fromlist=["extract_clause_evidence"]).extract_clause_evidence(document)
    source_id, target_id = clauses[0].clause_id, clauses[1].clause_id
    import json
    response = json.dumps([{
        "source": source_id, "target": target_id, "type": "triggers",
        "evidence": "The customer shall pay each invoice within 30 days.",
        "page": 1, "explanation": "Payment performance may relate to the late-payment section."
    }])
    result = build_clause_relationships(document, llm_invoke=lambda prompt: response)
    model_links = [
        item for item in result["relationships"]
        if item["confidence"] == "model_candidate_not_calibrated"
    ]
    assert len(model_links) == 1
    assert model_links[0]["type"] == "triggers"
    assert model_links[0]["source_type"] == "inferred"
    assert result["methodology"]["llm_status"] == "completed"


def test_llm_hallucinated_evidence_is_rejected():
    document = make_document(
        "1. PAYMENT\nThe customer shall pay within 30 days.\n"
        "2. LATE PAYMENT\nOverdue payments may accrue interest."
    )
    clauses = __import__("core.legal_intelligence", fromlist=["extract_clause_evidence"]).extract_clause_evidence(document)
    import json
    response = json.dumps([{
        "source": clauses[0].clause_id, "target": clauses[1].clause_id,
        "type": "triggers", "evidence": "The customer must pay a $10,000 penalty.",
        "page": 1, "explanation": "Invented quote."
    }])
    result = build_clause_relationships(document, llm_invoke=lambda prompt: response)
    assert not any(item["confidence"] == "model_candidate_not_calibrated"
                   for item in result["relationships"])
