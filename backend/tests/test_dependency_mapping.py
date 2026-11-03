import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from core.document_loader import LoadedDocument, PageData
from core.dependency_mapping import build_clause_relationships
from core.legal_intelligence import extract_clause_evidence


def make_document(text, page_two="4. GENERAL\nThis agreement is governed by its written terms."):
    return LoadedDocument(
        file_path="<memory>", file_name="agreement.pdf", total_pages=2,
        pages=[
            PageData(page_num=1, text=text, char_count=len(text), image_count=0),
            PageData(page_num=2, text=page_two, char_count=len(page_two), image_count=0),
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


def test_date_and_unclassified_preamble_are_not_graph_nodes_or_clauses():
    result = build_clause_relationships(make_document(
        "CLAUSEIQ TEST FIXTURE • FICTIONAL SAMPLE\n"
        "13 October 2026\n"
        "Background\nThe parties expect to discuss a possible integration.\n"
        "1. CONFIDENTIAL INFORMATION\nConfidential Information means non-public information.\n"
        "2. REMEDIES\nA party may seek relief for a breach."
    ))
    assert len(result["clauses"]) == 2
    assert all(node["label"] != "13 October 2026" for node in result["graph"]["nodes"])
    assert all(node["label"] != "Unclassified section" for node in result["graph"]["nodes"])


def test_no_keyword_only_relationships_are_invented_without_model_review():
    result = build_clause_relationships(make_document(
        "1. PAYMENT\nThe customer shall pay each invoice within 30 days.\n"
        "2. LATE PAYMENT\nOverdue payment may accrue interest after a delay."
    ))
    assert result["relationships"] == []
    assert result["graph"]["edges"] == []
    assert len(result["isolated_clauses"]) == 2
    assert result["graph"]["nodes"] == []


def test_graph_only_contains_clauses_participating_in_verified_relationships():
    result = build_clause_relationships(make_document(
        "1. PAYMENT\nSubject to Clause 2, fees are due monthly.\n"
        "2. LATE PAYMENT\nOverdue amounts accrue interest.\n"
        "3. GOVERNING LAW\nThe laws of Meridian apply."
    ))
    assert len(result["graph"]["nodes"]) == 2
    assert len(result["graph"]["edges"]) == 1
    assert len(result["isolated_clauses"]) == 1


def test_llm_relationship_requires_both_verbatim_quotes_and_correct_pages():
    document = make_document(
        "1. CONFIDENTIAL INFORMATION\nConfidential Information means non-public information disclosed by a party.\n"
        "2. REMEDIES AND LIABILITY\nA party may seek injunctive relief for an actual or threatened breach of confidentiality."
    )
    clauses = [c.to_dict() for c in extract_clause_evidence(document) if c.heading_detected]
    response = json.dumps([{
        "source": clauses[1]["clause_id"], "target": clauses[0]["clause_id"],
        "type": "depends_on",
        "source_evidence": "A party may seek injunctive relief for an actual or threatened breach of confidentiality.",
        "target_evidence": "Confidential Information means non-public information disclosed by a party.",
        "source_page": 1, "target_page": 1,
        "explanation": "The remedies provision concerns breach of the defined confidential information obligation."
    }])
    result = build_clause_relationships(document, llm_invoke=lambda prompt: response)
    inferred = [r for r in result["relationships"] if r["source_type"] == "inferred"]
    assert len(inferred) == 1
    assert inferred[0]["source"] == clauses[1]["clause_id"]
    assert inferred[0]["target"] == clauses[0]["clause_id"]
    assert inferred[0]["source_evidence"] and inferred[0]["target_evidence"]
    assert result["methodology"]["llm_status"] == "completed"


def test_llm_hallucinated_quote_from_either_clause_is_rejected():
    document = make_document(
        "1. PAYMENT\nThe customer shall pay within 30 days.\n"
        "2. LATE PAYMENT\nOverdue payments may accrue interest."
    )
    clauses = [c.to_dict() for c in extract_clause_evidence(document) if c.heading_detected]
    response = json.dumps([{
        "source": clauses[0]["clause_id"], "target": clauses[1]["clause_id"],
        "type": "triggers",
        "source_evidence": "The customer shall pay a $10,000 penalty.",
        "target_evidence": "Overdue payments may accrue interest.",
        "source_page": 1, "target_page": 1,
        "explanation": "Invented source evidence."
    }])
    result = build_clause_relationships(document, llm_invoke=lambda prompt: response)
    assert not any(r["source_type"] == "inferred" for r in result["relationships"])
