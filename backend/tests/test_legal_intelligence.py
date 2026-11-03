import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from core.document_loader import LoadedDocument, PageData
from core.legal_intelligence import extract_clause_evidence


def make_document(pages, name="sample-contract.pdf"):
    return LoadedDocument(
        file_path="<memory>",
        file_name=name,
        total_pages=len(pages),
        pages=[
            PageData(
                page_num=number,
                text=text,
                char_count=len(text),
                image_count=0,
            )
            for number, text in pages
        ],
    )


def test_extracts_numbered_and_article_headings_with_original_text():
    document = make_document([
        (1, "MASTER SERVICES AGREEMENT\n1. PAYMENT TERMS\nCustomer shall pay within 30 days.\n\nARTICLE II — DEFINITIONS\n“Services” means the services described in Schedule A."),
    ])

    clauses = extract_clause_evidence(document)

    assert len(clauses) == 3
    assert clauses[0].title == "Unclassified section"
    assert clauses[0].text == "MASTER SERVICES AGREEMENT"
    assert clauses[1].title == "1. PAYMENT TERMS"
    assert clauses[1].heading_detected is True
    assert clauses[1].text == "1. PAYMENT TERMS\nCustomer shall pay within 30 days."
    assert clauses[1].pages == [1]
    assert clauses[2].title == "ARTICLE II — DEFINITIONS"
    assert "“Services” means the services described in Schedule A." in clauses[2].text


def test_clause_continuing_on_next_page_keeps_both_page_references():
    document = make_document([
        (1, "2. CONFIDENTIALITY\nEach party shall protect all confidential information"),
        (2, "received from the other party and shall use it only to perform this Agreement."),
        (3, "3. TERM\nThis Agreement continues for one year."),
    ])

    clauses = extract_clause_evidence(document)

    confidentiality = clauses[0]
    assert confidentiality.title == "2. CONFIDENTIALITY"
    assert confidentiality.page_start == 1
    assert confidentiality.page_end == 2
    assert confidentiality.pages == [1, 2]
    assert "received from the other party" in confidentiality.text
    assert confidentiality.to_dict()["citations"] == ["[Page 1]", "[Page 2]"]
    assert clauses[1].title == "3. TERM"
    assert clauses[1].pages == [3]


def test_unheaded_text_is_retained_as_unclassified_evidence():
    document = make_document([
        (1, "This agreement is made between the parties.\nThe parties agree as follows."),
    ])

    clauses = extract_clause_evidence(document)

    assert len(clauses) == 1
    assert clauses[0].title == "Unclassified section"
    assert clauses[0].heading_detected is False
    assert clauses[0].section_path == []
    assert "This agreement is made between the parties." in clauses[0].text


def test_evaluation_artifact_pages_are_excluded():
    document = make_document([
        (1, "1. PAYMENT\nThe customer shall pay the fee."),
        (2, "EVALUATION DESIGN NOTES — DO NOT USE AS CONTRACT TERMS\nKnown evaluation targets"),
    ])

    clauses = extract_clause_evidence(document)

    assert len(clauses) == 1
    assert clauses[0].pages == [1]
    assert "evaluation" not in clauses[0].text.lower()


def test_ids_are_stable_for_same_document_content():
    pages = [(1, "1. PAYMENT\nThe fee is due within 30 days.")]
    first = extract_clause_evidence(make_document(pages, "same.pdf"))
    second = extract_clause_evidence(make_document(pages, "same.pdf"))

    assert first[0].clause_id == second[0].clause_id


def test_context_windows_are_neighbouring_evidence_only():
    document = make_document([
        (1, "1. PAYMENT\nPAYMENT BODY\n2. TERMINATION\nTERMINATION BODY"),
    ])

    clauses = extract_clause_evidence(document, context_chars=20)

    assert clauses[0].preceding_context == ""
    assert clauses[0].following_context == "2. TERMINATION\nTERMINATION BODY"[:20]
    assert clauses[1].preceding_context == "1. PAYMENT\nPAYMENT BODY"[-20:]
    assert clauses[1].following_context == ""


def test_negative_context_size_is_rejected():
    document = make_document([(1, "1. PAYMENT\nFee is due.")])

    try:
        extract_clause_evidence(document, context_chars=-1)
    except ValueError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("Expected ValueError for negative context size")
