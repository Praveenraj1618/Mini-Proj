import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from core.document_loader import LoadedDocument, PageData
from core.legal_intelligence import extract_clause_evidence
from core.legal_reviewer import LegalReviewer
import core.legal_reviewer as reviewer_module
import json


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
        (1, "1. PAYMENT\nPayment body\n2. TERMINATION\nTermination body"),
    ])

    clauses = extract_clause_evidence(document, context_chars=20)

    assert clauses[0].preceding_context == ""
    assert clauses[0].following_context == "2. TERMINATION\nTermination body"[:20]
    assert clauses[1].preceding_context == "1. PAYMENT\nPayment body"[-20:]
    assert clauses[1].following_context == ""


def test_negative_context_size_is_rejected():
    document = make_document([(1, "1. PAYMENT\nFee is due.")])

    try:
        extract_clause_evidence(document, context_chars=-1)
    except ValueError as error:
        assert "non-negative" in str(error)
    else:
        raise AssertionError("Expected ValueError for negative context size")



def test_clause_simplification_returns_grounded_fields_and_page_citations(monkeypatch):
    document = make_document([
        (1, "1. PAYMENT\nThe Buyer shall pay the Seller within a reasonable time after delivery.")
    ])
    reviewer = LegalReviewer(document, None)
    monkeypatch.setattr(reviewer_module, "check_api_key_configured", lambda: True)
    response = {
        "simplified_explanation": "The Buyer must pay after delivery, but the deadline is not specific. [Page 1]",
        "ambiguities": [{
            "issue": "reasonable time",
            "why_it_matters": "The payment deadline may be disputed.",
            "citations": ["[Page 1]"],
        }],
        "missing_information": [{
            "item": "A defined payment deadline",
            "reason": "No number of days is stated.",
            "citations": ["[Page 1]"],
        }],
        "clarification_questions": ["How many days after delivery is payment due?"],
        "uncertainty": {
            "explicit_terms": [{"statement": "Payment follows delivery.", "citations": ["[Page 1]"]}],
            "assumptions": [],
            "possible_interpretations": [],
        },
    }
    monkeypatch.setattr(reviewer, "_invoke_llm", lambda *args, **kwargs: json.dumps(response))

    result = reviewer.simplify_clause(extract_clause_evidence(document)[0].clause_id)

    assert result["status"] == "analyzed"
    assert result["pages"] == [1]
    assert result["ambiguities"][0]["issue"] == "reasonable time"
    assert result["clarification_questions"]
    assert result["uncertainty"]["explicit_terms"]


def test_clause_simplification_rejects_fabricated_page_citations(monkeypatch):
    document = make_document([(1, "1. PAYMENT\nPayment is due after delivery.")])
    reviewer = LegalReviewer(document, None)
    monkeypatch.setattr(reviewer_module, "check_api_key_configured", lambda: True)
    response = {
        "simplified_explanation": "Payment follows delivery. [Page 99]",
        "ambiguities": [],
        "missing_information": [],
        "clarification_questions": [],
        "uncertainty": {"explicit_terms": [], "assumptions": [], "possible_interpretations": []},
    }
    monkeypatch.setattr(reviewer, "_invoke_llm", lambda *args, **kwargs: json.dumps(response))

    result = reviewer.simplify_clause(extract_clause_evidence(document)[0].clause_id)

    assert result["status"] == "analysis_failed"
    assert "could not be validated" in result["error"]


def test_clause_simplification_reports_unknown_clause_id():
    reviewer = LegalReviewer(make_document([(1, "1. PAYMENT\nPayment is due.")]), None)

    result = reviewer.simplify_clause("not-a-real-clause")

    assert result["status"] == "not_found"


def test_recognizes_common_standalone_contract_headings():
    document = make_document([
        (1, "MUTUAL NON-DISCLOSURE AGREEMENT\nPARTIES\nParty A and Party B enter this agreement.\nEFFECTIVE DATE\nThis agreement starts on 1 January 2026.\nRECITALS\nThe parties wish to exchange information."),
    ])
    clauses = extract_clause_evidence(document)
    titles = [clause.title for clause in clauses]
    assert titles == ["Document information", "RECITALS"]
    metadata = clauses[0]
    assert "MUTUAL NON-DISCLOSURE AGREEMENT" in metadata.text
    assert "PARTIES" in metadata.text
    assert "EFFECTIVE DATE" in metadata.text
    assert "This agreement starts on 1 January 2026." in metadata.text
    assert clauses[1].heading_detected is True


def test_recognizes_uppercase_headings_across_contract_styles():
    samples = [
        ("NON-DISCLOSURE AGREEMENT", "CONFIDENTIALITY", "Each party protects information."),
        ("MASTER SERVICES AGREEMENT", "PAYMENT TERMS", "Invoices are due within 30 days."),
        ("EMPLOYMENT AGREEMENT", "TERMINATION", "Either party may terminate on notice."),
        ("LEASE AGREEMENT", "REPAIRS AND MAINTENANCE", "Tenant must keep the premises clean."),
    ]
    for filename, heading, body in samples:
        clauses = extract_clause_evidence(
            make_document([(1, f"{filename}\n{heading}\n{body}")], name=filename + ".pdf")
        )
        assert len(clauses) == 2
        assert clauses[1].title == heading
        assert body in clauses[1].text


def test_simplification_rejects_findings_without_valid_citations(monkeypatch):
    document = make_document([(1, "6. LIABILITY\nNeither party is liable for indirect loss.")])
    reviewer = LegalReviewer(document, None)
    monkeypatch.setattr(reviewer_module, "check_api_key_configured", lambda: True)
    response = {
        "simplified_explanation": "Indirect losses are excluded. [Page 1]",
        "ambiguities": [{
            "issue": "indirect loss",
            "why_it_matters": "The term may be interpreted differently.",
            "citations": ["[Page 99]"],
        }],
        "missing_information": [],
        "clarification_questions": [],
        "uncertainty": {"explicit_terms": [], "assumptions": [], "possible_interpretations": []},
    }
    monkeypatch.setattr(reviewer, "_invoke_llm", lambda *args, **kwargs: json.dumps(response))
    result = reviewer.simplify_clause(extract_clause_evidence(document)[0].clause_id)
    assert result["status"] == "analysis_failed"


def test_simplification_handles_provider_failure_without_crashing(monkeypatch):
    document = make_document([(1, "1. PAYMENT\nPayment is due within 30 days.")])
    reviewer = LegalReviewer(document, None)
    monkeypatch.setattr(reviewer_module, "check_api_key_configured", lambda: True)

    def fail_provider(*args, **kwargs):
        raise TimeoutError("simulated provider timeout")

    monkeypatch.setattr(reviewer, "_invoke_llm", fail_provider)
    result = reviewer.simplify_clause(extract_clause_evidence(document)[0].clause_id)
    assert result["status"] == "analysis_failed"
    assert result["diagnostic"]
    assert result["source_text"].startswith("1. PAYMENT")



def test_llm_retries_transient_gemini_503_then_succeeds(monkeypatch):
    from types import SimpleNamespace

    class TemporaryUnavailable(Exception):
        status_code = 503

    class FlakyClient:
        def __init__(self):
            self.calls = 0

        def bind(self, **kwargs):
            return self

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                raise TemporaryUnavailable("temporary upstream outage")
            return SimpleNamespace(content="OK")

    reviewer = LegalReviewer(make_document([(1, "1. PAYMENT\nPayment is due.")]), None)
    gemini = FlakyClient()
    monkeypatch.setattr(reviewer, "_candidate_clients", lambda vision=False: [("Gemini", gemini)])
    monkeypatch.setattr(reviewer_module.time, "sleep", lambda seconds: None)

    assert reviewer._invoke_llm([]) == "OK"
    assert gemini.calls == 2
    assert reviewer.last_llm_diagnostic == "Provider Gemini succeeded."


def test_llm_falls_back_after_three_transient_gemini_503_failures(monkeypatch):
    from types import SimpleNamespace

    class TemporaryUnavailable(Exception):
        status_code = 503

    class AlwaysUnavailableClient:
        def __init__(self):
            self.calls = 0

        def bind(self, **kwargs):
            return self

        def invoke(self, messages):
            self.calls += 1
            raise TemporaryUnavailable("temporary upstream outage")

    class HealthyClient:
        def __init__(self):
            self.calls = 0

        def bind(self, **kwargs):
            return self

        def invoke(self, messages):
            self.calls += 1
            return SimpleNamespace(content="OK from fallback")

    reviewer = LegalReviewer(make_document([(1, "1. PAYMENT\nPayment is due.")]), None)
    gemini = AlwaysUnavailableClient()
    groq = HealthyClient()
    monkeypatch.setattr(
        reviewer, "_candidate_clients",
        lambda vision=False: [("Gemini", gemini), ("Groq", groq)]
    )
    monkeypatch.setattr(reviewer_module.time, "sleep", lambda seconds: None)

    assert reviewer._invoke_llm([]) == "OK from fallback"
    assert gemini.calls == 3
    assert groq.calls == 1
    assert reviewer.last_llm_diagnostic == (
        "Fallback provider Groq succeeded after Gemini (TemporaryUnavailable)."
    )



def test_fixture_banners_are_removed_and_document_metadata_is_grouped():
    document = make_document([
        (1, "CLAUSEIQ TEST FIXTURE • FICTIONAL SAMPLE — NOT FOR LEGAL USE\n"
            "CLAUSEIQ • LEGAL DOCUMENT TEST SET\n"
            "Sample Civil Court Judgment\n"
            "PARTIES\nFictional Civil Court of Meridian City — Case No. CV-2026-0142\n"
            "EFFECTIVE / REFERENCE\nEFFECTIVE / REFERENCE\n"
            "DATE\n12 September 2026\n"
            "TEST DATA NOTICE. This document is fictional and created solely to test document upload, OCR, summaries, clause-level risk analysis, obligation extraction, grounded Q&A, and document comparison in ClauseIQ.")
    ])

    clauses = extract_clause_evidence(document)

    assert all("CLAUSEIQ TEST FIXTURE" not in clause.text for clause in clauses)
    assert all("TEST DATA NOTICE" not in clause.text for clause in clauses)
    metadata = [clause for clause in clauses if clause.title == "Document information"]
    assert len(metadata) == 1
    assert not any(clause.title == "Unclassified section" for clause in clauses)
    assert "PARTIES" in metadata[0].text
    assert "EFFECTIVE / REFERENCE" in metadata[0].text
    assert "12 September 2026" in metadata[0].text
    assert not any(clause.title in {"PARTIES", "DATE", "EFFECTIVE / REFERENCE"} for clause in clauses)



def test_page_labels_and_wrapped_test_notice_are_excluded():
    document = make_document([
        (1, "1. PAYMENT\nPayment is due within 30 days.\n"
            "TEST DATA NOTICE. This document is fictional.\n"
            "Names, addresses, dates, events, and terms are invented.\n"
            "It is not legal advice or a valid legal instrument.")
    ])
    clauses = extract_clause_evidence(document)
    assert len(clauses) == 1
    assert "TEST DATA NOTICE" not in clauses[0].text
    assert "Names, addresses" not in clauses[0].text


def test_signature_block_is_split_from_final_contract_clause():
    document = make_document([
        (1, "8. POLICIES AND GOVERNING LAW\n"
            "This agreement is governed by Indian law. Signatures (sample only) "
            "For Party A For Party B Name: ______ Date: ______")
    ])
    clauses = extract_clause_evidence(document)
    assert [clause.title for clause in clauses] == [
        "8. POLICIES AND GOVERNING LAW", "Signatures"
    ]
    assert "governed by Indian law" in clauses[0].text
    assert "For Party A" in clauses[1].text


def test_wrapped_fixture_notice_does_not_discard_contract_text_after_it():
    document = make_document([
        (1, "Document Title Agreement\n"
            "PARTIES\nCustomer and Provider\n"
            "TEST DATA NOTICE. This document is fictional and created solely\n"
            "to test document upload, OCR, summaries, and clause analysis.\n"
            "It is not legal advice or a valid legal instrument.\n"
            "BACKGROUND\nThe parties wish to work together.\n"
            "1. SCOPE AND DELIVERABLES\nProvider will deliver the services.\n"
            "2. PAYMENT\nCustomer will pay within 30 days."),
        (2, "3. TERM\nThe agreement lasts one year.\n"
            "4. CONFIDENTIALITY\nEach party must protect confidential information.")
    ])

    clauses = extract_clause_evidence(document)
    titles = [clause.title for clause in clauses]

    assert any(title.startswith("1. SCOPE") for title in titles)
    assert any(title.startswith("2. PAYMENT") for title in titles)
    assert any(title.startswith("3. TERM") for title in titles)
    assert any(title.startswith("4. CONFIDENTIALITY") for title in titles)
    combined_text = "\n".join(clause.text for clause in clauses)
    assert "Provider will deliver the services." in combined_text
    assert "Customer will pay within 30 days." in combined_text
    assert "TEST DATA NOTICE" not in combined_text
    assert "It is not legal advice" not in combined_text


def test_extracts_duplicated_numbered_headings_as_separate_clauses():
    document = make_document([
        (1, "RECITALS AND BACKGROUND\n"
            "This agreement governs the services.\n"
            "1. 1. Definitions and Interpretation\n"
            "Definitions apply throughout this Agreement.\n"
            "2. 2. Services and Scope\n"
            "Provider shall configure the platform.\n"
            "3. 3. Implementation Plan and Dependencies\n"
            "The parties shall follow the project plan.\n"
            "4. 4. Fees, Invoicing and Taxes\n"
            "Customer shall pay the fees.\n"
            "5. 5. Acceptance and Remediation\n"
            "Customer shall review each Deliverable."),
        (2, "6. 6. Availability and Support\n"
            "Provider shall target monthly availability.\n"
            "7. 7. Data Protection and Security\n"
            "Provider shall protect Customer Data.\n"
            "8. 8. Intellectual Property and Licence\n"
            "Each party retains its background intellectual property.\n"
            "9. 9. Change Control and Order of Precedence\n"
            "Changes require written approval.\n"
            "10. 10. Termination, Exit and Liability\n"
            "Either party may terminate for material breach."),
        (3, "The termination clause continues on this page.\n"
            "SCHEDULES AND CONTRACT PARTICULARS\n"
            "Schedule 1 – Contract particulars")
    ])

    clauses = extract_clause_evidence(document)
    titles = [clause.title for clause in clauses]

    expected = [
        "RECITALS AND BACKGROUND",
        "1. 1. Definitions and Interpretation",
        "2. 2. Services and Scope",
        "3. 3. Implementation Plan and Dependencies",
        "4. 4. Fees, Invoicing and Taxes",
        "5. 5. Acceptance and Remediation",
        "6. 6. Availability and Support",
        "7. 7. Data Protection and Security",
        "8. 8. Intellectual Property and Licence",
        "9. 9. Change Control and Order of Precedence",
        "10. 10. Termination, Exit and Liability",
        "SCHEDULES AND CONTRACT PARTICULARS",
    ]
    assert titles == expected
    termination = clauses[10]
    assert termination.pages == [2, 3]
    assert "continues on this page" in termination.text
