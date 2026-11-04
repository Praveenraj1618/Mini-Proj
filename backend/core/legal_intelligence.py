"""Deterministic, page-grounded clause and section evidence extraction.

This module deliberately uses conservative heading heuristics. It does not infer
missing clause boundaries or rewrite source wording; unrecognized text remains
in an explicitly unclassified section.
"""
from dataclasses import dataclass, field
import hashlib
import re
from typing import List, Optional

from core.document_analysis import contract_pages


_METADATA_HEADINGS = {
    "PARTIES", "EFFECTIVE DATE", "REFERENCE DATE", "EFFECTIVE / REFERENCE",
    "DATE", "CASE DETAILS", "CASE INFORMATION", "DOCUMENT DETAILS",
    "DOCUMENT INFORMATION",
}

_FIXTURE_NOISE_MARKERS = (
    "clauseiq test fixture",
    "clauseiq • legal document test set",
    "test data notice",
)

_HEADING_PATTERNS = (
    re.compile(r"^ARTICLE\s+(?:[0-9]+|[IVXLCDM]+)(?:\s*[:.\-–—]?\s*.*)?$", re.I),
    re.compile(r"^SECTION\s+[0-9]+(?:\.[0-9]+)*(?:\s*[:.\-–—]?\s*.*)?$", re.I),
    re.compile(r"^(?P<number>\d+(?:\.\d+)*)(?:[.)]?)\s+(?P<title>[A-Z][^\n]{0,140})$"),
    re.compile(r"^\([a-zivxlcdm]+\)\s+[A-Z][^\n]{0,100}$", re.I),
)


@dataclass
class ClauseEvidence:
    """A source-preserving section with explicit page provenance."""

    clause_id: str
    title: str
    text: str
    page_start: int
    page_end: int
    pages: List[int] = field(default_factory=list)
    section_path: List[str] = field(default_factory=list)
    heading_detected: bool = False
    preceding_context: str = ""
    following_context: str = ""

    def to_dict(self) -> dict:
        return {
            "clause_id": self.clause_id,
            "title": self.title,
            "text": self.text,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "pages": list(self.pages),
            "section_path": list(self.section_path),
            "heading_detected": self.heading_detected,
            "preceding_context": self.preceding_context,
            "following_context": self.following_context,
            "citations": [f"[Page {page}]" for page in self.pages],
        }


def _is_metadata_heading(line: str) -> bool:
    normalized = re.sub(r"\s+", " ", line.strip()).strip(" :.-–—").upper()
    return normalized in _METADATA_HEADINGS


def _is_fixture_noise(line: str) -> bool:
    normalized = " ".join(line.lower().split())
    return any(marker in normalized for marker in _FIXTURE_NOISE_MARKERS)


def _heading_title(line: str) -> Optional[str]:
    candidate = line.strip()
    if not candidate or len(candidate) > 180:
        return None
    # Dates such as "12 September 2026" match the generic numbered-heading
    # heuristic; they are metadata values, never clause headings.
    if re.fullmatch(
        r"(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|"
        r"\d{1,2}\s+[A-Za-z]+\s+\d{4}|"
        r"[A-Za-z]+\s+\d{1,2},?\s+\d{4})",
        candidate,
    ):
        return None
    for pattern in _HEADING_PATTERNS:
        if pattern.match(candidate):
            return candidate
    normalized = re.sub(r"\s+", " ", candidate).strip(" :.-–—").upper()
    if normalized in {"PARTIES", "EFFECTIVE DATE", "REFERENCE DATE", "RECITALS",
                      "BACKGROUND", "WHEREAS", "DEFINITIONS", "INTERPRETATION",
                      "SCHEDULE", "EXHIBIT", "ANNEX", "APPENDIX", "SIGNATURES",
                      "SIGNATURE PAGE", "PREAMBLE"}:
        return candidate
    # Document titles identify the instrument; they are not clause headings.
    if re.search(r"\b(?:AGREEMENT|CONTRACT|DEED|POLICY)\s*$", normalized):
        return None
    words = candidate.split()
    if (2 <= len(candidate) <= 72 and len(words) <= 8
            and candidate.upper() == candidate
            and re.fullmatch(r"[A-Z0-9][A-Z0-9 &()/',.:\\-–—]*", candidate)
            and not candidate.endswith((".", ";", ","))):
        return candidate
    return None


def _stable_id(document_name: str, ordinal: int, title: str, text: str) -> str:
    digest = hashlib.sha1(
        (document_name + "\0" + str(ordinal) + "\0" + title + "\0" + text).encode("utf-8")
    ).hexdigest()[:12]
    return f"clause-{ordinal:04d}-{digest}"


def extract_clause_evidence(document, context_chars: int = 400) -> List[ClauseEvidence]:
    """Extract conservative clause-sized evidence units from a LoadedDocument.

    Headings start a new unit. Text continues across page boundaries until the
    next recognized heading. Page numbers are copied from PageData and never
    estimated. Pages filtered by contract_pages() (e.g. evaluator appendices)
    are excluded consistently with the existing retrieval pipeline.
    """
    if context_chars < 0:
        raise ValueError("context_chars must be non-negative")

    sections: List[dict] = []
    current = None

    def start_section(title: str, page_num: int, heading_detected: bool, line: str):
        nonlocal current
        current = {
            "title": title,
            "text_parts": [line] if line else [],
            "page_start": page_num,
            "page_end": page_num,
            "pages": [page_num],
            "heading_detected": heading_detected,
            "is_metadata": title == "Document information",
        }
        sections.append(current)

    skipping_fixture_notice = False
    for page in contract_pages(document):
        lines = page.text.splitlines()
        if not lines:
            continue
        for line in lines:
            # A standalone PDF page label is already represented by page
            # provenance and should not be copied into the clause text.
            if re.fullmatch(r"page\s+\d+", line.strip(), re.I):
                continue

            # Benchmark banners are removed. Notices can wrap across lines;
            # skip only the notice paragraph, then resume at Background or a
            # recognized contractual heading instead of abandoning the page.
            normalized_line = " ".join(line.lower().split())
            if skipping_fixture_notice:
                if re.match(r"^background\b", line.strip(), re.I) or _heading_title(line):
                    skipping_fixture_notice = False
                else:
                    continue
            if "test data notice" in normalized_line:
                prefix = re.split(r"test data notice", line, maxsplit=1, flags=re.I)[0].strip()
                skipping_fixture_notice = True
                if not prefix:
                    continue
                line = prefix
            elif _is_fixture_noise(line):
                continue

            # PDF extraction can place a signature label at the end of the
            # preceding clause line. Split it into its own evidence section.
            signature = re.search(r"\b(signatures?\s*\(sample only\))", line, re.I)
            if signature:
                clause_text = line[:signature.start()].strip()
                signature_text = line[signature.start():].strip()
                if clause_text:
                    if current is None:
                        start_section("Unclassified section", page.page_num, False, clause_text)
                    else:
                        current["text_parts"].append(clause_text)
                start_section("Signatures", page.page_num, True, signature_text)
                continue

            # Parties/date/reference fields describe the document rather than
            # forming independent operative clauses. Keep their source text,
            # but group adjacent fields into one compact metadata section.
            if _is_metadata_heading(line):
                if current is None:
                    start_section("Document information", page.page_num, False, line)
                elif not current.get("is_metadata"):
                    # If the document opens with a title/preamble before its
                    # parties/date block, keep that material in the same card.
                    if len(sections) == 1 and current["title"] == "Unclassified section":
                        current["title"] = "Document information"
                        current["is_metadata"] = True
                        current["text_parts"].append(line)
                        current["page_end"] = page.page_num
                        if page.page_num not in current["pages"]:
                            current["pages"].append(page.page_num)
                    else:
                        start_section("Document information", page.page_num, False, line)
                elif line:
                    current["text_parts"].append(line)
                    current["page_end"] = page.page_num
                    if page.page_num not in current["pages"]:
                        current["pages"].append(page.page_num)
                continue

            heading = _heading_title(line)
            if heading:
                start_section(heading, page.page_num, True, line)
                continue
            if current is None:
                start_section("Unclassified section", page.page_num, False, line)
            elif line:
                current["text_parts"].append(line)
            else:
                current["text_parts"].append("")
            if current is not None:
                current["page_end"] = page.page_num
                if page.page_num not in current["pages"]:
                    current["pages"].append(page.page_num)

    evidence = []
    for index, section in enumerate(sections, 1):
        text = "\n".join(section["text_parts"]).strip()
        if not text:
            continue
        evidence.append(ClauseEvidence(
            clause_id=_stable_id(document.file_name, index, section["title"], text),
            title=section["title"],
            text=text,
            page_start=section["page_start"],
            page_end=section["page_end"],
            pages=section["pages"],
            section_path=[section["title"]] if section["heading_detected"] else [],
            heading_detected=section["heading_detected"],
        ))

    for index, item in enumerate(evidence):
        if context_chars == 0:
            continue
        if index:
            item.preceding_context = evidence[index - 1].text[-context_chars:]
        if index + 1 < len(evidence):
            item.following_context = evidence[index + 1].text[:context_chars]

    return evidence


__all__ = ["ClauseEvidence", "extract_clause_evidence"]
