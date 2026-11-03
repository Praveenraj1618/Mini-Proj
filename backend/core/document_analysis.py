"""Complete, bounded evidence windows for grounded document analysis."""
import re

from core.chunker import is_evaluation_artifact_text


def contract_pages(document):
    return [p for p in document.pages if p.text.strip() and not is_evaluation_artifact_text(p.text)]


def text_parts(text, maximum):
    """Split without discarding characters, including unusually long clauses."""
    while len(text) > maximum:
        boundary = text.rfind(" ", 0, maximum + 1)
        if boundary < maximum // 2:
            boundary = maximum
        yield text[:boundary]
        text = text[boundary:]
    if text:
        yield text


def evidence_windows(document, maximum):
    windows = []
    current = ""
    for page in contract_pages(document):
        label = f"[Page {page.page_num}]\n"
        for part in text_parts(page.text, maximum - len(label) - 2):
            entry = label + part
            if current and len(current) + len(entry) + 2 > maximum:
                windows.append(current)
                current = ""
            current += ("\n\n" if current else "") + entry
    if current:
        windows.append(current)
    return windows


