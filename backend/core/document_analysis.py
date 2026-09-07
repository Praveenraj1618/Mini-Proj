"""Complete, bounded evidence windows and deterministic version differences."""
import difflib
import re
from itertools import zip_longest

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


def document_units(document):
    """Sentence/paragraph units, with all text and original page references."""
    units = []
    for page in contract_pages(document):
        for sentence in re.split(r"(?<=[.;])\s+|\n\s*\n", page.text):
            normalized = " ".join(sentence.split())
            for part in text_parts(normalized, 2400):
                if part.strip():
                    units.append((part.strip(), page.page_num))
    return units


def contract_changes(first, second):
    left, right = document_units(first), document_units(second)
    matcher = difflib.SequenceMatcher(None, [x[0] for x in left], [x[0] for x in right], autojunk=False)
    changes = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        # Bounded pieces are all retained, never truncate a long changed clause.
        for before, after in zip_longest(left[i1:i2], right[j1:j2]):
            changes.append({
                "change_id": len(changes) + 1,
                "clause": f"Text change {len(changes) + 1}",
                "change_type": "replace" if before and after else "delete" if before else "insert",
                "v1_text": before[0] if before else "",
                "v2_text": after[0] if after else "",
                "v1_page": before[1] if before else None,
                "v2_page": after[1] if after else None,
                "impact": "REVIEW REQUIRED",
                "analysis": "Textual change detected; legal impact has not been assessed.",
            })
    return changes
