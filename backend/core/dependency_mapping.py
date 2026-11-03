"""Conservative, citation-grounded clause relationship mapping.

Explicit cross-references are resolved against extracted section headings. A small
rule set adds candidate thematic links; those are always labelled inferred and
must not be read as contractual dependencies or legal conclusions.
"""
import re
from collections import deque
from typing import Dict, Iterable, List

from core.legal_intelligence import extract_clause_evidence


_REFERENCE_RE = re.compile(
    r"(?P<cue>subject\s+to|notwithstanding|except\s+as\s+provided\s+in|"
    r"pursuant\s+to|in\s+accordance\s+with|under|as\s+set\s+out\s+in|"
    r"see|refer\s+to)?\s*"
    r"(?:clause|section|article|paragraph)\s+"
    r"(?P<number>\d+(?:\.\d+)*|[ivxlcdm]+)",
    re.IGNORECASE,
)
_CONCEPTS = {
    "payment": {"payment", "invoice", "fee", "amount", "pay", "paid"},
    "late_payment": {"late", "overdue", "interest", "default", "delayed"},
    "termination": {"terminat", "expiry", "expire", "end this agreement"},
    "breach": {"breach", "default", "violation", "failure to perform"},
    "notice": {"notice", "notify", "written notice", "days' notice"},
    "confidentiality": {"confidential", "disclos", "proprietary", "non-disclosure"},
    "liability": {"liability", "indemn", "damages", "limitation of liability"},
}


def _normal(value: str) -> str:
    return re.sub(r"[^a-z0-9.]+", " ", value.lower()).strip()


def _clause_number(clause: dict):
    match = re.match(r"^(?:clause|section|article)?\s*(\d+(?:\.\d+)*|[ivxlcdm]+)", clause["title"], re.I)
    return match.group(1).lower() if match else None


def _relationship_type(cue: str) -> str:
    cue = (cue or "").lower()
    if "subject to" in cue:
        return "depends_on"
    if "notwithstanding" in cue:
        return "overrides"
    if "except as provided" in cue:
        return "exception_to"
    return "references"


def build_clause_relationships(document) -> dict:
    """Return clauses, explicit references, conservative thematic candidates and graph data."""
    clauses = [item.to_dict() for item in extract_clause_evidence(document)]
    by_number: Dict[str, dict] = {}
    for clause in clauses:
        number = _clause_number(clause)
        if number and number not in by_number:
            by_number[number] = clause

    relationships: List[dict] = []
    seen = set()
    for source in clauses:
        for match in _REFERENCE_RE.finditer(source["text"]):
            target = by_number.get(match.group("number").lower())
            if not target or target["clause_id"] == source["clause_id"]:
                continue
            quote_start = max(0, match.start() - 70)
            quote_end = min(len(source["text"]), match.end() + 90)
            evidence = source["text"][quote_start:quote_end].strip()
            rel_type = _relationship_type(match.group("cue") or "")
            key = (source["clause_id"], target["clause_id"], rel_type, "explicit")
            if key in seen:
                continue
            seen.add(key)
            relationships.append({
                "source": source["clause_id"], "target": target["clause_id"],
                "type": rel_type, "source_type": "explicit",
                "confidence": "direct_reference", "evidence": evidence,
                "pages": source["pages"], "target_pages": target["pages"],
                "explanation": f'This clause expressly refers to “{target["title"]}”.',
            })

    # Inferred links are deliberately broad and labelled as candidates only.
    concept_matches = {}
    for clause in clauses:
        text = _normal(clause["title"] + " " + clause["text"])
        concepts = {
            concept for concept, terms in _CONCEPTS.items()
            if any(_normal(term) in text for term in terms)
        }
        concept_matches[clause["clause_id"]] = concepts

    for index, left in enumerate(clauses):
        for right in clauses[index + 1:]:
            shared = concept_matches[left["clause_id"]] & concept_matches[right["clause_id"]]
            # Only pair distinct clause topics through recognizable legal triggers.
            pair = shared & {"payment", "late_payment", "termination", "breach", "notice"}
            if not pair:
                continue
            if any(
                rel["source"] == left["clause_id"] and rel["target"] == right["clause_id"]
                for rel in relationships
            ):
                continue
            concept = sorted(pair)[0]
            relationships.append({
                "source": left["clause_id"], "target": right["clause_id"],
                "type": "related_to", "source_type": "inferred",
                "confidence": "heuristic_candidate", "evidence": None,
                "pages": left["pages"], "target_pages": right["pages"],
                "explanation": f"Both sections contain language associated with {concept.replace('_', ' ')}; review the original wording to confirm whether they interact.",
                "shared_concepts": sorted(pair),
            })

    nodes = [{
        "id": clause["clause_id"], "label": clause["title"],
        "pages": clause["pages"], "heading_detected": clause["heading_detected"],
    } for clause in clauses]
    edges = [{
        "id": f'edge-{index + 1}', "source": rel["source"],
        "target": rel["target"], "label": rel["type"],
        "source_type": rel["source_type"],
    } for index, rel in enumerate(relationships)]

    adjacency = {}
    for rel in relationships:
        adjacency.setdefault(rel["source"], []).append(rel["target"])
        if rel["source_type"] == "inferred":
            adjacency.setdefault(rel["target"], []).append(rel["source"])

    impact = []
    for clause in clauses:
        visited, queue = {clause["clause_id"]}, deque([clause["clause_id"]])
        while queue:
            current = queue.popleft()
            for neighbour in adjacency.get(current, []):
                if neighbour not in visited:
                    visited.add(neighbour)
                    queue.append(neighbour)
        linked = visited - {clause["clause_id"]}
        if linked:
            impact.append({
                "clause_id": clause["clause_id"],
                "related_clause_ids": sorted(linked),
                "note": "Linked clauses are review leads, not proof that a legal consequence automatically follows.",
            })

    return {
        "clauses": clauses, "relationships": relationships,
        "graph": {"nodes": nodes, "edges": edges},
        "impact_analysis": impact,
        "methodology": {
            "explicit": "Textual clause references resolved to detected section headings.",
            "inferred": "Conservative keyword co-occurrence candidates; not semantic proof.",
            "limitations": "Unrecognized headings, cross-document references, OCR errors and implicit legal dependencies may be missed. Verify every link against the cited pages.",
        },
    }


__all__ = ["build_clause_relationships"]
