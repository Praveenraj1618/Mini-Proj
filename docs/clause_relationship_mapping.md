# Clause relationship mapping

ClauseIQ exposes `GET /api/dependencies` for the currently uploaded document. The endpoint uses the same page-aware PDF extraction and section filtering as the existing review pipeline.

## Pipeline

1. **Section extraction** — `core.legal_intelligence.extract_clause_evidence` identifies conservative numbered/article/section headings, keeps original wording, retains page numbers across page breaks, and preserves unclassified text rather than silently dropping it.
2. **Explicit references** — `core.dependency_mapping` detects phrases such as “subject to Clause 2”, “notwithstanding Section 4”, and “see Article II”. A reference is emitted only when its identifier resolves to a detected section.
3. **Candidate thematic links** — a small keyword rule set identifies clauses sharing payment, late-payment, breach, termination, or notice concepts. These edges are always `source_type: inferred`, `type: related_to`, and `confidence: heuristic_candidate`. They are review prompts, not legal conclusions.
4. **Graph and traversal** — graph nodes represent extracted sections; directed edges preserve the direction of explicit textual references. Connected-clause paths are traversed in both directions so a section can show references made to it as well as references it makes.

## Relationship object

```json
{
  "source": "clause-0001-...",
  "target": "clause-0002-...",
  "type": "depends_on",
  "source_type": "explicit",
  "confidence": "direct_reference",
  "evidence": "Subject to Clause 2 ...",
  "pages": [1],
  "target_pages": [2],
  "explanation": "This clause expressly refers to ..."
}
```

Relationship types currently emitted:

- `depends_on` — the source uses “subject to”.
- `overrides` — the source uses “notwithstanding”.
- `exception_to` — the source uses “except as provided in”.
- `references` — another recognized cross-reference cue, or an uncued clause/section/article reference.
- `related_to` — inferred keyword co-occurrence only.

## Limitations and safe interpretation

- Heading recognition is heuristic. Unrecognized structures may be retained as unclassified sections or may not resolve as a reference target.
- Cross-document references, schedules/annexures without recognizable headings, OCR mistakes, and implicit dependencies may be missed.
- Keyword overlap does not prove a legal relationship. Inferred edges have no verbatim evidence quote and must be verified manually.
- The graph and connected-clause paths identify where a reviewer may want to look next. They do not establish that a breach, penalty, termination, or other legal consequence will occur.
- Page citations identify source pages, not legal authority. Review the original PDF and consult a qualified legal professional for legal interpretation.

## Tests

Run from the repository root:

```bash
python -m pytest backend/tests/test_legal_intelligence.py backend/tests/test_dependency_mapping.py -q
```
