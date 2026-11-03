# Clause relationship mapping

ClauseIQ exposes `GET /api/dependencies` for the currently uploaded document. The mapper preserves source wording and page provenance, then builds a deliberately conservative relationship map.

## Pipeline

1. **Clause selection** — `core.legal_intelligence.extract_clause_evidence` identifies candidate sections. The dependency mapper includes only recognized numbered, article, or section headings. Unclassified preamble text and date-like headings are excluded from the graph; recognized clauses remain in the clause inventory.
2. **Explicit references** — `core.dependency_mapping` detects phrases such as “subject to Clause 2”, “notwithstanding Section 4”, and “see Article II”. A relationship is emitted only when the referenced identifier resolves to another recognized clause.
3. **Model-assisted semantic review** — when an LLM provider is configured, up to 24 clause pairs are reviewed. No keyword-only edges are generated. A suggested relationship is accepted only when both clause IDs and the relationship type are valid, both quotations appear verbatim in their respective clauses, and each cited page contains its quotation. Model confidence remains uncalibrated.
4. **Graph and inventory** — the graph contains only clauses participating in accepted relationships. Clauses without a supported connection remain in `isolated_clauses` and the full clause inventory; they are not connected merely to make the graph look populated.
5. **Review paths** — connected-clause traversal is undirected for discovery, while each displayed relationship retains its source-to-target direction.

## Relationship object

```json
{
  "source": "clause-0002-...",
  "target": "clause-0001-...",
  "type": "depends_on",
  "source_type": "inferred",
  "confidence": "model_candidate_not_calibrated",
  "source_evidence": "Exact quotation from the source clause",
  "target_evidence": "Exact quotation from the target clause",
  "pages": [2],
  "target_pages": [1],
  "explanation": "Cautious explanation of the possible interaction"
}
```

Relationship direction is defined as follows:

- `depends_on` — source clause relies on the target clause.
- `defines` — source clause supplies a definition used by the target.
- `exception_to` / `overrides` — source clause is the exception or overriding rule; target is the baseline rule.
- `triggers` — source clause states a trigger; target contains the possible resulting provision.
- `modifies` — source clause changes how the target provision operates.
- `survives_termination` — source clause states a provision that continues after termination.
- `related_to` — a supported interaction is identified, but a more specific type is not justified.
- `references` — an explicit textual cross-reference without a stronger directional cue.

Explicit links are grounded in the contract's cross-reference wording. Inferred links are model candidates, not calibrated legal conclusions, even when both quotations are verified.

## Limitations and safe interpretation

- Heading recognition remains heuristic. Unrecognized headings, schedules, annexures, cross-document references, and OCR errors may be missed.
- Model review is bounded to the first 24 clause pairs; relationships outside that review budget may be missed.
- Exact quotations prove that the cited wording exists; they do not prove that the model's interpretation of the interaction is legally correct.
- An empty graph is a valid result when no relationship is sufficiently supported. It is preferable to fabricated or keyword-only connections.
- Graph links and review paths identify clauses worth reading together. They do not establish breach, penalties, termination, or other automatic legal consequences.
- Review the original PDF and consult a qualified legal professional for legal interpretation.

## Tests

Run from the repository root:

```bash
python -m pytest backend/tests/test_legal_intelligence.py backend/tests/test_dependency_mapping.py -q
```
