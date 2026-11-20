# Cross-document verification (Phase 5)

## Purpose and boundaries

`CrossDocumentVerificationService` compares statement pairs extracted from explicitly selected, immutable document versions belonging to different logical documents. It invokes the separately trained PyTorch ANN in `app/ml/inference.py` and stores an evidence-backed report. It does not use the ANN to determine which policy is authoritative, prove factual truth, or modify source documents, historical versions, or Pinecone records.

## Flow

1. Validate that at least two completed document versions are selected, with no more than one version per logical document.
2. Read each selected version's immutable stored file and extract sentence-level units from TXT/Markdown, DOCX, or PDF. Retain document ID, logical name, version ID/number, source filename, page/section/location and a chunk ID when the registry metadata can be mapped.
3. Generate candidate pairs only across different selected documents. Lexical Jaccard overlap ranks candidate pairs for evaluation; it is a candidate-ranking heuristic, not the ANN label or a contradiction decision.
4. Run the saved ANN predictor for each selected candidate. Preserve predicted label, raw model label, per-class probabilities, confidence, model version, threshold and review flag.
5. Persist verification run metadata and statement-pair findings in the configured SQLite audit database (`verification_runs`, `statement_pairs`, `verification_audit`).
6. An authorized application layer can call `review_pair` to append a human label. The original ANN output is never overwritten; the reviewer label, reviewer ID, notes and timestamp are stored separately.

## Limits and truncation

Defaults are 500 extracted statements per document and 2,000 evaluated pairs. A run selects at most 10 documents, 2,000 total extracted statements and 500,000 candidate pairs. When the evaluated-pair limit truncates the candidate set, the run is marked `partial`, and the report includes total candidate count, evaluated count and the lexical-overlap truncation policy. Raise `max_pairs` or narrow the document selection for broader coverage. A truncated run must not be represented as exhaustive verification.

The current sentence extraction is deterministic and source-file based. It does not yet perform Pinecone similarity search or LLM-based statement extraction. Chunk IDs are included when page/offset metadata can be matched; otherwise the more precise available page/line/paragraph location is retained.

## Python usage

```python
from app.services.verification import CrossDocumentVerificationService

service = CrossDocumentVerificationService()
report = service.verify_versions([
    "<version-id-from-document-registry>",
    "<another-document-version-id>",
])
print(report["run_id"], report["summary"])

# Retrieve the persisted report later:
report = service.get_report("<run-id>")

# Append a human review; the ANN label remains intact:
review = service.review_pair(
    "<pair-id>", reviewer="<authenticated-reviewer-id>",
    label="UNCERTAIN", notes="Refer to the policy owner",
)
```

The service method itself does not authenticate a reviewer. The API/reviewer authorization layer is intentionally part of the later API/UI phase; do not expose `review_pair` to untrusted callers directly.

## Configuration

- `ANN_ARTIFACT_DIR`: directory containing the trained ANN model artifact. Defaults to `artifacts/ann/ann-v1-seed42`.
- `ANN_CONFIDENCE_THRESHOLD`: optional threshold override in `[0, 1]`. Blank uses the threshold selected during ANN validation and stored in the artifact.
- `AUDIT_DB_PATH`: existing audit SQLite database path; verification tables are added without deleting existing audit records.

## Limitations

The current ANN was trained on the engineer-curated seed set documented in `docs/model_card.md` and `docs/evaluation.md`, not an independently human-reviewed enterprise benchmark. The ANN's test results were not better than the simple baseline on that small held-out set. Verification outputs are therefore review-support signals only. Domain owners must validate the dataset and model before production policy decisions.
