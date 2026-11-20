# Phase 7 — Document intelligence API and frontend

Phase 7 exposes version comparison and cross-document ANN verification through
admin-authenticated FastAPI routes and a browser-based Document Intelligence
workspace. The existing `/api/chat`, `/api/ingest`, `/api/documents` and
version-listing endpoints remain available.

## Authentication

All document management, comparison, verification and review endpoints require
`X-Admin-Key`. Configure a non-default `ADMIN_API_KEY` in `.env`; the default
`change-me` value is rejected. The browser workspace asks for the key at runtime
and sends it as a request header. Do not place production secrets in frontend
source code or commit `.env` files.

## Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/documents` | List logical documents and current version metadata |
| GET | `/api/documents/{document_id}/versions` | List retained versions for a document |
| POST | `/api/documents/compare` | Compare two versions of the same logical document |
| POST | `/api/verification/runs` | Run ANN verification across selected versions from distinct documents |
| GET | `/api/verification/runs/{run_id}` | Retrieve a persisted verification report |
| POST | `/api/verification/pairs/{pair_id}/review` | Store a separate human review decision |
| POST | `/api/documents/{document_id}/retire` | Retire a logical document without deleting its history |

### Compare versions

```http
POST /api/documents/compare
X-Admin-Key: <admin-key>
Content-Type: application/json
```

```json
{
  "old_version_id": "<older-version-id>",
  "new_version_id": "<newer-version-id>",
  "similarity_threshold": null
}
```

Both versions must exist and belong to the same logical document. The response
contains version summaries, alignment method, change counts and source-located
`ADDED`, `DELETED`, `MODIFIED` and `UNCHANGED` records. A text change is not by
itself a contradiction verdict.

### Start cross-document verification

```http
POST /api/verification/runs
X-Admin-Key: <admin-key>
Content-Type: application/json
```

```json
{
  "version_ids": ["<version-from-document-A>", "<version-from-document-B>"],
  "max_statements_per_version": 500,
  "max_pairs": 2000,
  "confidence_threshold": null
}
```

The service requires 2–10 selected versions, one version per logical document,
and completed ingestion. `max_statements_per_version` is capped at 2,000 and
`max_pairs` at 10,000. Partial runs are explicitly labelled when limits truncate
candidate coverage. ANN labels are decision-support signals, not a determination
of truth or policy authority.

### Human review

```http
POST /api/verification/pairs/<pair-id>/review
X-Admin-Key: <admin-key>
Content-Type: application/json
```

```json
{
  "reviewer": "reviewer-id",
  "label": "CONTRADICTION",
  "notes": "Reviewed against the current approved policy."
}
```

The reviewer label and notes are stored separately from the immutable ANN
prediction. The API currently treats the admin key as the authorization gate;
reviewer identity is a supplied audit label, not federated identity proof.

## Browser workspace

Use **Document Intelligence** in the left sidebar. Enter the admin key, refresh
the document list, and:

1. Select two versions of the same document to compare revisions.
2. Select one completed version from each of at least two distinct documents
   to run ANN verification.
3. Review each finding and save a separate reviewer label and notes.

The UI displays source statements, logical document names, version numbers,
source locations, ANN probabilities, confidence and review status. It does not
allow changing or hiding the original ANN output.

## Error handling and safety

- `401`: missing or invalid admin key.
- `400`: invalid operation, such as comparing versions from different logical documents.
- `404`: document version, run or finding not found.
- `422`: invalid input, incomplete ingestion or unsupported source content.
- `500`: generic service failure; internal exceptions are logged server-side.
- Historical versions, original files and Pinecone vectors are not deleted by
  comparison, verification or review operations.

## Verification

Run:

```powershell
python -m unittest discover -s tests -v
node --check static/js/app.js
python -m compileall -q app tests
```

The API route tests use lightweight service adapters. A deployment still needs
an end-to-end run with configured OpenAI, Pinecone and Tavily credentials and a
human-reviewed evaluation dataset before production use.
