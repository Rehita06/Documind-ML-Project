# Document registry and immutable versioning

The SQLite document registry is authoritative for logical document identity,
immutable version metadata, and chunk-to-version references.

- A logical document is identified by a case-insensitive `logical_name`.
- Each uploaded version receives a UUID, monotonic version number, SHA-256
  content hash, generated storage path, and upload timestamp.
- The original filename is retained as metadata only; it is never used to
  construct a storage path.
- Re-uploading identical bytes for the same logical document is an idempotent
  no-op.
- A changed upload is first stored and indexed as `pending`. It becomes the
  active version only after vector indexing and chunk metadata persistence
  succeed. The previous active version remains available if processing fails.
- Historical files and version rows are retained. No automatic purge is
  implemented.
- New vector metadata includes `document_id`, `version_id`, `logical_name`,
  `version_number`, `file_hash`, `source_filename`, and `chunk_id`.
- Ordinary retrieval filters registry-managed vectors to active version IDs.
  Legacy vectors without `version_id` remain searchable for backward
  compatibility.

The registry database defaults to `data/document_registry.db`. Uploaded files
are stored below the configured upload directory using generated UUID paths.
The upload limit defaults to 25 MiB and can be changed with
`MAX_UPLOAD_BYTES`.

This phase does not implement historical-version comparison, semantic change
detection, contradiction detection, or authorization beyond the existing
admin-key mechanism.


## Retirement

`POST /api/documents/{document_id}/retire` is an admin-key-protected soft
retirement operation. It removes the logical document from ordinary active
version retrieval by changing its registry status to `retired`. It does not
delete uploaded files, historical version rows, chunk metadata, or Pinecone
vectors. A `document_retired` event is written to the document audit table.
A retired logical name cannot be silently reactivated by another changed upload;
use a new logical name until an explicit restore workflow is introduced.
