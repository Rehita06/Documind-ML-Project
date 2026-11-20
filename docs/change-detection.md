# Phase 3 — Document change detection

## Scope

`app/services/change_detection.py` compares two retained document versions from the
Phase 2 SQLite registry. It does not modify the registry, upload files, vector
records, active-version pointer, or historical content. It can compare versions
of the same logical document or explicitly selected versions from different
logical documents.

## Comparison pipeline

1. Resolve both immutable version IDs through the registry and verify that both
   stored files still exist. Missing versions/files fail closed.
2. Extract source-locatable text units: TXT/Markdown line blocks, DOCX paragraphs,
   and PDF page/block pairs. Markdown headings and DOCX heading styles are retained
   as section labels. Each unit retains its ordinal and source location.
3. Perform exact alignment first using Unicode-normalized, case-insensitive text.
   Exact matches are `UNCHANGED` (formatting/case-only differences are ignored).
4. Align remaining units one-to-one by cosine similarity. Pass an embedding adapter
   exposing `embed_documents(texts)` (or a callable) to use sentence embeddings.
   If `OPENAI_API_KEY` is configured, the service attempts to use the existing
   `OPENAI` embedding model setting (`EMBEDDING_MODEL`). Otherwise, a deterministic
   local token/unigram+bigram cosine fallback is used. The result names the method
   actually used; the local fallback is lexical and is not represented as a
   sentence-embedding model.
5. A similarity-aligned but textually changed pair is `MODIFIED`; unmatched old
   units are `DELETED`; unmatched new units are `ADDED`. The report preserves both
   text values, old/new source locations, section labels, similarity and alignment
   method. It does not call wording changes contradictions.

## Python usage

```python
from app.services.change_detection import compare_document_versions
from app.services.document_registry import get_document_registry

report = compare_document_versions(
    get_document_registry(),
    old_version_id="<old-version-uuid>",
    new_version_id="<new-version-uuid>",
    similarity_threshold=0.72,
)
print(report["counts"])
for change in report["changes"]:
    print(change["change_type"], change["old_location"], change["new_location"])
```

An embedding adapter can be injected for deterministic tests or a separately
configured sentence-embedding deployment. `similarity_threshold` is a caller-
controlled cosine threshold in `[0, 1]`; tune it against representative labelled
version pairs before relying on semantic alignment in production. By default, the
service uses `0.72` for sentence embeddings and `0.45` for the local lexical fallback.

## API/UI status

The service and regression tests are implemented in Phase 3. The comparison HTTP
endpoint, request/response Pydantic schemas, audit event integration, and frontend
comparison view are intentionally deferred to the later API/UI phase in the
implementation sequence. No new LangGraph route or ANN classification is claimed
in this phase.

## Test fixtures and limitations

`data/fixtures/change_detection/vpn_policy_v1.txt` and `vpn_policy_v2.txt` provide
an illustrative VPN policy pair containing unchanged, modified, added and deleted
content. Tests use a deterministic injected embedding adapter and the local
fallback; this is implementation testing, not a benchmark or evidence of model
accuracy. PDF extraction quality depends on extractable text; OCR is not included.
DOCX table extraction is not included in this phase. Semantic alignment is a
similarity heuristic, not contradiction detection or an authority decision.
