# Phase 6 — LangGraph intent routing and workflow integration

## Scope

Phase 6 extends the existing `StateGraph`; it does not replace the original
private-knowledge RAG route. The router now supports four intents:

- `kb`: existing private Pinecone retrieval, evidence grading, controlled Tavily
  fallback, one query rewrite/retry, grounded answer or abstention.
- `direct`: greeting and casual-chat response.
- `verify_documents`: invoke the Phase 5 `CrossDocumentVerificationService` for
  explicitly selected immutable document version IDs.
- `compare_versions`: invoke the Phase 3 `compare_document_versions` service for
  explicitly selected old and new version IDs.

## Explicit selection contract

`ask()` accepts optional keyword arguments `intent`, `document_version_ids`,
`old_version_id`, and `new_version_id`. Explicit workflow intent takes precedence
over LLM routing. Document/version identifiers are passed as application state;
the LLM is never asked to invent or resolve them. A verification request without
at least two selected versions, or a comparison request without both version IDs,
returns an actionable selection message instead of calling a service.

If model-based intent routing fails, the graph safely falls back to the original
private-KB path. It does not turn an uncertain routing failure into an ungrounded
direct answer.

## Evidence and trace behaviour

Verification results preserve the ANN report, both source references, per-class
probabilities and human-review fields. Trace entries include the ANN model version,
number of evaluated pairs and label counts. The answer explicitly warns that ANN
labels do not establish policy authority.

Version comparison preserves the complete change report, old/new version metadata,
source locations, alignment method and change counts. It explicitly distinguishes
textual modifications from policy contradictions. A document audit event records
comparison metadata; document content is not copied into the audit event.

The existing RAG nodes and web fallback edges remain connected as before.

## Programmatic invocation

```python
from app.rag.workflow import ask

verification = ask(
    "Check these IT documents for conflicting statements",
    intent="verify_documents",
    document_version_ids=["VERSION_UUID_A", "VERSION_UUID_B"],
)

comparison = ask(
    "Compare the old and new VPN policy",
    intent="compare_versions",
    old_version_id="OLD_VERSION_UUID",
    new_version_id="NEW_VERSION_UUID",
)

ordinary_chat = ask("How do I connect to the company VPN?")
```

The dedicated request/response endpoints and frontend selection controls are Phase
7. The LangGraph interface is ready for those API adapters, but this phase does not
claim the UI already exposes the new operations.

## Validation and limitations

Offline unit tests verify routing decisions, fallback behaviour, graph wiring,
selection validation, result propagation, source citations and audit invocation.
The current environment does not have the pinned LangGraph, LangChain OpenAI,
LangChain Tavily or Pinecone dependencies installed, and no provider credentials
are configured. Consequently, live graph execution, live OpenAI structured output,
Pinecone retrieval and Tavily fallback remain unverified here.
