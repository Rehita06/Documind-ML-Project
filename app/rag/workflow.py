
import logging
from typing import Literal

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

from app.core.config import get_settings
from app.rag.state import AgentState, RouteDecision, EvidenceGrade
from app.rag.vectorstore import get_retriever

logger = logging.getLogger(__name__)
settings = get_settings()

_llm = None


def llm():
    """Return the locally running Ollama chat model."""
    global _llm

    if _llm is None:
        _llm = ChatOllama(
            model="qwen2.5:7b",
            temperature=0,
            base_url="http://localhost:11434",
        )

    return _llm


def add_trace(state: AgentState, message: str):
    return [*state.get("trace", []), message]


def route_question(state: AgentState):
    # Explicit workflow intent takes precedence over model classification.
    requested = (state.get("requested_intent") or "").strip().lower()

    supported = {
        "kb",
        "direct",
        "verify_documents",
        "compare_versions",
    }

    if requested:
        if requested not in supported:
            raise ValueError(f"Unsupported workflow intent: {requested}")

        decision = requested
        detail = "explicit request"

    else:
        try:
            router = llm().with_structured_output(
                RouteDecision,
                method="json_mode",
            )

            result = router.invoke(f"""
You route messages for an enterprise IT support assistant.

Choose exactly one route:

- kb: ordinary company IT questions, policies, VPN, password reset,
  MFA, laptop setup, software access, security, email, devices
  or troubleshooting.

- direct: greetings, thanks or casual chat needing no company knowledge.

- verify_documents: compare statements across multiple selected
  enterprise documents and identify contradiction, consistent
  or uncertain relationships.

- compare_versions: compare two selected versions of a document
  and identify additions, deletions, modifications and unchanged content.

Do not perform the task or invent document/version identifiers.

Question: {state['question']}

Return JSON with a route field.
""")

            decision = result.route
            detail = "model classification"

        except Exception:
            logger.exception(
                "Intent router failed; falling back to private KB route"
            )
            decision = "kb"
            detail = "safe KB fallback"

    return {
        "source_used": decision,
        "trace": add_trace(
            state,
            f"Router → {decision.upper()} ({detail})",
        ),
    }


def route_after_router(state: AgentState) -> Literal[
    "retrieve_kb",
    "direct_answer",
    "verify_documents",
    "compare_versions",
]:
    route = state.get("source_used", "kb")

    if route == "direct":
        return "direct_answer"

    if route == "verify_documents":
        return "verify_documents"

    if route == "compare_versions":
        return "compare_versions"

    return "retrieve_kb"


def verify_documents(state: AgentState):
    version_ids = state.get("document_version_ids") or []

    if len(version_ids) < 2:
        message = (
            "Document verification needs at least two selected document "
            "versions. Choose the documents to verify and submit again."
        )

        return {
            "answer": message,
            "source_used": "document_verification",
            "verification_report": None,
            "citations": [],
            "trace": add_trace(
                state,
                "Verification → waiting for at least two selected versions",
            ),
        }

    try:
        from app.services.verification import CrossDocumentVerificationService

        report = CrossDocumentVerificationService().verify_versions(
            version_ids
        )

    except (ValueError, FileNotFoundError) as exc:
        message = str(exc)

        return {
            "answer": (
                f"Document verification could not be completed: {message}"
            ),
            "source_used": "document_verification",
            "verification_report": None,
            "workflow_error": message,
            "citations": [],
            "trace": add_trace(
                state,
                f"Verification → validation stopped ({message})",
            ),
        }

    except Exception:
        logger.exception("Cross-document verification failed")

        message = (
            "The verification service could not complete this request. "
            "Check the model artifact and document registry configuration."
        )

        return {
            "answer": message,
            "source_used": "document_verification",
            "verification_report": None,
            "workflow_error": "verification_service_error",
            "citations": [],
            "trace": add_trace(state, "Verification → service error"),
        }

    summary = report.get("summary", {})
    counts = summary.get("label_counts", {})

    answer = (
        f"Cross-document verification {report.get('status', 'completed')}. "
        f"Evaluated {summary.get('evaluated_pairs', 0)} statement pairs: "
        f"{counts.get('CONTRADICTION', 0)} possible contradictions, "
        f"{counts.get('CONSISTENT', 0)} consistent, and "
        f"{counts.get('UNCERTAIN', 0)} uncertain. "
        f"{summary.get('human_review_required', 0)} findings require "
        "human review. ANN labels describe relationships between the "
        "supplied statements; they do not establish which document "
        "is authoritative."
    )

    citations = []
    seen = set()

    for selected in summary.get("selected_documents", []):
        identity = (selected.get("version_id"), None, None)

        if identity in seen:
            continue

        seen.add(identity)

        citations.append({
            "title": selected.get("logical_name", "Document"),
            "type": "document_verification",
            **selected,
        })

    for finding in report.get("findings", []):
        for source_key in ("source_a", "source_b"):
            source = finding.get(source_key) or {}

            identity = (
                source.get("version_id"),
                source.get("location"),
                source.get("chunk_id"),
            )

            if identity in seen:
                continue

            seen.add(identity)

            citations.append({
                "title": source.get("logical_name", "Document"),
                "type": "document_verification",
                **source,
            })

    trace = add_trace(
        state,
        f"Verification → ANN {report.get('model_version', 'unknown')} "
        f"evaluated {summary.get('evaluated_pairs', 0)} pairs",
    )

    trace = [
        *trace,
        f"ANN labels → {counts.get('CONTRADICTION', 0)} contradiction / "
        f"{counts.get('CONSISTENT', 0)} consistent / "
        f"{counts.get('UNCERTAIN', 0)} uncertain",
    ]

    return {
        "answer": answer,
        "source_used": "document_verification",
        "verification_report": report,
        "citations": citations,
        "trace": trace,
    }


def compare_versions(state: AgentState):
    old_id = state.get("old_version_id")
    new_id = state.get("new_version_id")

    if not old_id or not new_id:
        message = (
            "Version comparison needs an old version ID and a new version ID. "
            "Select both versions and submit the request again."
        )

        return {
            "answer": message,
            "source_used": "version_comparison",
            "comparison_report": None,
            "citations": [],
            "trace": add_trace(
                state,
                "Version comparison → waiting for two selected versions",
            ),
        }

    try:
        from app.services.change_detection import compare_document_versions
        from app.services.document_registry import get_document_registry

        report = compare_document_versions(
            get_document_registry(),
            old_id,
            new_id,
        )

    except (ValueError, FileNotFoundError) as exc:
        message = str(exc)

        return {
            "answer": (
                f"Version comparison could not be completed: {message}"
            ),
            "source_used": "version_comparison",
            "comparison_report": None,
            "workflow_error": message,
            "citations": [],
            "trace": add_trace(
                state,
                f"Version comparison → validation stopped ({message})",
            ),
        }

    except Exception:
        logger.exception("Document version comparison failed")

        return {
            "answer": (
                "The version comparison service could not complete "
                "this request."
            ),
            "source_used": "version_comparison",
            "comparison_report": None,
            "workflow_error": "comparison_service_error",
            "citations": [],
            "trace": add_trace(
                state,
                "Version comparison → service error",
            ),
        }

    counts = report.get("counts", {})

    old_version = report.get("old_version", {})
    new_version = report.get("new_version", {})

    answer = (
        f"Compared {old_version.get('logical_name', 'document')} version "
        f"{old_version.get('version_number', '?')} with version "
        f"{new_version.get('version_number', '?')}: "
        f"{counts.get('ADDED', 0)} added, "
        f"{counts.get('DELETED', 0)} deleted, "
        f"{counts.get('MODIFIED', 0)} modified, and "
        f"{counts.get('UNCHANGED', 0)} unchanged. "
        f"Alignment method: {report.get('alignment_method', 'unknown')}. "
        "A textual modification is not automatically a policy contradiction."
    )

    citations = []

    for key, version in (
        ("old_version", old_version),
        ("new_version", new_version),
    ):
        citations.append({
            "title": version.get("logical_name", key),
            "type": "document_version",
            **version,
        })

    try:
        from app.services.audit import write_document_audit

        write_document_audit(
            "document_versions_compared",
            report.get("document_id"),
            new_id,
            {
                "old_version_id": old_id,
                "new_version_id": new_id,
                "counts": counts,
                "alignment_method": report.get("alignment_method"),
            },
        )

    except Exception:
        logger.warning(
            "Could not persist version-comparison audit event"
        )

    trace = add_trace(
        state,
        f"Version comparison → "
        f"{report.get('alignment_method', 'unknown')} alignment",
    )

    trace = [
        *trace,
        f"Change counts → added {counts.get('ADDED', 0)} / "
        f"deleted {counts.get('DELETED', 0)} / "
        f"modified {counts.get('MODIFIED', 0)} / "
        f"unchanged {counts.get('UNCHANGED', 0)}",
    ]

    return {
        "answer": answer,
        "source_used": "version_comparison",
        "comparison_report": report,
        "citations": citations,
        "trace": trace,
    }


def retrieve_kb(state: AgentState):
    docs = get_retriever().invoke(state["current_query"])

    return {
        "kb_docs": docs,
        "trace": add_trace(
            state,
            f"Private KB retrieval → {len(docs)} chunks",
        ),
    }


def grade_kb(state: AgentState):
    grader = llm().with_structured_output(
        EvidenceGrade,
        method="json_mode",
    )

    context = "\n\n".join(
        f"Source: {d.metadata.get('source', 'unknown')}\n"
        f"{d.page_content}"
        for d in state["kb_docs"]
    )

    grade = grader.invoke(f"""
You grade evidence for an enterprise IT support assistant.

Question: {state['question']}

Private company KB evidence:
{context}

Return good only if the evidence is sufficient to answer confidently
and specifically.

Otherwise return weak.

Return JSON: {{"grade":"good"}} or {{"grade":"weak"}}.
""")

    return {
        "kb_grade": grade.grade,
        "trace": add_trace(
            state,
            f"KB evidence grade → {grade.grade.upper()}",
        ),
    }


def after_kb(
    state: AgentState,
) -> Literal["generate_from_kb", "insufficient"]:
    if state["kb_grade"] == "good":
        return "generate_from_kb"

    return "insufficient"


def generate_from_kb(state: AgentState):
    context = "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n"
        f"{d.page_content}"
        for d in state["kb_docs"]
    )

    answer = llm().invoke(f"""
You are an enterprise IT support copilot.

Answer ONLY from the private company knowledge base below.

Be concise, actionable, and safe.
If steps are present, present them clearly.
Do not invent policy details.
Mention that the answer is based on the company's private knowledge base.

If the evidence does not answer the question, clearly say so.

Question: {state['question']}

Private knowledge base:
{context}
""").content

    citations = []
    seen = set()

    for document in state["kb_docs"]:
        source = document.metadata.get("source", "Private KB")

        if source not in seen:
            seen.add(source)

            citations.append({
                "title": source.split("/")[-1],
                "url": "",
                "type": "private_kb",
            })

    return {
        "answer": answer,
        "source_used": "private_kb",
        "citations": citations,
        "trace": add_trace(
            state,
            "Answer generation → PRIVATE KB",
        ),
    }


def direct_answer(state: AgentState):
    answer = llm().invoke(
        f"Respond briefly and naturally to: {state['question']}"
    ).content

    return {
        "answer": answer,
        "source_used": "direct",
        "trace": add_trace(
            state,
            "Direct response → no retrieval",
        ),
    }


def insufficient(state: AgentState):
    return {
        "answer": (
            "I couldn't find enough reliable evidence in the company "
            "knowledge base to answer confidently. Please contact the "
            "IT help desk or provide more details."
        ),
        "source_used": "insufficient_evidence",
        "trace": add_trace(
            state,
            "Stopped → insufficient private KB evidence",
        ),
    }


def build_graph():
    graph = StateGraph(AgentState)

    for name, fn in {
        "route_question": route_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb": grade_kb,
        "generate_from_kb": generate_from_kb,
        "direct_answer": direct_answer,
        "insufficient": insufficient,
        "verify_documents": verify_documents,
        "compare_versions": compare_versions,
    }.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "route_question")

    graph.add_conditional_edges(
        "route_question",
        route_after_router,
        {
            "retrieve_kb": "retrieve_kb",
            "direct_answer": "direct_answer",
            "verify_documents": "verify_documents",
            "compare_versions": "compare_versions",
        },
    )

    graph.add_edge("retrieve_kb", "grade_kb")

    graph.add_conditional_edges(
        "grade_kb",
        after_kb,
        {
            "generate_from_kb": "generate_from_kb",
            "insufficient": "insufficient",
        },
    )

    graph.add_edge("generate_from_kb", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("insufficient", END)
    graph.add_edge("verify_documents", END)
    graph.add_edge("compare_versions", END)

    return graph.compile()


agent_graph = build_graph()


def ask(
    question: str,
    *,
    intent: str | None = None,
    document_version_ids: list[str] | None = None,
    old_version_id: str | None = None,
    new_version_id: str | None = None,
):
    """Run the integrated local LangGraph workflow."""

    initial: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
        "requested_intent": intent,
        "document_version_ids": list(document_version_ids or []),
        "old_version_id": old_version_id,
        "new_version_id": new_version_id,
        "verification_report": None,
        "comparison_report": None,
        "workflow_error": None,
    }

    return agent_graph.invoke(initial)
