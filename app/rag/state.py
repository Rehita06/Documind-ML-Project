from typing import Any, List, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document


class RouteDecision(BaseModel):
    route: Literal["kb", "direct", "verify_documents", "compare_versions"] = Field(
        description=(
            "kb for ordinary enterprise IT support; direct for greetings/casual chat; "
            "verify_documents for cross-document statement verification; "
            "compare_versions for comparing selected document versions"
        )
    )


class EvidenceGrade(BaseModel):
    grade: Literal["good", "weak"] = Field(description="Whether evidence is sufficient to answer")


class AgentState(TypedDict, total=False):
    question: str
    current_query: str
    kb_docs: List[Document]
    web_results: str
    kb_grade: str
    web_grade: str
    answer: str
    source_used: str
    retry_count: int
    trace: List[str]
    citations: List[dict]
    requested_intent: str | None
    document_version_ids: list[str]
    old_version_id: str | None
    new_version_id: str | None
    verification_report: dict[str, Any] | None
    comparison_report: dict[str, Any] | None
    workflow_error: str | None
