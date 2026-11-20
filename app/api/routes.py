import hashlib
import hmac
import logging
import uuid
from pathlib import Path
from datetime import date
import re
from typing import Literal

from fastapi import APIRouter, UploadFile, File, HTTPException, Header, Form
from pydantic import BaseModel, Field, field_validator

from app.core.config import get_settings
from app.rag.workflow import ask
from app.rag.vectorstore import add_documents
from app.services.ingestion import load_file, chunk_documents, SUPPORTED
from app.services.audit import write_audit, write_document_audit
from app.services.document_registry import get_document_registry

router = APIRouter(prefix="/api")
settings = get_settings()


class VerificationRunRequest(BaseModel):
    version_ids: list[str] = Field(min_length=2, max_length=10)
    max_statements_per_version: int = Field(default=500, ge=1, le=2000)
    max_pairs: int = Field(default=2000, ge=1, le=10000)
    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class VersionComparisonRequest(BaseModel):
    old_version_id: str = Field(min_length=1, max_length=128)
    new_version_id: str = Field(min_length=1, max_length=128)
    similarity_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class PairReviewRequest(BaseModel):
    reviewer: str = Field(min_length=1, max_length=128)
    label: Literal["CONTRADICTION", "CONSISTENT", "UNCERTAIN"]
    notes: str | None = Field(default=None, max_length=2000)

    @field_validator("reviewer")
    @classmethod
    def reviewer_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("reviewer must not be blank")
        return value


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=3000)


def require_admin(x_admin_key: str) -> None:
    configured = settings.admin_api_key or ""
    supplied = (x_admin_key or "").encode("utf-8")
    expected = configured.encode("utf-8")
    if not configured or configured == "change-me" or not hmac.compare_digest(
        supplied, expected
    ):
        raise HTTPException(status_code=401, detail="Invalid admin credentials")


@router.get("/health")
def health():
    return {"status": "ok", "service": settings.app_name}


@router.post("/chat")
def chat(payload: ChatRequest):
    try:
        result = ask(payload.question)
        write_audit(payload.question, result["source_used"], result.get("trace", []))
        return {
            "answer": result["answer"],
            "source_used": result["source_used"],
            "trace": result.get("trace", []),
            "citations": result.get("citations", []),
            "rewritten_query": result.get("current_query", payload.question),
        }
    except Exception as exc:
        # Do not expose provider credentials, stack traces, or raw internal errors.
        raise HTTPException(status_code=500, detail="Chat request failed") from exc


@router.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    x_admin_key: str = Header(default=""),
    logical_name: str | None = Form(default=None),
    effective_date: str | None = Form(default=None),
):
    require_admin(x_admin_key)
    # Treat both POSIX and Windows separators as untrusted path delimiters.
    original_filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    if not original_filename or len(original_filename) > 255:
        raise HTTPException(status_code=400, detail="Invalid filename")
    suffix = Path(original_filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(
            status_code=400,
            detail=f"Supported: {', '.join(sorted(SUPPORTED))}",
        )

    content = await file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.max_upload_bytes}-byte upload limit",
        )

    if effective_date:
        try:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", effective_date):
                raise ValueError("expected YYYY-MM-DD")
            date.fromisoformat(effective_date)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="effective_date must use YYYY-MM-DD format") from exc

    registry = get_document_registry()
    name = (logical_name or Path(original_filename).stem).strip()
    try:
        record = registry.create_version(
            logical_name=name,
            original_filename=original_filename,
            content=content,
            uploader="admin",
            effective_date=effective_date,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Could not register uploaded document") from exc

    if record["duplicate"]:
        try:
            write_document_audit("document_upload_duplicate", record["document_id"], record["version_id"], {"logical_name": name, "content_hash": record["content_hash"]})
        except Exception:
            logging.getLogger(__name__).warning("Could not persist duplicate-upload audit event")
        return {
            "message": "Identical content already exists; no new version was created",
            "duplicate": True,
            "document_id": record["document_id"],
            "version_id": record["version_id"],
            "version_number": record["version_number"],
            "content_hash": record["content_hash"],
            "ingestion_status": record["ingestion_status"],
        }

    version_id = record["version_id"]
    try:
        stored_path = Path(record["stored_path"])
        docs = load_file(stored_path)
        chunks = chunk_documents(docs)
        if not chunks:
            raise ValueError("No extractable text was found in the uploaded document")

        vector_ids = []
        registry_chunks = []
        for chunk in chunks:
            chunk_id = str(uuid.uuid4())
            metadata = dict(chunk.metadata)
            metadata.update({
                "document_id": record["document_id"],
                "version_id": version_id,
                "logical_name": name,
                "version_number": record["version_number"],
                "file_hash": record["content_hash"],
                "source_filename": original_filename,
                "chunk_id": chunk_id,
            })
            chunk.metadata = metadata
            vector_ids.append(chunk_id)
            text_hash = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()
            start = metadata.get("start_index")
            registry_chunks.append({
                "chunk_id": chunk_id,
                "vector_id": chunk_id,
                "page_number": metadata.get("page"),
                "section": metadata.get("section"),
                "start_offset": start,
                "end_offset": (start + len(chunk.page_content)) if isinstance(start, int) else None,
                "text_hash": text_hash,
                "metadata": metadata,
            })

        # The new version becomes active only after vector indexing and registry
        # chunk persistence both succeed. Older active versions remain retrievable
        # if this operation fails.
        ids_created = add_documents(chunks, ids=vector_ids)
        registry.complete_ingestion(version_id, registry_chunks)
        try:
            write_document_audit("document_version_ingested", record["document_id"], version_id, {"logical_name": name, "version_number": record["version_number"], "content_hash": record["content_hash"], "chunk_count": len(chunks)})
        except Exception:
            logging.getLogger(__name__).warning("Could not persist document-ingestion audit event")
        return {
            "message": "Document indexed",
            "duplicate": False,
            "file": original_filename,
            "chunks": len(chunks),
            "ids_created": len(ids_created or vector_ids),
            "document_id": record["document_id"],
            "version_id": version_id,
            "version_number": record["version_number"],
            "content_hash": record["content_hash"],
            "ingestion_status": "complete",
        }
    except Exception as exc:
        registry.mark_ingestion_failed(version_id)
        try:
            write_document_audit("document_ingestion_failed", record["document_id"], version_id, {"logical_name": name, "version_number": record["version_number"], "content_hash": record["content_hash"]})
        except Exception:
            logging.getLogger(__name__).warning("Could not persist failed-ingestion audit event")
        raise HTTPException(
            status_code=500,
            detail="Document was stored but ingestion did not complete; the previous active version was preserved",
        ) from exc


@router.get("/documents")
def list_documents(x_admin_key: str = Header(default="")):
    require_admin(x_admin_key)
    return {"documents": get_document_registry().list_documents()}


@router.get("/documents/{document_id}/versions")
def list_document_versions(document_id: str, x_admin_key: str = Header(default="")):
    require_admin(x_admin_key)
    versions = get_document_registry().list_versions(document_id)
    if not versions:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"document_id": document_id, "versions": versions}


@router.post("/documents/{document_id}/retire")
def retire_document(document_id: str, x_admin_key: str = Header(default="")):
    require_admin(x_admin_key)
    registry = get_document_registry()
    retired = registry.retire_document(document_id)
    if retired is None:
        raise HTTPException(status_code=404, detail="Document not found")
    try:
        write_document_audit(
            "document_retired",
            document_id,
            retired.get("active_version_id"),
            {"logical_name": retired.get("logical_name")},
        )
    except Exception:
        logging.getLogger(__name__).warning("Could not persist document-retirement audit event")
    return {
        "message": "Document retired from ordinary retrieval; historical files and versions were retained",
        "document_id": document_id,
        "status": "retired",
    }


@router.post("/documents/compare")
def compare_versions(payload: VersionComparisonRequest, x_admin_key: str = Header(default="")):
    require_admin(x_admin_key)
    from app.services.change_detection import compare_document_versions

    registry = get_document_registry()
    old_version = registry.get_version(payload.old_version_id)
    new_version = registry.get_version(payload.new_version_id)
    if old_version is None or new_version is None:
        raise HTTPException(status_code=404, detail="One or both document versions were not found")
    if old_version["document_id"] != new_version["document_id"]:
        raise HTTPException(status_code=400, detail="Version comparison requires two versions of the same logical document")
    if old_version.get("ingestion_status") != "complete" or new_version.get("ingestion_status") != "complete":
        raise HTTPException(status_code=422, detail="Both document versions must have completed ingestion")
    try:
        result = compare_document_versions(
            registry, payload.old_version_id, payload.new_version_id,
            similarity_threshold=payload.similarity_threshold,
        )
        return result
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logging.getLogger(__name__).exception("Document version comparison failed")
        raise HTTPException(status_code=500, detail="Document version comparison failed") from exc


@router.post("/verification/runs")
def create_verification_run(payload: VerificationRunRequest, x_admin_key: str = Header(default="")):
    require_admin(x_admin_key)
    from app.services.verification import CrossDocumentVerificationService

    try:
        service = CrossDocumentVerificationService()
        return service.verify_versions(
            payload.version_ids,
            confidence_threshold=payload.confidence_threshold,
            max_statements_per_version=payload.max_statements_per_version,
            max_pairs=payload.max_pairs,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logging.getLogger(__name__).exception("Cross-document verification failed")
        raise HTTPException(status_code=500, detail="Cross-document verification failed") from exc


@router.get("/verification/runs/{run_id}")
def get_verification_run(run_id: str, x_admin_key: str = Header(default="")):
    require_admin(x_admin_key)
    from app.services.verification import CrossDocumentVerificationService

    report = CrossDocumentVerificationService().get_report(run_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Verification run not found")
    return report


@router.post("/verification/pairs/{pair_id}/review")
def review_verification_pair(pair_id: str, payload: PairReviewRequest, x_admin_key: str = Header(default="")):
    require_admin(x_admin_key)
    from app.services.verification import CrossDocumentVerificationService

    try:
        return CrossDocumentVerificationService().review_pair(
            pair_id, reviewer=payload.reviewer, label=payload.label, notes=payload.notes
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Verification finding not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logging.getLogger(__name__).exception("Verification review failed")
        raise HTTPException(status_code=500, detail="Verification review failed") from exc
