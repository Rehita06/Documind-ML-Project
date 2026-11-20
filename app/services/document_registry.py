"""SQLite-backed logical document and immutable version registry.

The registry is authoritative for document identity/version metadata. Files are
stored under generated UUID paths; user-provided filenames are display metadata
only and are never used as filesystem paths.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import uuid
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _safe_extension(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix or '(missing extension)'}")
    return suffix


def validate_file_content(extension: str, content: bytes) -> None:
    """Check common file signatures/decodability before invoking document loaders."""
    if extension == ".pdf" and b"%PDF-" not in content[:1024]:
        raise ValueError("Uploaded content does not appear to be a PDF file")
    if extension in {".txt", ".md"}:
        try:
            content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("TXT/Markdown uploads must be UTF-8 text") from exc
    if extension == ".docx":
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                if "word/document.xml" not in archive.namelist():
                    raise ValueError("DOCX archive is missing word/document.xml")
        except (zipfile.BadZipFile, OSError) as exc:
            raise ValueError("Uploaded content is not a valid DOCX archive") from exc


class DocumentRegistry:
    def __init__(self, db_path: str | Path, upload_dir: str | Path):
        self.db_path = Path(db_path)
        self.upload_dir = Path(upload_dir)

    @contextmanager
    def _connect(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def initialize(self) -> None:
        with self._connect() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    logical_name TEXT NOT NULL,
                    logical_name_key TEXT NOT NULL UNIQUE,
                    original_filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    uploader TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    active_version_id TEXT,
                    status TEXT NOT NULL DEFAULT 'active'
                );

                CREATE TABLE IF NOT EXISTS document_versions (
                    version_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(document_id),
                    version_number INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    stored_path TEXT NOT NULL UNIQUE,
                    original_filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    uploaded_at TEXT NOT NULL,
                    effective_date TEXT,
                    status TEXT NOT NULL,
                    ingestion_status TEXT NOT NULL DEFAULT 'pending',
                    UNIQUE(document_id, version_number),
                    UNIQUE(document_id, content_hash)
                );

                CREATE INDEX IF NOT EXISTS idx_versions_document
                    ON document_versions(document_id, version_number);
                CREATE INDEX IF NOT EXISTS idx_versions_hash
                    ON document_versions(document_id, content_hash);

                CREATE TABLE IF NOT EXISTS document_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    version_id TEXT NOT NULL REFERENCES document_versions(version_id),
                    vector_id TEXT,
                    page_number INTEGER,
                    section TEXT,
                    start_offset INTEGER,
                    end_offset INTEGER,
                    text_hash TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_version
                    ON document_chunks(version_id);
            """)

    def create_version(
        self,
        logical_name: str,
        original_filename: str,
        content: bytes,
        uploader: str | None = None,
        effective_date: str | None = None,
    ) -> dict[str, Any]:
        name = (logical_name or "").strip()
        if not name:
            raise ValueError("logical_name must not be empty")
        if len(name) > 255:
            raise ValueError("logical_name must be 255 characters or fewer")
        if not content:
            raise ValueError("Uploaded file is empty")
        extension = _safe_extension(original_filename)
        validate_file_content(extension, content)
        digest = sha256_bytes(content)
        name_key = name.casefold()
        now = utc_now()

        self.initialize()
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            document = con.execute(
                "SELECT * FROM documents WHERE logical_name_key = ?", (name_key,)
            ).fetchone()

            if document:
                duplicate = con.execute(
                    """SELECT * FROM document_versions
                       WHERE document_id = ? AND content_hash = ?""",
                    (document["document_id"], digest),
                ).fetchone()
                if duplicate:
                    return {
                        "duplicate": True,
                        "document_id": document["document_id"],
                        "version_id": duplicate["version_id"],
                        "version_number": duplicate["version_number"],
                        "content_hash": digest,
                        "stored_path": duplicate["stored_path"],
                        "ingestion_status": duplicate["ingestion_status"],
                        "status": duplicate["status"],
                    }
                if document["status"] == "retired":
                    raise ValueError(
                        "This logical document is retired. Use a new logical name; "
                        "reactivation requires an explicit future workflow."
                    )

                document_id = document["document_id"]
                next_number = con.execute(
                    "SELECT COALESCE(MAX(version_number), 0) + 1 AS n "
                    "FROM document_versions WHERE document_id = ?",
                    (document_id,),
                ).fetchone()["n"]
            else:
                document_id = str(uuid.uuid4())
                next_number = 1
                con.execute(
                    """INSERT INTO documents
                       (document_id, logical_name, logical_name_key, original_filename,
                        file_type, uploader, created_at, updated_at, status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active')""",
                    (document_id, name, name_key, original_filename, extension,
                     uploader, now, now),
                )

            version_id = str(uuid.uuid4())
            relative_path = Path(document_id) / f"{version_id}{extension}"
            destination = self.upload_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            try:
                with open(temporary, "xb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
            except Exception:
                if temporary.exists():
                    temporary.unlink()
                raise

            con.execute(
                """INSERT INTO document_versions
                   (version_id, document_id, version_number, content_hash, stored_path,
                    original_filename, file_type, uploaded_at, effective_date,
                    status, ingestion_status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 'processing')""",
                (version_id, document_id, next_number, digest,
                 str(relative_path.as_posix()), original_filename, extension,
                 now, effective_date),
            )
            return {
                "duplicate": False,
                "document_id": document_id,
                "version_id": version_id,
                "version_number": next_number,
                "content_hash": digest,
                "stored_path": str(destination),
                "relative_path": str(relative_path.as_posix()),
                "ingestion_status": "processing",
                "status": "pending",
            }

    def complete_ingestion(self, version_id: str, chunks: Iterable[dict[str, Any]]) -> None:
        self.initialize()
        with self._connect() as con:
            version = con.execute(
                "SELECT document_id, status FROM document_versions WHERE version_id = ?",
                (version_id,),
            ).fetchone()
            if not version:
                raise ValueError("Unknown document version")
            if version["status"] != "pending":
                raise ValueError("Only a pending document version can be activated")
            document_id = version["document_id"]
            for item in chunks:
                con.execute(
                    """INSERT INTO document_chunks
                       (chunk_id, version_id, vector_id, page_number, section,
                        start_offset, end_offset, text_hash, metadata_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        item["chunk_id"], version_id, item.get("vector_id"),
                        item.get("page_number"), item.get("section"),
                        item.get("start_offset"), item.get("end_offset"),
                        item["text_hash"],
                        json.dumps(item.get("metadata", {}), sort_keys=True),
                    ),
                )
            # Promote only after vector indexing has succeeded. Failed uploads leave
            # the previous active version untouched.
            con.execute(
                "UPDATE document_versions SET status = 'superseded' "
                "WHERE document_id = ? AND status = 'active'", (document_id,)
            )
            con.execute(
                "UPDATE document_versions SET status = 'active', ingestion_status = 'complete' "
                "WHERE version_id = ?", (version_id,)
            )
            con.execute(
                """UPDATE documents SET active_version_id = ?, updated_at = ?,
                   original_filename = (SELECT original_filename FROM document_versions WHERE version_id = ?),
                   file_type = (SELECT file_type FROM document_versions WHERE version_id = ?)
                   WHERE document_id = ?""",
                (version_id, utc_now(), version_id, version_id, document_id),
            )

    def mark_ingestion_failed(self, version_id: str, error_code: str = "ingestion_failed") -> None:
        # Store only a bounded error category, never document text or credentials.
        with self._connect() as con:
            con.execute(
                "UPDATE document_versions SET ingestion_status = ?, status = 'failed' "
                "WHERE version_id = ? AND status = 'pending'",
                (error_code[:80], version_id),
            )

    def list_documents(self) -> list[dict[str, Any]]:
        self.initialize()
        with self._connect() as con:
            rows = con.execute("""
                SELECT d.document_id, d.logical_name, d.original_filename,
                       d.file_type, d.created_at, d.updated_at, d.status,
                       v.version_id AS latest_version_id,
                       v.version_number AS latest_version_number,
                       v.uploaded_at AS latest_uploaded_at,
                       v.ingestion_status AS latest_ingestion_status,
                       v.content_hash AS latest_content_hash
                FROM documents d
                LEFT JOIN document_versions v ON v.version_id = COALESCE(
                    d.active_version_id,
                    (SELECT version_id FROM document_versions
                     WHERE document_id = d.document_id
                     ORDER BY version_number DESC LIMIT 1)
                )
                ORDER BY d.updated_at DESC
            """).fetchall()
            return [dict(row) for row in rows]

    def list_versions(self, document_id: str) -> list[dict[str, Any]]:
        self.initialize()
        with self._connect() as con:
            rows = con.execute("""
                SELECT version_id, document_id, version_number, content_hash,
                       original_filename, file_type, uploaded_at, effective_date,
                       status, ingestion_status
                FROM document_versions WHERE document_id = ?
                ORDER BY version_number DESC
            """, (document_id,)).fetchall()
            return [dict(row) for row in rows]

    def get_version(self, version_id: str) -> dict[str, Any] | None:
        self.initialize()
        with self._connect() as con:
            row = con.execute("""
                SELECT v.*, d.logical_name, d.document_id
                FROM document_versions v
                JOIN documents d ON d.document_id = v.document_id
                WHERE v.version_id = ?
            """, (version_id,)).fetchone()
            if not row:
                return None
            result = dict(row)
            result["stored_path"] = str(self.upload_dir / result["stored_path"])
            return result

    def retire_document(self, document_id: str) -> dict[str, Any] | None:
        """Retire a logical document from ordinary retrieval without purging data."""
        self.initialize()
        with self._connect() as con:
            row = con.execute(
                "SELECT document_id, active_version_id, logical_name FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            if not row:
                return None
            con.execute(
                "UPDATE documents SET status = 'retired', updated_at = ? WHERE document_id = ?",
                (utc_now(), document_id),
            )
            con.execute(
                "UPDATE document_versions SET status = 'retired' "
                "WHERE document_id = ? AND status = 'active'",
                (document_id,),
            )
            return dict(row)

    def get_active_version_ids(self) -> list[str]:
        self.initialize()
        with self._connect() as con:
            rows = con.execute(
                "SELECT active_version_id FROM documents "
                "WHERE status = 'active' AND active_version_id IS NOT NULL"
            ).fetchall()
            return [row["active_version_id"] for row in rows]

    def get_version_chunks(self, version_id: str) -> list[dict[str, Any]]:
        self.initialize()
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM document_chunks WHERE version_id = ? ORDER BY rowid",
                (version_id,),
            ).fetchall()
            return [dict(row) for row in rows]


_registry_instance = None


def get_document_registry():
    """Return the configured process-wide registry."""
    global _registry_instance
    if _registry_instance is None:
        from app.core.config import get_settings
        settings = get_settings()
        _registry_instance = DocumentRegistry(
            settings.document_registry_db_path, settings.upload_dir
        )
        _registry_instance.initialize()
    return _registry_instance
