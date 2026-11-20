import json
import sqlite3
from datetime import datetime, timezone
from app.core.config import get_settings

settings = get_settings()

def init_db() -> None:
    con = sqlite3.connect(settings.audit_db_path)
    con.execute(
        """CREATE TABLE IF NOT EXISTS query_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            question TEXT NOT NULL,
            source_used TEXT NOT NULL,
            trace_json TEXT NOT NULL
        )"""
    )
    con.execute(
        """CREATE TABLE IF NOT EXISTS document_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            action TEXT NOT NULL,
            document_id TEXT,
            version_id TEXT,
            details_json TEXT NOT NULL
        )"""
    )
    con.commit()
    con.close()

def write_audit(question: str, source_used: str, trace: list[str]) -> None:
    con = sqlite3.connect(settings.audit_db_path)
    con.execute(
        "INSERT INTO query_audit(created_at, question, source_used, trace_json) VALUES (?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), question, source_used, json.dumps(trace)),
    )
    con.commit()
    con.close()


def write_document_audit(
    action: str,
    document_id: str | None,
    version_id: str | None,
    details: dict | None = None,
) -> None:
    """Record document lifecycle metadata without storing document contents."""
    con = sqlite3.connect(settings.audit_db_path)
    try:
        con.execute(
            """INSERT INTO document_audit
               (created_at, action, document_id, version_id, details_json)
               VALUES (?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                action,
                document_id,
                version_id,
                json.dumps(details or {}, sort_keys=True),
            ),
        )
        con.commit()
    finally:
        con.close()
