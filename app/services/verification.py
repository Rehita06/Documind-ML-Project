"""Evidence-backed cross-document statement verification using the trained ANN.

The service compares statements only across distinct logical documents. It stores
immutable model outputs separately from optional human review decisions. It does
not infer policy authority or modify document/version/vector data.
"""
from __future__ import annotations

import heapq
import itertools
import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from contextlib import closing
from typing import Any, Iterable

from app.core.config import get_settings
from app.ml.inference import load_predictor
from app.services.document_registry import get_document_registry

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
WORD = re.compile(r"[\w'-]+", re.UNICODE)
REVIEW_LABELS = {"CONTRADICTION", "CONSISTENT", "UNCERTAIN"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _split_statements(text: str) -> list[str]:
    parts = SENTENCE_SPLIT.split(text.strip())
    return [part.strip(" \t\r\n•-*") for part in parts if len(part.strip()) >= 12]


def _extract_units(version: dict[str, Any], max_statements: int, registry: Any) -> list[dict[str, Any]]:
    """Extract source-locatable statement units without requiring external APIs."""
    path = Path(version["stored_path"])
    if not path.is_file():
        raise ValueError(f"Stored source file is missing for version {version['version_id']}")
    suffix = path.suffix.lower()
    blocks: list[dict[str, Any]] = []
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8-sig")
        section = None
        offset = 0
        for line_no, line in enumerate(text.splitlines(keepends=True), start=1):
            value = line.strip()
            heading = HEADING.match(value)
            if heading:
                section = heading.group(1).strip()
            elif value:
                blocks.append({"text": value, "location": f"line:{line_no}",
                               "page_number": None, "section": section,
                               "start_offset": offset, "end_offset": offset + len(line)})
            offset += len(line)
    elif suffix == ".docx":
        from docx import Document
        doc = Document(str(path))
        section = None
        for index, paragraph in enumerate(doc.paragraphs, start=1):
            value = paragraph.text.strip()
            if not value:
                continue
            style = (paragraph.style.name or "") if paragraph.style else ""
            if style.lower().startswith("heading"):
                section = value
                continue
            blocks.append({"text": value, "location": f"paragraph:{index}",
                           "page_number": None, "section": section,
                           "start_offset": None, "end_offset": None})
    elif suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for line_no, line in enumerate(text.splitlines(), start=1):
                value = line.strip()
                if value:
                    blocks.append({"text": value, "location": f"page:{page_number}/line:{line_no}",
                                   "page_number": page_number, "section": None,
                                   "start_offset": None, "end_offset": None})
    else:
        raise ValueError(f"Unsupported document type: {suffix}")

    chunks = registry.get_version_chunks(version["version_id"])
    chunk_meta = []
    for chunk in chunks:
        try:
            metadata = json.loads(chunk["metadata_json"])
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        chunk_meta.append((chunk, metadata))

    units: list[dict[str, Any]] = []
    for block in blocks:
        for sentence in _split_statements(block["text"]):
            chunk_id = None
            for chunk, metadata in chunk_meta:
                metadata_page = metadata.get("page")
                page_match = (block["page_number"] is not None and
                              (metadata_page == block["page_number"] or
                               metadata_page == block["page_number"] - 1))
                offset_match = (isinstance(block["start_offset"], int) and
                                isinstance(chunk.get("start_offset"), int) and
                                chunk["start_offset"] <= block["start_offset"] <
                                (chunk.get("end_offset") or chunk["start_offset"]))
                if page_match or offset_match:
                    chunk_id = chunk["chunk_id"]
                    break
            units.append({
                "text": sentence,
                "source": {
                    "document_id": version["document_id"],
                    "logical_name": version["logical_name"],
                    "version_id": version["version_id"],
                    "version_number": version["version_number"],
                    "source_filename": version["original_filename"],
                    "page_number": block["page_number"],
                    "section": block["section"],
                    "location": block["location"],
                    "chunk_id": chunk_id,
                },
            })
            if len(units) >= max_statements:
                return units
    return units


def _lexical_overlap(a: str, b: str) -> float:
    left = {token.casefold() for token in WORD.findall(a)}
    right = {token.casefold() for token in WORD.findall(b)}
    union = left | right
    return len(left & right) / len(union) if union else 0.0


class CrossDocumentVerificationService:
    def __init__(self, *, registry: Any | None = None, predictor: Any | None = None,
                 db_path: str | Path | None = None, artifact_dir: str | Path | None = None):
        settings = get_settings()
        self.registry = registry or get_document_registry()
        self.predictor = predictor
        self.confidence_threshold = settings.ann_confidence_threshold
        self.artifact_dir = Path(artifact_dir or settings.ann_artifact_dir)
        self.db_path = Path(db_path or settings.audit_db_path)
        self.initialize()

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path)) as con:
            con.execute("PRAGMA foreign_keys = ON")
            con.executescript("""
                CREATE TABLE IF NOT EXISTS verification_runs (
                    run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                    selected_version_ids_json TEXT NOT NULL,
                    selected_document_ids_json TEXT NOT NULL,
                    model_version TEXT NOT NULL, status TEXT NOT NULL,
                    candidate_pairs INTEGER NOT NULL, evaluated_pairs INTEGER NOT NULL,
                    truncated INTEGER NOT NULL, summary_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS statement_pairs (
                    pair_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES verification_runs(run_id),
                    statement_a TEXT NOT NULL, statement_b TEXT NOT NULL,
                    source_a_json TEXT NOT NULL, source_b_json TEXT NOT NULL,
                    ann_label TEXT NOT NULL, model_label TEXT NOT NULL,
                    probabilities_json TEXT NOT NULL, confidence REAL NOT NULL,
                    model_version TEXT NOT NULL, review_required INTEGER NOT NULL,
                    review_status TEXT NOT NULL DEFAULT 'pending',
                    review_label TEXT, reviewer TEXT, review_notes TEXT, reviewed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_verification_pairs_run ON statement_pairs(run_id);
                CREATE INDEX IF NOT EXISTS idx_verification_pairs_review ON statement_pairs(review_status);
            """)
            con.commit()

    def _get_predictor(self):
        if self.predictor is None:
            self.predictor = load_predictor(self.artifact_dir)
        return self.predictor

    def verify_versions(self, version_ids: Iterable[str], *,
                        confidence_threshold: float | None = None,
                        max_statements_per_version: int = 500,
                        max_pairs: int = 2000) -> dict[str, Any]:
        selected = list(dict.fromkeys(str(value).strip() for value in version_ids if str(value).strip()))
        if len(selected) < 2:
            raise ValueError("Select at least two document versions")
        if len(selected) > 10:
            raise ValueError("Select no more than 10 documents per verification run")
        if max_statements_per_version < 1 or max_pairs < 1:
            raise ValueError("Statement and pair limits must be positive")
        if max_statements_per_version > 2000:
            raise ValueError("max_statements_per_version cannot exceed 2,000")
        if max_pairs > 10_000:
            raise ValueError("max_pairs cannot exceed 10,000 per verification run")
        versions = [self.registry.get_version(version_id) for version_id in selected]
        if any(version is None for version in versions):
            raise ValueError("One or more selected document versions do not exist")
        if len({version["document_id"] for version in versions}) != len(versions):
            raise ValueError("Cross-document verification requires one selected version per logical document")
        for version in versions:
            if version.get("ingestion_status") != "complete":
                raise ValueError(f"Document version {version['version_id']} has not completed ingestion")

        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        if confidence_threshold is not None and not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")

        predictor = self._get_predictor()
        model_version = predictor.metadata.get("model_version", "unknown")
        statements = []
        source_counts = []
        total_statements = 0
        for version in versions:
            units = _extract_units(version, max_statements_per_version, self.registry)
            if not units:
                raise ValueError(f"No extractable statements in {version['logical_name']}")
            statements.append(units)
            total_statements += len(units)
            if total_statements > 2000:
                raise ValueError("Verification selection exceeds 2,000 extracted statements; narrow the selected documents")
            source_counts.append({"version_id": version["version_id"],
                                  "document_id": version["document_id"],
                                  "logical_name": version["logical_name"],
                                  "statements": len(units),
                                  "statement_limit_reached": len(units) >= max_statements_per_version})

        candidate_heap = []
        total_candidates = 0
        for index_a, index_b in itertools.combinations(range(len(statements)), 2):
            for a in statements[index_a]:
                for b in statements[index_b]:
                    total_candidates += 1
                    if total_candidates > 500_000:
                        raise ValueError("Verification would generate more than 500,000 candidate pairs; narrow the selected documents")
                    # Keep only the best lexical-overlap candidates in bounded memory.
                    # The stable sequence number resolves ties without comparing dicts.
                    score = _lexical_overlap(a["text"], b["text"])
                    sequence = total_candidates
                    key = (score, -index_a, -index_b, -sequence)
                    item = (key, a, b)
                    if len(candidate_heap) < max_pairs:
                        heapq.heappush(candidate_heap, item)
                    elif key > candidate_heap[0][0]:
                        heapq.heapreplace(candidate_heap, item)
        # Lexical overlap ranks candidates; it is not a contradiction score.
        selected_candidates = sorted(candidate_heap, key=lambda item: item[0], reverse=True)
        run_id = str(uuid.uuid4())
        created_at = _now()
        findings = []
        for _, unit_a, unit_b in selected_candidates:
            prediction_kwargs = {}
            if confidence_threshold is not None:
                prediction_kwargs["confidence_threshold"] = confidence_threshold
            prediction = predictor.predict(unit_a["text"], unit_b["text"], **prediction_kwargs)
            finding = {
                "pair_id": str(uuid.uuid4()),
                "statement_a": unit_a["text"], "statement_b": unit_b["text"],
                "source_a": unit_a["source"], "source_b": unit_b["source"],
                "ann": prediction,
                "review_status": "pending",
                "review_label": None,
            }
            findings.append(finding)

        truncated = total_candidates > len(selected_candidates) or any(
            item["statement_limit_reached"] for item in source_counts)
        summary = {
            "selected_documents": source_counts,
            "candidate_pairs": total_candidates,
            "evaluated_pairs": len(findings),
            "truncated": truncated,
            "truncation_policy": "Highest lexical-overlap pairs are evaluated first; inspect all selected source documents or raise max_pairs for broader coverage.",
            "label_counts": {label: sum(f["ann"]["predicted_label"] == label for f in findings)
                             for label in ("CONTRADICTION", "CONSISTENT", "UNCERTAIN")},
            "human_review_required": sum(bool(f["ann"]["review_required"]) for f in findings),
            "model_version": model_version,
            "authority_warning": "ANN labels describe relationships between supplied statements; they do not establish factual truth or which policy is authoritative.",
        }
        with closing(sqlite3.connect(self.db_path)) as con:
            con.execute("PRAGMA foreign_keys = ON")
            con.execute("""INSERT INTO verification_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (run_id, created_at, json.dumps(selected),
                         json.dumps([v["document_id"] for v in versions]), model_version,
                         "partial" if truncated else "complete", total_candidates,
                         len(findings), int(truncated), json.dumps(summary, sort_keys=True)))
            for finding in findings:
                ann = finding["ann"]
                con.execute("""INSERT INTO statement_pairs
                    (pair_id, run_id, statement_a, statement_b, source_a_json, source_b_json,
                     ann_label, model_label, probabilities_json, confidence, model_version,
                     review_required, review_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (finding["pair_id"], run_id, finding["statement_a"], finding["statement_b"],
                     json.dumps(finding["source_a"], sort_keys=True),
                     json.dumps(finding["source_b"], sort_keys=True),
                     ann["predicted_label"], ann["model_label"],
                     json.dumps(ann["probabilities"], sort_keys=True), ann["confidence"],
                     ann["model_version"], int(ann["review_required"]), "pending"))
            con.execute("""CREATE TABLE IF NOT EXISTS verification_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
                action TEXT NOT NULL, run_id TEXT, pair_id TEXT, actor TEXT, details_json TEXT NOT NULL
            )""")
            con.execute("INSERT INTO verification_audit(created_at, action, run_id, actor, details_json) VALUES (?, ?, ?, ?, ?)",
                        (_now(), "verification_run_created", run_id, None, json.dumps(summary, sort_keys=True)))
            con.commit()

        return {"run_id": run_id, "created_at": created_at, "status": "partial" if truncated else "complete",
                "model_version": model_version, "summary": summary, "findings": findings}

    def get_report(self, run_id: str) -> dict[str, Any] | None:
        with closing(sqlite3.connect(self.db_path)) as con:
            con.row_factory = sqlite3.Row
            run = con.execute("SELECT * FROM verification_runs WHERE run_id = ?", (run_id,)).fetchone()
            if run is None:
                return None
            rows = con.execute("SELECT * FROM statement_pairs WHERE run_id = ? ORDER BY rowid", (run_id,)).fetchall()
        findings = []
        for row in rows:
            findings.append({
                "pair_id": row["pair_id"], "statement_a": row["statement_a"],
                "statement_b": row["statement_b"], "source_a": json.loads(row["source_a_json"]),
                "source_b": json.loads(row["source_b_json"]),
                "ann": {"predicted_label": row["ann_label"], "model_label": row["model_label"],
                        "probabilities": json.loads(row["probabilities_json"]),
                        "confidence": row["confidence"], "model_version": row["model_version"],
                        "review_required": bool(row["review_required"])},
                "review_status": row["review_status"], "review_label": row["review_label"],
                "reviewer": row["reviewer"], "review_notes": row["review_notes"],
                "reviewed_at": row["reviewed_at"],
            })
        return {"run_id": run["run_id"], "created_at": run["created_at"],
                "status": run["status"], "model_version": run["model_version"],
                "selected_version_ids": json.loads(run["selected_version_ids_json"]),
                "summary": json.loads(run["summary_json"]), "findings": findings}

    def review_pair(self, pair_id: str, *, reviewer: str, label: str,
                    notes: str | None = None) -> dict[str, Any]:
        if not reviewer.strip() or len(reviewer) > 128:
            raise ValueError("A valid reviewer identifier is required")
        normalized_label = label.strip().upper()
        if normalized_label not in REVIEW_LABELS:
            raise ValueError("Review label must be CONTRADICTION, CONSISTENT, or UNCERTAIN")
        if notes is not None and len(notes) > 2000:
            raise ValueError("Review notes must be 2000 characters or fewer")
        reviewed_at = _now()
        with closing(sqlite3.connect(self.db_path)) as con:
            con.execute("PRAGMA foreign_keys = ON")
            con.row_factory = sqlite3.Row
            row = con.execute("SELECT run_id, ann_label FROM statement_pairs WHERE pair_id = ?", (pair_id,)).fetchone()
            if row is None:
                raise KeyError("Unknown statement pair")
            con.execute("""UPDATE statement_pairs SET review_status = 'reviewed', review_label = ?,
                           reviewer = ?, review_notes = ?, reviewed_at = ? WHERE pair_id = ?""",
                        (normalized_label, reviewer.strip(), notes, reviewed_at, pair_id))
            con.execute("INSERT INTO verification_audit(created_at, action, run_id, pair_id, actor, details_json) VALUES (?, ?, ?, ?, ?, ?)",
                        (reviewed_at, "statement_pair_reviewed", row["run_id"], pair_id,
                         reviewer.strip(), json.dumps({"ann_label": row["ann_label"],
                                                       "review_label": normalized_label,
                                                       "notes": notes}, sort_keys=True)))
            con.commit()
        return {"pair_id": pair_id, "ann_label_preserved": row["ann_label"],
                "review_label": normalized_label, "review_status": "reviewed",
                "reviewer": reviewer.strip(), "reviewed_at": reviewed_at}
