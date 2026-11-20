import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services import audit


class DocumentAuditTests(unittest.TestCase):
    def test_document_event_records_metadata_without_document_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "audit.sqlite3"
            with patch.object(audit.settings, "audit_db_path", str(db)):
                audit.init_db()
                audit.write_document_audit(
                    "document_version_ingested",
                    "doc-123",
                    "version-456",
                    {"version_number": 2, "chunk_count": 4},
                )
            con = sqlite3.connect(db)
            try:
                row = con.execute(
                    "SELECT action, document_id, version_id, details_json FROM document_audit"
                ).fetchone()
            finally:
                con.close()
            self.assertEqual(row[0], "document_version_ingested")
            self.assertEqual(row[1], "doc-123")
            self.assertEqual(row[2], "version-456")
            self.assertEqual(json.loads(row[3]), {"chunk_count": 4, "version_number": 2})


if __name__ == "__main__":
    unittest.main()
