import asyncio
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException, UploadFile

from app.services.document_registry import DocumentRegistry


class FakeChunk:
    def __init__(self, text):
        self.page_content = text
        self.metadata = {"page": 0, "start_index": 0, "source": "source"}


def import_routes_with_lightweight_adapters():
    workflow = types.ModuleType("app.rag.workflow")
    workflow.ask = lambda question: {}
    vectorstore = types.ModuleType("app.rag.vectorstore")
    vectorstore.add_documents = lambda chunks, ids=None: ids or []
    ingestion = types.ModuleType("app.services.ingestion")
    ingestion.SUPPORTED = {".pdf", ".txt", ".md", ".docx"}
    ingestion.load_file = lambda path: [FakeChunk(Path(path).read_text(encoding="utf-8"))]
    ingestion.chunk_documents = lambda docs: docs
    with patch.dict(sys.modules, {
        "app.rag.workflow": workflow,
        "app.rag.vectorstore": vectorstore,
        "app.services.ingestion": ingestion,
    }):
        sys.modules.pop("app.api.routes", None)
        return __import__("app.api.routes", fromlist=["router"])


class IngestApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.registry = DocumentRegistry(root / "registry.sqlite3", root / "uploads")
        self.routes = import_routes_with_lightweight_adapters()
        self.routes.settings.admin_api_key = "test-admin-secret"
        self.routes.settings.max_upload_bytes = 1024
        self.routes.get_document_registry = lambda: self.registry
        self.routes.add_documents = lambda chunks, ids=None: ids or []
        self.routes.load_file = lambda path: [FakeChunk(Path(path).read_text(encoding="utf-8"))]
        self.routes.chunk_documents = lambda docs: docs
        self.audit_events = []
        self.routes.write_document_audit = lambda *args, **kwargs: self.audit_events.append(args)

    def tearDown(self):
        sys.modules.pop("app.api.routes", None)
        self.temp.cleanup()

    def upload(self, body, filename="vpn.txt", logical_name="VPN Policy"):
        file = UploadFile(filename=filename, file=io.BytesIO(body))
        return asyncio.run(self.routes.ingest(
            file=file,
            x_admin_key="test-admin-secret",
            logical_name=logical_name,
            effective_date=None,
        ))

    def test_upload_and_identical_reupload_are_idempotent(self):
        first = self.upload(b"VPN requires MFA")
        second = self.upload(b"VPN requires MFA", filename="renamed.txt")
        self.assertEqual(first["version_number"], 1)
        self.assertFalse(first["duplicate"])
        self.assertTrue(second["duplicate"])
        self.assertEqual(second["version_id"], first["version_id"])
        self.assertEqual(len(self.registry.list_versions(first["document_id"])), 1)
        self.assertEqual([event[0] for event in self.audit_events], ["document_version_ingested", "document_upload_duplicate"])

    def test_changed_upload_creates_new_active_version(self):
        first = self.upload(b"VPN requires MFA")
        second = self.upload(b"VPN requires device certificate")
        self.assertEqual(second["version_number"], 2)
        self.assertEqual(self.registry.get_active_version_ids(), [second["version_id"]])
        self.assertEqual(len(self.registry.list_versions(first["document_id"])), 2)

    def test_failed_vector_index_keeps_previous_active_version(self):
        first = self.upload(b"VPN requires MFA")
        self.routes.add_documents = lambda chunks, ids=None: (_ for _ in ()).throw(RuntimeError("provider down"))
        with self.assertRaises(HTTPException) as caught:
            self.upload(b"VPN requires a certificate")
        self.assertEqual(caught.exception.status_code, 500)
        self.assertEqual(self.registry.get_active_version_ids(), [first["version_id"]])

    def test_upload_requires_non_default_admin_key(self):
        self.routes.settings.admin_api_key = "change-me"
        file = UploadFile(filename="vpn.txt", file=io.BytesIO(b"text"))
        with self.assertRaises(HTTPException) as caught:
            asyncio.run(self.routes.ingest(file=file, x_admin_key="change-me"))
        self.assertEqual(caught.exception.status_code, 401)

    def test_upload_size_limit_is_enforced(self):
        with self.assertRaises(HTTPException) as caught:
            self.upload(b"x" * 1025)
        self.assertEqual(caught.exception.status_code, 413)


if __name__ == "__main__":
    unittest.main()
