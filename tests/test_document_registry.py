import hashlib
import tempfile
import unittest
from pathlib import Path

from app.services.document_registry import DocumentRegistry


class DocumentRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.registry = DocumentRegistry(root / "registry.sqlite3", root / "uploads")
        self.registry.initialize()

    def tearDown(self):
        self.temp.cleanup()

    def test_first_upload_creates_pending_version_and_safe_path(self):
        result = self.registry.create_version(
            "VPN Policy", "../../outside.txt", b"VPN access requires MFA"
        )
        self.assertFalse(result["duplicate"])
        self.assertEqual(result["version_number"], 1)
        self.assertEqual(result["ingestion_status"], "processing")
        stored = Path(result["stored_path"])
        self.assertTrue(stored.exists())
        self.assertEqual(stored.read_bytes(), b"VPN access requires MFA")
        self.assertNotIn("outside", str(stored))
        self.assertEqual(self.registry.get_active_version_ids(), [])

    def test_identical_reupload_is_idempotent(self):
        first = self.registry.create_version("VPN Policy", "vpn.txt", b"same bytes")
        self.registry.complete_ingestion(first["version_id"], [])
        duplicate = self.registry.create_version(
            "vpn policy", "renamed.txt", b"same bytes"
        )
        self.assertTrue(duplicate["duplicate"])
        self.assertEqual(duplicate["version_id"], first["version_id"])
        self.assertEqual(duplicate["version_number"], 1)
        self.assertEqual(len(self.registry.list_versions(first["document_id"])), 1)

    def test_changed_content_creates_immutable_version_and_promotes_after_ingest(self):
        first = self.registry.create_version("VPN Policy", "vpn.txt", b"version one")
        self.registry.complete_ingestion(first["version_id"], [{
            "chunk_id": "chunk-1", "vector_id": "vector-1",
            "text_hash": hashlib.sha256(b"version one").hexdigest(),
            "metadata": {"page": 1},
        }])
        second = self.registry.create_version("VPN Policy", "vpn.txt", b"version two")

        # A pending version must not displace the current active version.
        self.assertEqual(self.registry.get_active_version_ids(), [first["version_id"]])
        self.assertEqual(Path(self.registry.get_version(first["version_id"])["stored_path"]).read_bytes(), b"version one")

        self.registry.complete_ingestion(second["version_id"], [{
            "chunk_id": "chunk-2", "vector_id": "vector-2",
            "text_hash": hashlib.sha256(b"version two").hexdigest(),
            "metadata": {"page": 1},
        }])
        self.assertEqual(self.registry.get_active_version_ids(), [second["version_id"]])
        versions = self.registry.list_versions(first["document_id"])
        self.assertEqual({v["version_number"] for v in versions}, {1, 2})
        by_number = {v["version_number"]: v for v in versions}
        self.assertEqual(by_number[1]["status"], "superseded")
        self.assertEqual(by_number[2]["status"], "active")
        self.assertEqual(len(self.registry.get_version_chunks(first["version_id"])), 1)

    def test_failed_new_version_preserves_previous_active_version(self):
        first = self.registry.create_version("MFA Policy", "mfa.md", b"old")
        self.registry.complete_ingestion(first["version_id"], [])
        second = self.registry.create_version("MFA Policy", "mfa.md", b"new")
        self.registry.mark_ingestion_failed(second["version_id"])
        self.assertEqual(self.registry.get_active_version_ids(), [first["version_id"]])
        versions = self.registry.list_versions(first["document_id"])
        failed = next(v for v in versions if v["version_id"] == second["version_id"])
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["ingestion_status"], "ingestion_failed")

    def test_unsupported_extension_and_empty_content_are_rejected(self):
        with self.assertRaises(ValueError):
            self.registry.create_version("Doc", "file.exe", b"content")
        with self.assertRaises(ValueError):
            self.registry.create_version("Doc", "file.txt", b"")


    def test_file_content_is_checked_against_extension(self):
        with self.assertRaisesRegex(ValueError, "does not appear to be a PDF"):
            self.registry.create_version("Bad PDF", "policy.pdf", b"plain text")
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            self.registry.create_version("Bad Text", "policy.txt", bytes([0xff, 0xfe]))

    def test_retirement_excludes_document_from_retrieval_but_preserves_history(self):
        first = self.registry.create_version("Legacy VPN", "vpn.txt", b"retained policy")
        self.registry.complete_ingestion(first["version_id"], [])
        stored_path = Path(self.registry.get_version(first["version_id"])["stored_path"])
        retired = self.registry.retire_document(first["document_id"])
        self.assertEqual(retired["document_id"], first["document_id"])
        self.assertEqual(self.registry.get_active_version_ids(), [])
        self.assertTrue(stored_path.exists())
        self.assertEqual(len(self.registry.list_versions(first["document_id"])), 1)
        self.assertEqual(self.registry.list_documents()[0]["status"], "retired")
        self.assertEqual(self.registry.list_versions(first["document_id"])[0]["status"], "retired")

    def test_retired_logical_document_cannot_be_silently_reactivated(self):
        first = self.registry.create_version("Retired VPN", "vpn.txt", b"old")
        self.registry.complete_ingestion(first["version_id"], [])
        self.registry.retire_document(first["document_id"])
        with self.assertRaisesRegex(ValueError, "is retired"):
            self.registry.create_version("Retired VPN", "vpn.txt", b"new")
        self.assertEqual(self.registry.get_active_version_ids(), [])

    def test_documents_are_distinct_by_logical_name_and_list_current_version(self):
        one = self.registry.create_version("VPN Policy", "vpn.txt", b"same")
        self.registry.complete_ingestion(one["version_id"], [])
        two = self.registry.create_version("MFA Policy", "mfa.txt", b"same")
        self.registry.complete_ingestion(two["version_id"], [])
        documents = self.registry.list_documents()
        self.assertEqual(len(documents), 2)
        by_name = {item["logical_name"]: item for item in documents}
        self.assertEqual(by_name["VPN Policy"]["latest_version_id"], one["version_id"])
        self.assertEqual(by_name["MFA Policy"]["latest_version_id"], two["version_id"])

    def test_version_numbers_are_monotonic_after_failed_upload(self):
        one = self.registry.create_version("Password Policy", "p.txt", b"one")
        self.registry.complete_ingestion(one["version_id"], [])
        two = self.registry.create_version("Password Policy", "p.txt", b"two")
        self.registry.mark_ingestion_failed(two["version_id"])
        three = self.registry.create_version("Password Policy", "p.txt", b"three")
        self.assertEqual(three["version_number"], 3)


if __name__ == "__main__":
    unittest.main()
