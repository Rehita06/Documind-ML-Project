"""HTTP-level offline integration of upload, versioning, diff and real ANN inference."""
from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.security import SecurityHeadersMiddleware
from app.services.document_registry import DocumentRegistry
from app.services.verification import CrossDocumentVerificationService
from tests.test_ingest_api import FakeChunk, import_routes_with_lightweight_adapters


class EndToEndHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project_root = Path(__file__).resolve().parents[1]

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.registry = DocumentRegistry(root / "registry.sqlite3", root / "uploads")
        self.registry.initialize()
        self.routes = import_routes_with_lightweight_adapters()
        self.routes.settings.admin_api_key = "e2e-test-secret"
        self.routes.settings.max_upload_bytes = 1024 * 1024
        self.routes.get_document_registry = lambda: self.registry
        self.routes.add_documents = lambda chunks, ids=None: ids or []
        self.routes.load_file = lambda path: [FakeChunk(Path(path).read_text(encoding="utf-8"))]
        self.routes.chunk_documents = lambda docs: docs
        self.audit_events = []
        self.routes.write_document_audit = lambda *args, **kwargs: self.audit_events.append(args)

        self.verification = CrossDocumentVerificationService(
            registry=self.registry,
            db_path=root / "verification.sqlite3",
            artifact_dir=self.project_root / "artifacts" / "ann" / "ann-v1-seed42",
        )
        app = FastAPI()
        app.add_middleware(SecurityHeadersMiddleware)
        app.include_router(self.routes.router)
        self.client = TestClient(app)
        self.headers = {"X-Admin-Key": "e2e-test-secret"}

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()
        import sys
        sys.modules.pop("app.api.routes", None)

    def upload(self, content: str, filename: str, logical_name: str):
        return self.client.post(
            "/api/ingest",
            headers=self.headers,
            data={"logical_name": logical_name},
            files={"file": (filename, content.encode("utf-8"), "text/plain")},
        )

    def test_http_journey_upload_version_compare_verify_review(self):
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["status"], "ok")
        self.assertEqual(health.headers["x-content-type-options"], "nosniff")
        self.assertEqual(health.headers["x-frame-options"], "DENY")
        self.assertEqual(health.headers["cache-control"], "no-store")
        self.assertIn("frame-ancestors 'none'", health.headers["content-security-policy"])

        self.assertEqual(self.client.get("/api/documents").status_code, 401)

        old = self.upload(
            "# VPN Policy\nVPN access requires MFA for employees.",
            "..\\..\\vpn.txt", "VPN Policy",
        )
        self.assertEqual(old.status_code, 200, old.text)
        old_data = old.json()
        self.assertEqual(old_data["file"], "vpn.txt")
        self.assertEqual(old_data["version_number"], 1)
        self.assertEqual(old_data["ingestion_status"], "complete")

        duplicate = self.upload(
            "# VPN Policy\nVPN access requires MFA for employees.",
            "renamed.txt", "VPN Policy",
        )
        self.assertEqual(duplicate.status_code, 200)
        self.assertTrue(duplicate.json()["duplicate"])
        self.assertEqual(duplicate.json()["version_id"], old_data["version_id"])

        new = self.upload(
            "# VPN Policy\nVPN access requires MFA for employees.\nVPN access requires device certificates.",
            "vpn-revised.txt", "VPN Policy",
        )
        self.assertEqual(new.status_code, 200, new.text)
        new_data = new.json()
        self.assertEqual(new_data["version_number"], 2)

        other = self.upload(
            "# Remote Access\nRemote access must use multifactor authentication.",
            "remote-access.txt", "Remote Access Standard",
        )
        self.assertEqual(other.status_code, 200, other.text)
        other_data = other.json()

        listing = self.client.get("/api/documents", headers=self.headers)
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(len(listing.json()["documents"]), 2)

        versions = self.client.get(
            f"/api/documents/{old_data['document_id']}/versions", headers=self.headers
        )
        self.assertEqual(versions.status_code, 200)
        self.assertEqual(len(versions.json()["versions"]), 2)

        comparison = self.client.post(
            "/api/documents/compare", headers=self.headers,
            json={"old_version_id": old_data["version_id"], "new_version_id": new_data["version_id"]},
        )
        self.assertEqual(comparison.status_code, 200, comparison.text)
        self.assertGreaterEqual(comparison.json()["counts"]["ADDED"], 1)
        self.assertTrue(comparison.json()["same_logical_document"])

        with patch("app.services.verification.CrossDocumentVerificationService", return_value=self.verification):
            run = self.client.post(
                "/api/verification/runs", headers=self.headers,
                json={"version_ids": [new_data["version_id"], other_data["version_id"]]},
            )
            self.assertEqual(run.status_code, 200, run.text)
            report = run.json()
            self.assertIn(report["status"], {"complete", "partial"})
            self.assertEqual(report["model_version"], "ann-v1-seed42")
            self.assertGreaterEqual(report["summary"]["evaluated_pairs"], 1)
            finding = report["findings"][0]
            self.assertEqual(finding["source_a"]["document_id"], new_data["document_id"])
            self.assertEqual(finding["source_b"]["document_id"], other_data["document_id"])

            fetched = self.client.get(
                f"/api/verification/runs/{report['run_id']}", headers=self.headers
            )
            self.assertEqual(fetched.status_code, 200)
            self.assertEqual(fetched.json()["findings"][0]["ann"]["model_version"], "ann-v1-seed42")

            review = self.client.post(
                f"/api/verification/pairs/{finding['pair_id']}/review",
                headers=self.headers,
                json={"reviewer": "  qa-reviewer  ", "label": "UNCERTAIN", "notes": "Manual review required."},
            )
            self.assertEqual(review.status_code, 200, review.text)
            self.assertEqual(review.json()["ann_label_preserved"], finding["ann"]["predicted_label"])
            self.assertEqual(review.json()["review_label"], "UNCERTAIN")

            reviewed = self.client.get(
                f"/api/verification/runs/{report['run_id']}", headers=self.headers
            ).json()["findings"][0]
            self.assertEqual(reviewed["ann"]["predicted_label"], finding["ann"]["predicted_label"])
            self.assertEqual(reviewed["review_status"], "reviewed")
            self.assertEqual(reviewed["reviewer"], "qa-reviewer")

        self.assertGreaterEqual(len(self.audit_events), 3)
        self.assertTrue(all(len(event) >= 1 for event in self.audit_events))

        template = (self.project_root / "templates" / "index.html").read_text(encoding="utf-8")
        frontend = (self.project_root / "static" / "js" / "app.js").read_text(encoding="utf-8")
        self.assertIn("Document Intelligence", template)
        self.assertIn("/api/documents/compare", frontend)
        self.assertIn("/api/verification/runs", frontend)
        self.assertIn("submit-review", frontend)

    def test_http_validation_rejects_bad_effective_date_and_review_label(self):
        response = self.client.post(
            "/api/ingest", headers=self.headers,
            data={"logical_name": "Bad Date", "effective_date": "not-a-date"},
            files={"file": ("policy.txt", b"This is valid policy text.", "text/plain")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("YYYY-MM-DD", response.json()["detail"])

        compact_date = self.client.post(
            "/api/ingest", headers=self.headers,
            data={"logical_name": "Compact Date", "effective_date": "20261118"},
            files={"file": ("policy.txt", b"This is valid policy text.", "text/plain")},
        )
        self.assertEqual(compact_date.status_code, 400)

        response = self.client.post(
            "/api/verification/pairs/nope/review", headers=self.headers,
            json={"reviewer": "qa", "label": "OVERRIDE"},
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
