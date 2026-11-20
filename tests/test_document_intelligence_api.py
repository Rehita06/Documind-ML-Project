import sys
import types
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from tests.test_ingest_api import import_routes_with_lightweight_adapters


class FakeRegistry:
    def __init__(self):
        self.versions = {
            "v1": {"version_id": "v1", "document_id": "doc-a", "logical_name": "VPN Policy", "ingestion_status": "complete"},
            "v2": {"version_id": "v2", "document_id": "doc-a", "logical_name": "VPN Policy", "ingestion_status": "complete"},
            "v3": {"version_id": "v3", "document_id": "doc-b", "logical_name": "Access Standard", "ingestion_status": "complete"},
        }

    def get_version(self, version_id):
        return self.versions.get(version_id)


class FakeVerificationService:
    report = {"run_id": "run-1", "status": "complete", "findings": []}

    def __init__(self):
        pass

    def verify_versions(self, version_ids, **kwargs):
        return {**self.report, "selected_version_ids": version_ids}

    def get_report(self, run_id):
        return self.report if run_id == "run-1" else None

    def review_pair(self, pair_id, **kwargs):
        if pair_id != "pair-1":
            raise KeyError(pair_id)
        return {"pair_id": pair_id, "ann_label_preserved": "CONTRADICTION", "review_label": kwargs["label"]}


class DocumentIntelligenceApiTests(unittest.TestCase):
    def setUp(self):
        self.routes = import_routes_with_lightweight_adapters()
        self.routes.settings.admin_api_key = "api-test-secret"
        self.routes.get_document_registry = lambda: self.registry
        self.registry = FakeRegistry()

    def tearDown(self):
        sys.modules.pop("app.api.routes", None)

    def test_compare_versions_returns_report(self):
        change = types.ModuleType("app.services.change_detection")
        change.compare_document_versions = lambda registry, old, new, **kw: {
            "old_version": {"version_id": old}, "new_version": {"version_id": new},
            "counts": {"ADDED": 1, "DELETED": 0, "MODIFIED": 0, "UNCHANGED": 2}, "changes": []
        }
        with patch.dict(sys.modules, {"app.services.change_detection": change}):
            payload = self.routes.VersionComparisonRequest(old_version_id="v1", new_version_id="v2")
            result = self.routes.compare_versions(payload, "api-test-secret")
        self.assertEqual(result["counts"]["ADDED"], 1)

    def test_compare_rejects_versions_from_different_documents(self):
        payload = self.routes.VersionComparisonRequest(old_version_id="v1", new_version_id="v3")
        with self.assertRaises(HTTPException) as caught:
            self.routes.compare_versions(payload, "api-test-secret")
        self.assertEqual(caught.exception.status_code, 400)

    def test_compare_requires_admin_key(self):
        payload = self.routes.VersionComparisonRequest(old_version_id="v1", new_version_id="v2")
        with self.assertRaises(HTTPException) as caught:
            self.routes.compare_versions(payload, "wrong")
        self.assertEqual(caught.exception.status_code, 401)

    def test_verification_endpoint_calls_service(self):
        module = types.ModuleType("app.services.verification")
        module.CrossDocumentVerificationService = FakeVerificationService
        with patch.dict(sys.modules, {"app.services.verification": module}):
            payload = self.routes.VerificationRunRequest(version_ids=["v1", "v3"])
            result = self.routes.create_verification_run(payload, "api-test-secret")
        self.assertEqual(result["run_id"], "run-1")
        self.assertEqual(result["selected_version_ids"], ["v1", "v3"])

    def test_verification_report_not_found(self):
        module = types.ModuleType("app.services.verification")
        module.CrossDocumentVerificationService = FakeVerificationService
        with patch.dict(sys.modules, {"app.services.verification": module}):
            with self.assertRaises(HTTPException) as caught:
                self.routes.get_verification_run("missing", "api-test-secret")
        self.assertEqual(caught.exception.status_code, 404)

    def test_review_endpoint_preserves_ann_label(self):
        module = types.ModuleType("app.services.verification")
        module.CrossDocumentVerificationService = FakeVerificationService
        with patch.dict(sys.modules, {"app.services.verification": module}):
            payload = self.routes.PairReviewRequest(reviewer="reviewer-1", label="CONSISTENT")
            result = self.routes.review_verification_pair("pair-1", payload, "api-test-secret")
        self.assertEqual(result["ann_label_preserved"], "CONTRADICTION")
        self.assertEqual(result["review_label"], "CONSISTENT")


if __name__ == "__main__":
    unittest.main()
