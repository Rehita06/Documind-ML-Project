import tempfile
import unittest
from pathlib import Path

from app.services.document_registry import DocumentRegistry
from app.services.verification import CrossDocumentVerificationService


class FakePredictor:
    metadata = {"model_version": "fake-ann-test"}

    def predict(self, a, b, *, confidence_threshold=None):
        label = "CONTRADICTION" if ("not" in a.casefold()) != ("not" in b.casefold()) else "CONSISTENT"
        confidence = 0.91
        threshold = 0.8 if confidence_threshold is None else confidence_threshold
        low = confidence < threshold
        return {
            "predicted_label": "UNCERTAIN" if low else label,
            "model_label": label,
            "probabilities": {"CONTRADICTION": 0.91 if label == "CONTRADICTION" else 0.03,
                              "CONSISTENT": 0.91 if label == "CONSISTENT" else 0.03,
                              "UNCERTAIN": 0.06},
            "confidence": confidence,
            "model_version": self.metadata["model_version"],
            "review_required": low,
            "threshold": threshold,
        }


class CrossDocumentVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.registry = DocumentRegistry(root / "registry.db", root / "uploads")
        self.registry.initialize()
        self.db = root / "audit.db"
        self.predictor = FakePredictor()
        self.service = CrossDocumentVerificationService(
            registry=self.registry, predictor=self.predictor, db_path=self.db,
            artifact_dir=root / "unused-model")
        self.versions = []
        self.documents = []
        self._add_document("MFA Policy", "mfa.txt",
                           "All administrators must use MFA.\nVPN users must enroll a device.")
        self._add_document("Access Standard", "access.txt",
                           "Administrators must not use MFA.\nVPN users must enroll a device.")

    def tearDown(self):
        self.temp.cleanup()

    def _add_document(self, name, filename, content):
        record = self.registry.create_version(name, filename, content.encode())
        self.registry.complete_ingestion(record["version_id"], [])
        self.versions.append(record["version_id"])
        self.documents.append(record["document_id"])

    def test_verifies_only_cross_document_pairs_and_persists_source_references(self):
        report = self.service.verify_versions(self.versions)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["summary"]["candidate_pairs"], 4)
        self.assertEqual(report["summary"]["evaluated_pairs"], 4)
        self.assertEqual(report["model_version"], "fake-ann-test")
        finding = report["findings"][0]
        self.assertIn("document_id", finding["source_a"])
        self.assertIn("version_id", finding["source_b"])
        self.assertIn("location", finding["source_a"])
        loaded = self.service.get_report(report["run_id"])
        self.assertEqual(len(loaded["findings"]), 4)
        self.assertEqual(loaded["findings"][0]["ann"]["model_version"], "fake-ann-test")

    def test_same_logical_document_versions_are_not_compared_as_cross_documents(self):
        third = self.registry.create_version("MFA Policy", "mfa.txt", b"A different policy version.")
        self.registry.complete_ingestion(third["version_id"], [])
        with self.assertRaisesRegex(ValueError, "one selected version per logical document"):
            self.service.verify_versions([self.versions[0], third["version_id"]])

    def test_requires_two_existing_ingested_documents(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            self.service.verify_versions([self.versions[0]])
        with self.assertRaisesRegex(ValueError, "do not exist"):
            self.service.verify_versions([self.versions[0], "missing-version"])

    def test_pair_limit_is_reported_as_partial_not_silently_hidden(self):
        report = self.service.verify_versions(self.versions, max_pairs=2)
        self.assertEqual(report["status"], "partial")
        self.assertTrue(report["summary"]["truncated"])
        self.assertEqual(report["summary"]["evaluated_pairs"], 2)
        self.assertEqual(report["summary"]["candidate_pairs"], 4)

    def test_human_review_is_separate_from_original_ann_prediction(self):
        report = self.service.verify_versions(self.versions)
        finding = report["findings"][0]
        original = finding["ann"]["predicted_label"]
        reviewed = self.service.review_pair(
            finding["pair_id"], reviewer="reviewer-123", label="UNCERTAIN",
            notes="Needs policy owner review")
        self.assertEqual(reviewed["ann_label_preserved"], original)
        self.assertEqual(reviewed["review_label"], "UNCERTAIN")
        stored = self.service.get_report(report["run_id"])
        stored_finding = next(item for item in stored["findings"] if item["pair_id"] == finding["pair_id"])
        self.assertEqual(stored_finding["ann"]["predicted_label"], original)
        self.assertEqual(stored_finding["review_label"], "UNCERTAIN")
        self.assertEqual(stored_finding["review_status"], "reviewed")

    def test_invalid_review_label_and_unknown_pair_are_rejected(self):
        report = self.service.verify_versions(self.versions)
        with self.assertRaisesRegex(ValueError, "Review label"):
            self.service.review_pair(report["findings"][0]["pair_id"],
                                     reviewer="reviewer", label="OVERRIDE")
        with self.assertRaises(KeyError):
            self.service.review_pair("missing", reviewer="reviewer", label="UNCERTAIN")

    def test_low_confidence_threshold_routes_result_to_review(self):
        report = self.service.verify_versions(self.versions, confidence_threshold=0.99)
        self.assertTrue(all(item["ann"]["predicted_label"] == "UNCERTAIN"
                            for item in report["findings"]))
        self.assertEqual(report["summary"]["human_review_required"], 4)

    def test_chunk_ids_are_retained_when_registry_offsets_match(self):
        first_content = b"Administrators must always use MFA."
        second_content = b"Administrators must not use MFA."
        first = self.registry.create_version("Chunked MFA", "chunked-a.txt", first_content)
        second = self.registry.create_version("Chunked Access", "chunked-b.txt", second_content)
        for record, chunk_id, content in ((first, "chunk-a", first_content),
                                          (second, "chunk-b", second_content)):
            self.registry.complete_ingestion(record["version_id"], [{
                "chunk_id": chunk_id, "vector_id": chunk_id,
                "start_offset": 0, "end_offset": len(content),
                "text_hash": "test-hash", "metadata": {},
            }])
        report = self.service.verify_versions([first["version_id"], second["version_id"]])
        self.assertEqual(report["findings"][0]["source_a"]["chunk_id"], "chunk-a")
        self.assertEqual(report["findings"][0]["source_b"]["chunk_id"], "chunk-b")

    def test_real_saved_ann_artifact_can_be_used_by_service(self):
        from app.ml.inference import load_predictor
        root = Path(__file__).resolve().parents[1]
        predictor = load_predictor(root / "artifacts/ann/ann-v1-seed42")
        service = CrossDocumentVerificationService(
            registry=self.registry, predictor=predictor, db_path=Path(self.temp.name) / "real-ann.db")
        report = service.verify_versions(self.versions, max_pairs=1)
        self.assertEqual(report["model_version"], "ann-v1-seed42")
        prediction = report["findings"][0]["ann"]
        self.assertIn(prediction["predicted_label"], {"CONTRADICTION", "CONSISTENT", "UNCERTAIN"})
        self.assertAlmostEqual(sum(prediction["probabilities"].values()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
