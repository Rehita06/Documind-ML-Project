import tempfile
import unittest
from pathlib import Path

from app.services.change_detection import (
    TextUnit, compare_document_versions, compare_text_units,
)
from app.services.document_registry import DocumentRegistry


class TopicEmbedder:
    """Tiny deterministic adapter to test the embedding integration contract."""
    def embed_documents(self, texts):
        vectors = []
        for text in texts:
            lower = text.lower()
            if "mfa" in lower or "multi-factor" in lower:
                vectors.append([1.0, 0.0, 0.0, 0.0])
            elif "password" in lower:
                vectors.append([0.0, 1.0, 0.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 0.0, 1.0] if "account" in lower else [0.0, 0.0, 1.0, 0.0])
        return vectors


class ChangeDetectionTests(unittest.TestCase):
    def test_exact_alignment_and_all_four_change_types(self):
        old = [
            TextUnit("VPN access uses the corporate gateway.", "p1", "Access", 0),
            TextUnit("MFA is required for remote access.", "p2", "Access", 1),
            TextUnit("Passwords expire every 90 days.", "p3", "Credentials", 2),
            TextUnit("Inactive accounts are disabled after 30 days.", "p4", "Accounts", 3),
        ]
        new = [
            TextUnit("VPN access uses the corporate gateway.", "p1", "Access", 0),
            TextUnit("Multi-factor authentication is mandatory for remote access.", "p2", "Access", 1),
            TextUnit("Passwords expire every 60 days.", "p3", "Credentials", 2),
            TextUnit("All endpoints require approved protection software.", "p4", "Security", 3),
        ]
        result = compare_text_units(old, new, embedder=TopicEmbedder(), similarity_threshold=0.8)
        self.assertEqual(result["counts"], {
            "ADDED": 1, "DELETED": 1, "MODIFIED": 2, "UNCHANGED": 1,
        })
        modified = [item for item in result["changes"] if item["change_type"] == "MODIFIED"]
        self.assertEqual(len(modified), 2)
        self.assertTrue(all(item["old_text"] and item["new_text"] for item in modified))
        self.assertTrue(all(item["old_location"] and item["new_location"] for item in modified))

    def test_exact_match_is_case_and_whitespace_insensitive(self):
        result = compare_text_units(
            [TextUnit("MFA   is REQUIRED.", "old:1", None, 0)],
            [TextUnit("mfa is required.", "new:1", None, 0)],
        )
        self.assertEqual(result["counts"]["UNCHANGED"], 1)
        self.assertEqual(result["changes"][0]["alignment_method"], "exact")

    def test_empty_old_and_new_documents(self):
        result = compare_text_units([], [TextUnit("New policy.", "line:1", None, 0)])
        self.assertEqual(result["counts"]["ADDED"], 1)
        self.assertEqual(result["counts"]["DELETED"], 0)

    def test_threshold_validation(self):
        with self.assertRaises(ValueError):
            compare_text_units([], [], similarity_threshold=1.1)

    def test_registry_version_comparison_preserves_identity_and_provenance(self):
        root = Path(tempfile.mkdtemp())
        registry = DocumentRegistry(root / "registry.sqlite3", root / "uploads")
        registry.initialize()
        fixture_dir = Path(__file__).parent.parent / "data" / "fixtures" / "change_detection"
        old_bytes = (fixture_dir / "vpn_policy_v1.txt").read_bytes()
        new_bytes = (fixture_dir / "vpn_policy_v2.txt").read_bytes()
        old = registry.create_version("VPN Access Policy", "vpn_policy.txt", old_bytes)
        registry.complete_ingestion(old["version_id"], [])
        new = registry.create_version("VPN Access Policy", "vpn_policy.txt", new_bytes)
        registry.complete_ingestion(new["version_id"], [])

        result = compare_document_versions(registry, old["version_id"], new["version_id"])
        self.assertEqual(result["similarity_threshold"], 0.45)
        self.assertTrue(result["same_logical_document"])
        self.assertEqual(result["old_version"]["version_number"], 1)
        self.assertEqual(result["new_version"]["version_number"], 2)
        self.assertGreater(result["counts"]["UNCHANGED"], 0)
        self.assertGreater(result["counts"]["MODIFIED"], 0)
        self.assertGreater(result["counts"]["ADDED"], 0)
        self.assertGreater(result["counts"]["DELETED"], 0)
        self.assertTrue(all("old_location" in change and "new_location" in change
                            for change in result["changes"]))

    def test_missing_version_fails_closed(self):
        class EmptyRegistry:
            def get_version(self, _):
                return None
        with self.assertRaisesRegex(ValueError, "must exist"):
            compare_document_versions(EmptyRegistry(), "old", "new")

    def test_same_version_cannot_be_compared_to_itself(self):
        class Registry:
            def get_version(self, version_id):
                return {"version_id": version_id}
        with self.assertRaisesRegex(ValueError, "two different document versions"):
            compare_document_versions(Registry(), "same", "same")


if __name__ == "__main__":
    unittest.main()
