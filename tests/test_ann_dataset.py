import unittest
from collections import Counter
from pathlib import Path

from app.ml.dataset import LABELS, load_dataset, split_by_group

ROOT = Path(__file__).resolve().parents[1]


class AnnDatasetTests(unittest.TestCase):
    def test_seed_dataset_schema_and_class_balance(self):
        rows = load_dataset(ROOT / "data/ann/enterprise_it_statement_pairs.jsonl")
        self.assertEqual(len(rows), 90)
        self.assertEqual(set(Counter(row["label"] for row in rows)), set(LABELS))
        self.assertTrue(all(row["source_doc_a"] != row["source_doc_b"] for row in rows))

    def test_group_split_has_no_document_group_leakage(self):
        splits = split_by_group(load_dataset(ROOT / "data/ann/enterprise_it_statement_pairs.jsonl"), seed=42)
        group_sets = {name: {row["group_id"] for row in rows} for name, rows in splits.items()}
        self.assertFalse(group_sets["train"] & group_sets["validation"])
        self.assertFalse(group_sets["train"] & group_sets["test"])
        self.assertFalse(group_sets["validation"] & group_sets["test"])
        self.assertEqual(sum(len(rows) for rows in splits.values()), 90)

    
    def test_invalid_labels_are_rejected(self):
        import tempfile
        import json

        row = {
            "statement_a": "a",
            "statement_b": "b",
            "label": "MAYBE",
            "source_doc_a": "a",
            "source_doc_b": "b",
            "group_id": "g",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "invalid.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unsupported label"):
                load_dataset(path)

