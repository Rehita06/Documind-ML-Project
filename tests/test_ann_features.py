import unittest
from pathlib import Path

from app.ml.dataset import load_dataset, split_by_group
from app.ml.features import build_vocabulary, encode_texts, fit_scalar_scaler, structured_pair_features

ROOT = Path(__file__).resolve().parents[1]


class AnnFeatureTests(unittest.TestCase):
    def setUp(self):
        rows = load_dataset(ROOT / "data/ann/enterprise_it_statement_pairs.jsonl")
        self.splits = split_by_group(rows)

    def test_vocabulary_is_fitted_only_from_training_rows(self):
        vocabulary = build_vocabulary(self.splits["train"])
        self.assertNotIn("uniquetesttokenzz", vocabulary)
        test_text = "uniquetesttokenzz must be used"
        ids, lengths = encode_texts([test_text], vocabulary)
        self.assertEqual(ids.shape[0], 1)
        self.assertEqual(lengths[0], 4)
        self.assertIn(1, ids[0])  # unseen token maps to <UNK>

    def test_pair_features_are_label_independent_and_numeric(self):
        features = structured_pair_features(self.splits["train"][:4])
        self.assertEqual(features.shape, (4, 4))
        self.assertTrue((features >= 0).all())
        scaler = fit_scalar_scaler(self.splits["train"])
        self.assertEqual(scaler.n_features_in_, 4)
