import tempfile
import unittest
from pathlib import Path

from app.ml.inference import load_predictor
from app.ml.train import train

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/ann/enterprise_it_statement_pairs.jsonl"


class AnnTrainingTests(unittest.TestCase):
    def test_training_saves_loadable_model_and_baseline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report = train(DATASET, temp_dir, seed=17, embedding_dim=12,
                           hidden_dims=(24, 12), dropout=0.1,
                           learning_rate=0.005, batch_size=16, epochs=5, patience=3)
            artifact = Path(temp_dir) / report["model_version"]
            self.assertTrue((artifact / "model.pt").is_file())
            self.assertTrue((artifact / "evaluation_report.json").is_file())
            self.assertTrue((artifact / "baseline_logistic_regression.joblib").is_file())
            predictor = load_predictor(artifact)
            result = predictor.predict(
                "All privileged accounts must use multi-factor authentication.",
                "Administrators are required to complete MFA at sign-in.")
            self.assertIn(result["predicted_label"], {"CONTRADICTION", "CONSISTENT", "UNCERTAIN"})
            self.assertAlmostEqual(sum(result["probabilities"].values()), 1.0, places=5)
            self.assertEqual(result["model_version"], report["model_version"])
            self.assertIn("review_required", result)
            strict = predictor.predict("MFA is mandatory.", "MFA is mandatory.", confidence_threshold=1.0)
            self.assertEqual(strict["predicted_label"], "UNCERTAIN")
            self.assertTrue(strict["review_required"])
            with self.assertRaisesRegex(ValueError, "between 0 and 1"):
                predictor.predict("a", "b", confidence_threshold=1.5)
            self.assertIn("ann_thresholded", report["evaluation"])
            self.assertIn("logistic_regression_baseline", report["evaluation"])

    def test_empty_statement_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report = train(DATASET, temp_dir, embedding_dim=8, hidden_dims=(12,), epochs=2, patience=2)
            predictor = load_predictor(Path(temp_dir) / report["model_version"])
            with self.assertRaisesRegex(ValueError, "non-empty"):
                predictor.predict(" ", "second statement")
