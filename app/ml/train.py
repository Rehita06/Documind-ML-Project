"""Reproducible group-split training, baseline comparison, and artifact creation."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.ml.dataset import LABELS, load_dataset, split_by_group
from app.ml.features import build_vocabulary, encode_texts, fit_scalar_scaler, structured_pair_features
from app.ml.model import StatementPairMLP

LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _tensors(rows: list[dict[str, Any]], vocabulary: dict[str, int], scaler: StandardScaler):
    ids_a, _ = encode_texts([row["statement_a"] for row in rows], vocabulary)
    ids_b, _ = encode_texts([row["statement_b"] for row in rows], vocabulary)
    structured = scaler.transform(structured_pair_features(rows)).astype(np.float32)
    labels = np.asarray([LABEL_TO_ID[row["label"]] for row in rows], dtype=np.int64)
    return (torch.tensor(ids_a, dtype=torch.long), torch.tensor(ids_b, dtype=torch.long),
            torch.tensor(structured, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))


def _metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
        "classification_report": classification_report(y_true, y_pred, labels=[0, 1, 2],
                                                         target_names=list(LABELS), output_dict=True,
                                                         zero_division=0),
        "confusion_matrix_labels_order": list(LABELS),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def _calibrate_threshold(probabilities: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    best_threshold, best_score = 0.50, -1.0
    for threshold in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        predictions = probabilities.argmax(axis=1)
        predictions[probabilities.max(axis=1) < threshold] = LABEL_TO_ID["UNCERTAIN"]
        score = f1_score(y_true, predictions, average="macro", zero_division=0)
        if score > best_score:
            best_threshold, best_score = threshold, float(score)
    return best_threshold, best_score


def train(dataset_path: str | Path, artifact_root: str | Path, *, seed: int = 42,
          embedding_dim: int = 48, hidden_dims: tuple[int, ...] = (128, 64),
          dropout: float = 0.20, learning_rate: float = 0.003,
          batch_size: int = 16, epochs: int = 120, patience: int = 18) -> dict[str, Any]:
    seed_everything(seed)
    started = time.time()
    rows = load_dataset(dataset_path)
    splits = split_by_group(rows, seed=seed)
    train_rows, validation_rows, test_rows = splits["train"], splits["validation"], splits["test"]

    # All learned preprocessing is fitted on the training partition only.
    vocabulary = build_vocabulary(train_rows)
    scalar_scaler = fit_scalar_scaler(train_rows)
    train_tensors = _tensors(train_rows, vocabulary, scalar_scaler)
    validation_tensors = _tensors(validation_rows, vocabulary, scalar_scaler)
    test_tensors = _tensors(test_rows, vocabulary, scalar_scaler)

    train_dataset = TensorDataset(*train_tensors)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    model = StatementPairMLP(len(vocabulary), embedding_dim, hidden_dims, dropout)
    counts = Counter(row["label"] for row in train_rows)
    class_weights = torch.tensor([len(train_rows) / (3 * counts[label]) for label in LABELS], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    val_inputs = validation_tensors[:3]
    val_labels = validation_tensors[3]
    best_state, best_score, best_epoch, stale_epochs = None, -1.0, 0, 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for ids_a, ids_b, structured, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(ids_a, ids_b, structured)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val_logits = model(*val_inputs)
            val_loss = float(criterion(val_logits, val_labels))
            val_predictions = val_logits.argmax(dim=1).cpu().numpy()
            val_f1 = float(f1_score(val_labels.cpu().numpy(), val_predictions,
                                    average="macro", zero_division=0))
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)),
                        "validation_loss": val_loss, "validation_macro_f1": val_f1})
        if val_f1 > best_score or (val_f1 == best_score and (best_state is None or val_loss < best_state["validation_loss"])):
            best_score, best_epoch, stale_epochs = val_f1, epoch, 0
            best_state = {"weights": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                          "validation_loss": val_loss}
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint")
    model.load_state_dict(best_state["weights"])
    model.eval()

    with torch.no_grad():
        val_probabilities = torch.softmax(model(*val_inputs), dim=1).cpu().numpy()
        test_probabilities = torch.softmax(model(*test_tensors[:3]), dim=1).cpu().numpy()
    threshold, calibrated_validation_macro_f1 = _calibrate_threshold(
        val_probabilities, val_labels.cpu().numpy())
    test_y = test_tensors[3].cpu().numpy()
    raw_predictions = test_probabilities.argmax(axis=1)
    calibrated_predictions = raw_predictions.copy()
    calibrated_predictions[test_probabilities.max(axis=1) < threshold] = LABEL_TO_ID["UNCERTAIN"]

    # Logistic-regression baseline uses a separately fitted train-only TF-IDF vocabulary.
    baseline_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=512, sublinear_tf=True,
                                          token_pattern=r"(?u)\b\w+\b")
    train_text = [row["statement_a"] for row in train_rows] + [row["statement_b"] for row in train_rows]
    baseline_vectorizer.fit(train_text)
    def baseline_features(items: list[dict[str, Any]]) -> np.ndarray:
        a = baseline_vectorizer.transform([row["statement_a"] for row in items]).toarray()
        b = baseline_vectorizer.transform([row["statement_b"] for row in items]).toarray()
        scalar = scalar_scaler.transform(structured_pair_features(items))
        return np.concatenate([a, b, np.abs(a - b), a * b, scalar], axis=1)
    baseline = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    baseline.fit(baseline_features(train_rows), [LABEL_TO_ID[row["label"]] for row in train_rows])
    baseline_predictions = baseline.predict(baseline_features(test_rows))

    version = f"ann-v1-seed{seed}"
    artifact_dir = Path(artifact_root) / version
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), artifact_dir / "model.pt")
    joblib.dump(scalar_scaler, artifact_dir / "scalar_scaler.joblib")
    joblib.dump(baseline_vectorizer, artifact_dir / "baseline_vectorizer.joblib")
    joblib.dump(baseline, artifact_dir / "baseline_logistic_regression.joblib")
    (artifact_dir / "vocabulary.json").write_text(json.dumps(vocabulary, indent=2, sort_keys=True), encoding="utf-8")
    (artifact_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    source_hash = hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()
    report = {
        "model_version": version,
        "dataset": {"path": str(dataset_path), "rows": len(rows), "sha256": source_hash,
                    "label_status": "engineer_curated_seed_not_human_reviewed",
                    "real_world_performance_claim": False},
        "split": {name: {"rows": len(items), "groups": sorted({row["group_id"] for row in items}),
                         "label_counts": dict(Counter(row["label"] for row in items))}
                  for name, items in splits.items()},
        "training": {"seed": seed, "epochs_requested": epochs, "epochs_completed": len(history),
                     "best_epoch": best_epoch, "early_stopping_patience": patience,
                     "batch_size": batch_size, "learning_rate": learning_rate,
                     "class_weights": {label: float(class_weights[i]) for i, label in enumerate(LABELS)},
                     "elapsed_seconds": round(time.time() - started, 3),
                     "loss_history": history},
        "sentence_encoder": {"name": "task-trained-mean-pooled-word-embeddings",
                             "version": "1.0.0", "embedding_dim": embedding_dim,
                             "vocabulary_size": len(vocabulary),
                             "training_scope": "supervised_train_split_only"},
        "feature_schema": {"pair_features": ["embedding_a", "embedding_b", "absolute_difference",
                            "elementwise_product", "cosine_similarity", "numeric_date_mismatch",
                            "negation_mismatch", "token_jaccard_overlap", "statement_length_ratio"],
                           "structured_scaler_fit": "training_partition_only",
                           "embedding_dimension": embedding_dim,
                           "classifier_input_dimension": embedding_dim * 4 + 5},
        "architecture": {"type": "PyTorch feed-forward MLP", "hidden_dims": list(hidden_dims),
                          "dropout": dropout, "classes": list(LABELS)},
        "thresholds": {"low_confidence_to_uncertain": threshold,
                       "validation_macro_f1_after_threshold": round(calibrated_validation_macro_f1, 6)},
        "evaluation": {
            "test_set_status": "group-held-out curated seed set; not human-reviewed",
            "ann_raw": _metrics(test_y.tolist(), raw_predictions.tolist()),
            "ann_thresholded": _metrics(test_y.tolist(), calibrated_predictions.tolist()),
            "logistic_regression_baseline": _metrics(test_y.tolist(), baseline_predictions.tolist()),
        },
        "limitations": [
            "The labelled seed pairs are engineer-curated development examples, not independently human-reviewed domain annotations.",
            "The held-out test metrics are development diagnostics only and must not be presented as real-world enterprise accuracy.",
            "The task-trained word-embedding encoder is not a pretrained general-purpose semantic sentence encoder; it may generalize poorly to unseen vocabulary and paraphrases.",
            "A representative, independently human-reviewed enterprise test set is required before production quality claims or deployment decisions."
        ]
    }
    (artifact_dir / "model_metadata.json").write_text(json.dumps({
        "model_version": version, "vocab_size": len(vocabulary), "embedding_dim": embedding_dim,
        "hidden_dims": list(hidden_dims), "dropout": dropout, "label_to_id": LABEL_TO_ID,
        "threshold": threshold, "sentence_encoder": report["sentence_encoder"],
        "feature_schema": report["feature_schema"]
    }, indent=2), encoding="utf-8")
    (artifact_dir / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest = {}
    for path in sorted(artifact_dir.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            manifest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (artifact_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/ann/enterprise_it_statement_pairs.jsonl")
    parser.add_argument("--artifact-root", default="artifacts/ann")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-dim", type=int, default=48)
    parser.add_argument("--hidden-dims", default="128,64")
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    args = parser.parse_args()
    report = train(args.dataset, args.artifact_root, seed=args.seed,
                   embedding_dim=args.embedding_dim,
                   hidden_dims=tuple(int(x) for x in args.hidden_dims.split(",") if x.strip()),
                   dropout=args.dropout, learning_rate=args.learning_rate,
                   batch_size=args.batch_size, epochs=args.epochs, patience=args.patience)
    print(json.dumps({"model_version": report["model_version"],
                      "artifact": str(Path(args.artifact_root) / report["model_version"]),
                      "split": report["split"], "evaluation": report["evaluation"],
                      "thresholds": report["thresholds"]}, indent=2))

if __name__ == "__main__":
    main()
