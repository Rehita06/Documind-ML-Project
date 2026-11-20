"""Load and run the separately trained ANN without touching Pinecone embeddings."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from app.ml.features import encode_texts, structured_pair_features
from app.ml.model import StatementPairMLP


class StatementPairPredictor:
    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = Path(artifact_dir)
        self.metadata = json.loads((self.artifact_dir / "model_metadata.json").read_text(encoding="utf-8"))
        self.vocabulary = json.loads((self.artifact_dir / "vocabulary.json").read_text(encoding="utf-8"))
        self.scaler = joblib.load(self.artifact_dir / "scalar_scaler.joblib")
        self.label_to_id = self.metadata["label_to_id"]
        self.id_to_label = {int(index): label for label, index in self.label_to_id.items()}
        self.model = StatementPairMLP(
            vocab_size=int(self.metadata["vocab_size"]),
            embedding_dim=int(self.metadata["embedding_dim"]),
            hidden_dims=tuple(self.metadata["hidden_dims"]),
            dropout=float(self.metadata["dropout"]),
        )
        state = torch.load(self.artifact_dir / "model.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, statement_a: str, statement_b: str, *,
                confidence_threshold: float | None = None) -> dict[str, Any]:
        if not statement_a.strip() or not statement_b.strip():
            raise ValueError("Both statements must be non-empty")
        row = {"statement_a": statement_a, "statement_b": statement_b}
        ids_a, _ = encode_texts([statement_a], self.vocabulary)
        ids_b, _ = encode_texts([statement_b], self.vocabulary)
        structured = self.scaler.transform(structured_pair_features([row])).astype(np.float32)
        with torch.no_grad():
            logits = self.model(torch.tensor(ids_a, dtype=torch.long),
                                torch.tensor(ids_b, dtype=torch.long),
                                torch.tensor(structured, dtype=torch.float32))
            probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        confidence = float(probabilities.max())
        predicted_id = int(probabilities.argmax())
        threshold = float(self.metadata["threshold"] if confidence_threshold is None else confidence_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        low_confidence = confidence < threshold
        label = "UNCERTAIN" if low_confidence else self.id_to_label[predicted_id]
        return {
            "predicted_label": label,
            "model_label": self.id_to_label[predicted_id],
            "probabilities": {self.id_to_label[index]: round(float(probabilities[index]), 6)
                              for index in range(len(probabilities))},
            "confidence": round(confidence, 6),
            "model_version": self.metadata["model_version"],
            "review_required": low_confidence or label == "UNCERTAIN",
            "threshold": threshold,
            "sentence_encoder": self.metadata["sentence_encoder"],
            "warning": "This is a model prediction about statement relationship, not proof of factual truth or policy authority.",
        }


def load_predictor(artifact_dir: str | Path) -> StatementPairPredictor:
    return StatementPairPredictor(artifact_dir)
