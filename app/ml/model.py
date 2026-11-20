"""PyTorch mean-pooled sentence encoder plus feed-forward pair classifier."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class StatementPairMLP(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 48,
                 hidden_dims: tuple[int, ...] = (128, 64), dropout: float = 0.20):
        super().__init__()
        if vocab_size < 2 or embedding_dim < 4 or not hidden_dims:
            raise ValueError("Invalid ANN architecture dimensions")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.08)
        with torch.no_grad():
            self.embedding.weight[0].zero_()
        input_dim = embedding_dim * 4 + 5  # A, B, |A-B|, A*B, cosine, four scalars
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, width), nn.ReLU(), nn.Dropout(dropout)])
            previous = width
        layers.append(nn.Linear(previous, 3))
        self.classifier = nn.Sequential(*layers)

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        mask = (token_ids != 0).unsqueeze(-1)
        vectors = self.embedding(token_ids)
        pooled = (vectors * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return F.normalize(pooled, p=2, dim=1, eps=1e-8)

    def forward(self, ids_a: torch.Tensor, ids_b: torch.Tensor,
                scaled_structured: torch.Tensor) -> torch.Tensor:
        embedding_a = self.encode(ids_a)
        embedding_b = self.encode(ids_b)
        cosine = F.cosine_similarity(embedding_a, embedding_b, dim=1).unsqueeze(1)
        pair_features = torch.cat([
            embedding_a, embedding_b,
            torch.abs(embedding_a - embedding_b),
            embedding_a * embedding_b,
            cosine, scaled_structured,
        ], dim=1)
        return self.classifier(pair_features)
