"""Task-trained sentence embeddings and deterministic pair features.

The sentence encoder is a mean-pooled, trainable word-embedding model. It is
trained jointly with the MLP on the labelled statement-pair task; it is not a
pretrained general-purpose language model or the Pinecone embedding model.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*|\d+(?:\.\d+)?")
NUMBER_DATE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\b|\b(?:day|days|week|weeks|month|months|year|years|hour|hours|quarterly|monthly|weekly|daily|annually)\b",
    re.IGNORECASE,
)
NEGATIONS = {"no", "not", "never", "without", "cannot", "can't", "prohibited", "disallowed", "mustn't", "don't", "doesn't", "isn't", "aren't"}
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text)]


def build_vocabulary(training_rows: list[dict[str, Any]], min_frequency: int = 1) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in training_rows:
        counts.update(tokenize(row["statement_a"]))
        counts.update(tokenize(row["statement_b"]))
    vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, count in sorted(counts.items()):
        if count >= min_frequency and token not in vocabulary:
            vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_texts(texts: list[str], vocabulary: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    sequences = [[vocabulary.get(token, vocabulary[UNK_TOKEN]) for token in tokenize(text)] or [vocabulary[UNK_TOKEN]]
                 for text in texts]
    max_length = max(len(sequence) for sequence in sequences)
    ids = np.zeros((len(sequences), max_length), dtype=np.int64)
    for index, sequence in enumerate(sequences):
        ids[index, :len(sequence)] = sequence
    return ids, np.asarray([len(sequence) for sequence in sequences], dtype=np.int64)


def structured_pair_features(rows: list[dict[str, Any]]) -> np.ndarray:
    """Four label-independent scalars: numeric/date mismatch, negation mismatch,
    token Jaccard overlap, and statement-length ratio.
    """
    output = []
    for row in rows:
        a, b = row["statement_a"], row["statement_b"]
        tokens_a, tokens_b = set(tokenize(a)), set(tokenize(b))
        numeric_a = set(value.casefold() for value in NUMBER_DATE_RE.findall(a))
        numeric_b = set(value.casefold() for value in NUMBER_DATE_RE.findall(b))
        neg_a, neg_b = tokens_a & NEGATIONS, tokens_b & NEGATIONS
        union = tokens_a | tokens_b
        overlap = len(tokens_a & tokens_b) / len(union) if union else 0.0
        length_ratio = min(len(tokens_a), len(tokens_b)) / max(len(tokens_a), len(tokens_b), 1)
        numeric_mismatch = float(bool(numeric_a or numeric_b) and numeric_a != numeric_b)
        negation_mismatch = float(neg_a != neg_b)
        output.append([numeric_mismatch, negation_mismatch, overlap, length_ratio])
    return np.asarray(output, dtype=np.float32)


def fit_scalar_scaler(training_rows: list[dict[str, Any]]) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(structured_pair_features(training_rows))
    return scaler
