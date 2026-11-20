"""Dataset validation and document-group-aware splitting for ANN training."""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

LABELS = ("CONTRADICTION", "CONSISTENT", "UNCERTAIN")
REQUIRED_FIELDS = ("statement_a", "statement_b", "label", "source_doc_a", "source_doc_b", "group_id")


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_number}: {exc.msg}") from exc
            missing = [field for field in REQUIRED_FIELDS if not str(row.get(field, "")).strip()]
            if missing:
                raise ValueError(f"Line {line_number} is missing required fields: {', '.join(missing)}")
            label = str(row["label"]).upper()
            if label not in LABELS:
                raise ValueError(f"Line {line_number} has unsupported label {label!r}")
            if row["source_doc_a"] == row["source_doc_b"]:
                raise ValueError(f"Line {line_number} must reference two different source documents")
            row["label"] = label
            rows.append(row)
    if not rows:
        raise ValueError("ANN dataset is empty")
    return rows


def split_by_group(rows: list[dict[str, Any]], seed: int = 42,
                   train_fraction: float = 0.70, validation_fraction: float = 0.15
                   ) -> dict[str, list[dict[str, Any]]]:
    """Assign whole source-document groups to one split to reduce leakage."""
    groups = sorted({str(row["group_id"]) for row in rows})
    if len(groups) < 3:
        raise ValueError("At least three distinct document groups are required")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("Invalid split fractions")
    random.Random(seed).shuffle(groups)
    train_end = max(1, int(len(groups) * train_fraction))
    validation_end = max(train_end + 1, int(len(groups) * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, len(groups) - 1)
    assignment = {
        "train": set(groups[:train_end]),
        "validation": set(groups[train_end:validation_end]),
        "test": set(groups[validation_end:]),
    }
    result = {name: [row for row in rows if str(row["group_id"]) in selected]
              for name, selected in assignment.items()}
    if any(not split for split in result.values()):
        raise ValueError("Group split produced an empty train, validation, or test partition")
    if set().union(*(set(row["group_id"] for row in split) for split in result.values())) != set(groups):
        raise AssertionError("A document group was lost during splitting")
    if any(set(row["group_id"] for row in result[a]) & set(row["group_id"] for row in result[b])
           for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise AssertionError("Document group leakage detected")
    for name, split in result.items():
        counts = Counter(row["label"] for row in split)
        if set(counts) != set(LABELS):
            raise ValueError(f"{name} split must contain all labels; got {dict(counts)}")
    return result
