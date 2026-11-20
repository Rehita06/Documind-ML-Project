"""Evidence-preserving document version comparison.

The service performs exact text alignment first, then one-to-one similarity
alignment for remaining units. Similarity can be supplied by a sentence
embedding adapter; a deterministic token-vector fallback keeps local/offline
comparisons usable and is explicitly identified in the result.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


TOKEN_RE = re.compile(r"[\w.-]+", re.UNICODE)
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")


@dataclass(frozen=True)
class TextUnit:
    text: str
    location: str
    section: str | None
    ordinal: int


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(TOKEN_RE.findall(text))


def _lexical_vector(text: str) -> Counter[str]:
    # Token and adjacent-token features; deliberately deterministic and local.
    tokens = TOKEN_RE.findall(unicodedata.normalize("NFKC", text).casefold())
    features = Counter(tokens)
    features.update(f"{a}::{b}" for a, b in zip(tokens, tokens[1:]))
    return features


def _cosine_sparse(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(value * b.get(key, 0) for key, value in a.items())
    norm_a = math.sqrt(sum(value * value for value in a.values()))
    norm_b = math.sqrt(sum(value * value for value in b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _read_units(path: Path) -> list[TextUnit]:
    suffix = path.suffix.lower()
    raw: list[tuple[str, str, str | None]] = []
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8-sig")
        section = None
        for line_no, block in enumerate(re.split(r"\n+", text), start=1):
            block = block.strip()
            if not block:
                continue
            heading = HEADING_RE.match(block)
            if heading:
                section = heading.group(1).strip()
                continue
            raw.append((block, f"line-block:{line_no}", section))
    elif suffix == ".docx":
        from docx import Document as DocxDocument
        doc = DocxDocument(str(path))
        section = None
        for index, paragraph in enumerate(doc.paragraphs, start=1):
            value = paragraph.text.strip()
            if not value:
                continue
            style = (paragraph.style.name or "") if paragraph.style else ""
            if style.lower().startswith("heading"):
                section = value
                continue
            raw.append((value, f"paragraph:{index}", section))
    elif suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        for page_no, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for block_no, block in enumerate(re.split(r"\n+", text), start=1):
                value = block.strip()
                if value:
                    raw.append((value, f"page:{page_no}/block:{block_no}", None))
    else:
        raise ValueError(f"Unsupported document type for comparison: {suffix}")

    units: list[TextUnit] = []
    for text, location, section in raw:
        # A paragraph is a useful, source-locatable comparison unit. Avoid
        # merging across paragraphs because that would blur change provenance.
        units.append(TextUnit(text=text, location=location, section=section,
                              ordinal=len(units)))
    return units


def _embedding_similarity(
    old_texts: Sequence[str], new_texts: Sequence[str],
    embedder: Any,
) -> list[list[float]]:
    texts = list(old_texts) + list(new_texts)
    if hasattr(embedder, "embed_documents"):
        vectors = embedder.embed_documents(texts)
    elif callable(embedder):
        vectors = embedder(texts)
    else:
        raise TypeError("embedder must be callable or expose embed_documents")
    if len(vectors) != len(texts):
        raise ValueError("Embedding adapter returned an unexpected vector count")

    def cosine(a: Sequence[float], b: Sequence[float]) -> float:
        if len(a) != len(b):
            raise ValueError("Embedding vectors have inconsistent dimensions")
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    old_vectors = vectors[:len(old_texts)]
    new_vectors = vectors[len(old_texts):]
    return [[cosine(a, b) for b in new_vectors] for a in old_vectors]


def _similarity_matrix(old: Sequence[TextUnit], new: Sequence[TextUnit],
                       embedder: Any | None) -> tuple[list[list[float]], str]:
    if not old or not new:
        return [], "sentence_embeddings" if embedder is not None else "local_token_similarity_fallback"
    if embedder is not None:
        return _embedding_similarity([u.text for u in old], [u.text for u in new], embedder), "sentence_embeddings"
    old_vectors = [_lexical_vector(u.text) for u in old]
    new_vectors = [_lexical_vector(u.text) for u in new]
    return [[_cosine_sparse(a, b) for b in new_vectors] for a in old_vectors], "local_token_similarity_fallback"


def compare_text_units(
    old_units: Sequence[TextUnit], new_units: Sequence[TextUnit],
    *, embedder: Any | None = None, similarity_threshold: float | None = None,
) -> dict[str, Any]:
    """Align old/new units and return ADDED/DELETED/MODIFIED/UNCHANGED records."""
    if similarity_threshold is not None and not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1")

    old_unmatched = set(range(len(old_units)))
    new_unmatched = set(range(len(new_units)))
    records: list[dict[str, Any]] = []

    # Exact normalized-text matching is the first and highest-confidence pass.
    new_by_text: dict[str, list[int]] = defaultdict(list)
    for j, unit in enumerate(new_units):
        new_by_text[_normalize(unit.text)].append(j)
    for i, old_unit in enumerate(old_units):
        key = _normalize(old_unit.text)
        candidates = new_by_text.get(key, [])
        j = next((candidate for candidate in candidates if candidate in new_unmatched), None)
        if j is None:
            continue
        old_unmatched.remove(i)
        new_unmatched.remove(j)
        records.append(_record("UNCHANGED", old_unit, new_units[j], 1.0, "exact"))

    # Similarity alignment is one-to-one. Greedy highest-score matching avoids
    # pairing multiple old statements to the same new statement.
    old_indices = sorted(old_unmatched)
    new_indices = sorted(new_unmatched)
    matrix, alignment_method = _similarity_matrix(
        [old_units[i] for i in old_indices],
        [new_units[j] for j in new_indices], embedder,
    )
    effective_threshold = similarity_threshold if similarity_threshold is not None else (
        0.72 if alignment_method == "sentence_embeddings" else 0.45
    )
    candidates: list[tuple[float, int, int]] = []
    for oi, row in enumerate(matrix):
        for nj, score in enumerate(row):
            if score >= effective_threshold:
                candidates.append((score, oi, nj))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_old: set[int] = set()
    used_new: set[int] = set()
    for score, oi, nj in candidates:
        if oi in used_old or nj in used_new:
            continue
        i, j = old_indices[oi], new_indices[nj]
        used_old.add(oi)
        used_new.add(nj)
        old_unmatched.remove(i)
        new_unmatched.remove(j)
        records.append(_record("MODIFIED", old_units[i], new_units[j], score,
                               alignment_method))

    for i in sorted(old_unmatched):
        records.append(_record("DELETED", old_units[i], None, None, "unmatched"))
    for j in sorted(new_unmatched):
        records.append(_record("ADDED", None, new_units[j], None, "unmatched"))

    order = {"UNCHANGED": 0, "MODIFIED": 1, "DELETED": 2, "ADDED": 3}
    records.sort(key=lambda item: (
        item["new_ordinal"] if item["new_ordinal"] is not None else 10**9,
        item["old_ordinal"] if item["old_ordinal"] is not None else 10**9,
        order[item["change_type"]],
    ))
    counts = {kind: sum(r["change_type"] == kind for r in records)
              for kind in ("ADDED", "DELETED", "MODIFIED", "UNCHANGED")}
    return {"changes": records, "counts": counts,
            "alignment_method": alignment_method,
            "similarity_threshold": effective_threshold}


def _record(change_type: str, old: TextUnit | None, new: TextUnit | None,
            score: float | None, method: str) -> dict[str, Any]:
    return {
        "change_type": change_type,
        "old_text": old.text if old else None,
        "new_text": new.text if new else None,
        "old_location": old.location if old else None,
        "new_location": new.location if new else None,
        "old_section": old.section if old else None,
        "new_section": new.section if new else None,
        "old_ordinal": old.ordinal if old else None,
        "new_ordinal": new.ordinal if new else None,
        "similarity": round(score, 6) if score is not None else None,
        "alignment_method": method,
    }


def _configured_embedder() -> Any | None:
    """Use the configured OpenAI sentence embedding model when credentials exist."""
    try:
        from app.core.config import get_settings
        settings = get_settings()
        if not settings.openai_api_key:
            return None
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=settings.embedding_model,
                                api_key=settings.openai_api_key)
    except (ImportError, Exception):
        # The local fallback is intentional for offline tests and environments.
        return None


def compare_document_versions(
    registry: Any, old_version_id: str, new_version_id: str,
    *, embedder: Any | None = None, similarity_threshold: float | None = None,
) -> dict[str, Any]:
    """Compare two retained registry versions and preserve source provenance."""
    old_version = registry.get_version(old_version_id)
    new_version = registry.get_version(new_version_id)
    if old_version is None or new_version is None:
        raise ValueError("Both document versions must exist in the registry")
    if old_version["version_id"] == new_version["version_id"]:
        raise ValueError("Choose two different document versions")
    old_path, new_path = Path(old_version["stored_path"]), Path(new_version["stored_path"])
    if not old_path.is_file() or not new_path.is_file():
        raise FileNotFoundError("A retained version file is missing; no comparison was run")

    old_units, new_units = _read_units(old_path), _read_units(new_path)
    active_embedder = embedder if embedder is not None else _configured_embedder()
    result = compare_text_units(old_units, new_units, embedder=active_embedder,
                                similarity_threshold=similarity_threshold)
    result.update({
        "document_id": old_version["document_id"] if old_version["document_id"] == new_version["document_id"] else None,
        "old_version": _version_summary(old_version),
        "new_version": _version_summary(new_version),
        "same_logical_document": old_version["document_id"] == new_version["document_id"],
        "alignment_method": result["alignment_method"],
    })
    return result


def _version_summary(version: dict[str, Any]) -> dict[str, Any]:
    return {key: version.get(key) for key in (
        "document_id", "version_id", "version_number", "logical_name",
        "original_filename", "content_hash", "effective_date", "uploaded_at",
    )}
