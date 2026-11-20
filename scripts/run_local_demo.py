"""Run a reproducible offline demo of immutable versions, diff and real ANN inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.change_detection import compare_document_versions
from app.services.document_registry import DocumentRegistry
from app.services.verification import CrossDocumentVerificationService


def ingest_text(registry: DocumentRegistry, logical_name: str, filename: str, text: str) -> dict:
    content = text.encode("utf-8")
    record = registry.create_version(
        logical_name=logical_name,
        original_filename=filename,
        content=content,
        uploader="offline-demo",
    )
    if record["duplicate"]:
        return record
    chunk_id = str(uuid.uuid4())
    metadata = {"page": 1, "start_index": 0, "source": filename,
                "document_id": record["document_id"], "version_id": record["version_id"],
                "chunk_id": chunk_id}
    registry.complete_ingestion(record["version_id"], [{
        "chunk_id": chunk_id,
        "vector_id": None,
        "page_number": 1,
        "section": None,
        "start_offset": 0,
        "end_offset": len(text),
        "text_hash": hashlib.sha256(content).hexdigest(),
        "metadata": metadata,
    }])
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional JSON report output path")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/ann/ann-v1-seed42")
    args = parser.parse_args()

    old_text = (
        "# VPN Policy\n"
        "VPN access requires multifactor authentication for every employee.\n"
        "Remote access must use the approved company VPN client.\n"
    )
    new_text = (
        "# VPN Policy\n"
        "VPN access requires multifactor authentication for every employee.\n"
        "Remote access must use the approved company VPN client and a managed device certificate.\n"
    )
    second_doc = (
        "# Remote Access Standard\n"
        "All remote connections to corporate systems must use multifactor authentication.\n"
        "Remote connections must originate from a managed company device.\n"
    )

    with tempfile.TemporaryDirectory(prefix="enterprise-ai-copilot-demo-") as temp_dir:
        root = Path(temp_dir)
        registry = DocumentRegistry(root / "registry.sqlite3", root / "uploads")
        registry.initialize()
        old = ingest_text(registry, "VPN Policy", "vpn-policy-v1.txt", old_text)
        new = ingest_text(registry, "VPN Policy", "vpn-policy-v2.txt", new_text)
        other = ingest_text(registry, "Remote Access Standard", "remote-access.txt", second_doc)

        comparison = compare_document_versions(registry, old["version_id"], new["version_id"])
        verifier = CrossDocumentVerificationService(
            registry=registry,
            db_path=root / "verification.sqlite3",
            artifact_dir=args.artifact_dir,
        )
        verification = verifier.verify_versions([new["version_id"], other["version_id"]])

        report = {
            "mode": "offline_demo_no_external_provider_calls",
            "documents": [
                {"logical_name": name, "document_id": item["document_id"],
                 "version_id": item["version_id"], "version_number": item["version_number"]}
                for name, item in (("VPN Policy v1", old), ("VPN Policy v2", new),
                                   ("Remote Access Standard", other))
            ],
            "version_comparison": {
                "same_logical_document": comparison["same_logical_document"],
                "alignment_method": comparison["alignment_method"],
                "counts": comparison["counts"],
                "changes": comparison["changes"],
            },
            "cross_document_verification": {
                "run_id": verification["run_id"],
                "status": verification["status"],
                "model_version": verification["model_version"],
                "summary": verification["summary"],
                "findings": verification["findings"],
            },
            "limitations": [
                "This offline demo uses local sample text and the included trained seed ANN artifact.",
                "It does not exercise LangGraph, OpenAI, Pinecone, Tavily, or browser UI integration.",
                "ANN outputs are model predictions, not proof of factual truth or policy authority.",
            ],
        }
        serialized = json.dumps(report, indent=2, ensure_ascii=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n", encoding="utf-8")
            print(f"Demo report written to: {args.output.resolve()}")
        print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
