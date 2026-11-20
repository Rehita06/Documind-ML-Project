"""Check local prerequisites, environment configuration and ANN artifact integrity."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULES = {
    "fastapi": "FastAPI API server",
    "uvicorn": "ASGI server",
    "pydantic_settings": "Settings loader",
    "multipart": "Multipart upload parsing",
    "torch": "PyTorch ANN inference/training",
    "sklearn": "ANN preprocessing and baseline",
    "numpy": "Numerical features",
    "joblib": "Persisted preprocessing/baseline",
    "pypdf": "PDF loading",
    "docx": "DOCX loading",
    "langgraph": "LangGraph workflow",
    "langchain_core": "LangChain core types",
    "langchain_community": "LangChain document loaders",
    "langchain_text_splitters": "Document chunking",
    "openai": "OpenAI SDK",
    "langchain_openai": "OpenAI chat/embedding adapter",
    "langchain_tavily": "Tavily search adapter",
    "langchain_pinecone": "Pinecone LangChain adapter",
    "pinecone": "Pinecone client",
}


def check_artifact(artifact_dir: Path) -> tuple[bool, str]:
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        return False, f"manifest missing: {manifest_path}"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        missing, mismatched = [], []
        for filename, expected in manifest.items():
            path = artifact_dir / filename
            if not path.is_file():
                missing.append(filename)
                continue
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != expected:
                mismatched.append(filename)
        if missing or mismatched:
            return False, f"missing={missing}; checksum_mismatch={mismatched}"
        return True, f"{len(manifest)} files match SHA-256 manifest"
    except Exception as exc:
        return False, f"manifest could not be checked ({type(exc).__name__})"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="exit nonzero when full live-app requirements/config are missing")
    args = parser.parse_args()

    print(f"Project: {ROOT}")
    missing_modules = []
    for module, description in MODULES.items():
        available = importlib.util.find_spec(module) is not None
        print(f"{'OK' if available else 'MISSING':7} {module:22} {description}")
        if not available:
            missing_modules.append(module)

    env_file = ROOT / ".env"
    print(f"{'OK' if env_file.is_file() else 'MISSING':7} .env file")
    required_env = ("OPENAI_API_KEY", "PINECONE_API_KEY", "TAVILY_API_KEY", "ADMIN_API_KEY")
    configured_values = {name: os.getenv(name, "") for name in required_env}
    # Parse .env simply for reporting presence only; values are never printed.
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() in configured_values and not configured_values[key.strip()]:
                configured_values[key.strip()] = value.strip().strip('"').strip("'")
    placeholders = {"", "change-me", "replace-with-a-long-random-secret", "your-api-key", "<your-key>"}
    absent_env = []
    for name in required_env:
        is_set = configured_values[name].strip().casefold() not in placeholders
        print(f"{'OK' if is_set else 'MISSING':7} {name} (value not displayed)")
        if not is_set:
            absent_env.append(name)

    artifact_dir = Path(os.getenv("ANN_ARTIFACT_DIR", ROOT / "artifacts/ann/ann-v1-seed42"))
    if not artifact_dir.is_absolute():
        artifact_dir = ROOT / artifact_dir
    artifact_ok, artifact_message = check_artifact(artifact_dir)
    print(f"{'OK' if artifact_ok else 'FAIL':7} ANN artifact: {artifact_message}")

    if missing_modules:
        print("\nInstall the pinned requirements with: python -m pip install -r requirements.txt")
    if absent_env:
        print("Configure missing credentials in a local .env file. Do not paste secrets into chat.")
    print("\nOffline demo needs PyTorch, scikit-learn, NumPy, joblib and the saved ANN artifact; it does not call OpenAI, Pinecone or Tavily.")
    full_ready = not missing_modules and not absent_env and artifact_ok
    print(f"\nFull live application preflight: {'READY' if full_ready else 'NOT READY'}")
    return 0 if (full_ready or not args.strict) else 1


if __name__ == "__main__":
    raise SystemExit(main())
