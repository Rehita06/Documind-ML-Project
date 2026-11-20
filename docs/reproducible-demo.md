# Reproducible local demo (Phase 8)

The offline demo exercises the actual local document registry, version
comparison engine and saved PyTorch ANN artifact. It does not call OpenAI,
Pinecone or Tavily and does not require `.env` credentials.

## Requirements

- Python environment with the project's requirements installed, or at least
  `torch`, `numpy`, `scikit-learn`, `joblib` and the included ANN artifact.
- Run commands from the repository root.

## Commands

Windows PowerShell:

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH = (Get-Location).Path
python scripts/preflight.py
python scripts/run_local_demo.py --output demo-output/phase8-report.json
python -m unittest discover -s tests -v
node --check static/js/app.js
python -m compileall -q app tests scripts
```

The demo creates a temporary SQLite registry and temporary upload directory,
registers two immutable versions of a VPN policy plus a second logical
enterprise document, compares the two VPN versions, and runs the real saved ANN
against statements from the two different documents. It prints the version
change counts, model version, confidence/probabilities and source references.
The optional JSON output contains the complete local demo report. Temporary
SQLite data and uploaded demo files are removed automatically when the process
exits.

## Live application setup

1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`, `PINECONE_API_KEY`,
   `TAVILY_API_KEY` and a long random `ADMIN_API_KEY` locally. Do not commit `.env`.
2. Confirm `PINECONE_INDEX_NAME`, `PINECONE_NAMESPACE`, embedding model and index
   dimension are correct for the existing Pinecone index. A dimension mismatch
   fails safely; the application does not delete or recreate the existing index.
3. Run `python scripts/preflight.py --strict`. It exits nonzero until all
   configured dependencies, credentials and ANN artifact checks pass.
4. Run `python -m unittest discover -s tests -v` and `uvicorn app.main:app --reload`.
5. Open `http://127.0.0.1:8000`, upload sample IT policy documents using the
   admin key, then exercise chat, version comparison, cross-document
   verification and review.

The included seed ANN dataset is engineer-curated and not human-reviewed. Its
metrics are development diagnostics, not real-world enterprise performance.
The offline demo's model predictions are not policy authority. The live app
requires successful credentialed integration tests against the configured
providers and a human-reviewed evaluation dataset before production use.
