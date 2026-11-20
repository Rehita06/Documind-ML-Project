# Enterprise IT Support Agentic RAG Copilot

An end-to-end **Forward Deployed Engineer (FDE) project** that turns a notebook-style Agentic RAG workflow into a deployable internal product using **LangGraph, FastAPI, Pinecone, Groq, Tavily, HTML, CSS, and JavaScript**.

---

## 1. Business Problem

### Customer
**NovaRetail**, a fictional 3,000-employee retail company.

### Problem
The internal IT team maintains many documents: VPN instructions, password rules, MFA policy, software installation rules, laptop troubleshooting guides, and service-desk runbooks.

Employees still create repetitive support tickets because:

- They do not know where the correct document is.
- Traditional keyword search returns too many results.
- A normal chatbot may hallucinate an answer.
- Internal documents can be incomplete or outdated.
- Some questions require current vendor information from the public web.

### Example
An employee asks:

> **“How do I connect to the company VPN from home?”**

The answer exists in the private company KB, so the system should **not search the public internet**.

Another employee asks:

> **“What is the latest Microsoft Teams outage guidance?”**

The internal KB may not contain current outage information. The system should recognize weak private evidence, use an external search, grade that evidence, and answer with an external-source warning.

### Business Goal
Build a secure IT Support Copilot that:

1. Searches trusted private knowledge first.
2. Checks whether retrieved evidence is good enough.
3. Uses web search only when private knowledge is insufficient.
4. Rewrites weak queries and retries.
5. Generates grounded answers.
6. Shows the decision path for transparency and debugging.
7. Lets authorized staff add new company documents.

---

## 2. Why This Is an FDE Project

A Forward Deployed Engineer does more than create an LLM notebook. The FDE must turn the customer's problem into a usable product.

```text
Customer Problem
      ↓
Discovery & Requirements
      ↓
Solution Architecture
      ↓
Data / Knowledge Integration
      ↓
Agentic RAG Development
      ↓
API Development
      ↓
User Interface
      ↓
Security + Audit + Testing
      ↓
Deployment
      ↓
Observe + Improve
```

This repository demonstrates each layer.

---

## 3. Simple Architecture

```text
                   ┌─────────────────────┐
                   │   Employee / User   │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ HTML/CSS/JavaScript │
                   │      Web UI         │
                   └──────────┬──────────┘
                              │ POST /api/chat
                              ▼
                   ┌─────────────────────┐
                   │       FastAPI       │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │     LangGraph       │
                   │ Agentic RAG Control │
                   └──────────┬──────────┘
                              │
               ┌──────────────┴──────────────┐
               │                             │
               ▼                             ▼
       ┌───────────────┐              ┌──────────────┐
       │ Private KB    │              │ Tavily Web   │
       │ Pinecone      │              │ Search       │
       └───────┬───────┘              └──────┬───────┘
               │                             │
               └──────────────┬──────────────┘
                              ▼
                     ┌────────────────┐
                     │ Groq LLM       │
                     │ Grounded Answer│
                     └────────────────┘
```

---

## 4. Agentic RAG Workflow

This follows the same core implementation pattern as the reference notebook.

```text
Question
   ↓
[1] Route Question
   ├── Greeting / simple chat ─────────────→ Direct Answer
   │
   └── IT support question
                ↓
[2] Retrieve from Private Pinecone KB
                ↓
[3] Grade Private Evidence
       ┌────────┴────────┐
       │                 │
     GOOD               WEAK
       │                 │
       ▼                 ▼
Generate from KB    [4] Tavily Web Search
                         ↓
                  [5] Grade Web Evidence
                    ┌────┴─────┐
                    │          │
                  GOOD        WEAK
                    │          │
                    ▼          ▼
              Generate Web  [6] Rewrite Query
                               ↓
                         Retry Private KB
                               ↓
                        Max retry reached?
                               ↓
                    Insufficient Evidence
```

### Why Agentic?

Normal RAG does approximately this:

```text
Question → Retrieve → Generate
```

This project makes decisions:

```text
Question
 → Route
 → Retrieve
 → Evaluate evidence
 → Choose KB or Web
 → Rewrite if needed
 → Retry
 → Generate grounded answer
```

The system therefore controls **what to do next** based on its current state.

---

## 5. Main Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Agent workflow | LangGraph | Stateful routing and conditional decisions |
| LLM | Groq | Routing, grading, rewriting, answer generation |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` | Local 384-dimensional embeddings |
| Vector DB | Pinecone | Private enterprise knowledge base |
| External search | Tavily | Fallback when company KB is insufficient |
| API | FastAPI | Backend and REST endpoints |
| Frontend | HTML/CSS/JavaScript | Employee-facing interface |
| Audit | SQLite | Basic decision-path logging |
| Packaging | Docker | Reproducible deployment |

---

## 6. Project Structure

```text
FDE_Agentic_RAG_IT_Copilot/
│
├── app/
│   ├── api/
│   │   └── routes.py              # Chat, health and ingestion APIs
│   │
│   ├── core/
│   │   ├── config.py              # Environment configuration
│   │   └── logging.py             # Logging configuration
│   │
│   ├── rag/
│   │   ├── state.py               # LangGraph state + structured decisions
│   │   ├── vectorstore.py         # Pinecone + embeddings
│   │   └── workflow.py            # Complete Agentic RAG graph
│   │
│   ├── services/
│   │   ├── audit.py               # SQLite query audit
│   │   └── ingestion.py           # PDF/TXT/MD/DOCX loading + chunking
│   │
│   └── main.py                    # FastAPI application
│
├── data/
│   └── sample_kb/
│       ├── company_it_handbook.md
│       └── service_desk_runbook.md
│
├── static/
│   ├── css/style.css
│   └── js/app.js
│
├── templates/
│   └── index.html
│
├── tests/
│   └── test_ingestion.py
│
├── uploads/
├── .env.example
├── Dockerfile
├── ingest_sample_kb.py
├── requirements.txt
├── run.py
└── README.md
```

---

## 7. Setup

### Step 1 — Create virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Configure environment

Create a `.env` file in the project root directory with the following variables:

```env
# LLM API Keys
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# External Search
TAVILY_API_KEY=your_tavily_api_key_here

# Vector Database (Pinecone)
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=fde-it-support-rag
PINECONE_NAMESPACE=company-it-kb

# LLM Models
GROQ_MODEL=openai/gpt-oss-20b
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# Security
ADMIN_API_KEY=change-me-in-production

# Application Settings
APP_ENV=development
```

#### Environment Variable Reference

| Variable | Description | Example | Required |
|---|---|---|---|
| `GROQ_API_KEY` | API key for Groq LLM (routing, grading, generation) | `gsk_...` | ✅ Yes |
| `OPENAI_API_KEY` | API key for OpenAI (embeddings and fallback LLM) | `sk-...` | ✅ Yes |
| `TAVILY_API_KEY` | API key for Tavily web search | `tvly-...` | ✅ Yes |
| `PINECONE_API_KEY` | API key for Pinecone vector database | `pckey-...` | ✅ Yes |
| `PINECONE_INDEX_NAME` | Pinecone index name | `fde-it-support-rag` | ⚠️ Optional (default: `fde-it-support-rag`) |
| `PINECONE_NAMESPACE` | Pinecone namespace for document isolation | `company-it-kb` | ⚠️ Optional (default: `company-it-kb`) |
| `GROQ_MODEL` | Groq model identifier | `openai/gpt-oss-20b` | ⚠️ Optional (default: `openai/gpt-oss-20b`) |
| `OPENAI_MODEL` | OpenAI model identifier | `gpt-4o-mini` | ⚠️ Optional (default: `gpt-4o-mini`) |
| `EMBEDDING_MODEL` | Embedding model for vectorization | `text-embedding-3-small` | ⚠️ Optional (default: `text-embedding-3-small`) |
| `ADMIN_API_KEY` | Secret key for admin endpoints (document ingestion) | `your-secure-key` | ⚠️ Optional (default: `change-me`) |
| `APP_ENV` | Application environment | `development` or `production` | ⚠️ Optional (default: `development`) |

**⚠️ Important Security Notes:**

- Never commit `.env` to version control. Add it to `.gitignore`.
- Change `ADMIN_API_KEY` to a secure random value in production.
- Use environment variables instead of hardcoding secrets.
- Restrict API key access to only the services that need them.

### Step 4 — Load sample company knowledge

```bash
python ingest_sample_kb.py
```

This demonstrates the RAG ingestion pipeline:

```text
Company Documents
   ↓
Load Documents
   ↓
Chunk Text
   ↓
HuggingFace Embeddings
   ↓
Pinecone Vector Database
```

### Step 5 — Run the application

```bash
python run.py
```

Open:

```text
http://127.0.0.1:8000
```

FastAPI API docs:

```text
http://127.0.0.1:8000/docs
```

---

## 8. Classroom Demo Scenarios

### Demo A — Private KB Success

Ask:

> **How do I connect to the company VPN from home?**

Expected path:

```text
Router → KB
Private KB Retrieval
KB Grade → GOOD
Generate from Private KB
```

Teaching point:

> Trusted internal company knowledge is preferred. No public web call is necessary.

---

### Demo B — Company Policy Question

Ask:

> **Can IT support ask me to share my MFA code?**

The internal handbook says employees must never share passwords or MFA codes with support personnel.

Expected path:

```text
Router → KB
Private KB Retrieval
KB Grade → GOOD
Private KB Answer
```

Teaching point:

> RAG allows the model to answer using company-specific information it was never trained on.

---

### Demo C — External / Current Information

Ask:

> **What is the latest Microsoft Teams outage guidance?**

Expected path when internal documents do not answer it:

```text
Router → KB
Private KB Retrieval
KB Grade → WEAK
Tavily Search
Web Grade → GOOD
Web Answer
```

Teaching point:

> Agentic RAG can choose another information source rather than blindly answering from irrelevant vectors.

---

### Demo D — Query Rewrite

Ask a deliberately vague question such as:

> **My work communication app is acting strange after the new update. What should I do?**

If both KB and first external retrieval are weak, the workflow can rewrite the query and retry.

Teaching point:

> Retrieval failure does not immediately mean failure. An agent can improve the search query and try again while using a retry guard to avoid loops.

---

### Demo E — Direct Conversation

Ask:

> **Hello!**

Expected path:

```text
Router → DIRECT
Direct Answer
```

Teaching point:

> Not every message should trigger expensive vector search or web search.

---

## 9. Document Upload Demo

The sidebar contains **Add Company Document**.

Use the `ADMIN_API_KEY` from `.env`.

Upload one of:

- `.pdf`
- `.txt`
- `.md`
- `.docx`

The backend performs:

```text
Upload
  ↓
Validate Type
  ↓
Load Text
  ↓
Recursive Chunking
  ↓
Embeddings
  ↓
Pinecone Indexing
  ↓
Immediately Available for Retrieval
```

This is useful in an FDE demonstration because the customer does not want to edit Python every time a new policy is published.

---

## 10. API Endpoints

### `GET /api/health`

Health check.

### `POST /api/chat`

Request:

```json
{
  "question": "How do I reset my company password?"
}
```

Response contains:

```json
{
  "answer": "...",
  "source_used": "private_kb",
  "trace": [
    "Router → KB",
    "Private KB retrieval → 4 chunks",
    "KB evidence grade → GOOD",
    "Answer generation → PRIVATE KB"
  ],
  "citations": []
}
```

### `POST /api/ingest`

Protected by the `X-Admin-Key` request header.

Uploads a document and adds its chunks to Pinecone.

---



## 11. Immutable document versions (Phase 2)

The API now registers each uploaded file as a logical document with immutable
versions in SQLite. Each version has a UUID, monotonically increasing version
number, SHA-256 hash, generated storage path, upload timestamp and ingestion
status. Re-uploading identical bytes under the same logical name returns the
existing version instead of indexing a duplicate.

A changed upload is staged as `pending`; it becomes active only after chunk
embedding/indexing and registry metadata persistence succeed. If ingestion
fails, the prior active version remains active. Historical files and vector
records are retained.

The registry defaults to `data/document_registry.db`. New uploads are stored
under UUID-generated subdirectories in `uploads/`; the original filename is
display metadata only. The default upload limit is 25 MiB and can be configured
with `MAX_UPLOAD_BYTES`. The API checks supported extensions and basic PDF,
DOCX and UTF-8 text content signatures.

Admin endpoints:

- `POST /api/ingest` — upload; optional multipart `logical_name` and
  `effective_date` fields.
- `GET /api/documents` — list logical documents.
- `GET /api/documents/{document_id}/versions` — list immutable versions.
- `POST /api/documents/{document_id}/retire` — soft-retire from ordinary
  retrieval; historical files, versions and vectors are retained.

These endpoints use the `X-Admin-Key` header. Set a non-default
`ADMIN_API_KEY` in `.env`. A placeholder-only `.env.example` is provided.

Normal RAG retrieval filters registry-managed vectors to active version IDs
while retaining legacy vectors that predate version metadata. Version
comparison and ANN verification are later phases and are not implemented here.

## 12. Document change detection (Phase 3)

A comparison service is available for two retained version IDs. It runs exact
text alignment first, then similarity alignment (configured sentence embeddings
when available, otherwise a clearly labelled local lexical fallback with a lower
method-specific default threshold). Reports
classify units as `ADDED`, `DELETED`, `MODIFIED` or `UNCHANGED` and retain old/new
text, section labels and source locations. A text modification is not itself
reported as a contradiction.

See [docs/change-detection.md](docs/change-detection.md) for the service API,
threshold behaviour, fixture details and current limitations. The HTTP compare endpoint and frontend were added in Phase 7; the Phase 3
change-detection engine remains a separately tested Python service.

## 13. ANN statement relationship model (Phase 4)

Phase 4 adds a separately trained PyTorch MLP for classifying statement pairs
from different IT policy documents as `CONTRADICTION`, `CONSISTENT`, or
`UNCERTAIN`. It does not replace the OpenAI embeddings used by Pinecone and
makes no decision about which policy is authoritative.

The task-specific sentence encoder is a trainable, mean-pooled word-embedding
model jointly optimized with the MLP. Pair features include both sentence
vectors, absolute difference, element-wise product, cosine similarity, numeric
or date mismatch, negation mismatch, token overlap and length ratio. Vocabulary
and scalar normalization are fitted on training data only. Training uses a
seeded, document-group-disjoint train/validation/test split, class-weighted loss,
validation early stopping and validation-selected low-confidence threshold.
A TF-IDF logistic-regression baseline is trained and evaluated alongside it.

The 90-row dataset at `data/ann/enterprise_it_statement_pairs.jsonl` is a
curated development seed set, **not independently human-reviewed**. Its held-out
test metrics are diagnostics only and are not evidence of real-world enterprise
accuracy. A human-reviewed domain test set remains required before production
claims or deployment decisions. See [docs/model_card.md](docs/model_card.md) and
[docs/evaluation.md](docs/evaluation.md).

### Train on Windows PowerShell

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH = (Get-Location).Path
python -m app.ml.train --dataset data/ann/enterprise_it_statement_pairs.jsonl --artifact-root artifacts/ann --seed 42 --embedding-dim 48 --hidden-dims 128,64 --dropout 0.20 --learning-rate 0.003 --batch-size 16 --epochs 120 --patience 18
```

Artifacts are saved to a versioned directory under `artifacts/ann/`, which is
ignored by Git. See `docs/evaluation.md` for the inference smoke test and test
commands. Cross-document verification is implemented as a tested Python service in Phase 5. LangGraph routing was added in Phase 6 and user-facing API/UI integration in Phase 7.


## 14. Cross-document ANN verification service (Phase 5)

Phase 5 adds `CrossDocumentVerificationService` in
`app/services/verification.py`. It accepts explicitly selected immutable
version IDs from at least two distinct logical documents, extracts
source-locatable statements from the retained TXT/Markdown, DOCX or PDF files,
ranks cross-document candidate pairs, and calls the trained ANN artifact.
Each finding includes both source statements and document/version/page/section/
location/chunk references where available, the ANN label, raw model label,
per-class probabilities, confidence, model version and review flag.

Runs and findings are persisted in the existing audit SQLite database in
`verification_runs`, `statement_pairs` and `verification_audit`. Human review is
append-only relative to the ANN result: `review_pair` stores a separate reviewer
label, actor, notes and timestamp and never overwrites the model label. The service method itself does not authenticate reviewers; Phase 7 exposes it
only through the admin-key-protected API layer. Reviewer identity remains a
supplied audit label rather than federated identity proof.

Candidate pairs are ranked by lexical Jaccard overlap before ANN inference. This
ranking is only a compute-saving heuristic, not a contradiction classifier.
Runs are explicitly marked `partial` when pair or statement limits truncate
the selection. The current Phase 5 implementation reads selected immutable
source files directly; it does not yet perform Pinecone similarity retrieval,
LLM-based statement extraction, LangGraph routing or HTTP API/UI integration.
See [docs/cross-document-verification.md](docs/cross-document-verification.md)
for limits, Python usage and configuration.

### Windows PowerShell smoke test

After installing dependencies and configuring `.env`, set the Python path and
run the tests. To invoke verification, pass two or more `version_id` values
returned by the admin document-version listing to the Python service shown in
`docs/cross-document-verification.md`. The service loads the saved ANN from
`ANN_ARTIFACT_DIR` by default.

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -m unittest discover -s tests -v
```

---

## Phase 6 — Integrated LangGraph workflows

The existing RAG graph now has four intent routes while preserving its original
private-KB-first behaviour:

| Intent | Graph path | Result |
|---|---|---|
| `kb` | Private retrieval → evidence grade → grounded answer or Tavily fallback | Existing IT-support chat |
| `direct` | Direct response | Greeting/casual chat |
| `verify_documents` | Cross-document verification service → trained ANN | Evidence-backed statement-pair findings |
| `compare_versions` | Version change-detection service | Added/deleted/modified/unchanged report |

For UI/API-driven operations, selected version IDs are passed explicitly to
`ask()`; the LLM router does not invent document identifiers. If routing fails,
the workflow safely defaults to the original private-KB route. See
[`docs/langgraph-integration.md`](docs/langgraph-integration.md) for invocation,
trace behaviour, limitations and examples.

---

## Phase 7 — Document intelligence API and frontend

Phase 7 exposes document comparison and ANN verification through authenticated
FastAPI endpoints and the **Document Intelligence** browser workspace. The API
supports listing documents and versions, comparing two versions of the same
logical document, starting cross-document verification, retrieving persisted
reports, and recording separate human review decisions. All these endpoints
require a configured non-default `X-Admin-Key`.

The browser workspace lets an administrator select versions, compare revision
changes, run cross-document ANN verification, inspect source statements and
probabilities, and save reviewer labels/notes. Reviewer decisions remain
separate from the original ANN prediction. See
[`docs/api-frontend.md`](docs/api-frontend.md) for endpoint contracts, example
payloads, authentication and limitations.

### Run locally

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH = (Get-Location).Path
python -m unittest discover -s tests -v
uvicorn app.main:app --reload
```

Configure `.env` with a non-default `ADMIN_API_KEY` and the required provider
credentials before starting the application. API/UI route tests use local
adapters; live provider connectivity must be verified in the target deployment.

---

## Phase 8 — End-to-end validation, security review and reproducible demo

Phase 8 adds HTTP-level integration coverage for the main document lifecycle:
admin-authenticated upload, immutable version creation and duplicate detection,
version listing, actual change-detection comparison, actual saved ANN inference,
persisted report retrieval and human review. The new integration test uses
FastAPI's `TestClient`, a temporary SQLite registry and upload directory, and
the included trained ANN artifact. It does not substitute mock ANN predictions
for the real saved model.

The API now normalizes both Windows and POSIX upload path separators, strictly
validates `effective_date`, validates human-review labels and reviewer names,
and writes audit metadata for successful, duplicate and failed ingestion
attempts. A security-header middleware adds a restrictive same-origin CSP,
frame denial, MIME sniffing protection, referrer/permissions policies and
`no-store` for API responses.

### Reproducible offline demo

The offline demo exercises document versioning, actual change detection and the
real saved ANN without requiring OpenAI, Pinecone or Tavily credentials:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python scripts/preflight.py
python scripts/run_local_demo.py --output demo-output/phase8-report.json
python -m unittest discover -s tests -v
node --check static/js/app.js
python -m compileall -q app tests scripts
```

The demo uses temporary local files/SQLite and cleans them up automatically.
See [`docs/reproducible-demo.md`](docs/reproducible-demo.md) for the full flow,
expected outputs and live application setup. See
[`docs/security-review.md`](docs/security-review.md) for changes, residual
risks and deployment boundaries.

### Validation boundary

The Phase 8 HTTP integration tests exercise real FastAPI route handling, the
real document registry, real change detection and the included trained ANN
artifact. The full `app.main` startup and external OpenAI/Pinecone/Tavily
connections still require the pinned provider packages and valid local
credentials. The current environment did not contain the LangGraph,
LangChain-provider or Pinecone packages, and no live external-provider test is
claimed. The ANN's seed labels remain engineer-curated and not independently
human-reviewed; its held-out results are not production performance claims.

### Windows PowerShell live run

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env locally with valid credentials and a non-default ADMIN_API_KEY.
$env:PYTHONPATH = (Get-Location).Path
python scripts/preflight.py --strict
python -m unittest discover -s tests -v
uvicorn app.main:app --reload
```
