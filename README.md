# Documind – Enterprise Document Intelligence

An end-to-end **Machine Learning and Agentic RAG project** that transforms enterprise document management into an intelligent document intelligence system using **LangGraph, FastAPI, Ollama, ChromaDB, PyTorch, SQLite, HTML, CSS, and JavaScript**.

---

## 1. Business Problem

### Customer

**NovaRetail**, a fictional 3,000-employee retail company.

### Problem

The organization maintains a large collection of internal documents, including IT handbooks, VPN instructions, password policies, MFA guidelines, software installation procedures, laptop troubleshooting guides, and service-desk runbooks.

Employees and support teams face challenges when working with this information because:

- They do not always know where the relevant document is located.
- Traditional keyword search may return irrelevant results.
- Manual document comparison is time-consuming.
- Different versions of the same policy may contain conflicting information.
- Information distributed across multiple documents is difficult to verify.
- General-purpose chatbots may generate answers without sufficient supporting evidence.

### Example

An employee asks:

> **"How do I connect to the company VPN from home?"**

The answer exists in the internal company knowledge base. The system should retrieve the relevant document content and generate an answer grounded in that information.

Another example:

> **"What changed between the previous and current VPN policy?"**

The system should compare the selected document versions and identify added, deleted, modified, and unchanged content.

A further example:

> **"Do two company policies contain contradictory MFA requirements?"**

The system should extract relevant statements from the selected documents, generate statement-pair features, and use a trained ANN model to identify potentially contradictory, consistent, or uncertain relationships.

### Business Goal

Build an intelligent enterprise document assistant that:

1. Processes and indexes internal documents.
2. Retrieves relevant information using semantic search.
3. Generates context-aware answers using a locally running LLM.
4. Maintains document versions and historical records.
5. Identifies changes between document versions.
6. Detects potentially contradictory or consistent statements across documents.
7. Provides traceability through source information and audit records.
8. Supports human review of machine learning predictions.

---

## 2. Why This Is an ML Project

A Machine Learning Engineer does more than develop a model. The objective is to transform a business problem into a usable intelligent application.

This project demonstrates the integration of machine learning with software engineering and application development.

```text
Customer Problem
      |
      v
Problem Identification
      |
      v
Requirements Analysis
      |
      v
Solution Architecture
      |
      v
Data Collection and Preprocessing
      |
      v
Feature Engineering
      |
      v
Machine Learning Model Development
      |
      v
Agentic RAG Integration
      |
      v
API Development
      |
      v
User Interface
      |
      v
Security and Testing
      |
      v
Application Validation
      |
      v
Observe and Improve
```

The repository demonstrates the following ML engineering components:

- Document data ingestion and preprocessing.
- Text chunking and embedding generation.
- Semantic information retrieval.
- Agent workflow orchestration.
- ANN model development and inference.
- Statement-pair feature engineering.
- Document version comparison.
- Cross-document verification.
- API and frontend integration.
- Automated testing and validation.

---

## 3. Simple Architecture

```text
                  +----------------------+
                  |    Employee / User   |
                  +----------+-----------+
                             |
                             v
                  +----------------------+
                  |   HTML / CSS / JS    |
                  |      Web UI          |
                  +----------+-----------+
                             |
                             | HTTP Requests
                             v
                  +----------------------+
                  |       FastAPI        |
                  |     Application      |
                  +----------+-----------+
                             |
              +--------------+--------------+
              |              |              |
              v              v              v
       +-------------+ +-------------+ +-------------+
       |  Agentic    | |  Document   | |  Document   |
       |  RAG        | |  Management | |  Intelligence|
       +------+------+ +------+------+ +------+------+
              |               |               |
              v               v               v
       +-------------+ +-------------+ +-------------+
       |  LangGraph  | |   SQLite    | | Change      |
       |  Workflow   | |  Registry   | | Detection   |
       +------+------+ +-------------+ +-------------+
              |
       +------+----------------+
       |                       |
       v                       v
+--------------+       +----------------+
| Ollama       |       | ChromaDB        |
| Local LLM    |       | Vector Store    |
+--------------+       +----------------+
                               |
                               v
                       +---------------+
                       | Local         |
                       | Embeddings    |
                       +---------------+

              CROSS-DOCUMENT VERIFICATION
                          |
                          v
                 +-------------------+
                 | Statement         |
                 | Extraction        |
                 +---------+---------+
                           |
                           v
                 +-------------------+
                 | Feature Engineering|
                 +---------+---------+
                           |
                           v
                 +-------------------+
                 | Trained ANN Model |
                 +---------+---------+
                           |
                           v
                 +-------------------+
                 | Verification      |
                 | Findings          |
                 +---------+---------+
                           |
                           v
                 +-------------------+
                 | Human Review      |
                 +-------------------+
```

---

## 4. Agentic RAG Workflow

The Agentic RAG component uses LangGraph to coordinate question processing, retrieval, and response generation.

### Workflow

```text
User Question
      |
      v
[1] Query Processing
      |
      v
[2] Intent Routing
      |
      +-----------------------------+
      |                             |
      v                             v
 Direct Conversation          IT Support Question
      |                             |
      v                             v
 Direct Response             [3] Document Retrieval
                                    |
                                    v
                              ChromaDB Search
                                    |
                                    v
                              Retrieved Context
                                    |
                                    v
                              Local Ollama LLM
                                    |
                                    v
                              Grounded Response
```

The workflow is designed to coordinate different operations according to the user request.

The broader LangGraph integration also supports document comparison and cross-document verification operations.

### Why Agentic?

A conventional RAG workflow generally follows:

```text
Question
   |
   v
Retrieve Documents
   |
   v
Generate Answer
```

An agentic workflow introduces state management, routing, and conditional execution:

```text
Question
   |
   v
Intent Routing
   |
   v
Select Workflow
   |
   v
Retrieve / Process Documents
   |
   v
Execute Relevant Operation
   |
   v
Generate Result
   |
   v
Return Response
```

This enables the application to coordinate multiple document intelligence operations rather than treating every user request as a simple question-answering task.

---

## 5. Main Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Programming language | Python | Application and ML development |
| Agent workflow | LangGraph | Stateful workflow orchestration |
| LLM runtime | Ollama | Local language model execution |
| LLM | Qwen 2.5 7B | Language generation and reasoning |
| Embeddings | Nomic Embed Text | Local text embeddings |
| Vector database | ChromaDB | Semantic document retrieval |
| Machine Learning | PyTorch | ANN model development |
| Feature engineering | Python / NumPy | Statement-pair feature generation |
| API | FastAPI | Backend and REST endpoints |
| Frontend | HTML / CSS / JavaScript | User-facing interface |
| Database | SQLite | Document registry and audit records |
| Testing | Pytest / unittest | Application and model testing |
| Version control | Git / GitHub | Source code management |

### Local AI Architecture

The application uses local AI components instead of depending on paid external LLM API services.

| Component | Model / Technology |
|---|---|
| Language model | `qwen2.5:7b` |
| Embedding model | `nomic-embed-text` |
| LLM runtime | Ollama |
| Vector database | ChromaDB |

The language model and embedding model are executed locally through Ollama. ChromaDB maintains the persistent vector collection used for semantic retrieval.

---

## 6. Project Structure

```text
Documind-ML-Project/
|
+-- app/
|   +-- api/
|   |   +-- routes.py
|   |
|   +-- core/
|   |   +-- config.py
|   |   +-- logging.py
|   |   +-- security.py
|   |
|   +-- ml/
|   |   +-- dataset.py
|   |   +-- features.py
|   |   +-- inference.py
|   |   +-- model.py
|   |   +-- train.py
|   |
|   +-- rag/
|   |   +-- state.py
|   |   +-- vectorstore.py
|   |   +-- workflow.py
|   |
|   +-- services/
|   |   +-- audit.py
|   |   +-- change_detection.py
|   |   +-- document_registry.py
|   |   +-- ingestion.py
|   |   +-- verification.py
|   |
|   +-- main.py
|
+-- data/
|   +-- sample_kb/
|       +-- company_it_handbook.md
|       +-- service_desk_runbook.md
|
+-- docs/
|
+-- static/
|   +-- css/
|   |   +-- style.css
|   +-- js/
|   |   +-- app.js
|   +-- favicon.svg
|
+-- templates/
|   +-- index.html
|
+-- tests/
|
+-- scripts/
|   +-- preflight.py
|   +-- run_local_demo.py
|
+-- Agentic-RAG-demo/
+-- .env.example
+-- .gitignore
+-- Dockerfile
+-- ingest_sample_kb.py
+-- requirements.txt
+-- run.py
+-- README.md
```

---

## 7. Setup

### Prerequisites

- Python 3.11 or later.
- Ollama installed.
- Git.
- Recommended RAM: 16 GB or more for comfortable local execution with the 7B model.
- Windows, macOS, or Linux environment.

### Step 1 — Clone the repository

```bash
git clone https://github.com/Rehita06/Documind-ML-Project.git
cd Documind-ML-Project
```

### Step 2 — Create virtual environment

```powershell
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Step 3 — Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 — Install and configure Ollama

Install Ollama and make sure the Ollama service is running.

Download the required models:

```powershell
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

Verify the installed models:

```powershell
ollama list
```

### Step 5 — Configure environment

Create a local `.env` file in the project root using `.env.example` as a reference.

Configure the required application settings, including a secure `ADMIN_API_KEY`.

The `.env` file must remain local and should never be committed to version control.

### Step 6 — Run the application

```powershell
uvicorn app.main:app --reload
```

Open the application:

```text
http://127.0.0.1:8000
```

FastAPI API documentation:

```text
http://127.0.0.1:8000/docs
```

Health endpoint:

```text
http://127.0.0.1:8000/api/health
```

---

## 8. Classroom Demo Scenarios

### Demo A — Private Knowledge Retrieval

Ask:

> **How do I connect to the company VPN from home?**

Expected workflow:

```text
User Question
      |
      v
Intent Routing
      |
      v
ChromaDB Retrieval
      |
      v
Relevant IT Handbook Chunks
      |
      v
Ollama Answer Generation
      |
      v
Grounded Response
```

Teaching point:

> Semantic retrieval enables the system to identify relevant enterprise information based on meaning rather than relying exclusively on exact keyword matching.

### Demo B — Company Security Policy

Ask:

> **Can IT support ask me to share my MFA code?**

Expected behavior:

- Retrieve the relevant security policy.
- Identify the supporting document content.
- Generate an answer using the retrieved context.

Teaching point:

> RAG allows a language model to answer questions using organization-specific information that is not necessarily part of its original training data.

### Demo C — Document Upload

Upload a new company policy or service-desk document.

Expected workflow:

```text
Document Upload
      |
      v
File Validation
      |
      v
Text Extraction
      |
      v
Recursive Chunking
      |
      v
Embedding Generation
      |
      v
ChromaDB Indexing
      |
      v
Available for Retrieval
```

Teaching point:

> New knowledge can be incorporated into the retrieval system without manually modifying the language model.

### Demo D — Document Version Comparison

Select two versions of the same logical document.

Expected workflow:

```text
Select Old Version
        |
        v
Select New Version
        |
        v
Text Extraction
        |
        v
Content Alignment
        |
        v
Change Classification
        |
        v
Comparison Report
```

Teaching point:

> Document version comparison helps identify changes between historical and current enterprise policies.

### Demo E — Cross-Document Verification

Select document versions belonging to different logical documents.

Expected workflow:

```text
Selected Document Versions
          |
          v
Statement Extraction
          |
          v
Candidate Pair Generation
          |
          v
Feature Engineering
          |
          v
Trained ANN Inference
          |
          v
CONTRADICTION / CONSISTENT / UNCERTAIN
          |
          v
Human Review
```

Teaching point:

> Machine Learning can assist in identifying potentially conflicting statements across multiple enterprise documents, while human review remains important for final interpretation.

---

## 9. Document Upload and Ingestion

The application provides an administrator-facing document upload workflow.

Supported file formats:

- `.pdf`
- `.txt`
- `.md`
- `.docx`

The ingestion pipeline performs:

```text
Upload
  |
  v
Validate File Type
  |
  v
Extract Text
  |
  v
Recursive Chunking
  |
  v
Generate Embeddings
  |
  v
Store Vectors in ChromaDB
  |
  v
Available for Semantic Retrieval
```

The administrator must provide the configured `ADMIN_API_KEY` when using protected document-management operations.

This workflow allows new enterprise knowledge to be added without modifying the application source code.

---

## 10. API Endpoints

### `GET /api/health`

Checks application health and availability.

### `POST /api/chat`

Processes a user question through the application workflow.

Example request:

```json
{
  "question": "How do I reset my company password?"
}
```

The response includes the generated answer and available workflow information, such as source information and execution trace.

### `POST /api/ingest`

Uploads and ingests a document.

Protected using the `X-Admin-Key` request header.

### Document Management Endpoints

The document management functionality includes:

- `GET /api/documents` — list logical documents.
- `GET /api/documents/{document_id}/versions` — list document versions.
- `POST /api/documents/{document_id}/retire` — retire a document from ordinary retrieval.

### Document Intelligence Operations

The application also exposes document comparison and cross-document verification functionality through authenticated API operations.

Refer to the API implementation and `docs/api-frontend.md` for the available request and response contracts.

---

## 11. Immutable Document Versions (Phase 2)

The document registry maintains logical documents and their immutable versions in SQLite.

Each version includes:

- UUID-based version identifier.
- Monotonically increasing version number.
- SHA-256 content hash.
- Generated storage path.
- Upload timestamp.
- Ingestion status.

Re-uploading identical file content under the same logical document name is handled through duplicate detection.

Changed uploads are staged before becoming active. The previous active version is retained if the new ingestion operation fails.

Historical document versions and their associated records are retained.

The registry defaults to a local SQLite database under the `data/` directory.

Uploaded files are stored using generated UUID-based storage paths rather than relying on the original filename as a storage identifier.

The upload limit is configurable through `MAX_UPLOAD_BYTES`.

Normal retrieval uses registry-managed active document versions while retaining support for legacy vectors where applicable.

---

## 12. Document Change Detection (Phase 3)

The change detection service compares two retained versions of a logical document.

The implementation performs exact text alignment first, followed by similarity-based alignment.

The comparison results classify content into:

| Classification | Description |
|---|---|
| ADDED | Content present in the new version |
| DELETED | Content present in the old version but absent in the new version |
| MODIFIED | Content changed between versions |
| UNCHANGED | Content retained across versions |

The comparison report maintains old and new text, section information, and source locations where available.

A text modification is not automatically treated as a contradiction.

See [`docs/change-detection.md`](docs/change-detection.md) for implementation details and limitations.

---

## 13. ANN Statement Relationship Model (Phase 4)

Phase 4 introduces a separately trained Artificial Neural Network for classifying statement relationships across enterprise policy documents.

The model predicts three classes:

- `CONTRADICTION`
- `CONSISTENT`
- `UNCERTAIN`

It does not independently determine which document is authoritative.

### Feature Engineering

The model uses statement-pair representations and engineered features, including:

- Sentence vector representations.
- Absolute difference between statement vectors.
- Element-wise vector products.
- Cosine similarity.
- Numeric or date mismatch.
- Negation mismatch.
- Token overlap.
- Sentence length ratio.

### Model Development

The project includes:

- A trainable sentence representation component.
- A Multi-Layer Perceptron classifier.
- A document-group-disjoint training, validation, and test split.
- Class-weighted loss.
- Validation-based early stopping.
- Validation-selected confidence threshold.
- TF-IDF logistic regression baseline.

Vocabulary and scalar normalization are fitted using training data only.

### Evaluation Considerations

The included curated development dataset is a seed dataset and is not independently human-reviewed.

Therefore, held-out evaluation metrics are development diagnostics rather than evidence of production-level enterprise accuracy.

A larger independently reviewed dataset is required before making real-world accuracy claims.

See [`docs/model_card.md`](docs/model_card.md) and [`docs/evaluation.md`](docs/evaluation.md).

### Train on Windows PowerShell

```powershell
$env:PYTHONPATH = (Get-Location).Path

python -m app.ml.train --dataset data/ann/enterprise_it_statement_pairs.jsonl --artifact-root artifacts/ann --seed 42 --embedding-dim 48 --hidden-dims 128,64 --dropout 0.20 --learning-rate 0.003 --batch-size 16 --epochs 120 --patience 18
```

The trained artifacts are saved under the configured local artifact directory.

Local ANN artifacts are excluded from the Git repository.

---

## 14. Cross-Document ANN Verification Service (Phase 5)

Phase 5 introduces the `CrossDocumentVerificationService` in:

`app/services/verification.py`

The service accepts explicitly selected immutable document version identifiers from at least two distinct logical documents.

It performs:

1. Source document access.
2. Statement extraction.
3. Cross-document candidate pair generation.
4. Candidate ranking.
5. ANN inference.
6. Finding generation.
7. Audit persistence.

Each finding can include:

- Source statements.
- Document and version identifiers.
- Page, section, location, and chunk references where available.
- ANN classification.
- Class probabilities.
- Confidence information.
- Model version.
- Human-review flag.

### Human Review

Human review decisions are stored separately from the original ANN prediction.

The original model classification is not overwritten by reviewer feedback.

This supports traceability and preserves the distinction between machine-generated predictions and human decisions.

### Limitations

Candidate ranking is a computational optimization and is not itself a contradiction classifier.

Verification results should be treated as decision-support information rather than definitive policy judgments.

See [`docs/cross-document-verification.md`](docs/cross-document-verification.md).

---

## 15. Integrated LangGraph Workflows (Phase 6)

Phase 6 integrates the document intelligence operations into the LangGraph workflow.

The supported intent categories include:

| Intent | Operation | Purpose |
|---|---|---|
| `kb` | Document retrieval and answer generation | Enterprise question answering |
| `direct` | Direct response | Greetings and casual conversation |
| `verify_documents` | Cross-document verification | ANN-based statement relationship analysis |
| `compare_versions` | Document change detection | Version comparison |

Selected document version identifiers are passed explicitly to the relevant operations.

The workflow does not rely on the LLM to invent document identifiers.

This integration allows the application to coordinate conversational and document intelligence operations through a common workflow architecture.

See [`docs/langgraph-integration.md`](docs/langgraph-integration.md).

---

## 16. Document Intelligence API and Frontend (Phase 7)

Phase 7 exposes document comparison and cross-document verification through authenticated FastAPI endpoints and the Document Intelligence browser workspace.

The interface supports:

- Document listing.
- Document version selection.
- Version comparison.
- Cross-document ANN verification.
- Inspection of verification findings.
- Viewing statement-level information and probabilities.
- Recording human review decisions.

The browser interface provides an integrated way for administrators to access document intelligence operations without directly invoking Python services.

Protected operations use the configured `X-Admin-Key`.

See [`docs/api-frontend.md`](docs/api-frontend.md) for endpoint contracts, authentication, and implementation details.

---

## 17. End-to-End Validation, Security Review, and Reproducible Demo (Phase 8)

Phase 8 focuses on end-to-end validation, security improvements, and reproducible local demonstration.

The project includes automated tests for:

- Document ingestion.
- Immutable version creation.
- Duplicate detection.
- Version listing.
- Document change detection.
- ANN inference.
- Verification report retrieval.
- Human review.
- API integration.
- Vector-store safety.
- Workflow integration.

### Security Improvements

The application includes:

- Environment-based secret configuration.
- Administrator-key-protected operations.
- File validation.
- Upload-size restrictions.
- Security response headers.
- Content Security Policy.
- Audit logging.
- Separate human review records.

### Local Validation

Run the automated test suite:

```powershell
python -m unittest discover -s tests -v
```

Validate JavaScript syntax:

```powershell
node --check static/js/app.js
```

Compile Python modules:

```powershell
python -m compileall -q app tests scripts
```

The application has also been manually tested for the main document intelligence workflows, including document upload, question answering, source retrieval, document comparison, and ANN verification.

### Validation Boundary

The ANN development dataset is curated and not independently human-reviewed. Its evaluation results should not be presented as production performance claims.

Local execution depends on the required Ollama models, Python dependencies, and local application configuration.

See [`docs/reproducible-demo.md`](docs/reproducible-demo.md) and [`docs/security-review.md`](docs/security-review.md).

---

## 18. Security and Configuration

The application uses local configuration and environment variables to manage sensitive settings.

Important practices:

- Never commit `.env` to version control.
- Use a strong, non-default `ADMIN_API_KEY`.
- Keep local database files outside the public source repository.
- Keep local vector database data private.
- Keep ANN model artifacts and local runtime files separate from source code.
- Restrict administrative operations to authorized users.

The application is designed for local execution and demonstration. Production deployment would require additional authentication, authorization, infrastructure, and security review.

---

## 19. Limitations and Future Enhancements

### Current Limitations

- Language model response quality depends on model capability and available hardware.
- Semantic retrieval quality depends on document quality, chunking, and embedding performance.
- ANN classification depends on the quality and diversity of the training data.
- The seed dataset is not independently human-reviewed.
- ANN predictions require human review before operational decisions.
- Local model execution requires sufficient system resources.
- The current setup is designed for local application execution rather than managed multi-user cloud deployment.

### Future Enhancements

1. Expand the enterprise document dataset.
2. Introduce independently human-reviewed statement-pair datasets.
3. Improve ANN model performance and evaluation.
4. Enhance source citations and document-location references.
5. Extend document format support.
6. Implement role-based access control.
7. Introduce document-level permissions.
8. Improve retrieval evaluation and monitoring.
9. Develop advanced document intelligence analytics.
10. Explore scalable deployment infrastructure.

---

## 20. Conclusion

Documind demonstrates the practical integration of Machine Learning, Agentic Retrieval-Augmented Generation, semantic search, and enterprise document intelligence.

The project extends beyond conventional question-answering systems by combining document ingestion, local language models, semantic retrieval, immutable document versioning, change detection, and ANN-based cross-document verification.

It demonstrates the application of ML engineering principles across data preprocessing, feature engineering, model development, workflow orchestration, API integration, frontend development, testing, and security.

The resulting application provides a foundation for intelligent enterprise document management and evidence-assisted decision support.

---

## 21. Project Repository

**Project Name:** Documind – Enterprise Document Intelligence

**Project Domain:** Machine Learning / Artificial Intelligence / Natural Language Processing

**Core Concepts:**

- Agentic Retrieval-Augmented Generation
- Local Large Language Models
- Semantic Search
- Vector Databases
- Machine Learning
- Artificial Neural Networks
- Feature Engineering
- Document Version Management
- Document Change Detection
- Cross-Document Verification

**GitHub Repository:**

https://github.com/Rehita06/Documind-ML-Project
