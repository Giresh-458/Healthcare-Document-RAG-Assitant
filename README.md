# Healthcare Document Intelligence Platform (Adaptive RAG)

An educational Retrieval-Augmented Generation (RAG) prototype for synthetic clinical documents and laboratory reports. It is not clinically validated, does not guarantee de-identification, and is not production-ready. Built with **FastAPI**, **Streamlit**, **ChromaDB**, **BM25**, and **LangChain**, featuring hybrid retrieval, adaptive query routing, clinical entity extraction, automated PHI redaction, and lightweight diagnostic metrics.

---

## Architecture and Pipeline Overview

```
Clinical PDF
    │
    ▼
[PyMuPDF / OCR Fallback]
    │
    ▼
[PHI / PII Redaction Layer] ────► Best-effort pattern redaction (not comprehensive de-identification)
    │
    ▼
[Medical Named Entity Recognition] ──► Extract Clinical Terms & Lab Markers
    │
    ▼
[Text Chunking] ────► Attach page and document metadata
    │
    ├──► Dense Embeddings (HuggingFace all-MiniLM-L6-v2) ──► ChromaDB
    └──► Sparse Lexical Index ──────────────────────────────► BM25
                                                                 │
User Query                                                       │
    │                                                            │
    ▼                                                            │
[Adaptive Query Router] ──► Classify (FACTOID / AGGREGATION / IRRELEVANT)
    │                       - FACTOID: Standard Top-K window
    │                       - AGGREGATION: 3x Recall expansion
    │                       - IRRELEVANT: Pre-retrieval guardrail block
    │
    ▼
[Hybrid Retrieval Engine] ◄──────────────────────────────────────┘
    ├── BM25 Keyword Search (Exact numerical lab matches)
    └── Dense Semantic Search (Conceptual disease/symptom matching)
            │
            ▼
    [Reciprocal Rank Fusion (RRF)] ──► Score-invariant Rank Blending
            │
            ▼
[LLM Generation + Structured Output] (Groq / OpenAI / Gemini)
    │
    ├── Schema-Validated Output (Pydantic `MedicalAnswer`)
    │   ├── Context-grounded answer attempt
    │   ├── Confidence Level (HIGH / MEDIUM / LOW)
    │   ├── Extracted Clinical Entities
    │   └── Doctor Review Advisory Flag
    │
    ▼
[Evaluation & Observability Engine]
    ├── Query-token overlap heuristic
    ├── Answer/context token overlap heuristic
    └── End-to-End Latency Logging (JSONL)
```

---

## Core Technical Capabilities

| Capability | Engineering Implementation | Design Rationale |
| :--- | :--- | :--- |
| **Hybrid Search + RRF** | Combines **BM25 Okapi** with **ChromaDB** dense embeddings via **Reciprocal Rank Fusion (RRF)**. | Lexical matching can help retrieve exact values and units alongside semantic matches; quality must be measured on a labelled set. |
| **Adaptive Query Routing** | An LLM classifies incoming questions into `FACTOID`, `AGGREGATION`, or `IRRELEVANT` (an extra model call with added latency/cost). | The route changes retrieval depth or skips retrieval. This behavior has not yet been validated against a labelled routing dataset. |
| **PHI / PII Redaction** | Regex redaction cleans SSNs, phone numbers, and email patterns prior to vector storage. | Masks a limited set of patterns. This is not HIPAA compliance and can miss identifiers; use synthetic data unless a separately validated privacy process exists. |
| **Clinical NER** | Automatically identifies medical markers (`hemoglobin`, `rbc`, `glucose`, `hba1c`, etc.) and attaches them to chunk metadata. | Enhances metadata filtering and search index granularity. |
| **Structured Output** | Powered by Pydantic schemas (`MedicalAnswer`). | Requests schema-validated structured responses; model-generated confidence is not calibrated and this is not production-ready. |
| **Live RAG Metrics** | Heuristic token-overlap indicators and latency logging. The overlap values are not validated context-precision or faithfulness metrics. | Provides basic diagnostics only. Do not use token overlap as evidence of factual correctness or clinical safety. |
| **Integrated Demo Workflow** | One-click ingestion directly parses and embeds sample laboratory reports. | Enables frictionless evaluation and verification during technical reviews. |

---

## Technology Stack

- **Backend:** FastAPI, Uvicorn, Pydantic v2
- **Frontend:** Streamlit with custom CSS theme & metrics dashboard
- **Orchestration:** LangChain, LangChain-Community, LangChain-Groq
- **Vector & Lexical Search:** ChromaDB (Local persistent), HuggingFace Embeddings (`all-MiniLM-L6-v2`), Rank-BM25
- **Document Ingestion:** PyMuPDF (fitz), Tesseract OCR fallback (pytesseract)
- **Supported Models:** Groq (`openai/gpt-oss-120b`, `llama-3.3-70b-versatile`), OpenAI (`gpt-4o-mini`), Google Gemini (`gemini-1.5-flash`)

---

## Quickstart Guide

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/Giresh-458/Healthcare-Document-RAG-Assitant.git
cd Healthcare-Document-RAG-Assitant

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and edit values as needed. Keep secrets out of Git:
```env
OPENAI_API_KEY=your_key_here
CHUNK_SIZE=800
CHUNK_OVERLAP=100
TOP_K=4
MAX_UPLOAD_BYTES=5242880
MAX_PDF_PAGES=15
APP_API_TOKEN=replace-with-a-long-random-token-for-shared-token-gating
CORS_ALLOWED_ORIGINS=http://localhost:8501,http://127.0.0.1:8501
```

### Testing

Run the fast unit/API regression suite with `pip install -r requirements-test.txt` followed by `pytest -q`. Install `requirements.txt` for the full app, including local embedding and LLM-provider integrations. GitHub Actions runs compilation and the test suite on pull requests.

### 3. Launch Application
A single command boots up both the **FastAPI backend (port 8000)** and the **Streamlit dashboard (port 8501)**:
```bash
python run.py
```
Open your browser to `http://localhost:8501`.

---

## End-to-End Evaluation Workflow

1. Select your LLM Provider (e.g., **Groq**) and paste your API key in the sidebar.
2. Click **Load Demo Report (Hemoglobin Lab)** in the sidebar.
3. Common test queries:
   - *"What is the hemoglobin level?"* -> Answers only when the record scope is unambiguous; asks for clarification when multiple patient labels exist.
   - *"List all hemoglobin levels across these records"* -> Scans all indexed chunks for candidate hemoglobin values and cites each matched source; regex extraction is heuristic and must be verified.
   - *"Write a poem about space."* -> Routed as `IRRELEVANT`, blocked by triage guardrail.

---

## Repository Structure

```
Healthcare-Document-RAG-Assitant/
├── app/
│   ├── __init__.py
│   ├── config.py             # Environment configuration & TF guardrails
│   ├── schemas.py            # Pydantic structured output models
│   ├── main.py               # FastAPI REST API endpoints
│   ├── rag.py                # Core RAG pipeline orchestrator
│   ├── hybrid_retriever.py   # BM25 + Dense RRF fusion retriever
│   ├── query_router.py       # Triage classifier (Factoid/Aggregation/Irrelevant)
│   ├── phi_redactor.py       # Best-effort PHI/PII pattern redaction
│   ├── medical_ner.py        # Clinical entity extraction
│   ├── text_processor.py     # Clean chunking with metadata tagging
│   ├── vector_store.py       # ChromaDB persistence & singleton embeddings
│   ├── pdf_processor.py      # PyMuPDF parser
│   ├── ocr.py                # Tesseract OCR fallback
│   └── metrics.py            # Context Precision & Faithfulness evaluation
├── sample_documents/
│   └── hemoglobin-report-format.pdf # Standard test document
├── tests/
│   ├── test_api.py           # FastAPI endpoint tests
│   └── test_text.py          # Processing and chunking tests
├── streamlit_app.py          # Interactive web UI
├── run.py                    # Dual-process launcher
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## Patient and document scoping (prototype)

- Every new upload receives a unique `document_id`, so two PDFs with the same filename no longer overwrite each other. The PDF viewer and document filters use that ID rather than the filename.
- Uploaders may provide a `patient_label` (for example, a synthetic label such as `Patient A`). This is user-supplied metadata only; it is **not verified identity** and does not establish permission to view a record.
- A question such as "What is my hemoglobin?" is not answered by guessing that the uploader is the patient. The API asks for a patient scope. When a single-patient question spans multiple labelled patients, it asks for clarification instead of silently combining them.
- Explicit all-record hemoglobin questions scan all chunks in the current session/document/patient scope, rather than using top-k retrieval. The scan uses text patterns, may miss table/OCR layouts, and reports candidate matches—not validated clinical extraction. Check every source report. Other broad aggregation questions still use the normal bounded RAG retrieval path and must not be represented as exhaustive.
- The app currently has session isolation and an optional shared API token, **not verified user identity or patient-level authorization**. Do not expose real patient data or treat patient labels as access control. Use synthetic data only. A production system needs real authentication, authorization policy, verified patient identity/linkage, audit logging, and tested privacy controls.
- Existing indexed chunks created before unique IDs may appear as legacy records. Re-upload those PDFs to get unique IDs and a reliable viewer record. Files previously overwritten by same-name uploads cannot be recovered by this change.

---

## API token gate

If `APP_API_TOKEN` is set, configure the same value in the API and Streamlit process environments. The UI sends it in the `X-App-Token` header. The health endpoint remains public for basic process checks. This is a shared-token demo gate, not user-level authentication. If the token is unset, the API endpoints are unauthenticated and should remain on a trusted local network.

## Disclaimer
*This system is intended for document intelligence and educational research purposes. It is not a certified medical device and does not replace professional clinical diagnosis or judgment.*


## Known limitations and security notes

- This repository is an educational prototype, not a clinical product or medical device.
- PHI redaction uses regular expressions and is incomplete. It does not guarantee anonymization or HIPAA compliance. Use synthetic data for demos and tests.
- Session IDs are isolation labels, not authentication. `APP_API_TOKEN` optionally gates API endpoints with one shared token; it is not per-user identity or authorization. Do not expose this API to untrusted users without proper authentication/authorization, rate limiting, TLS, secrets management, and deployment-specific security controls.
- Dense candidate retrieval is bounded and BM25 is cached in-process by corpus fingerprint. The selected session's chunks are still loaded for lexical search on each request; large corpora need paginated/persistent lexical indexing and benchmarking.
- The displayed token-overlap values are heuristics only. Build a labelled dataset and measure retrieval recall/precision, answer correctness, citation support, abstention behavior, and latency before making quality claims.
- Upload size and page limits are enforced by the API. Configure \`MAX_UPLOAD_BYTES\`, \`MAX_PDF_PAGES\`, and \`CORS_ALLOWED_ORIGINS\` for your deployment.
- The project has not been validated for medical use. Use synthetic documents only. Never treat generated answers, confidence labels, or overlap heuristics as clinical decisions; qualified professionals must independently verify any medical interpretation.
