# Healthcare Document Intelligence Platform (Adaptive RAG)

A production-oriented, privacy-preserving Retrieval-Augmented Generation (RAG) system engineered for clinical documents and laboratory reports. Built with **FastAPI**, **Streamlit**, **ChromaDB**, **BM25**, and **LangChain**, featuring hybrid retrieval, adaptive query routing, clinical entity extraction, automated PHI redaction, and quantitative evaluation metrics.

---

## Architecture and Pipeline Overview

```
Clinical PDF
    │
    ▼
[PyMuPDF / OCR Fallback]
    │
    ▼
[PHI / PII Redaction Layer] ────► Mask Patient Demographics (HIPAA Alignment)
    │
    ▼
[Medical Named Entity Recognition] ──► Extract Clinical Terms & Lab Markers
    │
    ▼
[Healthcare-Aware Chunking] ────► Retain Table Cohesion & Metadata
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
    │   ├── Verified Response Content
    │   ├── Confidence Level (HIGH / MEDIUM / LOW)
    │   ├── Extracted Clinical Entities
    │   └── Doctor Review Advisory Flag
    │
    ▼
[Evaluation & Observability Engine]
    ├── Faithfulness Score (Factual Claim Grounding)
    ├── Context Precision Metric
    └── End-to-End Latency Logging (JSONL)
```

---

## Core Technical Capabilities

| Capability | Engineering Implementation | Design Rationale |
| :--- | :--- | :--- |
| **Hybrid Search + RRF** | Combines **BM25 Okapi** with **ChromaDB** dense embeddings via **Reciprocal Rank Fusion (RRF)**. | Dense models struggle with exact numeric metrics (e.g. `14 g/dl` vs `11 g/dl`). BM25 guarantees precision for numerical lab thresholds. |
| **Adaptive Query Routing** | Triage classification classifies incoming questions into `FACTOID`, `AGGREGATION`, or `IRRELEVANT`. | Aggregation questions (e.g., *"Are there any abnormal values?"*) require high recall, while factoids require high precision. Irrelevant queries are blocked prior to retrieval. |
| **PHI / PII Redaction** | Regex redaction cleans SSNs, phone numbers, and email patterns prior to vector storage. | Mitigates data leakage risks in compliance with HIPAA privacy standards. |
| **Clinical NER** | Automatically identifies medical markers (`hemoglobin`, `rbc`, `glucose`, `hba1c`, etc.) and attaches them to chunk metadata. | Enhances metadata filtering and search index granularity. |
| **Structured Output** | Powered by Pydantic schemas (`MedicalAnswer`). | Guarantees deterministic JSON contracts for production frontends, complete with self-assessed confidence and clinical review flags. |
| **Live RAG Metrics** | Deterministic evaluation of **Context Precision**, **Faithfulness**, and **Pipeline Latency** logged per interaction. | Provides quantitative visibility into answer grounding and latency trends. |
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
git clone https://github.com/your-username/Healthcare-Document-RAG-Assistant.git
cd Healthcare-Document-RAG-Assistant

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory (optional defaults are already configured in `app/config.py`):
```env
OPENAI_API_KEY=your_key_here
CHUNK_SIZE=800
CHUNK_OVERLAP=100
TOP_K=4
```

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
   - *"What is the hemoglobin level?"* -> Routed as `FACTOID`, answers `14 g/dl`.
   - *"Are there any abnormal values?"* -> Routed as `AGGREGATION`, verifies reference ranges (`13 - 17 g/dl`).
   - *"Write a poem about space."* -> Routed as `IRRELEVANT`, blocked by triage guardrail.

---

## Repository Structure

```
Healthcare-Document-RAG-Assistant/
├── app/
│   ├── __init__.py
│   ├── config.py             # Environment configuration & TF guardrails
│   ├── schemas.py            # Pydantic structured output models
│   ├── main.py               # FastAPI REST API endpoints
│   ├── rag.py                # Core RAG pipeline orchestrator
│   ├── hybrid_retriever.py   # BM25 + Dense RRF fusion retriever
│   ├── query_router.py       # Triage classifier (Factoid/Aggregation/Irrelevant)
│   ├── phi_redactor.py       # HIPAA-aligned PHI/PII redaction
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

## Disclaimer
*This system is intended for document intelligence and educational research purposes. It is not a certified medical device and does not replace professional clinical diagnosis or judgment.*
