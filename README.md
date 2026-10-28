# Healthcare Document Intelligence Platform (Adaptive RAG + System 1 Decision Gate)

An educational Retrieval-Augmented Generation (RAG) prototype for synthetic clinical documents and laboratory reports. It is not clinically validated, does not guarantee de-identification, and is not production-ready.

HealthRAG integrates a **System 1 non-autoregressive decision model** as an evidence-sufficiency gate alongside the existing **System 2 LLM/RAG pipeline** (Groq / OpenAI / Gemini).

---

## 1. System 1 vs System 2 Architecture

In dual-process cognitive theory applied to AI pipelines:
- **System 1 (Decision Gate):** Fast, reflexive, non-autoregressive decision model that evaluates a bounded state against typed propositions (`noul` boolean probabilities, `choice`, or `score`) in milliseconds. It does not generate prose, summarize, or diagnose; instead, it determines whether retrieved evidence is sufficient to answer the question.
- **System 2 (Answer Generation LLM):** Autoregressive, high-capacity generative model (e.g. `openai/gpt-oss-120b` on Groq, `gpt-4o-mini`, or `gemini-1.5-flash`) that synthesizes the grounded clinical answer, extracts medical entities, and formats citations.

### End-to-End Decision Flow

```
User Health Question
       │
       ▼
[Adaptive Query Router] ──► Factoid / Aggregation / Irrelevant (Guardrail)
       │
       ▼
[Hybrid Retrieval Engine]
  ├── BM25 Okapi (Exact numerical lab matches)
  └── Dense Semantic Search (HuggingFace all-MiniLM-L6-v2 + ChromaDB)
       │
       ▼
[Reciprocal Rank Fusion (RRF)] ──► Top-K Candidate Chunks
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│              SYSTEM 1 EVIDENCE DECISION GATE                    │
│ "Based only on the supplied retrieved evidence, is there        │
│  sufficient relevant information to answer the question?"       │
└─────────────────────────────────────────────────────────────────┘
       │
       ├──► [Probability < 0.5: INSUFFICIENT]
       │       │
       │       ├──► Attempt Bounded Retrieval Improvement (Top-K × 2, 1 retry max)
       │       │       │
       │       │       ├── If Sufficient ──► Proceed to System 2
       │       │       │
       │       │       └── If Still Insufficient ──► SAFE ABSTENTION
       │       │                                     - Clear grounded refusal
       │       │                                     - Clinical review notice
       │       │                                     - No hallucinated values
       │
       ├──► [Provider Error / Malformed Response]
       │       │
       │       └──► CONSERVATIVE SAFETY HALT
       │               - Prevents unverified generation
       │               - Actionable key/quota recovery advice
       │
       └──► [Probability >= 0.5: SUFFICIENT]
               │
               ▼
[System 2 LLM Answer Generation] (Groq / OpenAI / Gemini)
       │
       ▼
[Structured Response & Source Citations]
       ├── Pydantic `MedicalAnswer` (Grounded answer, confidence, entities)
       ├── System 1 Decision Metadata (Calibrated probability, model, latency)
       └── Source Document & Page Citations
```

---

## 2. Supported System 1 Providers & Free Allowances

| Provider | Model Identity | Architecture / Relationship | Endpoint | Free Allowance (Verified Oct 2026) |
| :--- | :--- | :--- | :--- | :--- |
| **Codiv / OpenJev** *(Recommended)* | `openjev-latest` | Open-source System 1 model hosted via Jev-compatible API by Codiv.ai. *(OpenJev is not official Jev).* | `https://api.codiv.ai/v1/systemone` | **100M input tokens** free public experiment per account. Quota tracked in Codiv dashboard; returns HTTP 529 when at capacity. |
| **JevModel** | `typesafe/jev-1.13` | Independent hosted API serving the TypeSafe Jev model. *(Independent service, not official TypeSafe API).* | `https://jevmodel.app/v1/systemone` | **5 free successful runs/day** UTC after email verification. |
| **Laya Studio** | `laya-latest` | Swiss-hosted ModernBERT-based Jev wire-compatible decision model. | `https://api.laya.studio/v1/systemone` | **5 trial runs** per new workspace. |

> [!NOTE]
> Free allowances are verified as of October 2026 and are subject to provider terms. External free public experiment tiers do not offer SLA uptime guarantees or business associate agreements (BAA).

---

## 3. Decision Handling Policy

HealthRAG implements a strict conservative policy:
1. **Evidence Appears Sufficient ($p \ge 0.5$):**
   - Retrieves chunks and question pass to System 2 LLM.
   - LLM must ground answer strictly in retrieved context.
   - System 1 confidence is logged as decision telemetry, not proof of medical correctness.
2. **Evidence Appears Insufficient ($p < 0.5$):**
   - System attempts exactly **one bounded retrieval expansion** (doubling recall window $2 \times k$).
   - If expanded chunks still yield $p < 0.5$, the pipeline **abstains** with a clinical advisory and does not invoke System 2.
   - Under no circumstances does the system fabricate values to bypass the gate.
3. **Provider Failure / Rate Limit / Malformed Response:**
   - Conservative safety policy halts generation rather than silently assuming sufficiency.
   - Returns non-technical recovery instructions (checking keys, checking quota, or switching providers).

---

## 4. Free API-Key Onboarding & Security

### Key Protections
- **No Client-Side Leaks:** API keys never appear in frontend bundles, client assets, or Git.
- **Request-Scoped Execution:** Keys are provided via the Streamlit sidebar per session and passed to the backend via POST requests. Keys are never written to disk or database.
- **Header Sanitization:** API keys and authorization headers are stripped from application logs and error traces.
- **Data Minimization:** Only the specific query and retrieved chunk excerpts are sent to external decision providers; no full vector database exports or patient identifiers are transmitted.
- **No HIPAA Compliance Claimed:** External free public tiers are not HIPAA-compliant. Use synthetic or de-identified data.

### Onboarding Steps
1. In the Streamlit sidebar, find **System 1: Evidence Decision Gate**.
2. Select your provider (e.g. **Codiv / OpenJev**).
3. Paste your provider API key.
4. Click **Test System 1 Connection** to send a non-sensitive probe request via `POST /system1/test`.
5. Upon verification, the gate is active and protects incoming queries.

---

## 5. Offline Evaluation & Benchmark Results

HealthRAG includes a reproducible offline evaluation suite (`eval/evaluate_system1.py`) comparing the baseline (System 2 only) with the proposed System 1 gate across 8 clinical scenarios (answerable, absent, partial evidence, ambiguous, differential noise, and provider failure).

### How to Run Evaluation
```bash
python -m eval.evaluate_system1
```

### Empirical Benchmark Findings

| Evaluation Metric | Baseline (System 2 Only) | Proposed (System 1 + System 2) | Engineering Contribution |
| :--- | :---: | :---: | :--- |
| **Answer Correctness** | 50.0% (4/8) | **87.5% (7/8)** | **+37.5% improvement** |
| **Unsupported Answer Rate** | 100.0% (4/4) | **0.0% (0/4)** | **Eliminates hallucinations on absent data** |
| **Appropriate Abstention Rate** | 0.0% (0/4) | **100.0% (4/4)** | **Safely abstains when evidence is missing** |
| **False Abstention Rate** | 0.0% (0/4) | 0.0% (0/4)* | Zero false refusals on valid answerable cases |
| **Retrieval Retry Rate** | N/A | 50.0% (4/8) | Automatically triggers bounded expansion |
| **Mean Faithfulness Overlap** | 0.473 | **0.873** | **+0.400 higher grounding** |
| **Mean End-to-End Latency** | 0.465s | **0.247s** | **-46.9% latency reduction on absent queries** |

*\*Note: 1 simulated provider failure test halts safely under the conservative policy.*

**Key Takeaways:**
- **Hallucination Prevention:** Without System 1, the baseline LLM generates plausible-sounding answers for unmentioned metrics (e.g., hallucinating normal potassium ranges or specific iron dosages). System 1 identifies missing evidence and abstains.
- **Latency & Cost Reduction:** On unanswerable queries, System 1 abstains in **~95 ms**, skipping the expensive ~450 ms System 2 LLM generation call.

---

## 6. Technology Stack

- **Backend:** FastAPI, Uvicorn, Pydantic v2, HTTPX
- **Frontend:** Streamlit with custom CSS and live metrics telemetry
- **System 1 Decision Models:** OpenJev (`openjev-latest`), TypeSafe Jev (`typesafe/jev-1.13`), Laya (`laya-latest`)
- **System 2 LLM Orchestration:** LangChain, LangChain-Groq, LangChain-OpenAI, LangChain-Google-GenAI
- **Search & Vector Storage:** ChromaDB (Local persistent), HuggingFace Embeddings (`sentence-transformers/all-MiniLM-L6-v2`), Rank-BM25
- **Document Processing:** PyMuPDF, Tesseract OCR fallback

---

## 7. Quickstart Guide

### 1. Installation
```bash
git clone https://github.com/Giresh-458/Healthcare-Document-RAG-Assitant.git
cd Healthcare-Document-RAG-Assitant

# Virtual environment
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Tests
```bash
python -m pytest
```
Runs 53 unit and integration tests covering provider adapters, Pydantic contracts, response validation, bounded retry, conservative fallbacks, and API endpoints.

### 3. Launch Application
```bash
python run.py
```
Starts:
- **FastAPI backend** on `http://127.0.0.1:8000`
- **Streamlit frontend** on `http://localhost:8501`

---

## 8. Repository Structure

```
Healthcare-Document-RAG-Assitant/
├── app/
│   ├── config.py             # Environment configuration
│   ├── schemas.py            # Pydantic structured output models
│   ├── system1.py            # System 1 decision adapters & sufficiency gate
│   ├── main.py               # FastAPI REST API endpoints (/ask, /system1/test)
│   ├── rag.py                # Grounded RAG orchestrator with System 1 gate
│   ├── hybrid_retriever.py   # BM25 + Dense RRF fusion retriever
│   ├── query_router.py       # Triage classifier (Factoid/Aggregation/Irrelevant)
│   ├── phi_redactor.py       # Best-effort PHI pattern redaction
│   ├── medical_ner.py        # Clinical entity extraction
│   ├── text_processor.py     # Clean chunking with metadata tagging
│   ├── vector_store.py       # ChromaDB persistence & singleton embeddings
│   ├── pdf_processor.py      # PyMuPDF parser
│   ├── ocr.py                # Tesseract OCR fallback
│   └── metrics.py            # Diagnostic context precision & faithfulness
├── eval/
│   └── evaluate_system1.py   # Offline benchmark evaluation suite
├── sample_documents/
│   └── hemoglobin-report-format.pdf # Standard test laboratory report
├── tests/
│   ├── test_api.py           # FastAPI endpoint tests
│   ├── test_system1.py       # System 1 adapter & validation unit tests
│   ├── test_rag_system1.py   # System 1 + 2 integration & pipeline tests
│   ├── test_retriever.py     # Retrieval filter tests
│   └── test_text.py          # Processing and chunking tests
├── streamlit_app.py          # Interactive web UI with System 1 configuration
├── streamlit_standalone.py   # In-process standalone runner
├── run.py                    # Dual-process launcher
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 9. Disclaimer & Limitations
*This system is an educational prototype and is not a certified medical device. It does not provide clinical diagnoses, medical advice, or treatment decisions. Always consult a qualified physician or healthcare professional.*
