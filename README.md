# HealthRAG: Healthcare Document RAG Assistant with System 1 Decision Gate

HealthRAG is a production-grade, grounded Retrieval-Augmented Generation (RAG) assistant designed for clinical and lab document analysis. It pairs a fast, non-autoregressive **System 1 decision gate** with a **System 2 LLM** (Groq / OpenAI / Gemini) to evaluate evidence sufficiency, reduce unsupported answers, and improve abstention behavior before invoking answer generation.

---

## 1. Primary Architecture: System 1 + System 2 Workflow

```
[User Health Question]
       │
       ▼
[Scoped Hybrid Retrieval] (Dense MiniLM + BM25 Lexical + Patient Scope)
       │
       ▼
[Retrieved Evidence Chunks]
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│               SYSTEM 1 DECISION GATE                            │
│  "Based only on the supplied retrieved evidence, is there       │
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
       │       │                                     - Clinical review advisory
       │       │                                     - S2 generation skipped
       │       │
       ├──► [Provider Error / Malformed Response]
       │       │
       │       └──► CONSERVATIVE SAFETY HALT
       │               - Prevents unverified generation
       │               - Actionable key/quota recovery advice
       │               - S2 generation skipped
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

## 2. Supported System 1 Providers & Verified API Contracts

| Provider | Model Identity / Routing | Official Endpoint | Request / Response Details | Free Allowance (Verified Oct 2026) |
| :--- | :--- | :--- | :--- | :--- |
| **Codiv / OpenJev** *(Recommended)* | `openjev-latest` | `https://api.codiv.ai/v1/systemone` | Sends `model: "openjev-latest"`. Response has root `answers`. *(OpenJev is open-source; not official Jev).* | **100M input tokens** free public experiment per account. Quota tracked on Codiv dashboard; returns HTTP 529 with Retry-After when at capacity. |
| **JevModel** | Server-fixed (`typesafe/jev-1.13`) | `https://jevmodel.app/api/v1/systemone` | Model parameter is fixed on server and **omitted** from request. Requires `criteria: {"true": "...", "false": "..."}` for noul. Responses nested under `data.answers`. | **5 free successful runs/day** UTC after email verification. Independent service, not official TypeSafe API. |
| **Laya Studio** | Automatic language routing (omits `model`) | `https://api.laya.studio/v1/systemone` | Omit `model` parameter to use automatic language routing. HealthRAG records the actual model checkpoint reported in the server response. | **5 trial runs** per new workspace. Swiss-hosted ModernBERT architecture. |

> [!NOTE]
> Free allowances are verified as of October 2026 and are subject to provider terms. External free public experiment tiers do not offer SLA uptime guarantees or business associate agreements (BAA).

---

## 3. Decision Handling Policy

HealthRAG implements a strict conservative policy:
1. **Evidence Appears Sufficient ($p \ge 0.5$):**
   - Retrieved chunks and question pass to System 2 LLM.
   - LLM must ground answer strictly in retrieved context.
   - System 1 confidence is logged as decision telemetry, not proof of medical correctness.
2. **Evidence Appears Insufficient ($p < 0.5$):**
   - System attempts exactly **one bounded retrieval expansion** (doubling recall window $2 \times k$).
   - If expanded chunks still yield $p < 0.5$, the pipeline **abstains** with a clinical advisory and does not invoke System 2.
   - Under no circumstances does the system fabricate values to bypass the gate.
3. **Provider Failure / Rate Limit / Malformed Response:**
   - Conservative safety policy halts generation rather than silently assuming sufficiency.
   - Returns non-technical recovery instructions (checking keys, checking quota, or switching providers).
4. **Missing Key Validation:**
   - When System 1 is enabled, a valid API key is mandatory. The pipeline immediately raises a configuration error rather than silently bypassing the decision gate.
5. **Exhaustive Hemoglobin Scan Routing (`EXHAUSTIVE_HEMOGLOBIN_SCAN`):**
   - For all-record hemoglobin aggregation queries across patient documents, full-corpus regex pattern matching extracts candidate matches across accessible chunks.
   - **System 1 Enabled:** Candidate chunks are submitted to the System 1 decision gate before presenting findings. If evaluated as insufficient or on provider failure, the conservative safety policy safely abstains or halts. If sufficient, candidate matches are returned with the verified decision metadata.
   - **System 1 Disabled:** Baseline deterministic regex candidates are returned directly without calling external models.

---

## 4. Free API-Key Onboarding & Security Controls

### Key Protections & Endpoint Security
- **Strict HTTPS Endpoint Allowlist:** Arbitrary caller-controlled endpoint overrides are prohibited. Requests are strictly confined to verified provider endpoints (`ALLOWED_SYSTEM1_ENDPOINTS`), preventing secret exfiltration attacks.
- **No Client-Side Leaks:** API keys never appear in frontend bundles, client assets, or Git.
- **Request-Scoped Execution:** Keys are provided via the Streamlit sidebar per session and passed in the JSON request body. Keys are never saved to disk or database.
- **Log Sanitization:** API keys and authorization headers are stripped from application logs and error traces.
- **Data Minimization:** Only the specific query and retrieved chunk excerpts are sent to external decision providers; no full vector database exports or patient identifiers are transmitted.
- **No HIPAA Compliance Claimed:** External free public tiers are not HIPAA-compliant. Use synthetic or de-identified data.

### Onboarding Steps
1. In the Streamlit sidebar, find **System 1: Evidence Decision Gate**.
2. Select your provider (e.g. **Codiv / OpenJev**).
3. Paste your provider API key.
4. Click **Test System 1 Connection** to send a non-sensitive probe request via `POST /system1/test`.
5. Upon verification, the gate is active and protects incoming queries.

---

## 5. Evaluation Harness & Testing Methodology

HealthRAG includes an evaluation framework (`eval/evaluate_system1.py`) supporting two distinct modes:

### Mode A: Deterministic Scenario Simulation (Default / Offline)
Executes synthetic scenario cases to verify pipeline routing, decision branching, bounded expansion, and safe abstention logic without incurring API fees or network flakiness.

```bash
python -m eval.evaluate_system1
```

*Note: Simulation outputs are strictly for architectural logic verification and do NOT represent empirical benchmark measurements.*

### Mode B: Live API Evaluation (`--live`)
Executes real end-to-end RAG retrieval, real System 1 decision calls (OpenJev/JevModel/Laya), and real System 2 LLM generation (Groq/OpenAI), measuring actual wall-clock latency and live model completions.

```bash
python -m eval.evaluate_system1 --live
```
*(Requires `SYSTEM1_API_KEY` and `GROQ_API_KEY` or `OPENAI_API_KEY`).*

### Proxy Metric Definitions & Disclaimers

> [!IMPORTANT]
> - **Proxy Keyword Correctness:** Evaluates whether predefined substring tokens appear in the generated output. It does **not** measure clinical accuracy, medical correctness, or diagnostic soundness.
> - **Proxy Token Overlap Faithfulness:** Measures unigram lexical overlap between generated response tokens and the actual evidence context chunks retrieved and used (accounting for expanded retrieval when retried). It is a lexical proximity heuristic, **not** clinically validated faithfulness or evidence grounding.
> - **No Medical Accuracy Claims:** Neither scenario simulations nor live proxy runs prove clinical efficacy, diagnostic safety, or complete elimination of hallucinations.

### Architectural Logic Verification Scenarios

| Scenario Category | Test Focus | Baseline Behavior (Ungated) | Gated Behavior (System 1 + 2) |
| :--- | :--- | :--- | :--- |
| **Answerable (Cases 1 & 2)** | Hemoglobin values & reference ranges | Generates answer | Passes gate -> System 2 generates answer |
| **Absent Metrics (Cases 3 & 4)** | Unmeasured lab tests (potassium, CT) | Tends to fabricate normal values | Insufficient evidence -> Safely abstains |
| **Partial Evidence (Case 5)** | Mentions condition, missing dosage | Suggests ungrounded dosages | Insufficient evidence -> Safely abstains |
| **Ambiguous Scope (Case 6)** | Normality check on lab values | Generates answer | Passes gate -> System 2 generates answer |
| **Contradictory / Noise (Case 7)** | Differential diagnosis list | Guesses specific diagnosis | Insufficient evidence -> Safely abstains |
| **Provider Failure (Case 8)** | HTTP 429 / Timeout | N/A | Conservative safety halt (no bypass) |

---

## 6. Technology Stack

- **Backend:** FastAPI, Uvicorn, Pydantic v2, HTTPX
- **Frontend:** Streamlit with custom CSS and live metrics telemetry
- **System 1 Decision Models:** OpenJev (`openjev-latest`), TypeSafe Jev (`typesafe/jev-1.13`), Laya Studio (Automatic routing)
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

### 2. Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your optional server-side keys (`GROQ_API_KEY`, `SYSTEM1_API_KEY`). Users can also enter keys in the Streamlit UI.

### 3. Running Backend & Frontend
Terminal 1 (FastAPI backend):
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2 (Streamlit UI):
```bash
streamlit run frontend/streamlit_app.py
```

### 4. Running Regression Tests
```bash
python -m pytest
```

---

## 8. Clinical & Regulatory Disclaimer

HealthRAG is an experimental prototype for research and evaluation. It is **not** a certified medical device and is **not** intended for primary diagnosis, treatment planning, or acute clinical decision-making. Always consult a licensed physician or healthcare professional for clinical advice.
