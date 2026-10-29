"""FastAPI endpoints for the local healthcare-document RAG demo.

Session IDs are isolation labels, not authentication. Configure APP_API_TOKEN
for a shared-token deployment gate and keep this service on a trusted network.
"""
import hmac
import os
import re
import uuid
from typing import Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.pdf_processor import process_pdf
from app.text_processor import chunk_text
from app.vector_store import add_documents, list_documents, list_document_records
from app.rag import answer_question

app = FastAPI(title="Healthcare Document RAG API")
_default_origins = "http://localhost:8501,http://127.0.0.1:8501"
_allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", _default_origins).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization", "X-App-Token"],
)

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "15"))
APP_API_TOKEN = os.getenv("APP_API_TOKEN", "").strip()
if MAX_UPLOAD_BYTES <= 0 or MAX_PDF_PAGES <= 0:
    raise RuntimeError("MAX_UPLOAD_BYTES and MAX_PDF_PAGES must be positive")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("data/metrics", exist_ok=True)


def require_app_token(x_app_token: Optional[str] = Header(default=None)):
    """Enforce an optional shared token; use real identity-based auth for multi-user deployments."""
    if APP_API_TOKEN and not hmac.compare_digest(x_app_token or "", APP_API_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid or missing application token")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    provider: Literal["openai", "groq", "gemini"] = "openai"
    api_key: str = Field(default="", max_length=512)
    document_names: Optional[List[str]] = None  # legacy filename filter
    document_ids: Optional[List[str]] = None
    patient_label: Optional[str] = Field(default=None, max_length=128)
    report_id: Optional[str] = Field(default=None, max_length=128)
    session_id: str = Field(default="global", min_length=1, max_length=128)
    system1_enabled: bool = False
    system1_provider: Literal["codiv", "jevmodel", "laya"] = "codiv"
    system1_api_key: str = Field(default="", max_length=512)
    system1_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    conversation_history: Optional[List[Dict[str, str]]] = None


class System1TestRequest(BaseModel):
    provider: Literal["codiv", "jevmodel", "laya"] = "codiv"
    api_key: str = Field(min_length=1, max_length=512)


def _safe_component(value: str, label: str) -> str:
    if not value or value in {".", ".."}:
        raise HTTPException(status_code=400, detail=f"Invalid {label}")
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", os.path.basename(value))
    if not safe or safe in {".", ".."}:
        raise HTTPException(status_code=400, detail=f"Invalid {label}")
    return safe[:128]


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/system1/test", dependencies=[Depends(require_app_token)])
def test_system1_endpoint(req: System1TestRequest):
    """Probe connection to the selected System 1 provider with a non-sensitive sample."""
    from app.system1 import get_system1_adapter, System1AuthError, System1QuotaError, System1Error
    try:
        adapter = get_system1_adapter(req.provider, api_key=req.api_key)
        return adapter.test_connection()
    except System1AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    except System1QuotaError as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    except System1Error as exc:
        status = exc.status_code if exc.status_code and 400 <= exc.status_code < 600 else 502
        raise HTTPException(status_code=status, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error testing System 1 connection: {type(exc).__name__}")


@app.post("/upload", dependencies=[Depends(require_app_token)])
async def upload_document(file: UploadFile = File(...), session_id: str = Form("global"), patient_label: str = Form("")):
    filename = _safe_component(file.filename or "", "filename")
    safe_session = _safe_component(session_id, "session_id")[:64]
    patient_label = (patient_label or "").strip()
    if len(patient_label) > 128:
        raise HTTPException(status_code=400, detail="Patient label must be 128 characters or fewer")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="PDF exceeds the configured upload limit")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    document_id = uuid.uuid4().hex
    final_path = os.path.join(UPLOAD_DIR, f"{safe_session}_{document_id}_{filename}")
    temp_path = final_path + ".uploading"
    total = 0
    try:
        with open(temp_path, "wb") as buffer:
            while True:
                block = await file.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="PDF exceeds the configured upload limit")
                buffer.write(block)

        with open(temp_path, "rb") as uploaded:
            if uploaded.read(5) != b"%PDF-":
                raise HTTPException(status_code=400, detail="File does not have a valid PDF header")

        pages = process_pdf(temp_path, filename, max_pages=MAX_PDF_PAGES)
        chunks = chunk_text(pages)
        if not chunks:
            raise HTTPException(status_code=422, detail="No readable text found in the PDF")

        add_documents(chunks, safe_session, document_id=document_id, patient_label=patient_label)
        os.replace(temp_path, final_path)
        return {
            "message": "Upload processed",
            "document_id": document_id,
            "filename": filename,
            "patient_label": patient_label.strip() or "unspecified",
            "chunks": len(chunks),
            "pages_processed": len(pages),
        }
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=422, detail="Could not process this PDF")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        await file.close()


@app.get("/document-pdf", dependencies=[Depends(require_app_token)])
def get_document_pdf(document_id: str, session_id: str = "global"):
    """Serve a PDF by unique upload ID, never by a potentially duplicated filename."""
    safe_id = _safe_component(document_id, "document_id")
    safe_session = _safe_component(session_id, "session_id")[:64]
    record = next(
        (item for item in list_document_records(safe_session) if item["document_id"] == safe_id),
        None,
    )
    if record is None or safe_id.startswith("legacy:"):
        raise HTTPException(status_code=404, detail="Document not found")
    filename = record["document_name"]
    pdf_path = os.path.join(UPLOAD_DIR, f"{safe_session}_{safe_id}_{filename}")
    if not os.path.isfile(pdf_path):
        raise HTTPException(status_code=404, detail="PDF file is unavailable")
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=filename,
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@app.post("/ask", dependencies=[Depends(require_app_token)])
def ask_question_endpoint(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    session_id = _safe_component(req.session_id, "session_id")[:64]
    if req.document_names is not None and len(req.document_names) > 100:
        raise HTTPException(status_code=400, detail="Too many document filters")
    if req.document_ids is not None and len(req.document_ids) > 500:
        raise HTTPException(status_code=400, detail="Too many document filters")

    # Phase 2 Case A: Explicit empty document selection fails closed
    if req.document_ids is not None and len(req.document_ids) == 0:
        return {
            "answer": "No documents are selected. Please select at least one PDF in Chat Context before asking a question.",
            "sources": [],
            "metrics": {"retrieved_chunks": 0},
            "route": "NO_SELECTION",
            "structured_data": {"confidence": "low", "medical_entities": [], "requires_doctor_review": False},
        }
    if req.document_names is not None and len(req.document_names) == 0:
        return {
            "answer": "No documents are selected. Please select at least one PDF in Chat Context before asking a question.",
            "sources": [],
            "metrics": {"retrieved_chunks": 0},
            "route": "NO_SELECTION",
            "structured_data": {"confidence": "low", "medical_entities": [], "requires_doctor_review": False},
        }

    records = list_document_records(session_id)
    if req.document_ids is not None:
        allowed_ids = {record["document_id"] for record in records}
        if any(doc_id not in allowed_ids for doc_id in req.document_ids):
            raise HTTPException(status_code=404, detail="One or more documents were not found in this session")

    patient_label = (req.patient_label or "").strip()
    available_labels = sorted({
        record["patient_label"] for record in records
        if record["patient_label"] and record["patient_label"] != "unspecified"
    })
    asks_about_self = bool(re.search(r"\b(my|mine|myself)\b", question, re.IGNORECASE))
    if asks_about_self:
        return {
            "answer": "This prototype has no verified user identity or patient authorization, so I cannot safely answer “my” from an uploaded file or selected label. Ask about a specific patient record you are authorized to access, or add real identity and authorization controls.",
            "clarification_required": True,
            "available_patient_labels": available_labels,
            "sources": [],
            "metrics": {"retrieved_chunks": 0},
            "route": "CLARIFICATION",
            "structured_data": {"confidence": "low", "medical_entities": [], "requires_doctor_review": True},
        }

    asks_multi_record = bool(re.search(r"\b(all|every|across|each)\b", question, re.IGNORECASE))
    selected_records = [record for record in records if req.document_ids and record["document_id"] in req.document_ids]
    selected_labels = {record["patient_label"] for record in selected_records if record["patient_label"] != "unspecified"}
    ambiguous_patient_scope = (
        not patient_label
        and not asks_multi_record
        and (len(available_labels) > 1 and not req.document_ids and not req.document_names
             or len(selected_labels) > 1)
    )
    if ambiguous_patient_scope:
        test_match = re.search(r"\b(hemoglobin|haemoglobin|hgb|hb|platelet|glucose|cholesterol|potassium|sodium|wbc|rbc)\b", question, re.IGNORECASE)
        test_name = "hemoglobin" if (test_match and test_match.group(1).lower() in ["hb", "hgb", "haemoglobin"]) else (test_match.group(1).lower() if test_match else None)
        clarification_msg = (
            f"Which patient's or report's {test_name} level would you like to know?"
            if test_name
            else "These records are labelled for multiple patients. Which patient's records should I use? Select a patient label or restrict the query to one upload record. I will not combine patient records for a single-patient question."
        )
        return {
            "answer": clarification_msg,
            "clarification_required": True,
            "available_patient_labels": sorted(selected_labels or set(available_labels)),
            "sources": [],
            "metrics": {"retrieved_chunks": 0},
            "route": "CLARIFICATION",
            "structured_data": {"confidence": "low", "medical_entities": [], "requires_doctor_review": True},
        }
    try:
        return answer_question(
            question, req.provider, req.api_key,
            req.document_names, session_id,
            document_ids=req.document_ids, patient_label=patient_label or None,
            system1_enabled=req.system1_enabled,
            system1_provider=req.system1_provider,
            system1_api_key=req.system1_api_key or None,
            system1_threshold=req.system1_threshold,
            conversation_history=req.conversation_history,
            report_id=req.report_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=502, detail="Question could not be processed")


@app.get("/documents", dependencies=[Depends(require_app_token)])
def get_documents(session_id: str = "global"):
    safe_session = _safe_component(session_id, "session_id")
    return {"documents": list_documents(safe_session), "records": list_document_records(safe_session)}
