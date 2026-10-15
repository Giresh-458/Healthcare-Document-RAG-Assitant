"""FastAPI endpoints for the local healthcare-document RAG demo.

Session IDs are isolation labels, not authentication. Configure APP_API_TOKEN
for a shared-token deployment gate and keep this service on a trusted network.
"""
import hmac
import os
import re
import uuid
from typing import List, Literal, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.pdf_processor import process_pdf
from app.text_processor import chunk_text
from app.vector_store import add_documents, list_documents
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
    document_names: Optional[List[str]] = None
    session_id: str = Field(default="global", min_length=1, max_length=128)


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


@app.post("/upload", dependencies=[Depends(require_app_token)])
async def upload_document(file: UploadFile = File(...), session_id: str = Form("global")):
    filename = _safe_component(file.filename or "", "filename")
    safe_session = _safe_component(session_id, "session_id")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="PDF exceeds the configured upload limit")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    final_path = os.path.join(UPLOAD_DIR, f"{safe_session}_{filename}")
    # Unique temporary names avoid concurrent uploads corrupting each other.
    temp_path = final_path + "." + uuid.uuid4().hex + ".uploading"
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

        # Index first; the vector store preserves the previous document if embedding fails.
        add_documents(chunks, safe_session)
        os.replace(temp_path, final_path)
        return {
            "message": "Upload processed",
            "filename": filename,
            "chunks": len(chunks),
            "pages_processed": len(pages),
        }
    except HTTPException:
        raise
    except Exception:
        # Do not expose local paths, provider details, or internal exception text.
        raise HTTPException(status_code=422, detail="Could not process this PDF")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        await file.close()


@app.post("/ask", dependencies=[Depends(require_app_token)])
async def ask_question_endpoint(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    session_id = _safe_component(req.session_id, "session_id")
    document_names = None
    if req.document_names is not None:
        if len(req.document_names) > 100:
            raise HTTPException(status_code=400, detail="Too many document filters")
        document_names = [_safe_component(name, "document name") for name in req.document_names]
    try:
        return answer_question(question, req.provider, req.api_key, document_names, session_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=502, detail="Question could not be processed")


@app.get("/documents", dependencies=[Depends(require_app_token)])
def get_documents(session_id: str = "global"):
    safe_session = _safe_component(session_id, "session_id")
    return {"documents": list_documents(safe_session)}
