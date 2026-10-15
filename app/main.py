import os
import re
import shutil
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.pdf_processor import process_pdf
from app.text_processor import chunk_text
from app.vector_store import add_documents, list_documents
from app.rag import answer_question

app = FastAPI(title="Healthcare Document RAG API")

# Configure explicit origins in deployment. Local development defaults to localhost.
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
    allow_headers=["Content-Type", "Authorization"],
)

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "15"))
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("data/metrics", exist_ok=True)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    provider: str = "openai"
    api_key: str = ""
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


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form("global"),
):
    filename = _safe_component(file.filename or "", "filename")
    safe_session = _safe_component(session_id, "session_id")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    # Starlette tracks the received file size. Also enforce a streaming cap while
    # copying so a missing/unknown size cannot bypass the limit.
    if file.size is not None and file.size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="PDF exceeds the configured upload limit")

    file_path = os.path.join(UPLOAD_DIR, f"{safe_session}_{filename}")
    total = 0
    try:
        with open(file_path, "wb") as buffer:
            while True:
                block = await file.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > MAX_UPLOAD_BYTES:
                    buffer.close()
                    os.remove(file_path)
                    raise HTTPException(
                        status_code=413, detail="PDF exceeds the configured upload limit"
                    )
                buffer.write(block)

        pages = process_pdf(file_path, filename, max_pages=MAX_PDF_PAGES)
        chunks = chunk_text(pages)
        add_documents(chunks, safe_session)
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
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=422, detail="Could not process this PDF")
    finally:
        await file.close()


@app.post("/ask")
async def ask_question_endpoint(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        return answer_question(
            question, req.provider, req.api_key, req.document_names, req.session_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=502, detail="Question could not be processed")


@app.get("/documents")
def get_documents(session_id: str = "global"):
    safe_session = _safe_component(session_id, "session_id")
    return {"documents": list_documents(safe_session)}
