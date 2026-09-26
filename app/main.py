from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
from typing import List, Optional

from app.pdf_processor import process_pdf
from app.text_processor import chunk_text
from app.vector_store import add_documents, list_documents
from app.rag import answer_question

app = FastAPI(title="Healthcare RAG API with Metrics")

# Enable CORS for the Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/metrics", exist_ok=True)

class AskRequest(BaseModel):
    question: str
    provider: str = "openai"
    api_key: str = ""
    document_names: Optional[List[str]] = None
    session_id: str = "global"

@app.get("/health")
def health_check():
    return {"status": "ok"}

import re

def sanitize_filename(name: str) -> str:
    """Sanitize strings to prevent directory traversal attacks."""
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', os.path.basename(name))

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), session_id: str = Form("global")):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # SECURITY: Prevent Path Traversal Attacks
    safe_session = sanitize_filename(session_id)
    safe_filename = sanitize_filename(file.filename)
    
    file_path = os.path.join("data", "uploads", f"{safe_session}_{safe_filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # PDF Processing Pipeline
    pages = process_pdf(file_path, safe_filename)
    chunks = chunk_text(pages)
    
    # Store to Vector DB
    add_documents(chunks, safe_session)
    
    return {"message": "Upload successful", "filename": safe_filename, "chunks": len(chunks)}

@app.post("/ask")
async def ask_question_endpoint(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        response = answer_question(req.question, req.provider, req.api_key, req.document_names, req.session_id)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
def get_documents(session_id: str = "global"):
    return {"documents": list_documents(session_id)}
