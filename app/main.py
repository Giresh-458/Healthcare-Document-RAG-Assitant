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

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    file_path = f"data/uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # PDF Processing Pipeline
    pages = process_pdf(file_path, file.filename)
    chunks = chunk_text(pages)
    
    # Store to Vector DB
    add_documents(chunks)
    
    return {"message": "Upload successful", "filename": file.filename, "chunks": len(chunks)}

@app.post("/ask")
async def ask_question_endpoint(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        response = answer_question(req.question, req.provider, req.api_key, req.document_names)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
def get_documents():
    return {"documents": list_documents()}
