# Healthcare Document Intelligence & RAG Assistant

## 1. Problem Statement
Many healthcare documents (such as medical reports, awareness materials, and prescriptions) are stored as PDFs, some of which are scanned images. Extracting specific information from these documents can be time-consuming for patients and professionals alike.

## 2. Solution
This project is an AI-powered RAG (Retrieval-Augmented Generation) assistant that allows users to upload healthcare PDFs, intelligently extracts the text (falling back to OCR for scanned images), and lets users ask questions in natural language. The system then finds the exact context and answers the question while citing the source document and page number.

## 3. Features
- **PDF Upload**: Easy web interface to upload medical PDFs.
- **OCR for Scanned PDFs**: Automatically detects scanned documents and uses Tesseract OCR to read the text.
- **Text Cleaning & Chunking**: Preprocesses text to remove noise and splits it intelligently using LangChain.
- **Embeddings & ChromaDB**: Converts text into vector embeddings using OpenAI and stores them locally via ChromaDB.
- **Semantic Search**: Understands the meaning of your question to retrieve the most relevant sections of your documents.
- **RAG + GPT Answers**: Uses the retrieved context to generate an accurate, grounded answer using OpenAI's GPT.
- **Source Citations**: Clearly shows which document and page the answer came from to prevent hallucinations.
- **FastAPI Backend & Simple Web Interface**: A robust REST API serving a clean, beginner-friendly HTML/JS frontend.

## 4. Architecture

```text
PDF
 ↓
PDF Text Extraction (PyMuPDF)
 ↓
OCR if needed (Tesseract)
 ↓
Text Cleaning
 ↓
Chunking (LangChain)
 ↓
Embeddings (OpenAI)
 ↓
ChromaDB (Vector Database)
 ↓
Similarity Search
 ↓
Relevant Context
 ↓
GPT (OpenAI)
 ↓
Answer + Sources (FastAPI -> Frontend)
```

## 5. Technologies
- **Python 3.11+**
- **FastAPI** (Backend framework)
- **LangChain** (RAG orchestration & chunking)
- **ChromaDB** (Local Vector Database)
- **OpenAI API** (Embeddings and LLM)
- **PyMuPDF / fitz** (PDF processing)
- **Tesseract OCR** (Image-to-text for scanned PDFs)

## 6. Installation
Open your terminal and run the following commands:
```bash
# Clone or navigate to the project directory
cd healthcare-rag-assistant

# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

## 7. Environment Setup
Create a `.env` file in the root folder based on `.env.example`:
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

CHUNK_SIZE=800
CHUNK_OVERLAP=100
TOP_K=4

OCR_ENABLED=true
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```
*Note: Replace `your_openai_api_key_here` with a valid OpenAI API key. Ensure `TESSERACT_CMD` points to your correct Tesseract installation path.*

## 8. Tesseract Installation (Windows)
1. Download the Windows installer from the [UB-Mannheim Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki).
2. Install it (usually to `C:\Program Files\Tesseract-OCR`).
3. The `.env` file in this project automatically points to this default directory.

## 9. Running the Project
Start the FastAPI application using Uvicorn:
```bash
python run.py
```
Open your browser and navigate to: http://127.0.0.1:8000

## 10. API Documentation
FastAPI automatically generates interactive Swagger API documentation.
Once the server is running, visit: http://127.0.0.1:8000/docs
Here you can directly test the `/upload`, `/ask`, and `/documents` endpoints.

## 11. Example Questions
I have included a demo PDF (`static/hemoglobin-report-format.pdf`) in the project for testing. It can also be downloaded directly from the web interface!

Upload it in the web interface and try asking:
- *"What is the hemoglobin level?"*
- *"Are there any abnormal values mentioned?"*
- *"What date was the test conducted?"*

## 12. Limitations
- **Educational Only**: This is an educational project and is **not** a substitute for professional medical advice or diagnosis.
- **Local Storage**: It uses local ChromaDB. In an enterprise system, a cloud vector database (like Pinecone or Milvus) would be used.
- **Language**: Currently optimized primarily for English medical documents.

## 13. Future Improvements
- Add conversational memory (chat history) so follow-up questions work better.
- Implement streaming responses for faster user feedback.
- Support for other file types like DOCX or Images directly.
- Add user authentication to separate documents per user.

---

