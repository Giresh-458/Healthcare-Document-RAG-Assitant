import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from typing import List, Dict
from app.phi_redactor import PHIRedactor
from app.medical_ner import MedicalNER

def clean_text(text: str) -> str:
    """
    Cleans the extracted text by removing excessive spaces and newlines
    while preserving important medical terms, numbers, and punctuation.
    """
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)
    return text.strip()

def chunk_text(pages: List[Dict]) -> List[Dict]:
    """
    Splits text into smaller chunks for embeddings.
    Applies PII/PHI redaction and extracts medical entities as metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = []
    for page_data in pages:
        # Phase 2: Redact PHI before storing
        redacted_text = PHIRedactor.redact(page_data["text"])
        
        cleaned_text = clean_text(redacted_text)
        if not cleaned_text:
            continue

        page_chunks = text_splitter.split_text(cleaned_text)
        for chunk in page_chunks:
            # Phase 2: Extract medical entities for metadata
            entities = MedicalNER.extract_entities(chunk)
            # Join into string so ChromaDB can store it in metadata
            entities_str = ",".join(entities) if entities else "none"
            
            chunks.append({
                "text": chunk,
                "metadata": {
                    "document_name": page_data["document_name"],
                    "page": page_data["page"],
                    "medical_entities": entities_str
                }
            })

    return chunks
