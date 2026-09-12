import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from typing import List, Dict

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
    Keeps metadata (document_name, page) attached to each chunk.
    
    Args:
        pages: A list of dicts with keys 'text', 'page', and 'document_name'.
        
    Returns:
        A list of chunk dicts containing 'text' and 'metadata'.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = []
    for page_data in pages:
        cleaned_text = clean_text(page_data["text"])
        if not cleaned_text:
            continue

        page_chunks = text_splitter.split_text(cleaned_text)
        for chunk in page_chunks:
            chunks.append({
                "text": chunk,
                "metadata": {
                    "document_name": page_data["document_name"],
                    "page": page_data["page"]
                }
            })

    return chunks
