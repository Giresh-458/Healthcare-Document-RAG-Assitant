import re
from typing import Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.phi_redactor import PHIRedactor
from app.medical_ner import MedicalNER


def clean_text(text: str) -> str:
    """
    Cleans the extracted text by removing excessive spaces and newlines
    while preserving important medical terms, numbers, and punctuation.
    """
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()


def _extract_page_metadata(text: str) -> Dict[str, str]:
    """Extract report ID and report date from page text if explicitly present."""
    meta = {"report_id": "", "report_date": ""}
    id_match = re.search(
        r"(?i)\b(?:report\s*(?:id|no|number)?|reg(?:istration)?\.?\s*(?:no|num|number)?|accession\s*(?:no|num|number)?)\s*[:#-]?\s*([A-Za-z0-9_-]+)",
        text,
    )
    if id_match:
        val = id_match.group(1).strip()
        if not val.startswith("[REDACTED"):
            meta["report_id"] = val

    date_match = re.search(
        r"(?i)\b(?:collected|received|reported|registered|date)\s*(?:on|date)?\s*[:#-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})",
        text,
    )
    if date_match:
        val = date_match.group(1).strip()
        if not val.startswith("[REDACTED"):
            meta["report_date"] = val
    return meta


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
        # Redact PHI before extracting metadata or storing chunks
        redacted_text = PHIRedactor.redact(page_data.get("text", ""))
        page_meta = _extract_page_metadata(redacted_text)
        
        cleaned_text = clean_text(redacted_text)
        if not cleaned_text:
            continue

        page_chunks = text_splitter.split_text(cleaned_text)
        for chunk in page_chunks:
            # Extract medical entities for metadata
            entities = MedicalNER.extract_entities(chunk)
            entities_str = ",".join(entities) if entities else "none"
            
            chunks.append({
                "text": chunk,
                "metadata": {
                    "document_name": page_data["document_name"],
                    "page": page_data["page"],
                    "medical_entities": entities_str,
                    "report_id": page_meta.get("report_id", ""),
                    "report_date": page_meta.get("report_date", ""),
                }
            })

    return chunks
