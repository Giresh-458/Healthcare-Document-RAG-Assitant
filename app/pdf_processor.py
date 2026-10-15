"""Extract text from PDF pages with optional OCR fallback."""
from typing import Dict, List

import pymupdf as fitz

from app.config import OCR_ENABLED
from app.ocr import extract_text_from_image


def process_pdf(
    file_path: str, filename: str, max_pages: int = 15
) -> List[Dict]:
    """Extract at most max_pages pages, keeping user-facing page numbers 1-indexed."""
    if max_pages < 1:
        raise ValueError("max_pages must be positive")

    extracted_pages: List[Dict] = []
    with fitz.open(file_path) as doc:
        for page_num in range(min(len(doc), max_pages)):
            page = doc.load_page(page_num)
            text = page.get_text("text").strip()
            if len(text) < 50 and OCR_ENABLED:
                pix = page.get_pixmap(dpi=150)
                text = extract_text_from_image(pix.tobytes("png"))
            if text:
                extracted_pages.append(
                    {
                        "text": text,
                        "page": page_num + 1,
                        "document_name": filename,
                    }
                )
    return extracted_pages
