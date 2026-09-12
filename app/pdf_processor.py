import pymupdf as fitz  # PyMuPDF
from app.ocr import extract_text_from_image
from app.config import OCR_ENABLED
from typing import List, Dict

def process_pdf(file_path: str, filename: str) -> List[Dict]:
    """
    Reads a PDF, extracts text page by page.
    If text is too short (likely scanned), falls back to OCR if enabled.
    
    Args:
        file_path (str): Path to the saved PDF file.
        filename (str): Name of the PDF file.
        
    Returns:
        List[Dict]: List of dictionaries containing 'text', 'page', and 'document_name'.
    """
    doc = fitz.open(file_path)
    extracted_pages = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text").strip()

        # If text is very small, it might be a scanned PDF
        if len(text) < 50 and OCR_ENABLED:
            # Render page to an image (pixmap) with 150 DPI for reasonable quality
            pix = page.get_pixmap(dpi=150)
            image_bytes = pix.tobytes("png")
            text = extract_text_from_image(image_bytes)

        if text:
            extracted_pages.append({
                "text": text,
                "page": page_num + 1,  # 1-indexed for user readability
                "document_name": filename
            })

    doc.close()
    return extracted_pages
