"""Unit tests for PDF text extraction and OCR fallback behavior.

These tests mock Tesseract itself: they verify that the application invokes the
OCR adapter correctly, not the accuracy of the installed OCR engine.
"""
import pymupdf
import pytest

import app.pdf_processor as pdf_processor


def _write_pdf(path, page_texts):
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def test_process_pdf_extracts_text_with_one_indexed_page_numbers(tmp_path, monkeypatch):
    path = tmp_path / "text-report.pdf"
    _write_pdf(path, [
        "Patient A laboratory report. Hemoglobin 12.4 g/dL. " * 2,
        "Patient A second page. Platelets 250 x10^9/L. " * 2,
    ])
    monkeypatch.setattr(pdf_processor, "OCR_ENABLED", False)

    pages = pdf_processor.process_pdf(str(path), "text-report.pdf")

    assert [page["page"] for page in pages] == [1, 2]
    assert all(page["document_name"] == "text-report.pdf" for page in pages)
    assert "Hemoglobin 12.4" in pages[0]["text"]
    assert "Platelets 250" in pages[1]["text"]


def test_process_pdf_uses_ocr_when_page_has_insufficient_embedded_text(tmp_path, monkeypatch):
    path = tmp_path / "scanned-report.pdf"
    _write_pdf(path, [""])
    calls = []
    monkeypatch.setattr(pdf_processor, "OCR_ENABLED", True)

    def fake_ocr(image_bytes):
        calls.append(image_bytes)
        return "Scanned hemoglobin result: 11.8 g/dL"

    monkeypatch.setattr(pdf_processor, "extract_text_from_image", fake_ocr)

    pages = pdf_processor.process_pdf(str(path), "scanned-report.pdf")

    assert len(calls) == 1
    assert calls[0].startswith(b"\x89PNG\r\n\x1a\n")
    assert pages == [{
        "text": "Scanned hemoglobin result: 11.8 g/dL",
        "page": 1,
        "document_name": "scanned-report.pdf",
    }]


def test_process_pdf_rejects_pdf_exceeding_max_pages(tmp_path, monkeypatch):
    path = tmp_path / "multi-page.pdf"
    _write_pdf(path, [
        "First page contains enough embedded text. " * 3,
        "Second page contains enough embedded text. " * 3,
    ])
    monkeypatch.setattr(pdf_processor, "OCR_ENABLED", False)

    with pytest.raises(ValueError, match=r"PDF contains 2 pages.*configured limit is 1"):
        pdf_processor.process_pdf(str(path), "multi-page.pdf", max_pages=1)


@pytest.mark.parametrize("max_pages", [0, -1])
def test_process_pdf_rejects_non_positive_page_limit(tmp_path, max_pages):
    path = tmp_path / "empty.pdf"
    _write_pdf(path, [""])

    with pytest.raises(ValueError, match="max_pages must be positive"):
        pdf_processor.process_pdf(str(path), "empty.pdf", max_pages=max_pages)
