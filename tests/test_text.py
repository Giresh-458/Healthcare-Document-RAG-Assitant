import pytest
from app.text_processor import clean_text, chunk_text

def test_clean_text():
    """Test the text cleaner function."""
    raw_text = "This   is \n\n\n a test   \n string."
    cleaned = clean_text(raw_text)
    assert cleaned == "This is \n a test \n string."

def test_chunking():
    """Test the document chunking logic."""
    pages = [
        {"text": "A" * 1000, "page": 1, "document_name": "test.pdf"}
    ]
    
    # We expect chunks to split at CHUNK_SIZE=800
    chunks = chunk_text(pages)
    
    assert len(chunks) > 1
    assert "metadata" in chunks[0]
    assert chunks[0]["metadata"]["document_name"] == "test.pdf"
    assert chunks[0]["metadata"]["page"] == 1


def test_phi_redaction_patterns():
    from app.phi_redactor import PHIRedactor

    text = "Email jane@example.com, DOB: 01/02/1990, MRN: ABC1234"
    redacted = PHIRedactor.redact(text)
    assert "jane@example.com" not in redacted
    assert "01/02/1990" not in redacted
    assert "ABC1234" not in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_DOB_LABELLED]" in redacted
    assert "[REDACTED_MRN_LABELLED]" in redacted


def test_empty_question_overlap_metric_is_zero():
    from app.metrics import RagMetrics

    assert RagMetrics.calculate_context_precision("", [{"text": "some text"}]) == 0.0


def test_overlap_metric_is_bounded():
    from app.metrics import RagMetrics

    score = RagMetrics.calculate_faithfulness(
        "Hemoglobin is 14 g/dL",
        [{"text": "Hemoglobin measured 14 g/dL"}],
    )
    assert 0.0 <= score <= 1.0
