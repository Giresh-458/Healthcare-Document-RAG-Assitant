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
