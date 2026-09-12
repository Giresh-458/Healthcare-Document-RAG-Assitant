import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    """Test the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_upload_invalid_file_type():
    """Test uploading a non-PDF file."""
    # Simulate a txt file upload
    files = {'file': ('test.txt', b'fake content')}
    response = client.post("/upload", files=files)
    assert response.status_code == 400
    assert "Only PDF files are allowed" in response.json()["detail"]

def test_ask_empty_question():
    """Test asking an empty question."""
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400
    assert "Question cannot be empty" in response.json()["detail"]
