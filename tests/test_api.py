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


def test_upload_rejects_fake_pdf_header(tmp_path, monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "APP_API_TOKEN", "")
    monkeypatch.setattr(main_module, "UPLOAD_DIR", str(tmp_path))
    response = client.post(
        "/upload",
        files={"file": ("fake.pdf", b"not a PDF", "application/pdf")},
        data={"session_id": "test-session"},
    )
    assert response.status_code == 400
    assert "PDF header" in response.json()["detail"]
    assert list(tmp_path.iterdir()) == []


def test_api_token_gate_rejects_missing_token(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "APP_API_TOKEN", "test-secret")
    response = client.get("/documents", params={"session_id": "test-session"})
    assert response.status_code == 401


def test_api_token_gate_accepts_correct_token(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "APP_API_TOKEN", "test-secret")
    response = client.get(
        "/documents",
        params={"session_id": "test-session"},
        headers={"X-App-Token": "test-secret"},
    )
    assert response.status_code == 200
    assert "documents" in response.json()
