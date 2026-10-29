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

def test_upload_requires_token_when_configured(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "APP_API_TOKEN", "test-secret")
    response = client.post(
        "/upload",
        files={"file": ("report.pdf", b"%PDF-1.4 fake", "application/pdf")},
        data={"session_id": "test-session"},
    )
    assert response.status_code == 401



def test_document_pdf_serves_unique_upload_by_id(tmp_path, monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "APP_API_TOKEN", "")
    monkeypatch.setattr(main_module, "UPLOAD_DIR", str(tmp_path))
    document_id = "abc123"
    monkeypatch.setattr(
        main_module,
        "list_document_records",
        lambda session_id: [{"document_id": document_id, "document_name": "report.pdf"}]
        if session_id == "session-a" else [],
    )
    (tmp_path / f"session-a_{document_id}_report.pdf").write_bytes(b"%PDF-1.4 test")
    response = client.get(
        "/document-pdf",
        params={"document_id": document_id, "session_id": "session-a"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content == b"%PDF-1.4 test"


def test_document_pdf_does_not_serve_another_session(tmp_path, monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "APP_API_TOKEN", "")
    monkeypatch.setattr(main_module, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(main_module, "list_document_records", lambda session_id: [])
    response = client.get(
        "/document-pdf",
        params={"document_id": "abc123", "session_id": "session-b"},
    )
    assert response.status_code == 404


def test_document_pdf_requires_shared_token_when_configured(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "APP_API_TOKEN", "test-secret")
    response = client.get(
        "/document-pdf",
        params={"document_id": "abc123", "session_id": "session-a"},
    )
    assert response.status_code == 401


def test_same_filename_uploads_get_distinct_document_ids(tmp_path, monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "APP_API_TOKEN", "")
    monkeypatch.setattr(main_module, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(main_module, "process_pdf", lambda path, filename, max_pages: [{"text": "Hemoglobin 12.4 g/dL", "page": 1, "document_name": filename}])
    monkeypatch.setattr(main_module, "chunk_text", lambda pages: [{"text": pages[0]["text"], "metadata": {"document_name": pages[0]["document_name"], "page": 1, "medical_entities": "hemoglobin"}}])
    indexed = []
    monkeypatch.setattr(main_module, "add_documents", lambda chunks, session_id, document_id, patient_label: indexed.append((document_id, patient_label)))

    first = client.post(
        "/upload",
        files={"file": ("same.pdf", b"%PDF-1.4 first", "application/pdf")},
        data={"session_id": "test-session", "patient_label": "Patient A"},
    )
    second = client.post(
        "/upload",
        files={"file": ("same.pdf", b"%PDF-1.4 second", "application/pdf")},
        data={"session_id": "test-session", "patient_label": "Patient B"},
    )

    assert first.status_code == second.status_code == 200
    first_data, second_data = first.json(), second.json()
    assert first_data["document_id"] != second_data["document_id"]
    assert first_data["filename"] == second_data["filename"] == "same.pdf"
    assert len(indexed) == 2
    assert indexed[0][1] == "Patient A"
    assert indexed[1][1] == "Patient B"
    assert (tmp_path / f"test-session_{first_data['document_id']}_same.pdf").exists()
    assert (tmp_path / f"test-session_{second_data['document_id']}_same.pdf").exists()


def test_ambiguous_multi_patient_question_requests_clarification(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "APP_API_TOKEN", "")
    monkeypatch.setattr(main_module, "list_document_records", lambda session_id: [
        {"document_id": "doc-a", "document_name": "same.pdf", "patient_label": "Patient A"},
        {"document_id": "doc-b", "document_name": "same.pdf", "patient_label": "Patient B"},
    ])
    monkeypatch.setattr(main_module, "answer_question", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not answer ambiguous question")))
    response = client.post("/ask", json={
        "question": "What is the hemoglobin level?",
        "session_id": "test-session",
        "document_ids": ["doc-a", "doc-b"],
    })
    assert response.status_code == 200
    assert response.json()["clarification_required"] is True
    assert response.json()["route"] == "CLARIFICATION"
    assert "Which patient's or report's hemoglobin level" in response.json()["answer"]
    assert "available_patient_labels" not in response.json()


def test_my_question_does_not_assume_uploader_is_patient(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "APP_API_TOKEN", "")
    monkeypatch.setattr(main_module, "list_document_records", lambda session_id: [
        {"document_id": "doc-a", "document_name": "report.pdf", "patient_label": "Patient A"},
    ])
    response = client.post("/ask", json={
        "question": "What is my hemoglobin level?",
        "session_id": "test-session",
        "document_ids": ["doc-a"],
    })
    assert response.status_code == 200
    assert response.json()["clarification_required"] is True


def test_all_hemoglobin_scan_checks_all_chunks_and_reports_sources():
    from app.rag import _all_hemoglobin_candidates

    chunks = [
        {"text": "Hemoglobin: 12.4 g/dL", "metadata": {"document_id": "doc-1", "document_name": "report.pdf", "patient_label": "Patient A", "page": 1}},
        {"text": "Hb 10.8 g/dL", "metadata": {"document_id": "doc-2", "document_name": "report.pdf", "patient_label": "Patient B", "page": 2}},
        {"text": "Platelets 250", "metadata": {"document_id": "doc-3", "document_name": "other.pdf", "patient_label": "Patient C", "page": 1}},
    ]
    result = _all_hemoglobin_candidates("List all hemoglobin levels", chunks, 0.0)
    assert result["route"] == "EXHAUSTIVE_HEMOGLOBIN_SCAN"
    assert result["metrics"]["documents_scanned"] == 3
    assert "12.4" in result["answer"]
    assert "10.8" in result["answer"]
    assert "Patient A" in result["answer"]
    assert "Patient B" in result["answer"]
    assert len(result["sources"]) == 2


def test_my_question_still_requires_verified_identity_even_with_selected_label(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "APP_API_TOKEN", "")
    monkeypatch.setattr(main_module, "list_document_records", lambda session_id: [
        {"document_id": "doc-a", "document_name": "report.pdf", "patient_label": "Patient A"},
    ])
    response = client.post(
        "/ask",
        json={"question": "What is my hemoglobin level?", "patient_label": "Patient A", "session_id": "test-session", "document_ids": ["doc-a"]},
    )
    assert response.status_code == 200
    assert response.json()["clarification_required"] is True


def test_explicit_all_hemoglobin_query_is_not_blocked_as_ambiguous(monkeypatch):
    import app.main as main_module

    monkeypatch.setattr(main_module, "APP_API_TOKEN", "")
    monkeypatch.setattr(main_module, "list_document_records", lambda session_id: [
        {"document_id": "doc-a", "document_name": "a.pdf", "patient_label": "Patient A"},
        {"document_id": "doc-b", "document_name": "b.pdf", "patient_label": "Patient B"},
    ])
    monkeypatch.setattr(main_module, "answer_question", lambda *args, **kwargs: {"route": "EXHAUSTIVE_HEMOGLOBIN_SCAN"})
    response = client.post("/ask", json={
        "question": "List all hemoglobin levels across all patients",
        "session_id": "test-session",
        "document_ids": ["doc-a", "doc-b"],
    })
    assert response.status_code == 200
    assert response.json()["route"] == "EXHAUSTIVE_HEMOGLOBIN_SCAN"



def test_ask_without_document_selection_fails_closed_before_lookup_or_generation(monkeypatch):
    """A missing selection must not inspect the index or call the RAG pipeline."""
    import app.main as main_module

    monkeypatch.setattr(main_module, "APP_API_TOKEN", "")
    monkeypatch.setattr(
        main_module,
        "list_document_records",
        lambda session_id: (_ for _ in ()).throw(AssertionError("must not inspect indexed records")),
    )
    monkeypatch.setattr(
        main_module,
        "answer_question",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not generate an answer")),
    )

    response = client.post(
        "/ask",
        json={"question": "Give hemoglobin level", "session_id": "test-session"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["route"] == "NO_SELECTION"
    assert "No documents are selected" in data["answer"]
    assert data["sources"] == []
    assert data["metrics"]["retrieved_chunks"] == 0
