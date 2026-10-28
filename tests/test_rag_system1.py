"""Integration tests for System 1 + System 2 HealthRAG pipeline, API endpoints, and evaluation."""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.rag import answer_question
from app.system1 import DecisionResult
from app.schemas import MedicalAnswer
from eval.evaluate_system1 import run_evaluation

client = TestClient(app)


def test_baseline_rag_without_system1(monkeypatch):
    """Verify original workflow runs directly to System 2 when System 1 is disabled."""
    mock_chat = MagicMock()
    mock_chat.with_structured_output.return_value.invoke.return_value = MedicalAnswer(
        answer="Hemoglobin is 14 g/dL according to the lab report.",
        confidence="high",
        medical_entities=["hemoglobin"],
        requires_doctor_review=True,
    )
    monkeypatch.setattr("app.rag.get_llm", lambda provider, key: mock_chat)
    monkeypatch.setattr("app.rag.QueryRouter.classify", lambda q, m: "FACTOID")
    monkeypatch.setattr("app.rag.HybridRetriever.search", lambda *args, **kwargs: [
        {"id": "c1", "text": "HEMOGLOBIN 14 g/dl", "metadata": {"document_name": "lab.pdf", "page": 1, "document_id": "doc1"}}
    ])

    result = answer_question(
        question="What is the hemoglobin level?",
        provider="groq",
        api_key="gsk-test-key",
        system1_enabled=False,
    )

    assert "Hemoglobin is 14" in result["answer"]
    assert len(result["sources"]) == 1
    assert "system1_decision" not in result["metrics"]
    assert "abstained_by_system1" not in result


def test_system1_sufficient_invokes_system2(monkeypatch):
    """When System 1 evaluates evidence as sufficient, System 2 generates the grounded answer."""
    mock_chat = MagicMock()
    mock_chat.with_structured_output.return_value.invoke.return_value = MedicalAnswer(
        answer="The hemoglobin measured 14 g/dL, which is within the normal 13-17 g/dL range.",
        confidence="high",
        medical_entities=["hemoglobin"],
        requires_doctor_review=True,
    )
    monkeypatch.setattr("app.rag.get_llm", lambda provider, key: mock_chat)
    monkeypatch.setattr("app.rag.QueryRouter.classify", lambda q, m: "FACTOID")
    monkeypatch.setattr("app.rag.HybridRetriever.search", lambda *args, **kwargs: [
        {"id": "c1", "text": "HEMOGLOBIN 14 g/dl (13-17)", "metadata": {"document_name": "lab.pdf", "page": 1, "document_id": "doc1"}}
    ])

    # System 1 mock decision: SUFFICIENT
    monkeypatch.setattr(
        "app.rag.evaluate_evidence_sufficiency",
        lambda question, chunks, **kwargs: DecisionResult(
            is_sufficient=True,
            probability=0.94,
            threshold=0.5,
            provider="codiv",
            model="openjev-latest",
            latency_sec=0.08,
            status="SUFFICIENT",
        )
    )

    result = answer_question(
        question="What is the hemoglobin level?",
        provider="groq",
        api_key="gsk-test-key",
        system1_enabled=True,
        system1_provider="codiv",
        system1_api_key="codiv-key-123",
    )

    assert "hemoglobin measured 14" in result["answer"].lower()
    assert result["metrics"]["system1_decision"]["status"] == "SUFFICIENT"
    assert result["metrics"]["system1_decision"]["probability"] == 0.94
    assert result["structured_data"]["system1_decision"]["model"] == "openjev-latest"
    assert len(result["sources"]) == 1


def test_system1_insufficient_triggers_bounded_retry_and_abstention(monkeypatch):
    """When evidence is insufficient, system attempts bounded retry, then abstains safely."""
    mock_chat = MagicMock()
    mock_chat.with_structured_output.return_value.invoke.side_effect = AssertionError("System 2 should not be invoked")
    monkeypatch.setattr("app.rag.get_llm", lambda provider, key: mock_chat)
    monkeypatch.setattr("app.rag.QueryRouter.classify", lambda q, m: "FACTOID")

    initial_chunk = [{"id": "c1", "text": "Patient was admitted for routine checkup.", "metadata": {"document_name": "report.pdf", "page": 1, "document_id": "d1"}}]
    monkeypatch.setattr("app.rag.HybridRetriever.search", lambda *args, **kwargs: initial_chunk)

    def mock_eval(question, chunks, **kwargs):
        return DecisionResult(
            is_sufficient=False,
            probability=0.12,
            threshold=0.5,
            provider="codiv",
            model="openjev-latest",
            latency_sec=0.06,
            status="INSUFFICIENT",
        )

    monkeypatch.setattr("app.rag.evaluate_evidence_sufficiency", mock_eval)

    result = answer_question(
        question="What was the troponin level?",
        provider="groq",
        api_key="gsk-key",
        system1_enabled=True,
        system1_provider="codiv",
        system1_api_key="codiv-key",
    )

    assert result["abstained_by_system1"] is True
    assert "insufficient evidence" in result["answer"].lower()
    assert "qualified healthcare professional" in result["answer"]
    assert result["metrics"]["system1_decision"]["status"] == "INSUFFICIENT"
    assert result["structured_data"]["confidence"] == "low"
    assert result["structured_data"]["requires_doctor_review"] is True


def test_system1_insufficient_bounded_retry_recovers_when_expanded(monkeypatch):
    """When initial chunks are insufficient, but expanded retrieval adds sufficient evidence, System 2 generates answer."""
    mock_chat = MagicMock()
    mock_chat.with_structured_output.return_value.invoke.return_value = MedicalAnswer(
        answer="The troponin I was 0.02 ng/mL (normal).",
        confidence="high",
        medical_entities=["troponin"],
        requires_doctor_review=True,
    )
    monkeypatch.setattr("app.rag.get_llm", lambda provider, key: mock_chat)
    monkeypatch.setattr("app.rag.QueryRouter.classify", lambda q, m: "FACTOID")

    c1 = {"id": "c1", "text": "Cardiac labs ordered.", "metadata": {"document_name": "d.pdf", "page": 1, "document_id": "1"}}
    c2 = {"id": "c2", "text": "Troponin I: 0.02 ng/mL", "metadata": {"document_name": "d.pdf", "page": 2, "document_id": "1"}}

    search_count = [0]
    def mock_search(*args, **kwargs):
        search_count[0] += 1
        return [c1] if search_count[0] == 1 else [c1, c2]

    monkeypatch.setattr("app.rag.HybridRetriever.search", mock_search)

    eval_count = [0]
    def mock_eval(question, chunks, **kwargs):
        eval_count[0] += 1
        if len(chunks) == 1:
            return DecisionResult(
                is_sufficient=False, probability=0.20, threshold=0.5, provider="codiv",
                model="openjev-latest", latency_sec=0.05, status="INSUFFICIENT"
            )
        else:
            return DecisionResult(
                is_sufficient=True, probability=0.91, threshold=0.5, provider="codiv",
                model="openjev-latest", latency_sec=0.06, status="SUFFICIENT"
            )

    monkeypatch.setattr("app.rag.evaluate_evidence_sufficiency", mock_eval)

    result = answer_question(
        question="What was the troponin level?",
        provider="groq",
        api_key="gsk-key",
        system1_enabled=True,
        system1_provider="codiv",
        system1_api_key="codiv-key",
    )

    assert eval_count[0] == 2
    assert "troponin I was 0.02" in result["answer"]
    assert result["metrics"]["system1_decision"]["status"] == "SUFFICIENT"
    assert result["metrics"]["system1_decision"]["retry_performed"] is True


def test_system1_error_triggers_conservative_safety_halt(monkeypatch):
    """When System 1 fails (e.g. rate limit/error), conservative policy halts ungrounded generation."""
    mock_chat = MagicMock()
    mock_chat.with_structured_output.return_value.invoke.side_effect = AssertionError("Should not run System 2 on gate error")
    monkeypatch.setattr("app.rag.get_llm", lambda provider, key: mock_chat)
    monkeypatch.setattr("app.rag.QueryRouter.classify", lambda q, m: "FACTOID")
    monkeypatch.setattr("app.rag.HybridRetriever.search", lambda *args, **kwargs: [
        {"id": "c1", "text": "Lab text", "metadata": {"document_name": "lab.pdf", "page": 1, "document_id": "d1"}}
    ])

    monkeypatch.setattr(
        "app.rag.evaluate_evidence_sufficiency",
        lambda *args, **kwargs: DecisionResult(
            is_sufficient=False,
            probability=0.0,
            threshold=0.5,
            provider="codiv",
            model="openjev-latest",
            latency_sec=0.02,
            status="ERROR",
            error_message="Authentication failed for Codiv (HTTP 401)",
        )
    )

    result = answer_question(
        question="What is the hemoglobin level?",
        provider="groq",
        api_key="gsk-key",
        system1_enabled=True,
        system1_provider="codiv",
        system1_api_key="bad-key",
    )

    assert result.get("system1_error") is True
    assert "provider issue" in result["answer"].lower()
    assert "401" in result["answer"]
    assert result["metrics"]["system1_decision"]["status"] == "ERROR"


def test_api_test_system1_endpoint_success(monkeypatch):
    """Verify POST /system1/test endpoint returns probe results for onboarding."""
    import app.system1 as s1_mod
    monkeypatch.setattr(
        s1_mod.CodivAdapter,
        "test_connection",
        lambda self: {
            "status": "ok",
            "provider": "codiv",
            "model": "openjev-latest",
            "latency_ms": 112.5,
            "sample_decision_probability": 0.98,
            "free_tier_terms": "100 million tokens free experiment",
        }
    )

    resp = client.post("/system1/test", json={"provider": "codiv", "api_key": "test-key-123"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model"] == "openjev-latest"
    assert data["sample_decision_probability"] == 0.98


def test_api_test_system1_endpoint_auth_failure(monkeypatch):
    """Verify POST /system1/test maps auth errors to HTTP 401."""
    import app.system1 as s1_mod
    def raise_auth(self):
        raise s1_mod.System1AuthError("Invalid API key for JevModel", provider="jevmodel", status_code=401)

    monkeypatch.setattr(s1_mod.JevModelAdapter, "test_connection", raise_auth)

    resp = client.post("/system1/test", json={"provider": "jevmodel", "api_key": "bad-key"})
    assert resp.status_code == 401
    assert "Invalid API key" in resp.json()["detail"]


def test_api_ask_endpoint_with_system1(monkeypatch):
    """Verify POST /ask endpoint accepts and forwards System 1 parameters."""
    import app.main as main_mod

    captured_kwargs = {}
    def mock_answer(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return {
            "answer": "Grounded answer",
            "sources": [],
            "metrics": {"latency_sec": 0.1, "system1_decision": {"status": "SUFFICIENT"}},
            "route": "FACTOID",
            "structured_data": {"confidence": "high"},
        }

    monkeypatch.setattr(main_mod, "answer_question", mock_answer)
    monkeypatch.setattr(main_mod, "list_document_records", lambda session_id: [])

    resp = client.post(
        "/ask",
        json={
            "question": "What is the hemoglobin level?",
            "provider": "groq",
            "api_key": "gsk-key",
            "system1_enabled": True,
            "system1_provider": "codiv",
            "system1_api_key": "codiv-test-key",
            "session_id": "test-session",
        }
    )

    assert resp.status_code == 200
    assert captured_kwargs.get("system1_enabled") is True
    assert captured_kwargs.get("system1_provider") == "codiv"
    assert captured_kwargs.get("system1_api_key") == "codiv-test-key"
    assert resp.json()["answer"] == "Grounded answer"


def test_offline_evaluation_suite():
    """Verify offline evaluation benchmark executes and computes metrics."""
    eval_result = run_evaluation()
    assert "summary" in eval_result
    assert "cases" in eval_result
    assert len(eval_result["cases"]) == 8
    summary = eval_result["summary"]
    metrics = summary["metrics"]
    assert "answer_correctness" in metrics
    assert "unsupported_answer_rate" in metrics
    assert "appropriate_abstention_rate" in metrics
