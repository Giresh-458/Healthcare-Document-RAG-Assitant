"""Integration tests for System 1 + System 2 HealthRAG pipeline, API endpoints, and evaluation."""
import pytest
from unittest.mock import MagicMock
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
        document_names=["test.pdf"],
        provider="groq",
        api_key="gsk-test-key",
        system1_enabled=False,
    )

    assert "Hemoglobin is 14" in result["answer"]
    assert len(result["sources"]) == 1
    assert "system1_decision" not in result["metrics"]
    assert "abstained_by_system1" not in result


def test_direct_rag_without_document_scope_fails_closed(monkeypatch):
    """No scope must stop processing before query routing, retrieval, or generation."""
    monkeypatch.setattr(
        "app.rag.get_llm",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("LLM must not be called")),
    )
    monkeypatch.setattr(
        "app.rag.HybridRetriever.search",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("retrieval must not be called")),
    )

    result = answer_question(
        question="Give hemoglobin level",
        provider="groq",
        api_key="gsk-test-key",
        system1_enabled=False,
    )

    assert result["route"] == "NO_SELECTION"
    assert result["sources"] == []
    assert result["metrics"]["retrieved_chunks"] == 0
    assert "No documents are selected" in result["answer"]


def test_missing_system1_key_raises_configuration_error():
    """Verify that enabling System 1 without supplying an API key raises a clear configuration error."""
    with pytest.raises(ValueError, match="System 1 evidence decision gate is enabled, but no API key was provided"):
        answer_question(
            question="What is the hemoglobin level?",
            document_names=["test.pdf"],
            provider="groq",
            api_key="gsk-test-key",
            system1_enabled=True,
            system1_provider="codiv",
            system1_api_key="",  # Missing key!
        )

    with pytest.raises(ValueError, match="System 1 evidence decision gate is enabled, but no API key was provided"):
        answer_question(
            question="What is the hemoglobin level?",
            document_names=["test.pdf"],
            provider="groq",
            api_key="gsk-test-key",
            system1_enabled=True,
            system1_provider="jevmodel",
            system1_api_key="   ",  # Whitespace key!
        )


def test_api_ask_endpoint_rejects_enabled_system1_with_missing_key(monkeypatch):
    """Verify POST /ask endpoint returns HTTP 400 when System 1 is enabled without a key."""
    import app.main as main_module
    monkeypatch.setattr(main_module, "APP_API_TOKEN", "")
    monkeypatch.setattr(main_module, "list_document_records", lambda session_id: [
        {"document_id": "doc-a", "document_name": "lab.pdf", "patient_label": "Patient A"}
    ])
    resp = client.post(
        "/ask",
        json={
            "question": "What is the hemoglobin level?",
            "provider": "groq",
            "api_key": "gsk-key",
            "system1_enabled": True,
            "system1_provider": "codiv",
            "system1_api_key": "",
            "session_id": "test-session",
            "document_ids": ["doc-a"],
        }
    )
    assert resp.status_code == 400
    assert "System 1 evidence decision gate is enabled, but no API key was provided" in resp.json()["detail"]


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
        document_names=["test.pdf"],
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
        document_names=["test.pdf"],
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
        document_names=["test.pdf"],
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
        document_names=["test.pdf"],
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
    monkeypatch.setattr(main_mod, "list_document_records", lambda session_id: [
        {"document_id": "doc-a", "document_name": "lab.pdf", "patient_label": "Patient A"}
    ])

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
            "document_ids": ["doc-a"],
        }
    )

    assert resp.status_code == 200
    assert captured_kwargs.get("system1_enabled") is True
    assert captured_kwargs.get("system1_provider") == "codiv"
    assert captured_kwargs.get("system1_api_key") == "codiv-test-key"
    assert resp.json()["answer"] == "Grounded answer"


def test_offline_evaluation_suite():
    """Verify offline evaluation harness executes and produces cleanly labeled simulation output."""
    eval_result = run_evaluation(mode="simulation")
    assert "summary" in eval_result
    assert "cases" in eval_result
    assert len(eval_result["cases"]) == 8
    summary = eval_result["summary"]
    assert summary["mode"] == "deterministic_scenario_simulation"
    assert "simulated_scenarios" in summary
    assert "proxy_metric_definitions" in summary
    assert "proxy_metrics_summary" in summary


# ─────────────────────────────────────────────────────────────────────────────
# ALL-HEMOGLOBIN EXHAUSTIVE SCAN REGRESSION TESTS (SYSTEM 1 ENABLED & DISABLED)
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_HEMOGLOBIN_CHUNKS = [
    {
        "id": "c1",
        "text": "HAEMATOLOGY REPORT\nPatient: John Doe | Hemoglobin: 14.2 g/dL | Normal: 13-17",
        "metadata": {"document_id": "doc-1", "document_name": "patient_a.pdf", "patient_label": "Patient A", "page": 1},
    },
    {
        "id": "c2",
        "text": "CBC Report\nPatient: Jane Smith | Hb 11.5 g/dL | RBC 4.2",
        "metadata": {"document_id": "doc-2", "document_name": "patient_b.pdf", "patient_label": "Patient B", "page": 2},
    },
    {
        "id": "c3",
        "text": "Platelet Count: 250,000 /mcL\nWBC: 6,800 /mcL",
        "metadata": {"document_id": "doc-3", "document_name": "patient_c.pdf", "patient_label": "Patient C", "page": 1},
    },
]


def test_all_hemoglobin_route_with_system1_disabled(monkeypatch):
    """When System 1 is disabled, all-hemoglobin query runs deterministic full-corpus regex scan."""
    monkeypatch.setattr("app.rag.HybridRetriever.get_all_chunks", lambda **kwargs: SAMPLE_HEMOGLOBIN_CHUNKS)

    result = answer_question(
        question="List all hemoglobin levels across all patients",
        document_names=["test.pdf"],
        provider="groq",
        api_key="gsk-key",
        system1_enabled=False,
    )

    assert result["route"] == "EXHAUSTIVE_HEMOGLOBIN_SCAN"
    assert "Patient A" in result["answer"]
    assert "Patient B" in result["answer"]
    assert "14.2" in result["answer"]
    assert "11.5" in result["answer"]
    assert len(result["sources"]) == 2
    assert "system1_decision" not in result.get("structured_data", {})
    assert "system1_decision" not in result.get("metrics", {})
    assert result.get("abstained_by_system1") is not True
    assert result.get("system1_error") is not True


def test_all_hemoglobin_route_with_system1_enabled_sufficient(monkeypatch):
    """When System 1 is enabled and evaluates candidate evidence as sufficient, returns verified candidate scan."""
    monkeypatch.setattr("app.rag.HybridRetriever.get_all_chunks", lambda **kwargs: SAMPLE_HEMOGLOBIN_CHUNKS)
    monkeypatch.setattr(
        "app.rag.evaluate_evidence_sufficiency",
        lambda question, chunks, **kwargs: DecisionResult(
            is_sufficient=True,
            probability=0.96,
            threshold=0.5,
            provider="codiv",
            model="openjev-latest",
            latency_sec=0.07,
            status="SUFFICIENT",
        ),
    )

    result = answer_question(
        question="List all hemoglobin levels across all patients",
        document_names=["test.pdf"],
        provider="groq",
        api_key="gsk-key",
        system1_enabled=True,
        system1_provider="codiv",
        system1_api_key="codiv-key-123",
    )

    assert result["route"] == "EXHAUSTIVE_HEMOGLOBIN_SCAN"
    assert "Patient A" in result["answer"]
    assert "Patient B" in result["answer"]
    assert "14.2" in result["answer"]
    assert result["metrics"]["system1_decision"]["status"] == "SUFFICIENT"
    assert result["metrics"]["system1_decision"]["probability"] == 0.96
    assert result["metrics"]["system1_evaluated"] is True
    assert result["structured_data"]["system1_decision"]["status"] == "SUFFICIENT"
    assert result.get("abstained_by_system1") is not True
    assert result.get("system1_error") is not True


def test_all_hemoglobin_route_with_system1_enabled_insufficient_abstains(monkeypatch):
    """When System 1 evaluates candidate evidence as insufficient, safe abstention is triggered."""
    monkeypatch.setattr("app.rag.HybridRetriever.get_all_chunks", lambda **kwargs: SAMPLE_HEMOGLOBIN_CHUNKS)
    monkeypatch.setattr(
        "app.rag.evaluate_evidence_sufficiency",
        lambda question, chunks, **kwargs: DecisionResult(
            is_sufficient=False,
            probability=0.15,
            threshold=0.5,
            provider="codiv",
            model="openjev-latest",
            latency_sec=0.05,
            status="INSUFFICIENT",
        ),
    )

    result = answer_question(
        question="List all hemoglobin levels across all patients",
        document_names=["test.pdf"],
        provider="groq",
        api_key="gsk-key",
        system1_enabled=True,
        system1_provider="codiv",
        system1_api_key="codiv-key-123",
    )

    assert result["route"] == "EXHAUSTIVE_HEMOGLOBIN_SCAN"
    assert result.get("abstained_by_system1") is True
    assert "insufficient evidence" in result["answer"].lower()
    assert "healthcare professional" in result["answer"].lower()
    assert result["metrics"]["system1_decision"]["status"] == "INSUFFICIENT"
    assert result["structured_data"]["system1_decision"]["status"] == "INSUFFICIENT"
    assert result["structured_data"]["confidence"] == "low"


def test_all_hemoglobin_route_with_system1_enabled_error_halts(monkeypatch):
    """When System 1 encounters an error during all-hemoglobin scan, conservative safety halts."""
    monkeypatch.setattr("app.rag.HybridRetriever.get_all_chunks", lambda **kwargs: SAMPLE_HEMOGLOBIN_CHUNKS)
    monkeypatch.setattr(
        "app.rag.evaluate_evidence_sufficiency",
        lambda question, chunks, **kwargs: DecisionResult(
            is_sufficient=False,
            probability=0.0,
            threshold=0.5,
            provider="codiv",
            model="openjev-latest",
            latency_sec=0.02,
            status="ERROR",
            error_message="HTTP 503 Service Unavailable",
        ),
    )

    result = answer_question(
        question="List all hemoglobin levels across all patients",
        document_names=["test.pdf"],
        provider="groq",
        api_key="gsk-key",
        system1_enabled=True,
        system1_provider="codiv",
        system1_api_key="codiv-key-123",
    )

    assert result["route"] == "EXHAUSTIVE_HEMOGLOBIN_SCAN"
    assert result.get("system1_error") is True
    assert "unable to verify the evidence" in result["answer"].lower()
    assert "503" in result["answer"]
    assert result["metrics"]["system1_decision"]["status"] == "ERROR"
    assert result["structured_data"]["system1_decision"]["status"] == "ERROR"


def test_all_hemoglobin_route_with_system1_enabled_missing_key_raises():
    """When all-hemoglobin scan is queried with System 1 enabled but key missing, fail-fast validation raises ValueError."""
    with pytest.raises(ValueError, match="System 1 evidence decision gate is enabled, but no API key was provided"):
        answer_question(
            question="List all hemoglobin levels across all patients",
            document_names=["test.pdf"],
            provider="groq",
            api_key="gsk-key",
            system1_enabled=True,
            system1_provider="codiv",
            system1_api_key="",
        )


def test_all_hemoglobin_route_empty_corpus(monkeypatch):
    """When all-hemoglobin scan finds no indexed chunks in scope, returns clear notice."""
    monkeypatch.setattr("app.rag.HybridRetriever.get_all_chunks", lambda **kwargs: [])

    result = answer_question(
        question="List all hemoglobin levels across all patients",
        document_names=["test.pdf"],
        provider="groq",
        api_key="gsk-key",
        system1_enabled=True,
        system1_api_key="codiv-key-123",
    )

    assert result["route"] == "EXHAUSTIVE_HEMOGLOBIN_SCAN"
    assert "No indexed chunks were found" in result["answer"]
    assert result["metrics"]["retrieved_chunks"] == 0
