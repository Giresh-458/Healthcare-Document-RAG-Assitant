"""Unit and integration tests for System 1 decision adapters and sufficiency gate."""
import pytest
import httpx
from unittest.mock import MagicMock, patch

from app.system1 import (
    CodivAdapter,
    JevModelAdapter,
    LayaAdapter,
    get_system1_adapter,
    format_evidence_state,
    evaluate_evidence_sufficiency,
    NoulAnswer,
    System1ResponsePayload,
    DecisionResult,
    System1Error,
    System1AuthError,
    System1QuotaError,
    System1ValidationError,
    System1ConnectionError,
    SYSTEM1_PROVIDERS,
)


def test_provider_adapter_metadata():
    """Verify provider identities, endpoints, and model names match official specs."""
    codiv = CodivAdapter(api_key="test-key")
    assert codiv.provider_id == "codiv"
    assert codiv.model_name == "openjev-latest"
    assert codiv.default_endpoint == "https://api.codiv.ai/v1/systemone"
    assert "OpenJev" in codiv.display_name
    assert "100 million" in codiv.free_tier_terms

    jev = JevModelAdapter(api_key="test-key")
    assert jev.provider_id == "jevmodel"
    assert jev.model_name == "typesafe/jev-1.13"
    assert jev.default_endpoint == "https://jevmodel.app/v1/systemone"
    assert "5 successful free" in jev.free_tier_terms

    laya = LayaAdapter(api_key="test-key")
    assert laya.provider_id == "laya"
    assert laya.model_name == "laya-latest"
    assert laya.default_endpoint == "https://api.laya.studio/v1/systemone"


def test_get_system1_adapter_factory():
    """Verify factory loads correct adapter class and rejects invalid providers."""
    assert isinstance(get_system1_adapter("codiv", "k"), CodivAdapter)
    assert isinstance(get_system1_adapter("jevmodel", "k"), JevModelAdapter)
    assert isinstance(get_system1_adapter("laya", "k"), LayaAdapter)

    with pytest.raises(ValueError, match="Unsupported System 1 provider"):
        get_system1_adapter("unsupported_provider", "k")


def test_build_headers_requires_key():
    """Verify adapter raises System1AuthError when API key is empty."""
    adapter = CodivAdapter(api_key="")
    with pytest.raises(System1AuthError, match="No API key provided"):
        adapter._build_headers()

    adapter_spaces = CodivAdapter(api_key="   ")
    with pytest.raises(System1AuthError, match="No API key provided"):
        adapter_spaces._build_headers()


def test_build_headers_structure():
    """Verify headers include Bearer authorization without altering secret."""
    adapter = CodivAdapter(api_key="sk-test-token-123")
    headers = adapter._build_headers()
    assert headers["Authorization"] == "Bearer sk-test-token-123"
    assert headers["Content-Type"] == "application/json"


def test_build_payload_noul_question():
    """Verify request payload uses Jev wire format with noul question primitive."""
    adapter = CodivAdapter(api_key="k")
    payload = adapter._build_payload("state text")
    assert payload["model"] == "openjev-latest"
    assert payload["state"] == "state text"
    assert "is_sufficient" in payload["questions"]
    assert payload["questions"]["is_sufficient"]["type"] == "noul"
    assert "sufficient" in payload["questions"]["is_sufficient"]["instructions"].lower()


def test_format_evidence_state_clean_and_bounded():
    """Verify evidence state contains question, passages, and truncates safely."""
    chunks = [
        {"text": "Hemoglobin 14 g/dL", "metadata": {"document_name": "report.pdf", "page": 1}},
        {"text": "Normal range 13-17 g/dL", "metadata": {"document_name": "report.pdf", "page": 1}},
    ]
    state = format_evidence_state("What is the hemoglobin level?", chunks)
    assert "User Health Question:" in state
    assert "What is the hemoglobin level?" in state
    assert "Passage 1 (Source: report.pdf, Page 1):" in state
    assert "Hemoglobin 14 g/dL" in state


def test_response_validation_success():
    """Verify valid Jev-compatible JSON response is parsed into calibrated probability."""
    raw = {
        "model": "openjev-latest",
        "answers": {
            "is_sufficient": {
                "type": "noul",
                "noul": 0.885,
                "confidence": 0.99,
            }
        }
    }
    parsed = System1ResponsePayload(**raw)
    noul_ans = parsed.get_sufficiency_noul("is_sufficient")
    assert noul_ans.type == "noul"
    assert noul_ans.noul == 0.885


def test_response_validation_missing_key():
    """Verify missing question key raises informative error."""
    raw = {"model": "openjev-latest", "answers": {}}
    parsed = System1ResponsePayload(**raw)
    with pytest.raises(ValueError, match="Question 'is_sufficient' not found"):
        parsed.get_sufficiency_noul("is_sufficient")


def test_response_validation_invalid_type():
    """Verify non-noul type raises validation error."""
    raw = {
        "model": "openjev-latest",
        "answers": {
            "is_sufficient": {
                "type": "choice",
                "choice": "yes",
            }
        }
    }
    parsed = System1ResponsePayload(**raw)
    with pytest.raises(Exception):
        parsed.get_sufficiency_noul("is_sufficient")


def test_response_validation_out_of_range():
    """Verify probability outside [0.0, 1.0] is rejected."""
    raw = {
        "model": "openjev-latest",
        "answers": {
            "is_sufficient": {
                "type": "noul",
                "noul": 1.5,
            }
        }
    }
    parsed = System1ResponsePayload(**raw)
    with pytest.raises(Exception):
        parsed.get_sufficiency_noul("is_sufficient")


def test_http_auth_failure_401(monkeypatch):
    """Verify HTTP 401 raises System1AuthError."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 401
    mock_resp.text = "Unauthorized"

    monkeypatch.setattr(httpx.Client, "post", lambda self, *args, **kwargs: mock_resp)
    adapter = CodivAdapter(api_key="invalid-key")

    with pytest.raises(System1AuthError, match="Authentication failed"):
        adapter.evaluate_state("test state", max_retries=0)


def test_http_rate_limit_429(monkeypatch):
    """Verify HTTP 429 raises System1QuotaError."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 429
    mock_resp.headers = {}
    mock_resp.text = "Rate limit reached"

    monkeypatch.setattr(httpx.Client, "post", lambda self, *args, **kwargs: mock_resp)
    adapter = JevModelAdapter(api_key="valid-key")

    with pytest.raises(System1QuotaError, match="Rate limit reached"):
        adapter.evaluate_state("test state", max_retries=0)


def test_http_overloaded_529_with_retry_after(monkeypatch):
    """Verify HTTP 529 (Codiv capacity overload) raises System1QuotaError."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 529
    mock_resp.headers = {"Retry-After": "2"}
    mock_resp.text = '{"error": "overloaded_error"}'

    monkeypatch.setattr(httpx.Client, "post", lambda self, *args, **kwargs: mock_resp)
    adapter = CodivAdapter(api_key="valid-key")

    with pytest.raises(System1QuotaError, match="Service is at capacity"):
        adapter.evaluate_state("test state", max_retries=0)


def test_http_server_error_500(monkeypatch):
    """Verify 500 server error raises System1Error."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 500
    mock_resp.text = "Internal error"

    monkeypatch.setattr(httpx.Client, "post", lambda self, *args, **kwargs: mock_resp)
    adapter = LayaAdapter(api_key="valid-key")

    with pytest.raises(System1Error, match="service error"):
        adapter.evaluate_state("test state", max_retries=0)


def test_http_network_timeout(monkeypatch):
    """Verify timeout exception raises System1ConnectionError."""
    def timeout_post(*args, **kwargs):
        raise httpx.TimeoutException("Connection timed out")

    monkeypatch.setattr(httpx.Client, "post", timeout_post)
    adapter = CodivAdapter(api_key="valid-key")

    with pytest.raises(System1ConnectionError, match="Could not connect"):
        adapter.evaluate_state("test state", max_retries=0)


def test_evaluate_evidence_sufficiency_empty_chunks():
    """Empty chunks return immediate INSUFFICIENT decision without API call."""
    res = evaluate_evidence_sufficiency("What is my Hb?", [], provider="codiv", api_key="test-key")
    assert res.is_sufficient is False
    assert res.probability == 0.0
    assert res.status == "INSUFFICIENT"


def test_evaluate_evidence_sufficiency_positive(monkeypatch):
    """Probability >= threshold returns SUFFICIENT status."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "model": "openjev-latest",
        "answers": {"is_sufficient": {"type": "noul", "noul": 0.92}}
    }
    monkeypatch.setattr(httpx.Client, "post", lambda self, *args, **kwargs: mock_resp)

    chunks = [{"text": "Hb is 14", "metadata": {"document_name": "lab.pdf", "page": 1}}]
    res = evaluate_evidence_sufficiency("Hb level?", chunks, provider="codiv", api_key="k", threshold=0.5)

    assert res.is_sufficient is True
    assert res.status == "SUFFICIENT"
    assert res.probability == 0.92
    assert res.model == "openjev-latest"


def test_evaluate_evidence_sufficiency_negative(monkeypatch):
    """Probability < threshold returns INSUFFICIENT status."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "model": "openjev-latest",
        "answers": {"is_sufficient": {"type": "noul", "noul": 0.15}}
    }
    monkeypatch.setattr(httpx.Client, "post", lambda self, *args, **kwargs: mock_resp)

    chunks = [{"text": "Platelets normal", "metadata": {"document_name": "lab.pdf", "page": 1}}]
    res = evaluate_evidence_sufficiency("What is the glucose?", chunks, provider="codiv", api_key="k", threshold=0.5)

    assert res.is_sufficient is False
    assert res.status == "INSUFFICIENT"
    assert res.probability == 0.15


def test_evaluate_evidence_sufficiency_error_conservative(monkeypatch):
    """Provider failure returns conservative status ERROR and is_sufficient=False."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 401
    mock_resp.text = "Bad Key"
    monkeypatch.setattr(httpx.Client, "post", lambda self, *args, **kwargs: mock_resp)

    chunks = [{"text": "Hb is 14", "metadata": {"document_name": "lab.pdf", "page": 1}}]
    res = evaluate_evidence_sufficiency("Hb level?", chunks, provider="codiv", api_key="bad-key")

    assert res.is_sufficient is False
    assert res.status == "ERROR"
    assert "Authentication failed" in (res.error_message or "")


def test_test_connection_probe(monkeypatch):
    """Verify test_connection sends non-sensitive probe and parses result."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "model": "openjev-latest",
        "answers": {"is_sufficient": {"type": "noul", "noul": 0.96}}
    }
    monkeypatch.setattr(httpx.Client, "post", lambda self, *args, **kwargs: mock_resp)

    adapter = CodivAdapter(api_key="valid-key")
    probe = adapter.test_connection()

    assert probe["status"] == "ok"
    assert probe["provider"] == "codiv"
    assert probe["model"] == "openjev-latest"
    assert probe["sample_decision_probability"] == 0.96
    assert "latency_ms" in probe
    assert "100 million" in probe["free_tier_terms"]
