"""Unit tests for System 1 decision adapters, endpoint security, and response validation."""
import pytest
import httpx
from unittest.mock import MagicMock

from app.system1 import (
    CodivAdapter,
    JevModelAdapter,
    LayaAdapter,
    get_system1_adapter,
    format_evidence_state,
    evaluate_evidence_sufficiency,
    validate_endpoint,
    ALLOWED_SYSTEM1_ENDPOINTS,
    System1SecurityError,
    System1Error,
    System1AuthError,
    System1QuotaError,
    System1ValidationError,
    System1ConnectionError,
    SYSTEM1_PROVIDERS,
    DECISION_QUESTION_KEY,
    NOUL_CRITERIA_TRUE,
    NOUL_CRITERIA_FALSE,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. ENDPOINT SECURITY ALLOWLIST TESTS (ISSUE 4)
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_endpoint_allows_verified_providers():
    """Verify all legitimate provider endpoints pass allowlist validation."""
    for ep in ALLOWED_SYSTEM1_ENDPOINTS:
        assert validate_endpoint(ep) == ep


def test_validate_endpoint_rejects_untrusted_urls():
    """Prove that an untrusted endpoint cannot receive requests or keys."""
    untrusted_urls = [
        "https://evil.attacker.com/steal-key",
        "http://api.codiv.ai/v1/systemone",  # Non-HTTPS
        "https://jevmodel.app.attacker.com/api/v1/systemone",
        "https://unknown-service.com/v1",
        "file:///etc/passwd",
    ]
    for bad_url in untrusted_urls:
        with pytest.raises(System1SecurityError, match="Untrusted System 1 endpoint"):
            validate_endpoint(bad_url)


def test_adapter_rejects_untrusted_endpoint_override():
    """Verify BaseSystem1Adapter constructor refuses to initialize with untrusted endpoint."""
    with pytest.raises(System1SecurityError, match="Untrusted System 1 endpoint"):
        CodivAdapter(api_key="secret-key-123", endpoint="https://attacker.com/capture")

    with pytest.raises(System1SecurityError, match="Untrusted System 1 endpoint"):
        JevModelAdapter(api_key="secret-key-123", endpoint="https://evil.site/api")


# ─────────────────────────────────────────────────────────────────────────────
# 2. JEVMODEL ADAPTER TESTS (ISSUE 2 - OFFICIAL DOCUMENTED SCHEMA)
# ─────────────────────────────────────────────────────────────────────────────

def test_jevmodel_adapter_endpoint():
    """Verify JevModel endpoint is https://jevmodel.app/api/v1/systemone."""
    jev = JevModelAdapter(api_key="test-key")
    assert jev.default_endpoint == "https://jevmodel.app/api/v1/systemone"
    assert jev.endpoint == "https://jevmodel.app/api/v1/systemone"


def test_jevmodel_payload_omits_model_and_includes_noul_criteria():
    """Verify JevModel payload does NOT send model and includes true/false criteria for noul."""
    jev = JevModelAdapter(api_key="test-key")
    payload = jev.build_payload("Test patient report state.")

    # Model is fixed on server; must NOT be present in request
    assert "model" not in payload

    # State must be present
    assert payload["state"] == "Test patient report state."

    # Question must be noul with criteria
    q = payload["questions"][DECISION_QUESTION_KEY]
    assert q["type"] == "noul"
    assert "criteria" in q
    assert q["criteria"]["true"] == NOUL_CRITERIA_TRUE
    assert q["criteria"]["false"] == NOUL_CRITERIA_FALSE


def test_jevmodel_parse_response_nested_under_data_answers():
    """Verify JevModel response parser extracts answers nested under data.answers."""
    raw_response = {
        "status": "success",
        "data": {
            "model": "typesafe/jev-1.13",
            "answers": {
                "is_sufficient": {
                    "type": "noul",
                    "noul": 0.912,
                    "confidence": 0.95,
                }
            }
        }
    }
    jev = JevModelAdapter(api_key="test-key")
    prob, model_name, ans_dict = jev.parse_response(raw_response)
    assert prob == 0.912
    assert model_name == "typesafe/jev-1.13"
    assert ans_dict["type"] == "noul"


def test_jevmodel_parse_response_missing_data_answers():
    """Verify JevModel raises validation error if data.answers is missing."""
    jev = JevModelAdapter(api_key="test-key")
    with pytest.raises(System1ValidationError, match="Missing answers in JevModel response"):
        jev.parse_response({"status": "success", "data": {}})


# ─────────────────────────────────────────────────────────────────────────────
# 3. LAYA STUDIO ADAPTER TESTS (ISSUE 5 - AUTOMATIC ROUTING & ACTUAL MODEL)
# ─────────────────────────────────────────────────────────────────────────────

def test_laya_adapter_automatic_routing_omits_model():
    """Verify Laya adapter omits model parameter to trigger automatic language routing."""
    laya = LayaAdapter(api_key="test-key")
    assert laya.default_endpoint == "https://api.laya.studio/v1/systemone"
    payload = laya.build_payload("Clinical notes state.")

    # Model is omitted for automatic routing
    assert "model" not in payload
    assert payload["state"] == "Clinical notes state."
    assert payload["questions"][DECISION_QUESTION_KEY]["type"] == "noul"


def test_laya_parse_response_records_actual_model_returned():
    """Verify Laya records the actual model returned by the router rather than a hardcoded claim."""
    raw_response = {
        "model": "laya-modernbert-en-v1",  # Actual returned checkpoint
        "answers": {
            "is_sufficient": {
                "type": "noul",
                "noul": 0.88,
            }
        }
    }
    laya = LayaAdapter(api_key="test-key")
    prob, reported_model, ans_dict = laya.parse_response(raw_response)
    assert prob == 0.88
    assert reported_model == "laya-modernbert-en-v1"


# ─────────────────────────────────────────────────────────────────────────────
# 4. CODIV / OPENJEV ADAPTER TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_codiv_adapter_metadata_and_payload():
    """Verify Codiv adapter targets https://api.codiv.ai/v1/systemone with openjev-latest."""
    codiv = CodivAdapter(api_key="test-key")
    assert codiv.default_endpoint == "https://api.codiv.ai/v1/systemone"
    payload = codiv.build_payload("Lab report state.")
    assert payload["model"] == "openjev-latest"
    assert payload["questions"][DECISION_QUESTION_KEY]["type"] == "noul"


def test_codiv_parse_response_root_answers():
    """Verify Codiv parses root answers object."""
    raw = {
        "model": "openjev-latest",
        "answers": {
            "is_sufficient": {
                "type": "noul",
                "noul": 0.95,
            }
        }
    }
    codiv = CodivAdapter(api_key="test-key")
    prob, model_name, ans = codiv.parse_response(raw)
    assert prob == 0.95
    assert model_name == "openjev-latest"


# ─────────────────────────────────────────────────────────────────────────────
# 5. ERROR HANDLING & HTTP PROTOCOL TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_auth_error_on_missing_key():
    """Verify adapter raises System1AuthError when key is empty or whitespace."""
    with pytest.raises(System1AuthError, match="No API key provided"):
        CodivAdapter(api_key="")._build_headers()

    with pytest.raises(System1AuthError, match="No API key provided"):
        CodivAdapter(api_key="   ")._build_headers()


def test_http_401_maps_to_auth_error(monkeypatch):
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 401
    mock_resp.text = "Unauthorized"
    monkeypatch.setattr(httpx.Client, "post", lambda self, *args, **kwargs: mock_resp)

    adapter = CodivAdapter(api_key="invalid-key")
    with pytest.raises(System1AuthError, match="Authentication failed"):
        adapter.evaluate_state("test", max_retries=0)


def test_http_429_maps_to_quota_error(monkeypatch):
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 429
    mock_resp.headers = {}
    mock_resp.text = "Rate limit reached"
    monkeypatch.setattr(httpx.Client, "post", lambda self, *args, **kwargs: mock_resp)

    adapter = JevModelAdapter(api_key="valid-key")
    with pytest.raises(System1QuotaError, match="Rate limit reached"):
        adapter.evaluate_state("test", max_retries=0)


def test_http_529_maps_to_quota_overload(monkeypatch):
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 529
    mock_resp.headers = {"Retry-After": "1"}
    mock_resp.text = '{"error": "overloaded_error"}'
    monkeypatch.setattr(httpx.Client, "post", lambda self, *args, **kwargs: mock_resp)

    adapter = CodivAdapter(api_key="valid-key")
    with pytest.raises(System1QuotaError, match="Service is at capacity"):
        adapter.evaluate_state("test", max_retries=0)


def test_http_500_maps_to_system1_error(monkeypatch):
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 500
    mock_resp.text = "Internal error"
    monkeypatch.setattr(httpx.Client, "post", lambda self, *args, **kwargs: mock_resp)

    adapter = LayaAdapter(api_key="valid-key")
    with pytest.raises(System1Error, match="service error"):
        adapter.evaluate_state("test", max_retries=0)


def test_http_network_timeout_maps_to_connection_error(monkeypatch):
    def raise_timeout(*args, **kwargs):
        raise httpx.TimeoutException("Read timed out")

    monkeypatch.setattr(httpx.Client, "post", raise_timeout)
    adapter = CodivAdapter(api_key="valid-key")
    with pytest.raises(System1ConnectionError, match="Could not connect"):
        adapter.evaluate_state("test", max_retries=0)


# ─────────────────────────────────────────────────────────────────────────────
# 6. DECISION GATE LOGIC TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_evaluate_evidence_sufficiency_empty_chunks():
    res = evaluate_evidence_sufficiency("What is my Hb?", [], provider="codiv", api_key="k")
    assert res.is_sufficient is False
    assert res.probability == 0.0
    assert res.status == "INSUFFICIENT"


def test_evaluate_evidence_sufficiency_positive(monkeypatch):
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "model": "openjev-latest",
        "answers": {"is_sufficient": {"type": "noul", "noul": 0.94}}
    }
    monkeypatch.setattr(httpx.Client, "post", lambda self, *args, **kwargs: mock_resp)

    chunks = [{"text": "Hb is 14", "metadata": {"document_name": "lab.pdf", "page": 1}}]
    res = evaluate_evidence_sufficiency("Hb level?", chunks, provider="codiv", api_key="k")
    assert res.is_sufficient is True
    assert res.status == "SUFFICIENT"
    assert res.probability == 0.94


def test_evaluate_evidence_sufficiency_negative(monkeypatch):
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "model": "openjev-latest",
        "answers": {"is_sufficient": {"type": "noul", "noul": 0.15}}
    }
    monkeypatch.setattr(httpx.Client, "post", lambda self, *args, **kwargs: mock_resp)

    chunks = [{"text": "Platelets normal", "metadata": {"document_name": "lab.pdf", "page": 1}}]
    res = evaluate_evidence_sufficiency("What is the glucose?", chunks, provider="codiv", api_key="k")
    assert res.is_sufficient is False
    assert res.status == "INSUFFICIENT"
    assert res.probability == 0.15


def test_evaluate_evidence_sufficiency_security_error_conservative():
    """Verify endpoint security violations result in conservative ERROR status without bypassing."""
    chunks = [{"text": "Hb is 14", "metadata": {"document_name": "lab.pdf", "page": 1}}]
    res = evaluate_evidence_sufficiency("Hb level?", chunks, provider="codiv", api_key="k", endpoint="https://attacker.site")
    assert res.is_sufficient is False
    assert res.status == "ERROR"
    assert "Endpoint Security Violation" in (res.error_message or "")


def test_test_connection_probe(monkeypatch):
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "model": "openjev-latest",
        "answers": {"is_sufficient": {"type": "noul", "noul": 0.98}}
    }
    monkeypatch.setattr(httpx.Client, "post", lambda self, *args, **kwargs: mock_resp)

    adapter = CodivAdapter(api_key="valid-key")
    probe = adapter.test_connection()
    assert probe["status"] == "ok"
    assert probe["model"] == "openjev-latest"
    assert probe["sample_decision_probability"] == 0.98
