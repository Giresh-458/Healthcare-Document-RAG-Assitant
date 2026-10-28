"""System 1 Decision-Making Module for HealthRAG.

Integrates fast, typed non-autoregressive decision models to act as an
evidence-sufficiency decision gate before invoking the System 2 LLM.

Supported Providers:
1. Codiv / OpenJev: Open-source System 1 model served via Jev-compatible wire format.
   Endpoint: https://api.codiv.ai/v1/systemone | Model: openjev-latest
2. JevModel: Independent hosted service for TypeSafe Jev.
   Endpoint: https://jevmodel.app/api/v1/systemone | Fixed server-side model | Requires criteria for noul
   Responses nested under data.answers.
3. Laya Studio: Swiss-hosted ModernBERT-based decision API.
   Endpoint: https://api.laya.studio/v1/systemone | Automatic language routing (omits model param)
   Extracts actual model returned by server.

Security & Endpoint Protection:
- Arbitrary caller-controlled endpoint overrides are prohibited.
- Endpoints must match the verified HTTPS provider allowlist (ALLOWED_SYSTEM1_ENDPOINTS).
- API keys are never persisted to disk or emitted in logs.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import httpx
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("healthrag.system1")

# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINT ALLOWLIST & SECURITY
# ─────────────────────────────────────────────────────────────────────────────

ALLOWED_SYSTEM1_ENDPOINTS = {
    "https://api.codiv.ai/v1/systemone",
    "https://jevmodel.app/api/v1/systemone",
    "https://api.laya.studio/v1/systemone",
}


class System1SecurityError(Exception):
    """Raised when an untrusted or non-allowlisted endpoint is requested."""
    pass


def validate_endpoint(endpoint: str, allowlist: Optional[set] = None) -> str:
    """Ensure destination URL belongs to the verified provider allowlist."""
    allowed = allowlist or ALLOWED_SYSTEM1_ENDPOINTS
    cleaned = (endpoint or "").strip()
    if cleaned not in allowed:
        raise System1SecurityError(
            f"Untrusted System 1 endpoint: '{cleaned}'. "
            f"Endpoint overrides are restricted to verified provider endpoints: {sorted(list(allowed))}"
        )
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# PROVIDER METADATA & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM1_PROVIDERS = {
    "codiv": {
        "name": "Codiv / OpenJev",
        "model": "openjev-latest",
        "default_endpoint": "https://api.codiv.ai/v1/systemone",
        "docs_url": "https://codiv.ai/docs/quickstart",
        "description": "Hosted OpenJev open-source System 1 model (Jev-compatible). Recommended for free testing.",
        "free_tier_terms": (
            "Free public experiment: 100 million input tokens per account (verified Oct 2026). "
            "Tracked via Codiv dashboard. Returns HTTP 529 with Retry-After when at capacity."
        ),
        "is_openjev": True,
    },
    "jevmodel": {
        "name": "JevModel",
        "model": "typesafe/jev-1.13",
        "default_endpoint": "https://jevmodel.app/api/v1/systemone",
        "docs_url": "https://jevmodel.app/jev-api",
        "description": "Independent hosted API serving the TypeSafe Jev model. Fixed model on server. Note: Independent service, not official TypeSafe API.",
        "free_tier_terms": (
            "5 successful free API runs per UTC day after email verification, subject to service capacity (verified Oct 2026)."
        ),
        "is_openjev": False,
    },
    "laya": {
        "name": "Laya Studio",
        "model": "automatic",
        "default_endpoint": "https://api.laya.studio/v1/systemone",
        "docs_url": "https://laya.studio/docs/api",
        "description": "Swiss-hosted ModernBERT-based System 1 decision model with automatic language routing (Jev wire-compatible).",
        "free_tier_terms": (
            "Limited trial: 5 free runs per new workspace (verified Oct 2026). Not a permanent high-volume free tier."
        ),
        "is_openjev": False,
    },
}

DEFAULT_SYSTEM1_PROVIDER = "codiv"
DEFAULT_SUFFICIENCY_THRESHOLD = 0.5
DECISION_QUESTION_KEY = "is_sufficient"
DECISION_INSTRUCTIONS = (
    "Based only on the supplied retrieved evidence, is there sufficient "
    "relevant information to answer the user's question accurately?"
)

# Standard criteria required by JevModel for noul decision primitives
NOUL_CRITERIA_TRUE = (
    "The supplied retrieved evidence contains sufficient relevant information "
    "to answer the user's question accurately."
)
NOUL_CRITERIA_FALSE = (
    "The supplied retrieved evidence is missing, partial, or insufficient "
    "to answer the user's question accurately."
)


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class System1Error(Exception):
    """Base exception for System 1 decision failures."""

    def __init__(self, message: str, provider: str = "unknown", status_code: Optional[int] = None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class System1AuthError(System1Error):
    """Raised when the provider API key is missing, invalid, or unauthorized."""
    pass


class System1QuotaError(System1Error):
    """Raised when rate limits or free-tier quotas are exhausted."""
    pass


class System1ValidationError(System1Error):
    """Raised when the provider response does not conform to the expected schema."""
    pass


class System1ConnectionError(System1Error):
    """Raised on network timeout, DNS resolution failure, or unreachable host."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC RESPONSE CONTRACTS
# ─────────────────────────────────────────────────────────────────────────────

class NoulAnswer(BaseModel):
    """Contract for Jev-compatible boolean/probability decision primitive."""
    model_config = ConfigDict(extra="ignore")

    type: Literal["noul"]
    noul: float = Field(ge=0.0, le=1.0, description="Calibrated probability of proposition being true [0.0 - 1.0]")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# DECISION RESULT DATA CLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DecisionResult:
    """Standardized decision outcome for the HealthRAG pipeline."""
    is_sufficient: bool
    probability: float
    threshold: float
    provider: str
    model: str
    latency_sec: float
    status: Literal["SUFFICIENT", "INSUFFICIENT", "ERROR", "SKIPPED"]
    error_message: Optional[str] = None
    retry_performed: bool = False
    raw_answer: Optional[Dict[str, Any]] = None
    allowance_note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_sufficient": self.is_sufficient,
            "probability": round(self.probability, 4),
            "threshold": self.threshold,
            "provider": self.provider,
            "model": self.model,
            "latency_sec": round(self.latency_sec, 3),
            "status": self.status,
            "error_message": self.error_message,
            "retry_performed": self.retry_performed,
            "allowance_note": self.allowance_note,
        }


# ─────────────────────────────────────────────────────────────────────────────
# PROVIDER ADAPTER BASE & IMPLEMENTATIONS
# ─────────────────────────────────────────────────────────────────────────────

class BaseSystem1Adapter:
    """Base adapter defining the contract for System 1 decision providers."""

    def __init__(self, api_key: str, endpoint: Optional[str] = None, timeout: float = 10.0):
        self.api_key = (api_key or "").strip()
        self.timeout = timeout
        target_endpoint = (endpoint or self.default_endpoint).strip()
        # Security: Enforce strict allowlist check on all endpoints
        self.endpoint = validate_endpoint(target_endpoint)

    @property
    def provider_id(self) -> str:
        raise NotImplementedError

    @property
    def display_name(self) -> str:
        raise NotImplementedError

    @property
    def default_endpoint(self) -> str:
        raise NotImplementedError

    @property
    def free_tier_terms(self) -> str:
        raise NotImplementedError

    def build_payload(self, state: str) -> Dict[str, Any]:
        raise NotImplementedError

    def parse_response(self, raw_json: Dict[str, Any]) -> Tuple[float, str, Dict[str, Any]]:
        """Parse raw response JSON into (probability, model_name, raw_answer_dict)."""
        raise NotImplementedError

    def _build_headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise System1AuthError(
                f"No API key provided for {self.display_name}. Please configure your API key in the sidebar.",
                provider=self.provider_id,
            )
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "HealthRAG-System1/1.0",
        }

    def _execute_http_request(self, payload: Dict[str, Any], max_retries: int = 1) -> Tuple[Dict[str, Any], float]:
        """Execute HTTP POST with bounded exponential backoff and Retry-After handling."""
        headers = self._build_headers()
        start_time = time.time()
        last_exception: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(self.endpoint, headers=headers, json=payload)

                if response.status_code == 200:
                    try:
                        data = response.json()
                    except Exception as json_err:
                        raise System1ValidationError(
                            f"Provider {self.display_name} returned non-JSON response: {json_err}",
                            provider=self.provider_id,
                            status_code=200,
                        )
                    elapsed = time.time() - start_time
                    return data, elapsed

                if response.status_code in (401, 403):
                    raise System1AuthError(
                        f"Authentication failed for {self.display_name} (HTTP {response.status_code}). "
                        "Please verify that your API key is active and correct.",
                        provider=self.provider_id,
                        status_code=response.status_code,
                    )

                if response.status_code in (429, 529):
                    retry_after_header = response.headers.get("Retry-After")
                    wait_sec = 1.0
                    if retry_after_header and retry_after_header.isdigit():
                        wait_sec = min(float(retry_after_header), 3.0)

                    if attempt < max_retries:
                        time.sleep(wait_sec)
                        continue

                    detail = "Service is at capacity (HTTP 529)" if response.status_code == 529 else "Rate limit reached (HTTP 429)"
                    raise System1QuotaError(
                        f"{self.display_name} {detail}. Free allowance may be exhausted or service is overloaded.",
                        provider=self.provider_id,
                        status_code=response.status_code,
                    )

                if 500 <= response.status_code < 600:
                    if attempt < max_retries:
                        time.sleep(0.5 * (2 ** attempt))
                        continue
                    raise System1Error(
                        f"Provider {self.display_name} service error (HTTP {response.status_code}).",
                        provider=self.provider_id,
                        status_code=response.status_code,
                    )

                # Other 4xx
                error_body = response.text[:200]
                raise System1Error(
                    f"Provider {self.display_name} request rejected (HTTP {response.status_code}): {error_body}",
                    provider=self.provider_id,
                    status_code=response.status_code,
                )

            except (httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError) as net_err:
                last_exception = net_err
                if attempt < max_retries:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                raise System1ConnectionError(
                    f"Could not connect to {self.display_name} at {self.endpoint}: {type(net_err).__name__}",
                    provider=self.provider_id,
                )

        raise System1Error(
            f"Failed to communicate with {self.display_name}: {last_exception}",
            provider=self.provider_id,
        )

    def evaluate_state(self, state: str, max_retries: int = 1) -> Tuple[float, float, str, Dict[str, Any]]:
        """Evaluate a state string against the sufficiency proposition.

        Returns (noul_probability, latency_sec, reported_model, raw_response).
        """
        payload = self.build_payload(state)
        raw_json, latency = self._execute_http_request(payload, max_retries=max_retries)

        try:
            prob, model_name, answer_dict = self.parse_response(raw_json)
            return prob, latency, model_name, raw_json
        except Exception as exc:
            if isinstance(exc, (System1Error, System1ValidationError)):
                raise
            raise System1ValidationError(
                f"Malformed response envelope from {self.display_name}: {exc}",
                provider=self.provider_id,
            )

    def test_connection(self) -> Dict[str, Any]:
        """Perform a minimal, non-sensitive probe request to verify key and connectivity."""
        probe_state = (
            "Lab result: Fasting Blood Glucose: 92 mg/dL. Reference Range: 70 - 99 mg/dL. "
            "Report status: Final."
        )
        prob, latency, model_name, _ = self.evaluate_state(probe_state, max_retries=0)
        return {
            "status": "ok",
            "provider": self.provider_id,
            "display_name": self.display_name,
            "model": model_name,
            "latency_ms": round(latency * 1000, 1),
            "sample_decision_probability": round(prob, 4),
            "free_tier_terms": self.free_tier_terms,
            "verified_at": "2026-10-28",
        }


class CodivAdapter(BaseSystem1Adapter):
    """Adapter for Codiv.ai hosting the OpenJev model.

    OpenJev is an open-source System 1 model served through a Jev-compatible API.
    Identified clearly as OpenJev (not the official TypeSafe Jev model).
    Endpoint: https://api.codiv.ai/v1/systemone
    Request schema includes model: 'openjev-latest'.
    """
    @property
    def provider_id(self) -> str:
        return "codiv"

    @property
    def display_name(self) -> str:
        return SYSTEM1_PROVIDERS["codiv"]["name"]

    @property
    def default_endpoint(self) -> str:
        return SYSTEM1_PROVIDERS["codiv"]["default_endpoint"]

    @property
    def free_tier_terms(self) -> str:
        return SYSTEM1_PROVIDERS["codiv"]["free_tier_terms"]

    def build_payload(self, state: str) -> Dict[str, Any]:
        return {
            "model": "openjev-latest",
            "state": state,
            "questions": {
                DECISION_QUESTION_KEY: {
                    "type": "noul",
                    "instructions": DECISION_INSTRUCTIONS,
                }
            },
        }

    def parse_response(self, raw_json: Dict[str, Any]) -> Tuple[float, str, Dict[str, Any]]:
        if not isinstance(raw_json, dict):
            raise System1ValidationError(f"Expected JSON object, got {type(raw_json).__name__}", provider=self.provider_id)
        model_name = str(raw_json.get("model") or "openjev-latest")
        answers = raw_json.get("answers")
        if not isinstance(answers, dict):
            raise System1ValidationError(f"Missing 'answers' object in Codiv response: {list(raw_json.keys())}", provider=self.provider_id)
        if DECISION_QUESTION_KEY not in answers:
            raise System1ValidationError(f"Question '{DECISION_QUESTION_KEY}' not found in answers: {list(answers.keys())}", provider=self.provider_id)
        ans = answers[DECISION_QUESTION_KEY]
        if not isinstance(ans, dict):
            raise System1ValidationError(f"Expected dict for answer '{DECISION_QUESTION_KEY}'", provider=self.provider_id)
        noul_obj = NoulAnswer(**ans)
        return noul_obj.noul, model_name, ans


class JevModelAdapter(BaseSystem1Adapter):
    """Adapter for JevModel (https://jevmodel.app/jev-api).

    Verified official documentation requirements:
    1. Endpoint: https://jevmodel.app/api/v1/systemone
    2. Fixed model on server: do NOT send 'model' in the request payload.
    3. Noul questions require criteria: { 'true': '...', 'false': '...' }.
    4. Successful response nests answers under data.answers:
       { 'data': { 'model': 'typesafe/jev-1.13', 'answers': { ... } } }
    """
    @property
    def provider_id(self) -> str:
        return "jevmodel"

    @property
    def display_name(self) -> str:
        return SYSTEM1_PROVIDERS["jevmodel"]["name"]

    @property
    def default_endpoint(self) -> str:
        return SYSTEM1_PROVIDERS["jevmodel"]["default_endpoint"]

    @property
    def free_tier_terms(self) -> str:
        return SYSTEM1_PROVIDERS["jevmodel"]["free_tier_terms"]

    def build_payload(self, state: str) -> Dict[str, Any]:
        # Note: 'model' is fixed on the backend and must NOT be included in request
        return {
            "state": state,
            "questions": {
                DECISION_QUESTION_KEY: {
                    "type": "noul",
                    "instructions": DECISION_INSTRUCTIONS,
                    "criteria": {
                        "true": NOUL_CRITERIA_TRUE,
                        "false": NOUL_CRITERIA_FALSE,
                    },
                }
            },
        }

    def parse_response(self, raw_json: Dict[str, Any]) -> Tuple[float, str, Dict[str, Any]]:
        if not isinstance(raw_json, dict):
            raise System1ValidationError(f"Expected JSON object, got {type(raw_json).__name__}", provider=self.provider_id)

        # Official JevModel response envelope nests payload under 'data'
        data_block = raw_json.get("data")
        if isinstance(data_block, dict):
            model_name = str(data_block.get("model") or raw_json.get("model") or "typesafe/jev-1.13")
            answers = data_block.get("answers")
        else:
            # Fallback to top-level if present
            model_name = str(raw_json.get("model") or "typesafe/jev-1.13")
            answers = raw_json.get("answers")

        if not isinstance(answers, dict):
            raise System1ValidationError(
                f"Missing answers in JevModel response (expected under data.answers): {list(raw_json.keys())}",
                provider=self.provider_id,
            )

        if DECISION_QUESTION_KEY not in answers:
            raise System1ValidationError(
                f"Question '{DECISION_QUESTION_KEY}' not found in JevModel answers: {list(answers.keys())}",
                provider=self.provider_id,
            )

        ans = answers[DECISION_QUESTION_KEY]
        if not isinstance(ans, dict):
            raise System1ValidationError(f"Expected dict for answer '{DECISION_QUESTION_KEY}'", provider=self.provider_id)
        noul_obj = NoulAnswer(**ans)
        return noul_obj.noul, model_name, ans


class LayaAdapter(BaseSystem1Adapter):
    """Adapter for Laya Studio (https://laya.studio/docs/api).

    Verified documentation requirements:
    1. Endpoint: https://api.laya.studio/v1/systemone
    2. Model selection: Automatic language routing is supported by omitting the
       'model' parameter in the request payload.
    3. Response envelope: Returns the actual model checkpoint chosen by the backend
       in {"model": "<actual_checkpoint>", "answers": { ... }}.
       HealthRAG records the actual returned model rather than claiming a static identity.
    """
    @property
    def provider_id(self) -> str:
        return "laya"

    @property
    def display_name(self) -> str:
        return SYSTEM1_PROVIDERS["laya"]["name"]

    @property
    def default_endpoint(self) -> str:
        return SYSTEM1_PROVIDERS["laya"]["default_endpoint"]

    @property
    def free_tier_terms(self) -> str:
        return SYSTEM1_PROVIDERS["laya"]["free_tier_terms"]

    def build_payload(self, state: str) -> Dict[str, Any]:
        # Automatic language routing: omit 'model' parameter per official docs
        return {
            "state": state,
            "questions": {
                DECISION_QUESTION_KEY: {
                    "type": "noul",
                    "instructions": DECISION_INSTRUCTIONS,
                }
            },
        }

    def parse_response(self, raw_json: Dict[str, Any]) -> Tuple[float, str, Dict[str, Any]]:
        if not isinstance(raw_json, dict):
            raise System1ValidationError(f"Expected JSON object, got {type(raw_json).__name__}", provider=self.provider_id)

        # Record actual model reported by Laya Studio's router
        model_name = str(raw_json.get("model") or "laya-routed")
        answers = raw_json.get("answers")
        if not isinstance(answers, dict):
            raise System1ValidationError(f"Missing 'answers' in Laya response: {list(raw_json.keys())}", provider=self.provider_id)

        if DECISION_QUESTION_KEY not in answers:
            raise System1ValidationError(f"Question '{DECISION_QUESTION_KEY}' not found in answers: {list(answers.keys())}", provider=self.provider_id)

        ans = answers[DECISION_QUESTION_KEY]
        if not isinstance(ans, dict):
            raise System1ValidationError(f"Expected dict for answer '{DECISION_QUESTION_KEY}'", provider=self.provider_id)
        noul_obj = NoulAnswer(**ans)
        return noul_obj.noul, model_name, ans


def get_system1_adapter(
    provider: str = DEFAULT_SYSTEM1_PROVIDER,
    api_key: str = "",
    endpoint: Optional[str] = None,
    timeout: float = 10.0,
) -> BaseSystem1Adapter:
    """Factory creating the appropriate provider adapter with security validation."""
    prov_key = (provider or "").strip().lower()
    if prov_key == "codiv":
        return CodivAdapter(api_key=api_key, endpoint=endpoint, timeout=timeout)
    if prov_key == "jevmodel":
        return JevModelAdapter(api_key=api_key, endpoint=endpoint, timeout=timeout)
    if prov_key == "laya":
        return LayaAdapter(api_key=api_key, endpoint=endpoint, timeout=timeout)
    raise ValueError(f"Unsupported System 1 provider: '{provider}'. Supported options: {list(SYSTEM1_PROVIDERS.keys())}")


# ─────────────────────────────────────────────────────────────────────────────
# EVIDENCE-SUFFICIENCY DECISION GATE
# ─────────────────────────────────────────────────────────────────────────────

def format_evidence_state(question: str, chunks: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    """Format question and retrieved chunks into a bounded state representation.

    Omits unnecessary internal database metadata to protect document boundaries
    and avoid token-limit rejections.
    """
    formatted_chunks = []
    for i, c in enumerate(chunks, start=1):
        meta = c.get("metadata", {})
        doc_name = meta.get("document_name", "document")
        page = meta.get("page", "?")
        text = c.get("text", "").strip()
        formatted_chunks.append(f"Passage {i} (Source: {doc_name}, Page {page}):\n{text}")

    evidence_text = "\n\n".join(formatted_chunks)
    if len(evidence_text) > max_chars:
        evidence_text = evidence_text[:max_chars] + "\n[Evidence truncated to safe size]"

    return (
        f"User Health Question:\n{question.strip()}\n\n"
        f"Retrieved Document Evidence:\n{evidence_text}"
    )


def evaluate_evidence_sufficiency(
    question: str,
    chunks: List[Dict[str, Any]],
    provider: str = DEFAULT_SYSTEM1_PROVIDER,
    api_key: str = "",
    threshold: float = DEFAULT_SUFFICIENCY_THRESHOLD,
    endpoint: Optional[str] = None,
    max_retries: int = 1,
) -> DecisionResult:
    """Evidence-sufficiency decision gate.

    Conservative Policy:
    - If no chunks were retrieved, returns INSUFFICIENT immediately (0.0).
    - If probability >= threshold, returns SUFFICIENT.
    - If probability < threshold, returns INSUFFICIENT.
    - If provider returns an error, malformed JSON, or fails auth:
      Returns ERROR with conservative is_sufficient=False (never silently bypasses).
    """
    start_time = time.time()
    prov_key = (provider or DEFAULT_SYSTEM1_PROVIDER).strip().lower()
    prov_meta = SYSTEM1_PROVIDERS.get(prov_key, {})
    default_model = prov_meta.get("model", "unknown")

    if not chunks:
        return DecisionResult(
            is_sufficient=False,
            probability=0.0,
            threshold=threshold,
            provider=prov_key,
            model=default_model,
            latency_sec=round(time.time() - start_time, 3),
            status="INSUFFICIENT",
            error_message="No evidence chunks were retrieved.",
            allowance_note=prov_meta.get("free_tier_terms"),
        )

    try:
        adapter = get_system1_adapter(prov_key, api_key=api_key, endpoint=endpoint)
        state = format_evidence_state(question, chunks)
        prob, latency, actual_model, raw_json = adapter.evaluate_state(state, max_retries=max_retries)

        is_sufficient = prob >= threshold
        status = "SUFFICIENT" if is_sufficient else "INSUFFICIENT"

        # Extract question answer dict depending on root vs data envelope
        answers_dict = raw_json.get("data", {}).get("answers") if isinstance(raw_json.get("data"), dict) else raw_json.get("answers", {})

        return DecisionResult(
            is_sufficient=is_sufficient,
            probability=prob,
            threshold=threshold,
            provider=prov_key,
            model=actual_model,
            latency_sec=latency,
            status=status,
            raw_answer=answers_dict.get(DECISION_QUESTION_KEY) if isinstance(answers_dict, dict) else None,
            allowance_note=adapter.free_tier_terms,
        )

    except System1SecurityError as sec_err:
        logger.error("System 1 endpoint security rejection: %s", str(sec_err))
        return DecisionResult(
            is_sufficient=False,
            probability=0.0,
            threshold=threshold,
            provider=prov_key,
            model=default_model,
            latency_sec=round(time.time() - start_time, 3),
            status="ERROR",
            error_message=f"Endpoint Security Violation: {sec_err}",
            allowance_note=prov_meta.get("free_tier_terms"),
        )
    except System1AuthError as auth_err:
        logger.warning("System 1 authentication failed for provider %s: %s", prov_key, str(auth_err))
        return DecisionResult(
            is_sufficient=False,
            probability=0.0,
            threshold=threshold,
            provider=prov_key,
            model=default_model,
            latency_sec=round(time.time() - start_time, 3),
            status="ERROR",
            error_message=str(auth_err),
            allowance_note=prov_meta.get("free_tier_terms"),
        )
    except System1QuotaError as quota_err:
        logger.warning("System 1 quota/rate limit error for provider %s: %s", prov_key, str(quota_err))
        return DecisionResult(
            is_sufficient=False,
            probability=0.0,
            threshold=threshold,
            provider=prov_key,
            model=default_model,
            latency_sec=round(time.time() - start_time, 3),
            status="ERROR",
            error_message=str(quota_err),
            allowance_note=prov_meta.get("free_tier_terms"),
        )
    except (System1ValidationError, System1ConnectionError, System1Error) as sys_err:
        logger.warning("System 1 execution error for provider %s: %s", prov_key, str(sys_err))
        return DecisionResult(
            is_sufficient=False,
            probability=0.0,
            threshold=threshold,
            provider=prov_key,
            model=default_model,
            latency_sec=round(time.time() - start_time, 3),
            status="ERROR",
            error_message=str(sys_err),
            allowance_note=prov_meta.get("free_tier_terms"),
        )
    except Exception as unk_err:
        logger.error("Unexpected System 1 failure: %s", str(unk_err))
        return DecisionResult(
            is_sufficient=False,
            probability=0.0,
            threshold=threshold,
            provider=prov_key,
            model=default_model,
            latency_sec=round(time.time() - start_time, 3),
            status="ERROR",
            error_message=f"Unexpected internal decision error: {type(unk_err).__name__}",
            allowance_note=prov_meta.get("free_tier_terms"),
        )
