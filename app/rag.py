"""Grounded RAG orchestration with explicit patient/document scope."""
import re
import time
from typing import Any, Dict, Optional, List

from langchain_core.messages import HumanMessage, SystemMessage
from app.hybrid_retriever import HybridRetriever
from app.metrics import RagMetrics
from app.query_router import QueryRouter
from app.config import TOP_K
from app.schemas import MedicalAnswer
from app.system1 import evaluate_evidence_sufficiency, DecisionResult, DEFAULT_SUFFICIENCY_THRESHOLD


def get_llm(provider: str, api_key: str):
    """Instantiate the selected provider model."""
    if not api_key.strip():
        raise ValueError(f"API key for {provider} is required.")
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(api_key=api_key, model_name="openai/gpt-oss-120b", temperature=0)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-1.5-flash", temperature=0)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(api_key=api_key, model_name="gpt-4o-mini", temperature=0)
    raise ValueError(f"Unsupported AI provider: {provider}")


def _source(chunk):
    metadata = chunk["metadata"]
    return {
        "document": metadata.get("document_name", "unknown"),
        "document_id": metadata.get("document_id", ""),
        "patient_label": metadata.get("patient_label", "unspecified"),
        "page": metadata.get("page", "?"),
    }


def _all_hemoglobin_candidates(question: str, chunks: List[Dict], start_time: float):
    """Deterministic, transparent candidate scan; values still require source verification."""
    value_patterns = [
        re.compile(r"\b(?:hemoglobin|haemoglobin|hgb|hb)\b\s*(?:[:=]\s*)?(\d{1,3}(?:\.\d+)?)(?!\s*[-–]\s*\d)\s*(g\s*/\s*d?l|g\s*/\s*l)?\b", re.I),
        re.compile(r"\b(?:hemoglobin|haemoglobin|hgb|hb)\b\s*\([^\)\n]{1,20}\)\s*[:=]?\s*(\d{1,3}(?:\.\d+)?)(?!\s*[-–]\s*\d)\s*(g\s*/\s*d?l|g\s*/\s*l)?\b", re.I),
        re.compile(r"\b(\d{1,3}(?:\.\d+)?)\s*(g\s*/\s*d?l|g\s*/\s*l)\s*(?:of\s+)?(?:hemoglobin|haemoglobin|hgb|hb)\b", re.I),
    ]
    rows, seen = [], set()
    for chunk in chunks:
        metadata = chunk["metadata"]
        for pattern_index, pattern in enumerate(value_patterns):
            for match in pattern.finditer(chunk["text"]):
                if pattern_index == 0:
                    value = match.group(1)
                    unit = (match.group(2) or "unit not stated").replace(" ", "")
                else:
                    value = match.group(1)
                    unit = match.group(2).replace(" ", "")
                key = (metadata.get("document_id"), metadata.get("page"), value, unit.lower())
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "patient_label": metadata.get("patient_label", "unspecified"),
                    "value": value,
                    "unit": unit,
                    "document": metadata.get("document_name", "unknown"),
                    "document_id": metadata.get("document_id", ""),
                    "page": metadata.get("page", "?"),
                    "evidence": match.group(0).strip(),
                })
    rows.sort(key=lambda row: (str(row["patient_label"]).casefold(), str(row["document"]).casefold(), str(row["page"])))
    if rows:
        lines = [
            "Candidate hemoglobin values found by scanning all accessible indexed chunks (not just top-k retrieval). These are text matches, not clinically validated results; verify each cited report.",
            f"Scanned {len({c['metadata'].get('document_id') or c['metadata'].get('document_name') for c in chunks})} indexed document(s); found {len(rows)} candidate value(s).",
            "",
        ]
        for row in rows:
            lines.append(
                f"- Patient label: {row['patient_label']} | Hemoglobin candidate: {row['value']} {row['unit']} | "
                f"Document: {row['document']} | Page: {row['page']} | Evidence: “{row['evidence']}”"
            )
        answer = "\n".join(lines)
    else:
        answer = (
            f"Scanned {len({c['metadata'].get('document_id') or c['metadata'].get('document_name') for c in chunks})} "
            "accessible indexed document(s), but found no reliably patterned hemoglobin value. "
            "The scan may miss unusual table layouts or OCR errors; inspect the source reports."
        )
    sources = []
    seen_sources = set()
    for row in rows:
        key = (row["document_id"], row["document"], row["page"])
        if key not in seen_sources:
            seen_sources.add(key)
            sources.append({"document": row["document"], "document_id": row["document_id"], "patient_label": row["patient_label"], "page": row["page"]})
    return {
        "answer": answer,
        "sources": sources,
        "metrics": {
            "latency_sec": round(time.time() - start_time, 2),
            "retrieved_chunks": len(chunks),
            "documents_scanned": len({c["metadata"].get("document_id") or c["metadata"].get("document_name") for c in chunks}),
            "aggregation_method": "full-corpus regex candidate scan",
        },
        "route": "EXHAUSTIVE_HEMOGLOBIN_SCAN",
        "structured_data": {"confidence": "low", "medical_entities": ["hemoglobin"], "requires_doctor_review": True},
    }


def answer_question(
    question: str,
    provider: str,
    api_key: str,
    document_names: Optional[list] = None,
    session_id: str = "global",
    document_ids: Optional[List[str]] = None,
    patient_label: Optional[str] = None,
    system1_enabled: bool = False,
    system1_provider: str = "codiv",
    system1_api_key: Optional[str] = None,
    system1_threshold: float = DEFAULT_SUFFICIENCY_THRESHOLD,
    system1_endpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """Answer from the selected session and optional patient/document scope.

    Workflow:
    User Question -> Retrieve Evidence -> System 1 Decision Gate (if enabled)
    -> System 2 LLM generates grounded answer -> Display answer with sources.
    """
    start_time = time.time()

    # Fast configuration check: If System 1 is enabled, key cannot be missing or blank
    if system1_enabled:
        if not system1_api_key or not system1_api_key.strip():
            raise ValueError(
                f"System 1 evidence decision gate is enabled, but no API key was provided for provider '{system1_provider}'. "
                "Please configure a valid System 1 API key or disable the System 1 gate to proceed with baseline System 2."
            )

    asks_all = bool(re.search(r"\b(all|every|across|each)\b", question, re.I))
    asks_hemoglobin = bool(re.search(r"\b(hemoglobin|haemoglobin|hgb|hb)\b", question, re.I))

    # Explicit all-record hemoglobin questions use a full-corpus scan, not top-k RAG.
    if asks_all and asks_hemoglobin:
        chunks = HybridRetriever.get_all_chunks(
            document_names=document_names, session_id=session_id,
            document_ids=document_ids, patient_label=patient_label,
        )
        if not chunks:
            return {
                "answer": "No indexed chunks were found in the selected scope.",
                "sources": [], "metrics": {"retrieved_chunks": 0},
                "route": "EXHAUSTIVE_HEMOGLOBIN_SCAN",
                "structured_data": {"confidence": "low", "medical_entities": ["hemoglobin"], "requires_doctor_review": True},
            }
        return _all_hemoglobin_candidates(question, chunks, start_time)

    chat = get_llm(provider, api_key)
    route = QueryRouter.classify(question, chat)
    if route == "IRRELEVANT":
        return {
            "answer": "This question does not appear to be related to the uploaded healthcare documents. Please ask a medical or report-related question.",
            "sources": [],
            "metrics": {"latency_sec": round(time.time() - start_time, 2), "query_token_overlap_heuristic": 0, "answer_context_token_overlap_heuristic": 0, "retrieved_chunks": 0},
            "route": route,
        }

    k_value = TOP_K * 3 if route == "AGGREGATION" else TOP_K
    relevant_chunks = HybridRetriever.search(
        question, top_k=k_value, document_names=document_names, session_id=session_id,
        document_ids=document_ids, patient_label=patient_label,
    )
    if not relevant_chunks:
        return {
            "answer": "I could not find any relevant context in the selected document and patient scope.",
            "sources": [],
            "metrics": {"latency_sec": round(time.time() - start_time, 2), "query_token_overlap_heuristic": 0, "answer_context_token_overlap_heuristic": 0, "retrieved_chunks": 0},
            "route": route,
            "structured_data": {"confidence": "low", "medical_entities": [], "requires_doctor_review": True},
        }

    # ─────────────────────────────────────────────────────────────────────────
    # SYSTEM 1 EVIDENCE-SUFFICIENCY DECISION GATE
    # ─────────────────────────────────────────────────────────────────────────
    system1_decision_dict: Optional[Dict[str, Any]] = None

    if system1_enabled:
        decision = evaluate_evidence_sufficiency(
            question=question,
            chunks=relevant_chunks,
            provider=system1_provider,
            api_key=system1_api_key.strip(),
            threshold=system1_threshold,
            endpoint=system1_endpoint,
        )

        # Conservative Decision Handling Policy:
        # B. If evidence is INSUFFICIENT, attempt a bounded retrieval improvement ONCE.
        if decision.status == "INSUFFICIENT":
            expanded_k = k_value * 2
            expanded_chunks = HybridRetriever.search(
                question, top_k=expanded_k, document_names=document_names,
                session_id=session_id, document_ids=document_ids, patient_label=patient_label,
            )
            # If new chunks were uncovered, evaluate once more with System 1
            if len(expanded_chunks) > len(relevant_chunks):
                retry_decision = evaluate_evidence_sufficiency(
                    question=question,
                    chunks=expanded_chunks,
                    provider=system1_provider,
                    api_key=system1_api_key.strip(),
                    threshold=system1_threshold,
                    endpoint=system1_endpoint,
                )
                retry_decision.retry_performed = True
                if retry_decision.status == "SUFFICIENT":
                    relevant_chunks = expanded_chunks
                    decision = retry_decision
                else:
                    decision = retry_decision

            # If still insufficient after bounded retry: abstain with clear clinical guidance
            if decision.status == "INSUFFICIENT":
                sources, seen = [], set()
                for chunk in relevant_chunks:
                    source = _source(chunk)
                    key = (source["document_id"], source["document"], source["page"])
                    if key not in seen:
                        seen.add(key)
                        sources.append(source)

                abstention_answer = (
                    "Based on the available healthcare documents in your selected scope, "
                    "there is insufficient evidence to answer your question accurately.\n\n"
                    "Clinical Recommendation: Please consult a qualified healthcare professional "
                    "or physician for medical advice, diagnosis, or interpretation of health records. "
                    "You may also verify whether the uploaded document contains the specific lab test or report section."
                )
                total_latency = round(time.time() - start_time, 2)
                metrics = RagMetrics.log_metrics(question, abstention_answer, relevant_chunks, total_latency)
                metrics["system1_decision"] = decision.to_dict()

                return {
                    "answer": abstention_answer,
                    "sources": sources,
                    "metrics": metrics,
                    "route": route,
                    "structured_data": {
                        "confidence": "low",
                        "medical_entities": [],
                        "requires_doctor_review": True,
                        "system1_decision": decision.to_dict(),
                    },
                    "abstained_by_system1": True,
                }

        # C & D. If System 1 returned an error or malformed response:
        # Conservative policy halts unverified generation and provides clear recovery guidance.
        if decision.status == "ERROR":
            sources, seen = [], set()
            for chunk in relevant_chunks:
                source = _source(chunk)
                key = (source["document_id"], source["document"], source["page"])
                if key not in seen:
                    seen.add(key)
                    sources.append(source)

            error_notice = (
                f"The System 1 evidence-sufficiency gate was unable to verify the evidence "
                f"due to a provider issue ({decision.error_message}). "
                "Under HealthRAG's conservative safety policy, unverified answer generation was prevented.\n\n"
                "Recovery Guidance:\n"
                "- Verify your System 1 API key in the sidebar configuration.\n"
                "- Check the provider status and your remaining free quota.\n"
                "- Alternatively, uncheck the System 1 gate to test the baseline System 2 LLM directly.\n\n"
                "Clinical Recommendation: Always consult a licensed healthcare provider for clinical interpretation."
            )
            total_latency = round(time.time() - start_time, 2)
            metrics = RagMetrics.log_metrics(question, error_notice, relevant_chunks, total_latency)
            metrics["system1_decision"] = decision.to_dict()

            return {
                "answer": error_notice,
                "sources": sources,
                "metrics": metrics,
                "route": route,
                "structured_data": {
                    "confidence": "low",
                    "medical_entities": [],
                    "requires_doctor_review": True,
                    "system1_decision": decision.to_dict(),
                },
                "system1_error": True,
            }

        # Case A: SUFFICIENT
        system1_decision_dict = decision.to_dict()

    # ─────────────────────────────────────────────────────────────────────────
    # SYSTEM 2 LLM GROUNDED ANSWER GENERATION
    # ─────────────────────────────────────────────────────────────────────────
    context_text = "\n\n---\n\n".join(
        [f"Source: {chunk['metadata']['document_name']} (Patient label: {chunk['metadata'].get('patient_label', 'unspecified')}; Page {chunk['metadata']['page']})\nContent:\n{chunk['text']}" for chunk in relevant_chunks]
    )
    system_prompt = (
        "You are a healthcare document assistant. Answer only from the provided document context. "
        "Do not invent information. If the answer is absent, say so. Cite source document and page. "
        "Do not infer that the uploader is the patient. Patient labels are user-supplied metadata, not verified identity. "
        "Do not combine different patients' results unless the user explicitly asks for a multi-patient summary. "
        "You are not a doctor and must not diagnose or replace professional medical advice."
    )
    user_prompt = f"Context documents:\n{context_text}\n\nQuestion: {question}"
    try:
        structured_response = chat.with_structured_output(MedicalAnswer).invoke([
            SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)
        ])
        answer_text = structured_response.answer
        structured_data = {
            "confidence": structured_response.confidence,
            "medical_entities": structured_response.medical_entities,
            "requires_doctor_review": structured_response.requires_doctor_review,
        }
    except Exception:
        response = chat.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        answer_text = response.content
        structured_data = {"confidence": "unknown", "medical_entities": [], "requires_doctor_review": True}

    if system1_decision_dict:
        structured_data["system1_decision"] = system1_decision_dict

    sources, seen = [], set()
    for chunk in relevant_chunks:
        source = _source(chunk)
        key = (source["document_id"], source["document"], source["page"])
        if key not in seen:
            seen.add(key)
            sources.append(source)
    latency = round(time.time() - start_time, 2)
    metrics = RagMetrics.log_metrics(question, answer_text, relevant_chunks, latency)
    if system1_decision_dict:
        metrics["system1_decision"] = system1_decision_dict

    return {"answer": answer_text, "sources": sources, "metrics": metrics, "route": route, "structured_data": structured_data}
