"""HealthRAG Evaluation Suite: System 1 Decision Gate vs Baseline.

Supports two evaluation modes:
1. Live Evaluation (--live):
   Executes real end-to-end RAG retrieval, real System 1 decision calls (OpenJev/JevModel/Laya),
   and real System 2 LLM generation (Groq/OpenAI), measuring actual wall-clock latency,
   live token probabilities, and live model completions.
   Requires: SYSTEM1_API_KEY and GROQ_API_KEY (or OPENAI_API_KEY).

2. Deterministic Scenario Simulation (default / offline):
   Runs synthetic scenario cases to verify pipeline logic, decision branching, bounded
   retrieval retry, and safe abstention routing without external API costs or network flakiness.
   Clearly labeled as a deterministic simulation — not empirical benchmark results.

Evaluation Metrics Note (Proxy Metrics):
   - Correctness checks use heuristic substring keyword matching.
   - Faithfulness scores use unigram token overlap with the actual retrieved evidence context
     chunks used (incorporating expanded chunks when retrieval retry occurs).
   - These are proxy metrics for pipeline routing verification. They do NOT evaluate clinical
     accuracy, medical validity, or validated clinical faithfulness.

Usage:
  python -m eval.evaluate_system1            # Offline deterministic simulation
  python -m eval.evaluate_system1 --live     # Live API evaluation (requires API keys)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC BENCHMARK DATASET
# ─────────────────────────────────────────────────────────────────────────────

DEMO_CORPUS_CHUNKS = [
    {
        "id": "chunk-1",
        "text": (
            "HAEMATOLOGY REPORT\n"
            "PATIENT: Mr. Saubhik Bhaumik | AGE/SEX: 28 YRS / M | REG NO: 1015\n"
            "REFERRED BY: Dr. Sachin Patil (MBBS) | DATE: 28/10/2024\n"
            "TEST: HEMOGLOBIN | VALUE: 14 | UNIT: g/dl | REFERENCE: 13 - 17\n"
            "Hemoglobin is the major protein of erythrocytes that transports oxygen from lungs to peripheral tissues."
        ),
        "metadata": {"document_name": "hemoglobin-report.pdf", "page": 1, "document_id": "doc-demo-1"},
    },
    {
        "id": "chunk-2",
        "text": (
            "METHODOLOGY & INTERPRETATION:\n"
            "Measured by spectrophotometry on automated instruments after lysis of red cells and conversion to cyanmethemoglobin.\n"
            "Increased in:\n"
            "1. Hemoconcentration (dehydration, burns, vomiting)\n"
            "2. Polycythemia (erythrocytosis)\n"
            "3. Extreme physical exercise.\n"
            "Decreased in:\n"
            "1. Macrocytic anemia (liver disease, hypothyroidism, vitamin B12 deficiency, folate deficiency)\n"
            "2. Normocytic anemia (early iron deficiency, anemia of chronic disease, acute hemorrhage)\n"
            "3. Microcytic anemia (iron deficiency, thalassemia)."
        ),
        "metadata": {"document_name": "hemoglobin-report.pdf", "page": 1, "document_id": "doc-demo-1"},
    },
    {
        "id": "chunk-3",
        "text": (
            "LABORATORY NOTES & CLINICAL ESCALATION:\n"
            "The cyanmethemoglobin technique is the method of choice selected by ICSH.\n"
            "Hypertriglyceridemia and high white blood cell counts can cause false elevations of Hb.\n"
            "Reported on: 28/10/2024 06:04 PM | Lab Incharge: Mr. Sachin Sharma | Pathologist: Dr. A. K. Asthana, MBBS, MD."
        ),
        "metadata": {"document_name": "hemoglobin-report.pdf", "page": 1, "document_id": "doc-demo-1"},
    },
]

BENCHMARK_CASES = [
    {
        "id": "CASE-1",
        "category": "answerable",
        "question": "What is the patient's hemoglobin value in the lab report?",
        "expected_answerable": True,
        "ground_truth_contains": ["14"],
        "retrieved_chunk_ids": ["chunk-1"],
        "expanded_chunk_ids": ["chunk-1", "chunk-2"],
        "simulated_s1_prob": 0.96,
    },
    {
        "id": "CASE-2",
        "category": "answerable",
        "question": "What is the normal reference range for hemoglobin reported?",
        "expected_answerable": True,
        "ground_truth_contains": ["13", "17"],
        "retrieved_chunk_ids": ["chunk-1"],
        "expanded_chunk_ids": ["chunk-1", "chunk-2"],
        "simulated_s1_prob": 0.94,
    },
    {
        "id": "CASE-3",
        "category": "absent",
        "question": "What was the patient's serum potassium and troponin I level?",
        "expected_answerable": False,
        "ground_truth_contains": ["insufficient", "absent", "not"],
        "retrieved_chunk_ids": ["chunk-1"],
        "expanded_chunk_ids": ["chunk-1", "chunk-2", "chunk-3"],
        "simulated_s1_prob": 0.08,
    },
    {
        "id": "CASE-4",
        "category": "absent",
        "question": "What were the findings on the patient's chest CT scan?",
        "expected_answerable": False,
        "ground_truth_contains": ["insufficient", "absent", "not"],
        "retrieved_chunk_ids": ["chunk-2"],
        "expanded_chunk_ids": ["chunk-1", "chunk-2", "chunk-3"],
        "simulated_s1_prob": 0.04,
    },
    {
        "id": "CASE-5",
        "category": "partially_relevant",
        "question": "What daily dosage of iron supplements was prescribed for this patient?",
        "expected_answerable": False,
        "ground_truth_contains": ["insufficient", "not", "prescribed"],
        "retrieved_chunk_ids": ["chunk-2"],
        "expanded_chunk_ids": ["chunk-1", "chunk-2", "chunk-3"],
        "simulated_s1_prob": 0.18,
    },
    {
        "id": "CASE-6",
        "category": "ambiguous",
        "question": "Is the patient's test result considered dangerous or abnormal?",
        "expected_answerable": True,
        "ground_truth_contains": ["normal", "14", "not abnormal"],
        "retrieved_chunk_ids": ["chunk-1"],
        "expanded_chunk_ids": ["chunk-1", "chunk-2"],
        "simulated_s1_prob": 0.88,
    },
    {
        "id": "CASE-7",
        "category": "contradictory_differential",
        "question": "Does the patient have polycythemia or thalassemia based on this test?",
        "expected_answerable": False,
        "ground_truth_contains": ["insufficient", "normal", "differential"],
        "retrieved_chunk_ids": ["chunk-2"],
        "expanded_chunk_ids": ["chunk-1", "chunk-2"],
        "simulated_s1_prob": 0.22,
    },
    {
        "id": "CASE-8",
        "category": "provider_failure_fallback",
        "question": "Who was the pathologist who signed the report?",
        "expected_answerable": True,
        "ground_truth_contains": ["asthana"],
        "retrieved_chunk_ids": ["chunk-3"],
        "expanded_chunk_ids": ["chunk-1", "chunk-3"],
        "simulated_s1_prob": -1.0,
    },
]


def _calculate_token_overlap(answer: str, chunks: List[Dict[str, Any]]) -> float:
    """Calculate lexical unigram token overlap between generated answer and context chunks (Proxy Metric).

    This calculates lexical token intersection. It is a proximity heuristic only,
    NOT clinically validated faithfulness, factual grounding, or medical correctness.
    """
    if not answer or not chunks:
        return 0.0
    context_tokens = set(re.findall(r"[a-z0-9]+", " ".join(c.get("text", "") for c in chunks).lower()))
    answer_tokens = re.findall(r"[a-z0-9]+", answer.lower())
    if not answer_tokens:
        return 1.0
    matches = sum(1 for t in answer_tokens if t in context_tokens)
    return round(matches / len(answer_tokens), 4)


@dataclass
class TestCaseResult:
    case_id: str
    category: str
    question: str
    expected_answerable: bool
    baseline_answer: str
    baseline_latency: float
    baseline_proxy_faithfulness: float  # Lexical token overlap with actual evidence (Proxy Metric)
    baseline_proxy_correctness: bool   # Substring keyword match (Proxy Metric)
    baseline_unsupported: bool
    baseline_abstained: bool
    proposed_answer: str
    proposed_latency: float
    proposed_system1_prob: float
    proposed_system1_status: str
    proposed_retry_performed: bool
    proposed_proxy_faithfulness: float  # Lexical token overlap with actual evidence (Proxy Metric)
    proposed_proxy_correctness: bool   # Substring keyword match (Proxy Metric)
    proposed_unsupported: bool
    proposed_abstained: bool

    # Backward-compatible property aliases for existing consumers
    @property
    def baseline_correct(self) -> bool:
        return self.baseline_proxy_correctness

    @property
    def baseline_faithfulness(self) -> float:
        return self.baseline_proxy_faithfulness

    @property
    def proposed_correct(self) -> bool:
        return self.proposed_proxy_correctness

    @property
    def proposed_faithfulness(self) -> float:
        return self.proposed_proxy_faithfulness


# ─────────────────────────────────────────────────────────────────────────────
# LIVE EVALUATION (REAL MODELS & RETRIEVAL)
# ─────────────────────────────────────────────────────────────────────────────

def run_live_evaluation(
    system1_provider: str = "codiv",
    system1_key: str = "",
    llm_provider: str = "groq",
    llm_key: str = "",
) -> Dict[str, Any]:
    """Execute live evaluation with actual network calls to System 1 and System 2."""
    from app.rag import answer_question
    from app.vector_store import add_documents

    # Ingest benchmark chunks into dedicated evaluation session
    eval_session = f"eval_live_{int(time.time())}"
    add_documents(DEMO_CORPUS_CHUNKS, session_id=eval_session)

    results: List[TestCaseResult] = []
    chunk_map = {c["id"]: c for c in DEMO_CORPUS_CHUNKS}

    for case in BENCHMARK_CASES:
        q = case["question"]
        expected_ans = case["expected_answerable"]
        initial_chunks = [chunk_map[ch_id] for ch_id in case["retrieved_chunk_ids"]]
        expanded_chunks = [chunk_map[ch_id] for ch_id in case["expanded_chunk_ids"]]

        # 1. Real Baseline Call (System 2 only)
        t0 = time.time()
        b_res = answer_question(
            question=q,
            provider=llm_provider,
            api_key=llm_key,
            session_id=eval_session,
            system1_enabled=False,
        )
        b_lat = round(time.time() - t0, 3)
        b_ans = b_res.get("answer", "")
        # Measure token overlap against the actual retrieved context chunks used by the RAG pipeline
        b_faith = b_res.get("metrics", {}).get(
            "answer_context_token_overlap_heuristic",
            _calculate_token_overlap(b_ans, initial_chunks)
        )
        b_abstained = "insufficient" in b_ans.lower() or "not find any relevant" in b_ans.lower()
        if expected_ans:
            b_correct = all(kw in b_ans.lower() for kw in case["ground_truth_contains"]) and not b_abstained
            b_unsupported = False
        else:
            b_correct = b_abstained
            b_unsupported = not b_abstained

        # 2. Real Proposed Call (System 1 Gate + System 2)
        t1 = time.time()
        p_res = answer_question(
            question=q,
            provider=llm_provider,
            api_key=llm_key,
            session_id=eval_session,
            system1_enabled=True,
            system1_provider=system1_provider,
            system1_api_key=system1_key,
        )
        p_lat = round(time.time() - t1, 3)
        p_ans = p_res.get("answer", "")
        s1_meta = p_res.get("metrics", {}).get("system1_decision", {})
        p_prob = s1_meta.get("probability", 0.0)
        p_status = s1_meta.get("status", "UNKNOWN")
        p_retry = s1_meta.get("retry_performed", False)
        p_abstained = p_res.get("abstained_by_system1", False) or "insufficient" in p_ans.lower()

        # Ensure correct evidence context is used for evaluation:
        # If retrieval was expanded during retry, evaluation must use the expanded evidence set.
        # The RAG pipeline computes 'answer_context_token_overlap_heuristic' directly on the
        # active relevant chunks (whether initial or expanded).
        actual_proposed_chunks = expanded_chunks if p_retry else initial_chunks
        p_faith = p_res.get("metrics", {}).get(
            "answer_context_token_overlap_heuristic",
            _calculate_token_overlap(p_ans, actual_proposed_chunks)
        )
        if expected_ans:
            p_correct = all(kw in p_ans.lower() for kw in case["ground_truth_contains"]) and not p_abstained
            p_unsupported = False
        else:
            p_correct = p_abstained
            p_unsupported = not p_abstained

        results.append(TestCaseResult(
            case_id=case["id"],
            category=case["category"],
            question=q,
            expected_answerable=expected_ans,
            baseline_answer=b_ans,
            baseline_latency=b_lat,
            baseline_proxy_faithfulness=b_faith,
            baseline_proxy_correctness=b_correct,
            baseline_unsupported=b_unsupported,
            baseline_abstained=b_abstained,
            proposed_answer=p_ans,
            proposed_latency=p_lat,
            proposed_system1_prob=p_prob,
            proposed_system1_status=p_status,
            proposed_retry_performed=p_retry,
            proposed_proxy_faithfulness=p_faith,
            proposed_proxy_correctness=p_correct,
            proposed_unsupported=p_unsupported,
            proposed_abstained=p_abstained,
        ))

    return _aggregate_results(results, mode="live_api_evaluation")


# ─────────────────────────────────────────────────────────────────────────────
# DETERMINISTIC SCENARIO SIMULATION (FOR OFFLINE PIPELINE VERIFICATION)
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation() -> Dict[str, Any]:
    """Execute deterministic scenario simulation to verify branching logic."""
    results: List[TestCaseResult] = []
    chunk_map = {c["id"]: c for c in DEMO_CORPUS_CHUNKS}

    for case in BENCHMARK_CASES:
        cid = case["id"]
        q = case["question"]
        expected_ans = case["expected_answerable"]
        initial_chunks = [chunk_map[ch_id] for ch_id in case["retrieved_chunk_ids"]]
        expanded_chunks = [chunk_map[ch_id] for ch_id in case["expanded_chunk_ids"]]
        sim_prob = case["simulated_s1_prob"]

        # Simulated baseline
        if "hemoglobin" in q.lower() and "value" in q.lower():
            b_ans = "Based on the hematology report, the patient's hemoglobin value is 14 g/dl (Page 1)."
        elif "reference range" in q.lower():
            b_ans = "The reference range for hemoglobin in this report is 13 - 17 g/dl."
        elif "pathologist" in q.lower():
            b_ans = "The report was signed by pathologist Dr. A. K. Asthana, MBBS, MD (Page 1)."
        elif "dangerous or abnormal" in q.lower():
            b_ans = "The hemoglobin level of 14 g/dl is within the normal reference range (13 - 17 g/dl) and is not abnormal."
        elif "potassium" in q.lower() or "troponin" in q.lower():
            b_ans = "The patient's potassium level appears to be in the normal range around 4.0 mEq/L and troponin is negative."
        elif "ct scan" in q.lower():
            b_ans = "The chest CT scan shows clear lungs with no acute cardiopulmonary disease noted."
        elif "dosage of iron" in q.lower():
            b_ans = "A typical iron supplement dosage of ferrous sulfate 325 mg once daily is recommended for microcytic anemia."
        else:
            b_ans = "The report notes decreased hemoglobin in thalassemia and increased in polycythemia, suggesting mild thalassemia trait."

        b_lat = 0.465
        # Ensure correct evidence context is used: baseline retrieves initial_chunks
        b_faith = _calculate_token_overlap(b_ans, initial_chunks)
        b_abstained = "insufficient" in b_ans.lower()
        if expected_ans:
            b_correct = all(kw in b_ans.lower() for kw in case["ground_truth_contains"]) and not b_abstained
            b_unsupported = False
        else:
            b_correct = b_abstained
            b_unsupported = not b_abstained

        # Simulated proposed
        p_retry = False
        if sim_prob < 0:
            p_status = "ERROR"
            p_ans = "The System 1 evidence-sufficiency gate was unable to verify the evidence due to a provider issue (HTTP 429)."
            p_lat = 0.120
            p_abstained = True
            p_correct = not expected_ans
            p_unsupported = False
            p_faith = 1.0
            p_prob = 0.0
        elif sim_prob < 0.5:
            p_retry = True
            p_status = "INSUFFICIENT"
            p_ans = "Based on the available healthcare documents in your selected scope, there is insufficient evidence to answer your question accurately."
            p_lat = 0.095
            p_abstained = True
            p_correct = not expected_ans
            p_unsupported = False
            p_faith = 1.0
            p_prob = sim_prob
        else:
            p_status = "SUFFICIENT"
            p_ans = b_ans
            p_lat = 0.492
            p_abstained = False
            p_correct = all(kw in p_ans.lower() for kw in case["ground_truth_contains"])
            p_unsupported = False
            # Ensure correct evidence context is used: if retry occurred, use expanded_chunks
            actual_p_chunks = expanded_chunks if p_retry else initial_chunks
            p_faith = _calculate_token_overlap(p_ans, actual_p_chunks)
            p_prob = sim_prob

        results.append(TestCaseResult(
            case_id=cid,
            category=case["category"],
            question=q,
            expected_answerable=expected_ans,
            baseline_answer=b_ans,
            baseline_latency=b_lat,
            baseline_proxy_faithfulness=b_faith,
            baseline_proxy_correctness=b_correct,
            baseline_unsupported=b_unsupported,
            baseline_abstained=b_abstained,
            proposed_answer=p_ans,
            proposed_latency=p_lat,
            proposed_system1_prob=p_prob,
            proposed_system1_status=p_status,
            proposed_retry_performed=p_retry,
            proposed_proxy_faithfulness=p_faith,
            proposed_proxy_correctness=p_correct,
            proposed_unsupported=p_unsupported,
            proposed_abstained=p_abstained,
        ))

    return _aggregate_results(results, mode="deterministic_scenario_simulation")


def _aggregate_results(results: List[TestCaseResult], mode: str) -> Dict[str, Any]:
    total = len(results)
    unans = [r for r in results if not r.expected_answerable]
    ans = [r for r in results if r.expected_answerable]

    b_corr = sum(1 for r in results if r.baseline_proxy_correctness)
    p_corr = sum(1 for r in results if r.proposed_proxy_correctness)
    b_unsupp = sum(1 for r in unans if r.baseline_unsupported)
    p_unsupp = sum(1 for r in unans if r.proposed_unsupported)
    b_abst = sum(1 for r in unans if r.baseline_abstained)
    p_abst = sum(1 for r in unans if r.proposed_abstained)
    b_false_abst = sum(1 for r in ans if r.baseline_abstained)
    p_false_abst = sum(1 for r in ans if r.proposed_abstained)
    retry_cnt = sum(1 for r in results if r.proposed_retry_performed)
    b_mean_faith = round(sum(r.baseline_proxy_faithfulness for r in results) / total, 4) if total else 0.0
    p_mean_faith = round(sum(r.proposed_proxy_faithfulness for r in results) / total, 4) if total else 0.0

    summary = {
        "mode": mode,
        "is_empirical_benchmark": (mode == "live_api_evaluation"),
        "disclaimer": (
            "Empirical results obtained via live API execution using heuristic proxy metrics. "
            "Keyword presence and unigram token overlap are heuristic proxies and DO NOT constitute "
            "clinical accuracy, medical validity, or validated faithfulness."
            if mode == "live_api_evaluation"
            else "Notice: Deterministic scenario simulation for routing logic verification; NOT an empirical benchmark measurement."
        ),
        "proxy_metric_definitions": {
            "proxy_keyword_correctness": (
                "Evaluates presence of expected key tokens in the answer string. "
                "Proxy heuristic only; does NOT assess medical or clinical accuracy."
            ),
            "proxy_token_overlap_faithfulness": (
                "Evaluates unigram lexical overlap between generated response and the actual retrieved "
                "evidence context chunks used (including expanded chunks if retry occurred). "
                "Proximity heuristic only; does NOT assess factual truthfulness or clinical grounding."
            ),
        },
        "total_test_cases": total,
        "answerable_count": len(ans),
        "unanswerable_count": len(unans),
        "proxy_metrics_summary": {
            "proxy_keyword_correctness": {"baseline": f"{b_corr}/{total}", "proposed": f"{p_corr}/{total}"},
            "unsupported_answer_rate": {"baseline": f"{b_unsupp}/{len(unans)}", "proposed": f"{p_unsupp}/{len(unans)}"},
            "appropriate_abstention_rate": {"baseline": f"{b_abst}/{len(unans)}", "proposed": f"{p_abst}/{len(unans)}"},
            "false_abstention_rate": {"baseline": f"{b_false_abst}/{len(ans)}", "proposed": f"{p_false_abst}/{len(ans)}"},
            "retrieval_retry_rate": {"proposed": f"{retry_cnt}/{total}"},
            "mean_proxy_faithfulness": {"baseline": b_mean_faith, "proposed": p_mean_faith},
        },
        # For backward compatibility with existing test scripts:
        "simulated_scenarios": {
            "answer_correctness": {"baseline": f"{b_corr}/{total}", "proposed": f"{p_corr}/{total}"},
            "proxy_keyword_correctness": {"baseline": f"{b_corr}/{total}", "proposed": f"{p_corr}/{total}"},
            "unsupported_answer_rate": {"baseline": f"{b_unsupp}/{len(unans)}", "proposed": f"{p_unsupp}/{len(unans)}"},
            "appropriate_abstention_rate": {"baseline": f"{b_abst}/{len(unans)}", "proposed": f"{p_abst}/{len(unans)}"},
            "false_abstention_rate": {"baseline": f"{b_false_abst}/{len(ans)}", "proposed": f"{p_false_abst}/{len(ans)}"},
            "retrieval_retry_rate": {"proposed": f"{retry_cnt}/{total}"},
        },
    }
    return {"summary": summary, "cases": [asdict(r) for r in results]}


def run_evaluation(mode: str = "simulation") -> Dict[str, Any]:
    """Unified entry point."""
    if mode == "live":
        s1_key = os.getenv("SYSTEM1_API_KEY", "")
        llm_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        if not s1_key or not llm_key:
            print("[WARN] SYSTEM1_API_KEY or GROQ_API_KEY missing. Falling back to deterministic simulation.")
            return run_simulation()
        return run_live_evaluation(
            system1_provider=os.getenv("SYSTEM1_DEFAULT_PROVIDER", "codiv"),
            system1_key=s1_key,
            llm_provider="groq" if os.getenv("GROQ_API_KEY") else "openai",
            llm_key=llm_key,
        )
    return run_simulation()


def main():
    parser = argparse.ArgumentParser(description="HealthRAG System 1 Decision Gate Evaluation")
    parser.add_argument("--live", action="store_true", help="Run live evaluation against actual APIs")
    args = parser.parse_args()

    mode = "live" if args.live else "simulation"
    eval_data = run_evaluation(mode=mode)
    summary = eval_data["summary"]

    print("\n" + "=" * 80)
    if summary["is_empirical_benchmark"]:
        print(" HEALTHRAG LIVE API EVALUATION (REAL MODELS & RETRIEVAL)")
        print(" Notice: Correctness and Faithfulness are evaluated using heuristic proxy metrics.")
        print("         Do not interpret these results as validated clinical accuracy.")
    else:
        print(" HEALTHRAG DETERMINISTIC SCENARIO SIMULATION")
        print(" Notice: Architectural simulation for routing logic; NOT empirical measurements.")
    print("=" * 80)
    print(f"Mode: {summary['mode']} | Test Cases: {summary['total_test_cases']}\n")

    metrics_dict = summary.get("proxy_metrics_summary", summary.get("simulated_scenarios", {}))
    for k, v in metrics_dict.items():
        label = k.replace("_", " ").title()
        print(f"  {label:<32}: Baseline={v.get('baseline', 'N/A')} | Proposed={v.get('proposed', 'N/A')}")

    os.makedirs("data", exist_ok=True)
    with open("data/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2)

    report_md = f"""# HealthRAG System 1 Decision Gate: Evaluation & Simulation Documentation

**Evaluation Mode:** {summary['mode']}  
**Status:** {'Empirical live API run (Proxy Metrics)' if summary['is_empirical_benchmark'] else 'Deterministic scenario simulation for logic verification'}  
**Disclaimer:** {summary['disclaimer']}

## Proxy Metric Definitions & Limitations

> [!IMPORTANT]
> The metrics reported below are **proxy indicators** used solely for pipeline flow and routing verification:
> - **Proxy Keyword Correctness:** Evaluates whether predefined substring tokens appear in the generated output. It does **not** measure clinical accuracy, medical correctness, or diagnostic soundness.
> - **Proxy Token Overlap Faithfulness:** Measures unigram lexical overlap between generated response tokens and the actual evidence context chunks used (accounting for expanded retrieval when retried). It is a lexical proximity heuristic, **not** clinically validated faithfulness or evidence grounding.
> - **No Medical Accuracy Claims:** Neither scenario simulations nor live proxy runs prove clinical efficacy, diagnostic safety, or complete elimination of hallucinations.

## Methodology & Architectural Hypotheses

HealthRAG integrates a fast System 1 non-autoregressive decision model as an evidence sufficiency gate prior to invoking the System 2 LLM.

### Hypotheses Evaluated:
1. **Evidence-Sufficiency Gate:** Can a fast decision model evaluate retrieved passages and flag when critical information is missing?
2. **Abstention Behavior:** Does gating prevent unsupported generation on out-of-domain or absent metrics?
3. **Bounded Retrieval Expansion:** Can low initial confidence trigger a single bounded retry to recover missed evidence?
4. **Conservative Fallback:** When the external decision provider encounters a rate limit or network error, does the system safely halt rather than silently generating unverified claims?

## Test Scenarios & Logic Verification Results

| Scenario Category | Test Focus | Baseline Behavior | Gated Behavior |
| :--- | :--- | :--- | :--- |
| **Answerable (Cases 1 & 2)** | Hemoglobin values & reference ranges | Generates answer | Passes gate -> Generates answer |
| **Absent Metrics (Cases 3 & 4)** | Unmeasured lab tests (potassium, CT) | Tends to hallucinate | Fails gate -> Safely abstains |
| **Partial Evidence (Case 5)** | Mentions condition, missing prescription | Generates advice | Fails gate -> Safely abstains |
| **Ambiguous Scope (Case 6)** | Borderline normality check | Generates answer | Passes gate -> Generates answer |
| **Contradictory / Noise (Case 7)** | Differential diagnosis list | Selects diagnosis | Fails gate -> Safely abstains |
| **Provider Failure (Case 8)** | HTTP 429 / Timeout | N/A | Conservative safety halt |

*Note: Clinical validity must be established through ongoing evaluation on licensed, de-identified clinical datasets with independent physician review.*
"""
    with open("data/evaluation_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)
    print("\n[INFO] Saved results to data/evaluation_results.json and data/evaluation_report.md")


if __name__ == "__main__":
    main()
