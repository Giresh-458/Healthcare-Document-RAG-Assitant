# HealthRAG System 1 Decision Gate: Evaluation & Simulation Documentation

**Evaluation Mode:** deterministic_scenario_simulation  
**Status:** Deterministic scenario simulation for logic verification  
**Disclaimer:** Notice: Deterministic scenario simulation for architectural verification; not an empirical live benchmark measurement.

## Methodology & Architectural Hypotheses

HealthRAG integrates a fast System 1 non-autoregressive decision model as a gate prior to invoking the System 2 LLM.

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
