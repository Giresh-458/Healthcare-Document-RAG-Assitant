import pytest

from app.rag import _resolve_contextual_followup


@pytest.mark.parametrize(
    "followup",
    [
        "from the uploaded records",
        "from uploaded records",
        "in the selected documents",
        "the uploaded reports",
    ],
)
def test_document_reference_followup_reuses_previous_clinical_question(followup):
    history = [
        {"role": "user", "content": "what is the hemoglobin level?"},
        {"role": "assistant", "content": "Which patient's or report's hemoglobin level would you like to know?"},
    ]

    resolved = _resolve_contextual_followup(followup, history)

    assert "hemoglobin" in resolved.casefold()
    assert "selected uploaded records" in resolved.casefold()


def test_context_only_followup_does_not_inherit_unrelated_or_stale_question():
    history = [
        {"role": "user", "content": "What is the hemoglobin level?"},
        {"role": "assistant", "content": "Which report?"},
        {"role": "user", "content": "Explain Python decorators."},
    ]

    assert _resolve_contextual_followup("from the uploaded records", history) == "from the uploaded records"


def test_standalone_healthcare_question_is_not_rewritten():
    question = "What is the hemoglobin level?"
    history = [{"role": "user", "content": "What is the RBC count?"}]

    assert _resolve_contextual_followup(question, history) == question


def test_context_only_followup_without_history_remains_unresolved():
    question = "from the uploaded records"

    assert _resolve_contextual_followup(question, []) == question
