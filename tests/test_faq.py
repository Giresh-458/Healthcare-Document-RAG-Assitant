"""Unit tests for the FAQ / Help module and configuration."""

import pytest
from app.faq_content import (
    OFFICIAL_LINKS,
    SUPPORTED_PROVIDERS,
    COMMON_PROBLEMS_FAQS,
    get_faq_data,
)


def test_faq_links_match_requirements():
    """Verify official links match user specifications exactly."""
    expected_links = {
        "groq": "https://console.groq.com/",
        "openai": "https://platform.openai.com/api-keys",
        "gemini": "https://aistudio.google.com/apikey",
        "codiv": "https://codiv.ai/",
        "jevmodel": "https://jevmodel.app/",
        "laya": "https://laya.studio/",
    }
    for provider, url in expected_links.items():
        assert OFFICIAL_LINKS[provider] == url, f"Mismatch for {provider}"


def test_system1_models_exact_configuration():
    """Verify System 1 includes only models configured in this repository."""
    s1_providers = SUPPORTED_PROVIDERS["system1"]
    assert len(s1_providers) == 3
    s1_ids = {p["id"] for p in s1_providers}
    assert s1_ids == {"codiv", "jevmodel", "laya"}

    codiv_entry = next(p for p in s1_providers if p["id"] == "codiv")
    assert codiv_entry["model"] == "openjev-latest"

    jev_entry = next(p for p in s1_providers if p["id"] == "jevmodel")
    assert jev_entry["model"] == "typesafe/jev-1.13"

    laya_entry = next(p for p in s1_providers if p["id"] == "laya")
    assert "automatic" in laya_entry["model"]


def test_system2_models_exact_configuration():
    """Verify System 2 includes only models configured in this repository."""
    s2_providers = SUPPORTED_PROVIDERS["system2"]
    assert len(s2_providers) == 3
    s2_ids = {p["id"] for p in s2_providers}
    assert s2_ids == {"groq", "gemini", "openai"}

    groq_entry = next(p for p in s2_providers if p["id"] == "groq")
    assert groq_entry["model"] == "openai/gpt-oss-120b"

    gemini_entry = next(p for p in s2_providers if p["id"] == "gemini")
    assert gemini_entry["model"] == "gemini-1.5-flash"

    openai_entry = next(p for p in s2_providers if p["id"] == "openai")
    assert openai_entry["model"] == "gpt-4o-mini"


def test_all_12_troubleshooting_questions_present():
    """Verify all 12 required troubleshooting and conceptual questions are present."""
    required_questions = [
        "What is an API key?",
        "Why is my API key rejected?",
        "Why am I getting a quota or rate-limit error?",
        "Why does System 1 connection testing fail?",
        "Why is the backend unavailable?",
        "Why is my answer incorrect or missing?",
        "What is the difference between System 1 and System 2?",
        "Can I use HealthRAG without System 1?",
        "Does the app diagnose diseases?",
        "Where can I see the source citations?",
        "Are API keys saved?",
        "Is my healthcare data sent to external providers?",
    ]

    faq_questions = [item["q"] for item in COMMON_PROBLEMS_FAQS]
    for rq in required_questions:
        assert rq in faq_questions, f"Missing troubleshooting FAQ question: {rq}"


def test_free_allowance_explanations():
    """Verify that free allowance claims contain verified information and caveats."""
    s1_dict = {p["id"]: p for p in SUPPORTED_PROVIDERS["system1"]}
    s2_dict = {p["id"]: p for p in SUPPORTED_PROVIDERS["system2"]}

    # Codiv: 100M input tokens public experiment
    assert "100 million" in s1_dict["codiv"]["free_allowance"] or "100M" in s1_dict["codiv"]["free_allowance"]
    assert "529" in s1_dict["codiv"]["free_allowance"]

    # JevModel: 5 free runs/day
    assert "5" in s1_dict["jevmodel"]["free_allowance"]
    assert "UTC" in s1_dict["jevmodel"]["free_allowance"]

    # Laya: 5 free runs
    assert "5" in s1_dict["laya"]["free_allowance"]

    # Groq: free tier
    assert "Free" in s2_dict["groq"]["free_allowance"]
    assert "credit card" in s2_dict["groq"]["free_allowance"]

    # OpenAI: ChatGPT subscription note
    assert "ChatGPT Plus" in s2_dict["openai"]["free_allowance"] or "ChatGPT subscription" in s2_dict["openai"]["free_allowance"]


def test_get_faq_data_helper():
    """Verify get_faq_data returns all top-level keys."""
    data = get_faq_data()
    assert "links" in data
    assert "providers" in data
    assert "troubleshooting" in data


def test_streamlit_app_faq_button_and_dialog():
    """Verify Streamlit app renders FAQ button and expands dialog upon click."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("../streamlit_app.py", default_timeout=15)
    at.run()
    assert not at.exception

    faq_button = next((b for b in at.button if "FAQ" in b.label), None)
    assert faq_button is not None, "FAQ button not found in streamlit_app.py"

    faq_button.click().run()
    assert not at.exception
    # When dialog opens, it should render the provider & troubleshooting expanders
    assert len(at.expander) >= 18


def test_streamlit_standalone_faq_button():
    """Verify standalone entrypoint also includes FAQ button."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("../streamlit_standalone.py", default_timeout=15)
    at.run()
    assert not at.exception

    faq_button = next((b for b in at.button if "FAQ" in b.label), None)
    assert faq_button is not None, "FAQ button not found in streamlit_standalone.py"

