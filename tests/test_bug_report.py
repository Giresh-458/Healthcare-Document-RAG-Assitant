"""Unit and integration tests for the Bug Reporting module."""

from unittest.mock import MagicMock, patch
import pytest

from app.bug_report import (
    sanitize_header_value,
    validate_email_address,
    generate_reference_id,
    get_email_config,
    check_email_configuration,
    build_email_contents,
    submit_bug_report,
    BugReportResult,
    MAX_LEN_TITLE,
    MAX_LEN_DESC,
)


def test_sanitize_header_value():
    """Verify that carriage returns and newlines are stripped from headers."""
    dirty = "Title with\r\nHeader\nInjection\rAttempt"
    cleaned = sanitize_header_value(dirty)
    assert "\r" not in cleaned
    assert "\n" not in cleaned
    assert cleaned == "Title with Header Injection Attempt"


def test_validate_email_address():
    """Verify email format validation."""
    assert validate_email_address("") is True  # Optional
    assert validate_email_address("doctor@hospital.org") is True
    assert validate_email_address("invalid-email") is False
    assert validate_email_address("name@domain") is False
    assert validate_email_address("test\n@domain.com") is False


def test_generate_reference_id():
    """Verify generated reference ID format."""
    ref_id = generate_reference_id()
    assert ref_id.startswith("HRAG-BUG-")
    assert len(ref_id) >= 16


def test_build_email_contents_excludes_sensitive_data():
    """Verify that email body contains only form fields and no patient/API-key leaks."""
    form_data = {
        "title": "Upload button unclickable",
        "issue_type": "UI Problem",
        "name": "Dr. Smith",
        "email": "smith@clinic.org",
        "description": "The upload button does not react on Firefox.",
        "steps": "1. Open app\n2. Click upload",
        "expected": "File dialog opens",
        "actual": "Nothing happens",
        "browser_device": "Firefox 120 on Windows 11",
        "consent_contact": True,
    }

    subject, plain_body, html_body = build_email_contents(
        form_data, "HRAG-BUG-TEST01", "2026-10-29 10:00:00 UTC", "Git commit 123456"
    )

    # Subject format check
    assert subject == "[HealthRAG Bug Report] [UI Problem] Upload button unclickable"

    # Ensure submitted fields are present
    assert "Dr. Smith" in plain_body
    assert "smith@clinic.org" in plain_body
    assert "The upload button does not react on Firefox." in plain_body
    assert "Firefox 120 on Windows 11" in plain_body
    assert "HRAG-BUG-TEST01" in plain_body

    # Explicit check: no medical documents or API keys injected
    assert "sk-" not in plain_body
    assert "gsk_" not in plain_body
    assert "AIza" not in plain_body
    assert "hemoglobin-report-format.pdf" not in plain_body


def test_submit_bug_report_required_fields():
    """Verify title and description validation."""
    # Missing title
    res_no_title = submit_bug_report({"title": "", "description": "Some description"})
    assert res_no_title.success is False
    assert res_no_title.status == "validation_error"
    assert "short title" in res_no_title.message

    # Missing description
    res_no_desc = submit_bug_report({"title": "A Title", "description": "   "})
    assert res_no_desc.success is False
    assert res_no_desc.status == "validation_error"
    assert "description" in res_no_desc.message

    # Overlong title
    overlong_title = "A" * (MAX_LEN_TITLE + 1)
    res_long_title = submit_bug_report({"title": overlong_title, "description": "Valid"})
    assert res_long_title.success is False
    assert res_long_title.status == "validation_error"

    # Overlong description
    overlong_desc = "D" * (MAX_LEN_DESC + 1)
    res_long_desc = submit_bug_report({"title": "Valid", "description": overlong_desc})
    assert res_long_desc.success is False
    assert res_long_desc.status == "validation_error"


def test_submit_bug_report_honeypot_trapped():
    """Verify anti-spam honeypot rejects automated bots."""
    bot_payload = {
        "title": "Spam Title",
        "description": "Spam Body",
        "honeypot": "I am a bot filling every input field",
    }
    res = submit_bug_report(bot_payload)
    assert res.success is False
    assert res.status == "validation_error"
    assert "automated security filter" in res.message


def test_submit_bug_report_missing_configuration(monkeypatch):
    """Verify helpful diagnostic when email is unconfigured."""
    # Clear all email environment variables
    for var in [
        "BUG_REPORT_TO_EMAIL", "REPORT_TO_EMAIL", "SMTP_HOST",
        "SMTP_USERNAME", "SMTP_PASSWORD", "RESEND_API_KEY",
    ]:
        monkeypatch.delenv(var, raising=False)

    valid_payload = {
        "title": "Valid Bug Title",
        "description": "Valid detailed description of the problem.",
    }

    res = submit_bug_report(valid_payload)
    assert res.success is False
    assert res.status == "not_configured"
    assert "not configured" in res.message
    assert res.missing_config is not None
    assert any("BUG_REPORT_TO_EMAIL" in c for c in res.missing_config)


def test_submit_bug_report_smtp_success(monkeypatch):
    """Verify successful submission using mocked SMTP provider."""
    monkeypatch.setenv("BUG_REPORT_TO_EMAIL", "maintainer@healthrag.example.com")
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_USERNAME", "mailer@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "super-secret-pass")

    payload = {
        "title": "TypeError in hybrid retriever",
        "issue_type": "API Error",
        "description": "A TypeError occurs when BM25 score is NaN.",
        "name": "Tester",
        "email": "tester@example.com",
        "consent_contact": True,
    }

    mock_smtp_instance = MagicMock()
    with patch("smtplib.SMTP", return_value=mock_smtp_instance) as mock_smtp_class:
        res = submit_bug_report(payload)
        assert res.success is True
        assert res.status == "success"
        assert res.reference_id.startswith("HRAG-BUG-")
        assert "sent to the project maintainer" in res.message

        mock_smtp_class.assert_called_once_with("smtp.example.com", 587, timeout=15)
        mock_smtp_instance.starttls.assert_called_once()
        mock_smtp_instance.login.assert_called_once_with("mailer@example.com", "super-secret-pass")
        mock_smtp_instance.send_message.assert_called_once()
        mock_smtp_instance.quit.assert_called_once()


def test_submit_bug_report_smtp_failure_handling(monkeypatch):
    """Verify that SMTP failure returns delivery_failed without leaking credentials."""
    monkeypatch.setenv("BUG_REPORT_TO_EMAIL", "maintainer@healthrag.example.com")
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_USERNAME", "mailer@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "super-secret-pass")

    payload = {
        "title": "Network failure test",
        "description": "Testing provider exception handling.",
    }

    with patch("smtplib.SMTP", side_effect=TimeoutError("Connection timed out")):
        res = submit_bug_report(payload)
        assert res.success is False
        assert res.status == "delivery_failed"
        assert "Unable to deliver report" in res.message
        # Crucial: verify passwords are not in the error message
        assert "super-secret-pass" not in res.message


def test_submit_bug_report_resend_api_success(monkeypatch):
    """Verify transactional Resend API dispatch when RESEND_API_KEY is configured."""
    monkeypatch.setenv("BUG_REPORT_TO_EMAIL", "maintainer@healthrag.example.com")
    monkeypatch.setenv("RESEND_API_KEY", "re_test_key_12345")

    payload = {
        "title": "Resend API test",
        "description": "Testing Resend HTTP integration.",
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": "resend_id_123"}

    with patch("requests.post", return_value=mock_response) as mock_post:
        res = submit_bug_report(payload)
        assert res.success is True
        assert res.status == "success"
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert "Authorization" in call_kwargs["headers"]
        assert call_kwargs["json"]["to"] == ["maintainer@healthrag.example.com"]


def test_rate_limiting_cooldown(monkeypatch):
    """Verify rate limit cooldown between submissions."""
    monkeypatch.setenv("BUG_REPORT_TO_EMAIL", "maintainer@healthrag.example.com")
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_USERNAME", "mailer@example.com")
    monkeypatch.setenv("SMTP_PASSWORD", "pass")

    class FakeSessionState:
        last_bug_report_time = 0.0
        bug_reports_count = 0

    session = FakeSessionState()
    payload = {"title": "First", "description": "First submission"}

    with patch("smtplib.SMTP"):
        res1 = submit_bug_report(payload, session_state=session)
        assert res1.success is True

        # Immediate second submission should be blocked by cooldown
        res2 = submit_bug_report(payload, session_state=session)
        assert res2.success is False
        assert res2.status == "rate_limited"
        assert "wait" in res2.message


def test_streamlit_app_bug_report_button_and_dialog():
    """Verify Streamlit app renders Bug Report button and opens dialog upon click."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("streamlit_app.py", default_timeout=15)
    at.run()
    assert not at.exception

    bug_button = next((b for b in at.button if "Report Bug" in b.label or "Report a Bug" in b.label), None)
    assert bug_button is not None, "Report Bug button not found in streamlit_app.py"

    bug_button.click().run()
    assert not at.exception
    # Verify the dialog renders text inputs and areas from the form
    assert len(at.text_input) >= 2
    assert len(at.text_area) >= 1
    # Check that privacy warning notice is rendered
    markdown_texts = [m.value for m in at.markdown]
    assert any("Security & Medical Privacy Notice" in text or "patient names" in text for text in markdown_texts)


def test_streamlit_standalone_bug_report_button():
    """Verify standalone entrypoint also includes Bug Report button."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("streamlit_standalone.py", default_timeout=15)
    at.run()
    assert not at.exception

    bug_button = next((b for b in at.button if "Report Bug" in b.label or "Report a Bug" in b.label), None)
    assert bug_button is not None, "Report Bug button not found in streamlit_standalone.py"

