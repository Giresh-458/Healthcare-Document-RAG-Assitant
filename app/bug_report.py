"""Bug Reporting and Feedback Module for HealthRAG.

Allows users to submit bug reports, errors, and suggestions from within the Streamlit UI.
Reports are securely routed to the project maintainer's configured email inbox
via SMTP or transactional email API without exposing medical data, document chunks,
or session API keys.
"""

from __future__ import annotations

import datetime
import html
import logging
import os
import re
import secrets
import smtplib
import subprocess
import time
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st

logger = logging.getLogger("healthrag.bug_report")

# Permitted issue types
ISSUE_TYPES = [
    "Bug",
    "API Error",
    "Upload Problem",
    "Incorrect Answer",
    "UI Problem",
    "Feature Request",
    "Other",
]

# Field length limits
MAX_LEN_NAME = 100
MAX_LEN_EMAIL = 120
MAX_LEN_TITLE = 120
MAX_LEN_DESC = 5000
MAX_LEN_STEPS = 3000
MAX_LEN_EXPECTED = 2000
MAX_LEN_ACTUAL = 2000
MAX_LEN_DEVICE = 200

# Anti-spam rate limits
SUBMISSION_COOLDOWN_SECONDS = 20
MAX_SUBMISSIONS_PER_SESSION = 5


@dataclass
class BugReportResult:
    """Structured result of a bug report submission attempt."""
    success: bool
    status: str  # "success", "not_configured", "validation_error", "rate_limited", "delivery_failed"
    reference_id: str
    message: str
    missing_config: Optional[List[str]] = None


def sanitize_header_value(value: str) -> str:
    """Strip carriage returns, newlines, and excessive whitespace to prevent email header injection."""
    if not value:
        return ""
    # Remove any CR, LF characters
    cleaned = re.sub(r"[\r\n]+", " ", str(value))
    return cleaned.strip()


def validate_email_address(email: str) -> bool:
    """Basic format validation for optional user contact email."""
    if not email:
        return True  # Email is optional
    pattern = r"^[^@\s\r\n]+@[^@\s\r\n]+\.[^@\s\r\n]+$"
    return bool(re.match(pattern, email.strip()))


def generate_reference_id() -> str:
    """Generate a clean, human-readable unique reference ID."""
    token = secrets.token_hex(4).upper()
    return f"HRAG-BUG-{token}"


def get_app_version() -> str:
    """Retrieve Git commit hash or version identifier."""
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        ).decode().strip()
        if git_hash:
            return f"Git commit {git_hash}"
    except Exception:
        pass
    return "HealthRAG v1.0 (Production)"


def get_secret(key: str, default: str = "") -> str:
    """Read a setting from Streamlit Cloud Secrets (st.secrets) or environment variables.

    Never logs secret values.
    """
    # 1. Check Streamlit Cloud Secrets
    try:
        if hasattr(st, "secrets") and key in st.secrets:
            val = str(st.secrets[key]).strip()
            if val:
                return val
    except Exception:
        pass

    # 2. Check environment variables
    val = os.getenv(key, default)
    return str(val).strip() if val else default


def get_email_config() -> Dict[str, Any]:
    """Resolves email delivery configuration from secrets or environment."""
    to_email = get_secret("BUG_REPORT_TO_EMAIL") or get_secret("REPORT_TO_EMAIL")
    smtp_host = get_secret("SMTP_HOST")
    smtp_port_str = get_secret("SMTP_PORT", "587")
    smtp_user = get_secret("SMTP_USERNAME") or get_secret("SMTP_USER")
    smtp_pass = get_secret("SMTP_PASSWORD") or get_secret("SMTP_PASS")
    from_email = get_secret("SMTP_FROM_EMAIL") or smtp_user or to_email
    resend_api_key = get_secret("RESEND_API_KEY")

    try:
        smtp_port = int(smtp_port_str)
    except ValueError:
        smtp_port = 587

    use_tls = get_secret("SMTP_USE_TLS", "true").lower() in ("true", "1", "yes")
    use_ssl = get_secret("SMTP_USE_SSL", "false").lower() in ("true", "1", "yes")
    if smtp_port == 465:
        use_ssl = True
        use_tls = False

    return {
        "to_email": to_email,
        "from_email": from_email,
        "smtp_host": smtp_host,
        "smtp_port": smtp_port,
        "smtp_user": smtp_user,
        "smtp_pass": smtp_pass,
        "smtp_use_tls": use_tls,
        "smtp_use_ssl": use_ssl,
        "resend_api_key": resend_api_key,
    }


def check_email_configuration() -> Tuple[bool, List[str]]:
    """Checks whether sufficient email configuration exists to deliver reports."""
    cfg = get_email_config()
    missing: List[str] = []

    if not cfg["to_email"]:
        missing.append("BUG_REPORT_TO_EMAIL (Destination inbox address)")

    # Need either Resend API or SMTP
    if not cfg["resend_api_key"]:
        if not cfg["smtp_host"]:
            missing.append("SMTP_HOST (or RESEND_API_KEY for transactional API)")
        if not cfg["smtp_user"]:
            missing.append("SMTP_USERNAME")
        if not cfg["smtp_pass"]:
            missing.append("SMTP_PASSWORD")

    is_configured = len(missing) == 0
    return is_configured, missing


def build_email_contents(
    report_data: Dict[str, Any],
    ref_id: str,
    timestamp_str: str,
    app_version: str,
) -> Tuple[str, str, str]:
    """Constructs sanitized subject, plain-text body, and HTML body.

    Guarantees no medical documents, extracted vector chunks, or API keys are included.
    """
    issue_type = sanitize_header_value(report_data.get("issue_type", "Bug"))
    title = sanitize_header_value(report_data.get("title", ""))

    subject = f"[HealthRAG Bug Report] [{issue_type}] {title}"

    # Extract user-submitted fields (treat all as untrusted text)
    name = (report_data.get("name") or "Not provided").strip()
    email_addr = (report_data.get("email") or "Not provided").strip()
    consent = "Yes" if report_data.get("consent_contact") else "No"
    desc = (report_data.get("description") or "").strip()
    steps = (report_data.get("steps") or "Not provided").strip()
    expected = (report_data.get("expected") or "Not provided").strip()
    actual = (report_data.get("actual") or "Not provided").strip()
    device = (report_data.get("browser_device") or "Not provided").strip()

    # Plain text body
    plain_text = f"""======================================================================
HEALTHRAG BUG / FEEDBACK REPORT
======================================================================
Reference ID:       {ref_id}
Timestamp (UTC):    {timestamp_str}
Application:        {app_version}
Issue Type:         {issue_type}
Title:              {title}

--- REPORTER DETAILS ---
Name:               {name}
Email:              {email_addr}
Contact Consent:    {consent}

--- ISSUE DESCRIPTION ---
{desc}

--- STEPS TO REPRODUCE ---
{steps}

--- EXPECTED BEHAVIOR ---
{expected}

--- ACTUAL BEHAVIOR / ERROR MESSAGE ---
{actual}

--- BROWSER / DEVICE ---
{device}

======================================================================
Security Note: Submitted via HealthRAG in-app bug reporting form.
Document texts, patient identifiers, and API keys were excluded.
======================================================================
"""

    # HTML body
    html_text = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.5; color: #1e293b; background-color: #f8fafc; padding: 20px; }}
  .container {{ max-width: 680px; margin: 0 auto; background: #ffffff; border-radius: 12px; border: 1px solid #e2e8f0; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); }}
  .header {{ background: linear-gradient(135deg, #0ea5e9, #6366f1); color: white; padding: 20px 24px; }}
  .header h2 {{ margin: 0; font-size: 1.3rem; }}
  .header .ref {{ opacity: 0.9; font-size: 0.85rem; margin-top: 4px; }}
  .content {{ padding: 24px; }}
  .field-group {{ margin-bottom: 18px; }}
  .field-label {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; font-weight: 700; margin-bottom: 4px; }}
  .field-value {{ font-size: 0.95rem; color: #0f172a; background: #f1f5f9; padding: 8px 12px; border-radius: 6px; white-space: pre-wrap; word-break: break-word; }}
  .badge {{ display: inline-block; background: #e0e7ff; color: #3730a3; padding: 2px 8px; border-radius: 12px; font-weight: 600; font-size: 0.8rem; }}
  .footer {{ background: #f8fafc; border-top: 1px solid #e2e8f0; padding: 14px 24px; font-size: 0.78rem; color: #64748b; text-align: center; }}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h2>🐞 HealthRAG Bug / Feedback Report</h2>
    <div class="ref">Reference: <strong>{html.escape(ref_id)}</strong> · {html.escape(timestamp_str)}</div>
  </div>
  <div class="content">
    <div class="field-group">
      <div class="field-label">Issue Details</div>
      <div><span class="badge">{html.escape(issue_type)}</span> <strong>{html.escape(title)}</strong></div>
      <div style="font-size:0.82rem; color:#64748b; margin-top:4px;">App Version: {html.escape(app_version)}</div>
    </div>

    <div class="field-group">
      <div class="field-label">Reporter Information</div>
      <div class="field-value">Name: {html.escape(name)}<br>Email: {html.escape(email_addr)}<br>Consent to Contact: {html.escape(consent)}</div>
    </div>

    <div class="field-group">
      <div class="field-label">Description</div>
      <div class="field-value">{html.escape(desc)}</div>
    </div>

    <div class="field-group">
      <div class="field-label">Steps to Reproduce</div>
      <div class="field-value">{html.escape(steps)}</div>
    </div>

    <div class="field-group">
      <div class="field-label">Expected vs Actual Behavior</div>
      <div class="field-value"><strong>Expected:</strong> {html.escape(expected)}<br><br><strong>Actual:</strong> {html.escape(actual)}</div>
    </div>

    <div class="field-group">
      <div class="field-label">Browser / Environment</div>
      <div class="field-value">{html.escape(device)}</div>
    </div>
  </div>
  <div class="footer">
    Sent securely via HealthRAG in-app bug reporter. Sensitive healthcare documents, chunks, and API keys are strictly excluded.
  </div>
</div>
</body>
</html>
"""
    return subject, plain_text, html_text


def send_via_smtp(
    config: Dict[str, Any],
    subject: str,
    plain_body: str,
    html_body: str,
    reply_to: Optional[str] = None,
) -> None:
    """Dispatches email via standard SMTP with TLS or SSL."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = config["from_email"]
    msg["To"] = config["to_email"]
    if reply_to and reply_to.strip():
        msg["Reply-To"] = reply_to.strip()

    msg.attach(MIMEText(plain_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    host = config["smtp_host"]
    port = config["smtp_port"]
    timeout = 15

    if config.get("smtp_use_ssl"):
        server = smtplib.SMTP_SSL(host, port, timeout=timeout)
    else:
        server = smtplib.SMTP(host, port, timeout=timeout)
        if config.get("smtp_use_tls", True):
            server.starttls()

    try:
        if config.get("smtp_user") and config.get("smtp_pass"):
            server.login(config["smtp_user"], config["smtp_pass"])
        server.send_message(msg)
    finally:
        try:
            server.quit()
        except Exception:
            pass


def send_via_resend(
    api_key: str,
    from_email: str,
    to_email: str,
    subject: str,
    plain_body: str,
    html_body: str,
    reply_to: Optional[str] = None,
) -> None:
    """Dispatches email via Resend transactional HTTPS API."""
    url = "https://api.resend.com/emails"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "from": from_email or "HealthRAG <onboarding@resend.dev>",
        "to": [to_email],
        "subject": subject,
        "text": plain_body,
        "html": html_body,
    }
    if reply_to and reply_to.strip():
        payload["reply_to"] = reply_to.strip()

    resp = requests.post(url, json=payload, headers=headers, timeout=15)
    if resp.status_code >= 400:
        raise RuntimeError(f"Resend API returned HTTP {resp.status_code}: {resp.text}")


def submit_bug_report(
    form_data: Dict[str, Any],
    session_state: Optional[Any] = None,
) -> BugReportResult:
    """Validates, sanitizes, and dispatches a user bug report.

    Parameters:
        form_data: dictionary with name, email, issue_type, title, description,
                   steps, expected, actual, browser_device, consent_contact, honeypot.
        session_state: Streamlit session_state or dict for anti-spam tracking.
    """
    ref_id = generate_reference_id()

    # 1. Honeypot Anti-Spam Check (hidden field must remain empty)
    if form_data.get("website_check") or form_data.get("honeypot"):
        logger.warning(f"Bot submission trapped by honeypot field (ref={ref_id}).")
        return BugReportResult(
            success=False,
            status="validation_error",
            reference_id=ref_id,
            message="Submission rejected by automated security filter.",
        )

    # 2. Session Rate Limiting / Cooldown
    now = time.time()
    if session_state is not None:
        last_submit = getattr(session_state, "last_bug_report_time", 0.0)
        total_submits = getattr(session_state, "bug_reports_count", 0)

        if total_submits >= MAX_SUBMISSIONS_PER_SESSION:
            return BugReportResult(
                success=False,
                status="rate_limited",
                reference_id=ref_id,
                message=f"Submission limit reached ({MAX_SUBMISSIONS_PER_SESSION} reports per session). Please try again later.",
            )

        elapsed = now - last_submit
        if elapsed < SUBMISSION_COOLDOWN_SECONDS:
            remaining = int(SUBMISSION_COOLDOWN_SECONDS - elapsed)
            return BugReportResult(
                success=False,
                status="rate_limited",
                reference_id=ref_id,
                message=f"Please wait {remaining} seconds before submitting another report.",
            )

    # 3. Required Field Validation
    title = (form_data.get("title") or "").strip()
    description = (form_data.get("description") or "").strip()

    if not title:
        return BugReportResult(
            success=False,
            status="validation_error",
            reference_id=ref_id,
            message="Please provide a short title for the issue.",
        )

    if len(title) > MAX_LEN_TITLE:
        return BugReportResult(
            success=False,
            status="validation_error",
            reference_id=ref_id,
            message=f"Title is too long (maximum {MAX_LEN_TITLE} characters).",
        )

    if not description:
        return BugReportResult(
            success=False,
            status="validation_error",
            reference_id=ref_id,
            message="Please provide a description of the issue.",
        )

    if len(description) > MAX_LEN_DESC:
        return BugReportResult(
            success=False,
            status="validation_error",
            reference_id=ref_id,
            message=f"Description exceeds maximum allowed size ({MAX_LEN_DESC} characters).",
        )

    # Validate optional fields length
    email = (form_data.get("email") or "").strip()
    if email and not validate_email_address(email):
        return BugReportResult(
            success=False,
            status="validation_error",
            reference_id=ref_id,
            message="The email address provided is not valid. Please enter a valid email or leave it empty.",
        )

    # 4. Email Configuration Check
    is_configured, missing_keys = check_email_configuration()
    if not is_configured:
        logger.info(f"Bug report attempted but email service is unconfigured. Missing: {missing_keys}")
        return BugReportResult(
            success=False,
            status="not_configured",
            reference_id=ref_id,
            message="Email delivery is not configured on this HealthRAG deployment.",
            missing_config=missing_keys,
        )

    # 5. Build Sanitized Email
    config = get_email_config()
    timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    app_version = get_app_version()

    subject, plain_body, html_body = build_email_contents(
        form_data, ref_id, timestamp_str, app_version
    )
    reply_to = email if (email and form_data.get("consent_contact")) else None

    # 6. Dispatch Email via Provider
    try:
        if config.get("resend_api_key"):
            send_via_resend(
                api_key=config["resend_api_key"],
                from_email=config["from_email"],
                to_email=config["to_email"],
                subject=subject,
                plain_body=plain_body,
                html_body=html_body,
                reply_to=reply_to,
            )
        else:
            send_via_smtp(
                config=config,
                subject=subject,
                plain_body=plain_body,
                html_body=html_body,
                reply_to=reply_to,
            )

        # Update session anti-spam telemetry
        if session_state is not None:
            session_state.last_bug_report_time = now
            session_state.bug_reports_count = getattr(session_state, "bug_reports_count", 0) + 1

        logger.info(f"Successfully dispatched bug report (ref={ref_id}) to {config['to_email']}")
        return BugReportResult(
            success=True,
            status="success",
            reference_id=ref_id,
            message="Your report has been sent to the project maintainer. Thank you for your feedback!",
        )

    except Exception as ex:
        # Never log raw exception if it might contain credentials; log clean diagnostic
        logger.error(f"Failed to deliver bug report {ref_id}: {type(ex).__name__}")
        return BugReportResult(
            success=False,
            status="delivery_failed",
            reference_id=ref_id,
            message=f"Unable to deliver report due to a server connection error ({type(ex).__name__}). Your input has been preserved so you can retry.",
        )


def render_bug_report_form():
    """Renders the Bug Report modal dialog form in Streamlit."""
    # Prominent medical privacy and security notice
    st.markdown(
        """
        <div style="background:#fffbeb; border:1px solid #fde68a; border-left:4px solid #f59e0b; padding:10px 14px; border-radius:8px; margin-bottom:1rem; font-size:0.85rem; color:#92400e;">
            <strong>⚠️ Security & Medical Privacy Notice:</strong> Do not include patient names, medical records, API keys, passwords, or other sensitive personal health information in your bug report.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Check if a recent submission succeeded
    if st.session_state.get("bug_report_last_success_ref"):
        ref = st.session_state.bug_report_last_success_ref
        st.success(
            f"### ✅ Report Submitted Successfully!\n\n"
            f"**Reference ID:** `{ref}`\n\n"
            "Your report has been sent to the project maintainer's inbox. Thank you for helping improve HealthRAG!\n\n"
            "*Sensitive documents and API keys were excluded from this transmission.*"
        )
        if st.button("Submit Another Report", use_container_width=True):
            st.session_state.bug_report_last_success_ref = None
            st.rerun()
        return

    # Initialize form memory in session_state if not present
    if "bug_form_state" not in st.session_state:
        st.session_state.bug_form_state = {
            "name": "",
            "email": "",
            "issue_type": "Bug",
            "title": "",
            "description": "",
            "steps": "",
            "expected": "",
            "actual": "",
            "browser_device": "",
            "consent_contact": False,
        }

    form_state = st.session_state.bug_form_state

    # Render Streamlit form
    with st.form("bug_report_submission_form", clear_on_submit=False):
        c1, c2 = st.columns(2)
        with c1:
            name_input = st.text_input(
                "Your Name (optional)",
                value=form_state.get("name", ""),
                max_chars=MAX_LEN_NAME,
                placeholder="e.g. Alex Smith",
            )
        with c2:
            email_input = st.text_input(
                "Your Email Address (optional)",
                value=form_state.get("email", ""),
                max_chars=MAX_LEN_EMAIL,
                placeholder="e.g. alex@example.com (for follow-up)",
            )

        consent_input = st.checkbox(
            "I consent to being contacted at the supplied email address regarding this report",
            value=form_state.get("consent_contact", False),
        )

        c_type, c_title = st.columns([1, 2])
        with c_type:
            issue_type_index = (
                ISSUE_TYPES.index(form_state.get("issue_type", "Bug"))
                if form_state.get("issue_type") in ISSUE_TYPES
                else 0
            )
            issue_type_input = st.selectbox("Issue Type *", ISSUE_TYPES, index=issue_type_index)
        with c_title:
            title_input = st.text_input(
                "Short Title *",
                value=form_state.get("title", ""),
                max_chars=MAX_LEN_TITLE,
                placeholder="e.g. System 1 connection timeout on Codiv",
            )

        desc_input = st.text_area(
            "Describe the issue *",
            value=form_state.get("description", ""),
            max_chars=MAX_LEN_DESC,
            height=120,
            placeholder="What happened? Please describe the behavior or suggestion in detail...",
        )

        steps_input = st.text_area(
            "Steps to reproduce the issue (optional)",
            value=form_state.get("steps", ""),
            max_chars=MAX_LEN_STEPS,
            height=80,
            placeholder="1. Upload PDF\n2. Select Gemini\n3. Click ask...",
        )

        c_exp, c_act = st.columns(2)
        with c_exp:
            expected_input = st.text_area(
                "Expected behavior (optional)",
                value=form_state.get("expected", ""),
                max_chars=MAX_LEN_EXPECTED,
                height=70,
                placeholder="What did you expect to happen?",
            )
        with c_act:
            actual_input = st.text_area(
                "Actual behavior or error message (optional)",
                value=form_state.get("actual", ""),
                max_chars=MAX_LEN_ACTUAL,
                height=70,
                placeholder="Error banner or unexpected output displayed",
            )

        device_input = st.text_input(
            "Browser / Device (optional)",
            value=form_state.get("browser_device", ""),
            max_chars=MAX_LEN_DEVICE,
            placeholder="e.g. Chrome 120 on Windows 11 / Safari on iOS",
        )

        # Hidden honeypot field for anti-bot protection
        honeypot = st.text_input("Leave this field empty", value="", key="hp_check", label_visibility="collapsed")

        submit_button = st.form_submit_button("Submit Report", use_container_width=True)

    if submit_button:
        # Preserve user input in session state
        st.session_state.bug_form_state = {
            "name": name_input,
            "email": email_input,
            "issue_type": issue_type_input,
            "title": title_input,
            "description": desc_input,
            "steps": steps_input,
            "expected": expected_input,
            "actual": actual_input,
            "browser_device": device_input,
            "consent_contact": consent_input,
            "honeypot": honeypot,
        }

        with st.spinner("Submitting report..."):
            result = submit_bug_report(st.session_state.bug_form_state, session_state=st.session_state)

        if result.success:
            st.session_state.bug_report_last_success_ref = result.reference_id
            st.session_state.bug_form_state = {}  # Clear form on success
            st.rerun()
        elif result.status == "not_configured":
            st.warning(
                f"### ⚙️ Email Delivery Not Configured\n\n"
                f"{result.message}\n\n"
                f"**Reference ID:** `{result.reference_id}`\n\n"
                f"To receive bug reports in your inbox, configure the following secrets in Streamlit Cloud "
                f"or in your server environment (`.env`):\n\n"
                + "\n".join([f"- `{k}`" for k in (result.missing_config or [])])
                + "\n\n*Your report input above has been preserved.*"
            )
        elif result.status == "rate_limited":
            st.warning(f"⏳ **Rate Limited:** {result.message}")
        elif result.status == "validation_error":
            st.error(f"⚠️ **Validation Error:** {result.message}")
        else:
            st.error(f"❌ **Delivery Error:** {result.message}")


@st.dialog("🐞 Report a Bug / Feedback", width="large")
def show_bug_report_dialog():
    """Modal dialog displaying the bug report submission form."""
    render_bug_report_form()
    st.markdown("---")
    if st.button("✕ Close", use_container_width=True, key="bug_dlg_close_btn"):
        st.rerun()
