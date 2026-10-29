import streamlit as st
import requests
import os
import base64
import uuid
import html
from dotenv import load_dotenv

from app.faq_content import show_faq_dialog, render_faq_panel
from app.bug_report import show_bug_report_dialog

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
APP_API_TOKEN = os.getenv("APP_API_TOKEN", "").strip()


def _api_headers():
    return {"X-App-Token": APP_API_TOKEN} if APP_API_TOKEN else {}

# ──────────────────────── SESSION ISOLATION ────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# ──────────────────────── PAGE CONFIG ────────────────────────
st.set_page_config(
    page_title="Healthcare Document RAG Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────── CUSTOM CSS ────────────────────────
st.markdown("""
<style>
    /* Global font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    /* Keep Streamlit's header available so the sidebar expand/collapse control remains usable. */
    header[data-testid="stHeader"] { visibility: visible !important; background: transparent !important; }
    button[data-testid="collapsedControl"] { visibility: visible !important; display: flex !important; }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stTextInput label,
    section[data-testid="stSidebar"] .stCheckbox label,
    section[data-testid="stSidebar"] .stFileUploader label {
        color: #94a3b8 !important;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 50%, #8b5cf6 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .hero-banner h1 { margin: 0; font-size: 1.8rem; font-weight: 700; letter-spacing: -0.02em; }
    .hero-banner p { margin: 0.3rem 0 0 0; opacity: 0.88; font-size: 0.95rem; }

    /* Pipeline badge pills */
    .pipeline-bar {
        display: flex; gap: 6px; flex-wrap: wrap;
        margin-bottom: 1rem;
    }
    .pipeline-badge {
        background: #1e293b;
        color: #38bdf8;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        border: 1px solid #334155;
    }
    .pipeline-badge-highlight {
        background: #1e1b4b;
        color: #818cf8;
        border: 1px solid #4338ca;
    }

    /* Metrics card row */
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 12px 16px;
        text-align: center;
    }
    .metric-card .label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card .value { font-size: 1.3rem; font-weight: 700; margin-top: 2px; }
    .metric-card .value.high { color: #10b981; }
    .metric-card .value.medium { color: #f59e0b; }
    .metric-card .value.low { color: #ef4444; }
    .metric-card .value.blue { color: #3b82f6; }

    /* Entity tags */
    .entity-tag {
        display: inline-block;
        background: #ede9fe;
        color: #6d28d9;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px 3px;
    }

    /* Doctor review banner */
    .doctor-banner {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 10px 16px;
        border-radius: 8px;
        font-size: 0.85rem;
        color: #92400e;
        margin-top: 8px;
    }

    /* Source chip */
    .source-chip {
        display: inline-block;
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 5px 12px;
        font-size: 0.78rem;
        margin: 3px 4px;
        color: #334155;
    }

    /* Sidebar demo button */
    section[data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1rem !important;
        transition: transform 0.15s, box-shadow 0.15s;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99,102,241,0.4) !important;
    }

    /* Chat input styling */
    .stChatInput textarea { border-radius: 12px !important; }

    /* Sidebar doc list */
    .doc-item {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 6px 12px;
        margin: 4px 0;
        font-size: 0.82rem;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────── HERO BANNER ────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>Healthcare Document Intelligence Platform</h1>
    <p>Clinical report ingestion, adaptive hybrid retrieval, System 1 decision-making gate, and grounded System 2 LLM answers.</p>
</div>
""", unsafe_allow_html=True)

# Pipeline feature badges
st.markdown("""
<div class="pipeline-bar">
    <span class="pipeline-badge pipeline-badge-highlight">⚡ System 1 Decision Gate</span>
    <span class="pipeline-badge">Hybrid BM25 + Dense Retrieval</span>
    <span class="pipeline-badge">Reciprocal Rank Fusion</span>
    <span class="pipeline-badge">Adaptive Query Routing</span>
    <span class="pipeline-badge">PHI Redaction</span>
    <span class="pipeline-badge">Clinical NER</span>
    <span class="pipeline-badge">Live RAG Metrics</span>
    <span class="pipeline-badge">Structured Pydantic Output</span>
</div>
""", unsafe_allow_html=True)


# ──────────────────────── SIDEBAR ────────────────────────
with st.sidebar:
    col_help, col_bug = st.columns(2)
    with col_help:
        if st.button("❓ FAQ / Help", use_container_width=True, help="Open FAQ, API key setup instructions, free tier comparison, and troubleshooting"):
            show_faq_dialog()
    with col_bug:
        if st.button("🐞 Report Bug", use_container_width=True, help="Report a bug, error, or feedback directly to the maintainer"):
            show_bug_report_dialog()

    st.markdown("### System 2: Answer LLM")
    st.caption("Generates grounded medical answers and citations.")
    provider = st.selectbox("LLM Provider", ["groq", "openai", "gemini"], index=0)
    api_key = st.text_input("LLM API Key", type="password", placeholder="Paste Groq/OpenAI/Gemini key...")

    st.markdown("---")
    st.markdown("### System 1: Evidence Decision Gate")
    st.caption("Fast non-autoregressive decision model evaluating evidence sufficiency before LLM generation.")
    
    system1_enabled = st.checkbox("Enable System 1 Gate", value=True, help="When checked, System 1 validates evidence sufficiency and triggers bounded retrieval or safe abstention.")

    s1_provider_options = {
        "Codiv / OpenJev (openjev-latest) [Recommended]": "codiv",
        "JevModel (typesafe/jev-1.13)": "jevmodel",
        "Laya Studio (laya-latest)": "laya",
    }
    s1_label = st.selectbox("System 1 Provider", list(s1_provider_options.keys()), index=0)
    system1_provider = s1_provider_options[s1_label]

    if system1_provider == "codiv":
        st.info(
            "**Provider:** Codiv.ai\n\n"
            "**Model:** `openjev-latest`\n\n"
            "**Free Allowance:** Public experiment with 100M free input tokens per account (verified Oct 2026). "
            "Quota tracked on Codiv dashboard. Returns HTTP 529 when at capacity.\n\n"
            "*Note: OpenJev is an open-source model served via Jev-compatible API, not official Jev.*"
        )
    elif system1_provider == "jevmodel":
        st.info(
            "**Provider:** JevModel (`jevmodel.app`)\n\n"
            "**Model:** `typesafe/jev-1.13`\n\n"
            "**Free Allowance:** 5 successful free API runs per UTC day after email verification (verified Oct 2026), subject to capacity.\n\n"
            "*Note: JevModel is an independent service, not official TypeSafe API.*"
        )
    elif system1_provider == "laya":
        st.info(
            "**Provider:** Laya Studio (`laya.studio`)\n\n"
            "**Model:** `laya-latest`\n\n"
            "**Free Allowance:** Limited trial of 5 free runs per new workspace (verified Oct 2026). "
            "Swiss-hosted ModernBERT architecture with Jev wire compatibility."
        )

    system1_api_key = st.text_input("System 1 API Key", type="password", placeholder="Paste System 1 key...")

    if st.button("Test System 1 Connection", use_container_width=True):
        if not system1_api_key.strip():
            st.error("Please enter a System 1 API key above to test connection.")
        else:
            with st.spinner("Sending probe request..."):
                try:
                    probe_payload = {
                        "provider": system1_provider,
                        "api_key": system1_api_key.strip(),
                    }
                    probe_res = requests.post(
                        f"{API_URL}/system1/test",
                        json=probe_payload,
                        headers=_api_headers(),
                        timeout=15,
                    )
                    if probe_res.status_code == 200:
                        pdata = probe_res.json()
                        st.success(
                            f"Connection Verified!\n\n"
                            f"- **Model:** `{pdata['model']}`\n"
                            f"- **Latency:** {pdata['latency_ms']} ms\n"
                            f"- **Test Decision:** {pdata['sample_decision_probability']*100:.1f}% sufficiency\n\n"
                            f"**Allowance:** {pdata.get('free_tier_terms', 'Standard free terms apply.')}"
                        )
                    else:
                        try:
                            err_msg = probe_res.json().get("detail", probe_res.text)
                        except Exception:
                            err_msg = probe_res.text
                        st.error(f"Test Failed (HTTP {probe_res.status_code}):\n{err_msg}")
                except Exception as ex:
                    st.error(f"Could not reach backend: {ex}")

    st.caption("🔒 **Security & Privacy:** Keys are request-scoped and never stored to disk or emitted in logs. Queries and retrieved passages are transmitted to the chosen provider. External free tiers are not HIPAA-compliant; do not upload real personal health information.")

    st.markdown("---")
    st.markdown("### Document Management")
    st.caption("Documents are filtered by session ID. IDs are not authentication. Use synthetic data; configure APP_API_TOKEN for a shared-token API gate.")

    # ONE-CLICK DEMO LOADER
    if st.button("Load Demo Report (Hemoglobin Lab)", use_container_width=True):
        demo_path = os.path.join("sample_documents", "hemoglobin-report-format.pdf")
        if os.path.exists(demo_path):
            with st.spinner("Indexing demo document..."):
                with open(demo_path, "rb") as f:
                    files = {"file": ("hemoglobin-report-format.pdf", f.read(), "application/pdf")}
                    data = {"session_id": st.session_state.session_id, "patient_label": "Demo Patient"}
                    try:
                        res = requests.post(f"{API_URL}/upload", files=files, data=data, headers=_api_headers(), timeout=60)
                        if res.status_code == 200:
                            st.success("Demo report loaded successfully.")
                        else:
                            st.error(f"Upload error: {res.text}")
                    except Exception:
                        st.error("Backend offline. Ensure FastAPI is running on port 8000.")
        else:
            st.error("Demo file not found in sample_documents/ directory.")

    patient_label_input = st.text_input("Patient label for this upload (optional)", placeholder="e.g., Patient A; not identity verification")
    st.caption("Upload multiple synthetic clinical PDFs (5MB maximum per file).")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Select as many PDFs as you need. Each file must be 5MB or smaller."
    )
    max_file_bytes = 5 * 1024 * 1024
    if uploaded_files:
        oversized = [file.name for file in uploaded_files if file.size > max_file_bytes]
        if oversized:
            st.error("These files exceed the 5MB per-file limit: " + ", ".join(oversized))
        else:
            st.caption(f"{len(uploaded_files)} PDF(s) selected · {sum(file.size for file in uploaded_files) / (1024 * 1024):.1f} MB total")
            if st.button("Upload Selected PDFs", use_container_width=True):
                succeeded = []
                failed = []
                progress = st.progress(0, text="Preparing uploads...")
                for index, uploaded_file in enumerate(uploaded_files, start=1):
                    progress.progress((index - 1) / len(uploaded_files), text=f"Uploading {index}/{len(uploaded_files)}: {uploaded_file.name}")
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    data = {
                        "session_id": st.session_state.session_id,
                        "patient_label": patient_label_input.strip(),
                    }
                    try:
                        res = requests.post(
                            f"{API_URL}/upload",
                            files=files,
                            data=data,
                            headers=_api_headers(),
                            timeout=120,
                        )
                        if res.status_code == 200:
                            succeeded.append(uploaded_file.name)
                        else:
                            try:
                                detail = res.json().get("detail", res.text)
                            except ValueError:
                                detail = res.text
                            failed.append(f"{uploaded_file.name}: HTTP {res.status_code} — {detail}")
                    except requests.RequestException as exc:
                        failed.append(f"{uploaded_file.name}: backend/network error ({type(exc).__name__})")
                    progress.progress(index / len(uploaded_files), text=f"Processed {index}/{len(uploaded_files)} files")
                progress.empty()
                if succeeded:
                    st.success(f"Uploaded {len(succeeded)} PDF(s): " + ", ".join(succeeded))
                if failed:
                    st.error("Some uploads failed:\n" + "\n".join(failed))
                if succeeded:
                    st.rerun()

    st.markdown("---")
    st.markdown("### Chat Context")
    st.caption("Patient labels help scope a demo query; they are not verified identity or access control.")
    try:
        docs_response = requests.get(
            f"{API_URL}/documents",
            params={"session_id": st.session_state.session_id},
            headers=_api_headers(),
            timeout=3,
        )
        docs_payload = docs_response.json() if docs_response.status_code == 200 else {}
        records = docs_payload.get("records", [])
        docs = docs_payload.get("documents", [])
        labels = sorted({r.get("patient_label", "unspecified") for r in records if r.get("patient_label") != "unspecified"})
        patient_scope_options = ["No patient selected"] + labels
        selected_patient_scope = st.selectbox("Patient scope", patient_scope_options, index=0)
        patient_label_to_send = None if selected_patient_scope == "No patient selected" else selected_patient_scope
        if records:
            st.caption("Search within specific upload records (duplicate filenames remain separate):")
            record_labels = {
                f"{r['document_name']} · {r.get('patient_label', 'unspecified')} · {r['document_id'][-8:]}": r["document_id"]
                for r in records
            }
            selected_record_labels = st.multiselect(
                "Document Filter", list(record_labels.keys()), default=[],
                placeholder="Select documents to search...", label_visibility="collapsed"
            )
            docs_to_send = [record_labels[label] for label in selected_record_labels]
            st.markdown("**Currently Indexed Uploads:**")
            for record in records:
                st.markdown(
                    f'<div class="doc-item">{html.escape(str(record["document_name"]))} · '
                    f'{html.escape(str(record.get("patient_label", "unspecified")))} · '
                    f'{html.escape(str(record["document_id"][-8:]))}</div>',
                    unsafe_allow_html=True,
                )
        else:
            docs_to_send = []
            patient_label_to_send = None
            st.caption("No documents indexed yet.")
    except (requests.RequestException, ValueError, KeyError):
        docs, records, docs_to_send, patient_label_to_send = [], [], [], None
        st.caption("Backend offline or document index unavailable.")

    st.markdown("---")
    st.markdown("### Sample Queries")
    st.caption("Common evaluation queries:")
    sample_qs = [
        "What is the hemoglobin level?",
        "Are there any abnormal values?",
        "What was the patient's potassium level?",
        "Summarize the full report",
    ]
    for q in sample_qs:
        st.code(q, language=None)


# ──────────────────────── HELPER: RENDER ASSISTANT EXTRAS ────────────────────────
def render_assistant_extras(structured_data, route, metrics, sources):
    """Renders rich UI elements below an assistant message, including System 1 decisions."""

    # Doctor review warning
    if structured_data and structured_data.get("requires_doctor_review"):
        st.markdown(
            '<div class="doctor-banner"><strong>Clinical Notice:</strong> This response discusses medical metrics '
            'and requires independent clinical verification.</div>',
            unsafe_allow_html=True
        )

    # System 1 Gate Decision Banner
    s1_dec = (structured_data or {}).get("system1_decision") or (metrics or {}).get("system1_decision")
    if s1_dec:
        s1_status = s1_dec.get("status", "UNKNOWN")
        s1_prob = s1_dec.get("probability", 0.0)
        s1_model = s1_dec.get("model", "")
        s1_lat = s1_dec.get("latency_sec", 0.0)
        s1_retried = s1_dec.get("retry_performed", False)
        s1_err = s1_dec.get("error_message")

        if s1_status == "SUFFICIENT":
            border_color = "#10b981"
            status_text = "EVIDENCE SUFFICIENT"
            badge_bg = "#ecfdf5"
            badge_fg = "#065f46"
        elif s1_status == "INSUFFICIENT":
            border_color = "#f59e0b"
            status_text = "EVIDENCE INSUFFICIENT (ABSTAINED)"
            badge_bg = "#fffbeb"
            badge_fg = "#92400e"
        else:
            border_color = "#ef4444"
            status_text = f"DECISION GATE ERROR ({s1_err or 'Provider Error'})"
            badge_bg = "#fef2f2"
            badge_fg = "#991b1b"

        retry_html = ' <span style="background:#e0e7ff; color:#3730a3; padding:2px 8px; border-radius:12px; font-size:0.72rem; font-weight:600;">Bounded Retry Triggered</span>' if s1_retried else ""

        st.markdown(
            f'<div style="background:{badge_bg}; border-left:4px solid {border_color}; padding:8px 14px; border-radius:8px; margin:8px 0; font-size:0.83rem; color:{badge_fg};">'
            f'<strong>⚡ System 1 Gate:</strong> <strong>{status_text}</strong> · '
            f'Model: <code>{html.escape(str(s1_model))}</code> · '
            f'Calibrated Sufficiency: <strong>{s1_prob*100:.1f}%</strong> · '
            f'Gate Latency: {s1_lat:.2f}s{retry_html}</div>',
            unsafe_allow_html=True,
        )

    # Metrics cards row
    if metrics or structured_data:
        cols = st.columns(4)
        with cols[0]:
            conf = (structured_data or {}).get("confidence", "—")
            color_class = {"high": "high", "medium": "medium", "low": "low"}.get(conf, "blue")
            st.markdown(
                f'<div class="metric-card"><div class="label">Confidence</div>'
                f'<div class="value {color_class}">{conf.upper() if conf != "—" else "—"}</div></div>',
                unsafe_allow_html=True
            )
        with cols[1]:
            st.markdown(
                f'<div class="metric-card"><div class="label">Route</div>'
                f'<div class="value blue">{route}</div></div>',
                unsafe_allow_html=True
            )
        with cols[2]:
            fp = (metrics or {}).get("answer_context_token_overlap_heuristic", "—")
            st.markdown(
                f'<div class="metric-card"><div class="label">Token Overlap (Heuristic)</div>'
                f'<div class="value blue">{fp}</div></div>',
                unsafe_allow_html=True
            )
        with cols[3]:
            lat = (metrics or {}).get("latency_sec", "—")
            st.markdown(
                f'<div class="metric-card"><div class="label">Total Latency</div>'
                f'<div class="value blue">{lat}s</div></div>',
                unsafe_allow_html=True
            )

    # Entity tags
    entities = (structured_data or {}).get("medical_entities", [])
    if entities:
        tags_html = " ".join([f'<span class="entity-tag">{html.escape(str(e))}</span>' for e in entities])
        st.markdown(f"**Entities Identified:** {tags_html}", unsafe_allow_html=True)

    # Source chips
    if sources:
        with st.expander("Source Citations"):
            chips = "".join([f'<span class="source-chip">{html.escape(str(s["document"]))} | Page {html.escape(str(s["page"]))}</span>' for s in sources])
            st.markdown(chips, unsafe_allow_html=True)


# ──────────────────────── TABS (CHAT & VIEWER) ────────────────────────
tab1, tab2 = st.tabs(["💬 Chat Interface", "📄 Document Viewer"])

with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_container = st.container()

    # Render history
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    render_assistant_extras(
                        msg.get("structured_data", {}),
                        msg.get("route", ""),
                        msg.get("metrics", {}),
                        msg.get("sources", [])
                    )

    # User input
    if prompt := st.chat_input("Ask about patient findings, lab values, or reference ranges..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                if not docs:
                    st.error("⚠️ Please upload a document or load the demo report first.")
                elif not docs_to_send:
                    st.warning("⚠️ No documents are selected. Please select at least one PDF in Chat Context before asking a question.")
                elif not api_key:
                    st.error("⚠️ Please provide a System 2 LLM API key in the sidebar configuration to proceed.")
                elif system1_enabled and not system1_api_key.strip():
                    st.warning("⚠️ System 1 Gate is enabled but no System 1 API key was provided. Enter a System 1 key in the sidebar, or uncheck the gate to run baseline System 2.")
                else:
                    with st.spinner("Retrieving evidence, evaluating System 1 gate, generating answer..."):
                        recent_history = [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages[-4:]
                            if m.get("role") in {"user", "assistant"}
                        ]
                        payload = {
                            "question": prompt,
                            "provider": provider,
                            "api_key": api_key,
                            "document_ids": docs_to_send,
                            "patient_label": patient_label_to_send,
                            "session_id": st.session_state.session_id,
                            "system1_enabled": system1_enabled,
                            "system1_provider": system1_provider,
                            "system1_api_key": system1_api_key.strip(),
                            "conversation_history": recent_history,
                        }
                        try:
                            res = requests.post(f"{API_URL}/ask", json=payload, headers=_api_headers(), timeout=60)
                            if res.status_code == 200:
                                data = res.json()
                                answer = data.get("answer", "")
                                sources = data.get("sources", [])
                                metrics = data.get("metrics", {})
                                route = data.get("route", "FACTOID")
                                structured_data = data.get("structured_data", {})

                                st.markdown(answer)
                                render_assistant_extras(structured_data, route, metrics, sources)

                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": answer,
                                    "sources": sources,
                                    "metrics": metrics,
                                    "route": route,
                                    "structured_data": structured_data
                                })
                            else:
                                st.error(f"Backend error: {res.text}")
                        except requests.exceptions.ConnectionError:
                            st.error("Unable to connect to FastAPI backend at port 8000.")
                        except Exception as e:
                            st.error(f"Unexpected error: {e}")

with tab2:
    st.markdown("### View Indexed Documents")
    try:
        docs_res = requests.get(f"{API_URL}/documents", params={"session_id": st.session_state.session_id}, headers=_api_headers(), timeout=3)
        if docs_res.status_code == 200:
            doc_records = docs_res.json().get("records", [])
            if doc_records:
                viewer_options = {f"{r['document_name']} · {r.get('patient_label', 'unspecified')} · {r['document_id'][-8:]}": r for r in doc_records}
                selected_doc_label = st.selectbox("Select an upload to view", list(viewer_options.keys()))
                selected_doc = viewer_options[selected_doc_label]

                try:
                    pdf_res = requests.get(
                        f"{API_URL}/document-pdf",
                        params={
                            "document_id": selected_doc["document_id"],
                            "session_id": st.session_state.session_id,
                        },
                        headers=_api_headers(),
                        timeout=15,
                    )
                    if pdf_res.status_code == 200 and pdf_res.headers.get("content-type", "").lower().startswith("application/pdf"):
                        base64_pdf = base64.b64encode(pdf_res.content).decode("ascii")
                        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
                        st.markdown(pdf_display, unsafe_allow_html=True)
                    elif pdf_res.status_code == 404:
                        st.warning("The PDF file is not available in backend storage for this upload. Re-upload the document and try again.")
                    else:
                        st.error(f"Could not load PDF preview (HTTP {pdf_res.status_code}).")
                except requests.RequestException:
                    st.error("Could not connect to the PDF service. Please retry in a moment.")
            else:
                st.info("No documents uploaded yet. Please upload a document or load the demo from the sidebar.")
        else:
            st.error("Could not fetch documents from backend.")
    except Exception:
        st.error("Backend offline. Cannot fetch documents.")
