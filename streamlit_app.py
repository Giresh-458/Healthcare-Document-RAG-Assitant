import streamlit as st
import requests
import os
import base64
import uuid
import html
from dotenv import load_dotenv

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
    #MainMenu, footer, header { visibility: hidden; }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stTextInput label,
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
    <p>Clinical report ingestion, adaptive hybrid retrieval, and grounded question answering with source traceability.</p>
</div>
""", unsafe_allow_html=True)

# Pipeline feature badges
st.markdown("""
<div class="pipeline-bar">
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
    st.markdown("### LLM Configuration")
    provider = st.selectbox("Provider", ["groq", "openai", "gemini"], index=0)
    api_key = st.text_input("API Key", type="password", placeholder="Paste API key here...")

    st.markdown("---")
    st.markdown("### Document Management")
    st.caption("Documents are filtered by session ID. IDs are not authentication. Use synthetic data; configure APP_API_TOKEN for a shared-token API gate.")

    # ONE-CLICK DEMO LOADER
    if st.button("Load Demo Report (Hemoglobin Lab)", use_container_width=True):
        demo_path = os.path.join("sample_documents", "hemoglobin-report-format.pdf")
        if os.path.exists(demo_path):
            with st.spinner("Indexing demo document..."):
                with open(demo_path, "rb") as f:
                    # Index the demo into this browser session, not a shared global corpus
                    files = {"file": ("hemoglobin-report-format.pdf", f.read(), "application/pdf")}
                    data = {"session_id": st.session_state.session_id, "patient_label": "Demo Patient"}
                    try:
                        res = requests.post(f"{API_URL}/upload", files=files, data=data, headers=_api_headers(), timeout=60)
                        if res.status_code == 200:
                            st.success("Demo report loaded successfully.")
                        else:
                            st.error(f"Upload error: {res.text}")
                    except:
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
                placeholder="All records in this session", label_visibility="collapsed"
            )
            docs_to_send = [record_labels[label] for label in selected_record_labels] or None
            st.markdown("**Currently Indexed Uploads:**")
            for record in records:
                st.markdown(
                    f'<div class="doc-item">{html.escape(str(record["document_name"]))} · '
                    f'{html.escape(str(record.get("patient_label", "unspecified")))} · '
                    f'{html.escape(str(record["document_id"][-8:]))}</div>',
                    unsafe_allow_html=True,
                )
        else:
            docs_to_send = None
            patient_label_to_send = None
            st.caption("No documents indexed yet.")
    except (requests.RequestException, ValueError, KeyError):
        docs, records, docs_to_send, patient_label_to_send = [], [], None, None
        st.caption("Backend offline or document index unavailable.")


    st.markdown("---")
    st.markdown("### Sample Queries")
    st.caption("Common evaluation queries:")
    sample_qs = [
        "What is the hemoglobin level?",
        "Are there any abnormal values?",
        "Summarize the full report",
    ]
    for q in sample_qs:
        st.code(q, language=None)


# ──────────────────────── HELPER: RENDER ASSISTANT EXTRAS ────────────────────────
def render_assistant_extras(structured_data, route, metrics, sources):
    """Renders the rich UI elements below an assistant message."""

    # Doctor review warning
    if structured_data and structured_data.get("requires_doctor_review"):
        st.markdown(
            '<div class="doctor-banner"><strong>Clinical Notice:</strong> This response discusses medical metrics '
            'and requires independent clinical verification.</div>',
            unsafe_allow_html=True
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
                f'<div class="metric-card"><div class="label">Answer/context token overlap (heuristic)</div>'
                f'<div class="value blue">{fp}</div></div>',
                unsafe_allow_html=True
            )
        with cols[3]:
            lat = (metrics or {}).get("latency_sec", "—")
            st.markdown(
                f'<div class="metric-card"><div class="label">Latency</div>'
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
                elif not api_key:
                    st.error("⚠️ Please provide an API key in the sidebar configuration to proceed.")
                else:
                    with st.spinner("Routing query, retrieving context, generating response..."):
                        payload = {
                            "question": prompt, 
                            "provider": provider, 
                            "api_key": api_key,
                            "document_ids": docs_to_send,
                            "patient_label": patient_label_to_send,
                            "session_id": st.session_state.session_id
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
                
                # Fetch the PDF from the API that owns the upload. The Streamlit host
                # may not share a filesystem with the FastAPI service.
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
    except:
        st.error("Backend offline. Cannot fetch documents.")
