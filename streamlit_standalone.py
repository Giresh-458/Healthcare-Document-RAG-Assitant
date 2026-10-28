import streamlit as st
import os
import uuid
import html
import base64

# Import our backend logic directly instead of using requests!
from app.pdf_processor import process_pdf
from app.text_processor import chunk_text
from app.vector_store import add_documents, list_documents
from app.rag import answer_question
from app.system1 import get_system1_adapter, SYSTEM1_PROVIDERS
from app.config import OCR_ENABLED

# ──────────────────────── SESSION ISOLATION ────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Ensure directories exist
os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/metrics", exist_ok=True)

# ──────────────────────── PAGE CONFIG ────────────────────────
st.set_page_config(
    page_title="Healthcare Document RAG Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────── CUSTOM CSS ────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { visibility: visible !important; background: transparent !important; }
    button[data-testid="collapsedControl"] { visibility: visible !important; display: flex !important; }
    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    .hero-banner { background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 50%, #8b5cf6 100%); padding: 1.5rem 2rem; border-radius: 16px; margin-bottom: 1.5rem; color: white; }
    .hero-banner h1 { margin: 0; font-size: 1.8rem; font-weight: 700; }
    .pipeline-bar { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 1rem; }
    .pipeline-badge { background: #1e293b; color: #38bdf8; padding: 4px 10px; border-radius: 20px; font-size: 0.7rem; font-weight: 600; border: 1px solid #334155; }
    .metric-card { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px 16px; text-align: center; }
    .metric-card .label { font-size: 0.7rem; color: #64748b; text-transform: uppercase; }
    .metric-card .value { font-size: 1.3rem; font-weight: 700; margin-top: 2px; }
    .metric-card .value.high { color: #10b981; }
    .entity-tag { display: inline-block; background: #ede9fe; color: #6d28d9; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; margin: 2px 3px; }
    .source-chip { display: inline-block; background: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 8px; padding: 5px 12px; font-size: 0.78rem; margin: 3px 4px; color: #334155; }
    .doc-item { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 6px 12px; margin: 4px 0; font-size: 0.82rem; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────── HERO BANNER ────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>Healthcare Document Intelligence Platform</h1>
    <p>Clinical report ingestion, adaptive hybrid retrieval, and grounded question answering with System 1 decision-making.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="pipeline-bar">
    <span class="pipeline-badge" style="background:#1e1b4b; color:#818cf8; border-color:#4338ca;">⚡ System 1 Decision Gate</span>
    <span class="pipeline-badge">Hybrid BM25 + Dense Retrieval</span>
    <span class="pipeline-badge">Reciprocal Rank Fusion</span>
    <span class="pipeline-badge">Adaptive Query Routing</span>
    <span class="pipeline-badge">PHI Redaction</span>
    <span class="pipeline-badge">Clinical NER</span>
    <span class="pipeline-badge">Live RAG Metrics</span>
</div>
""", unsafe_allow_html=True)

# ──────────────────────── SIDEBAR ────────────────────────
with st.sidebar:
    st.markdown("### System 2: Answer LLM")
    provider = st.selectbox("Provider", ["groq", "openai", "gemini"], index=0)
    api_key = st.text_input("LLM API Key", type="password", placeholder="Paste API key here...")

    st.markdown("---")
    st.markdown("### System 1: Evidence Gate")
    system1_enabled = st.checkbox("Enable System 1 Gate", value=True)
    s1_options = {
        "Codiv / OpenJev (openjev-latest) [Recommended]": "codiv",
        "JevModel (typesafe/jev-1.13)": "jevmodel",
        "Laya Studio (laya-latest)": "laya",
    }
    s1_label = st.selectbox("System 1 Provider", list(s1_options.keys()), index=0)
    system1_provider = s1_options[s1_label]
    system1_api_key = st.text_input("System 1 API Key", type="password", placeholder="Paste provider API key here...")

    if st.button("Test System 1 Connection", use_container_width=True):
        if not system1_api_key.strip():
            st.error("Please enter a System 1 API key above.")
        else:
            with st.spinner("Testing connection..."):
                try:
                    adapter = get_system1_adapter(system1_provider, api_key=system1_api_key.strip())
                    probe = adapter.test_connection()
                    st.success(f"Verified: {probe['model']} ({probe['latency_ms']}ms, {probe['sample_decision_probability']*100:.1f}%)")
                except Exception as ex:
                    st.error(f"Connection failed: {ex}")

    st.markdown("---")
    st.markdown("### Document Management")
    st.caption("Local standalone mode: session IDs are isolation labels, not authentication. Use synthetic documents only.")

    if st.button("Load Demo Report (Hemoglobin Lab)", use_container_width=True):
        demo_path = os.path.join("sample_documents", "hemoglobin-report-format.pdf")
        if os.path.exists(demo_path):
            with st.spinner("Indexing demo document..."):
                pages = process_pdf(demo_path, "hemoglobin-report-format.pdf")
                chunks = chunk_text(pages)
                add_documents(chunks, st.session_state.session_id)
                st.success("Demo report loaded successfully.")

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
            total_mb = sum(file.size for file in uploaded_files) / (1024 * 1024)
            st.caption(f"{len(uploaded_files)} PDF(s) selected · {total_mb:.1f} MB total")
            if st.button("Upload Selected PDFs", use_container_width=True):
                succeeded = []
                failed = []
                progress = st.progress(0, text="Preparing uploads...")
                for index, uploaded_file in enumerate(uploaded_files, start=1):
                    progress.progress(
                        (index - 1) / len(uploaded_files),
                        text=f"Processing {index}/{len(uploaded_files)}: {uploaded_file.name}"
                    )
                    safe_name = uploaded_file.name.replace(" ", "_")
                    unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
                    file_path = f"data/uploads/{st.session_state.session_id}_{unique_name}"
                    try:
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        pages = process_pdf(file_path, safe_name)
                        if not pages:
                            raise ValueError("No readable pages were extracted.")
                        if len(pages) > 15:
                            st.warning(
                                f"{safe_name} has {len(pages)} pages; only the first 15 will be processed."
                            )
                            pages = pages[:15]
                        chunks = chunk_text(pages)
                        if not chunks:
                            raise ValueError("No text chunks could be created from this PDF.")
                        add_documents(chunks, st.session_state.session_id)
                        succeeded.append(safe_name)
                    except Exception as exc:
                        failed.append(f"{safe_name}: {type(exc).__name__}: {exc}")
                    progress.progress(
                        index / len(uploaded_files),
                        text=f"Processed {index}/{len(uploaded_files)} files"
                    )
                progress.empty()
                if succeeded:
                    st.success(f"Processed {len(succeeded)} PDF(s): " + ", ".join(succeeded))
                if failed:
                    st.error("Some files failed:\n" + "\n".join(failed))
                if succeeded:
                    st.rerun()

    st.markdown("---")
    st.markdown("### Chat Context")
    docs = list_documents(st.session_state.session_id)
    if docs:
        st.caption("Search within specific documents:")
        selected_chat_docs = st.multiselect("Document Filter", docs, default=[], label_visibility="collapsed")
        docs_to_send = selected_chat_docs if len(selected_chat_docs) > 0 else None

        st.markdown("**Currently Indexed:**")
        for d in docs:
            st.markdown(f'<div class="doc-item">{html.escape(str(d))}</div>', unsafe_allow_html=True)
    else:
        docs_to_send = None
        st.caption("No documents indexed yet.")

# ──────────────────────── HELPER ────────────────────────
def render_assistant_extras(structured_data, route, metrics, sources):
    s1_dec = (structured_data or {}).get("system1_decision") or (metrics or {}).get("system1_decision")
    if s1_dec:
        s1_status = s1_dec.get("status", "UNKNOWN")
        s1_prob = s1_dec.get("probability", 0.0)
        s1_model = s1_dec.get("model", "")
        s1_lat = s1_dec.get("latency_sec", 0.0)
        s1_retried = s1_dec.get("retry_performed", False)
        border_col = "#10b981" if s1_status == "SUFFICIENT" else ("#f59e0b" if s1_status == "INSUFFICIENT" else "#ef4444")
        retry_msg = " [Bounded Retry Used]" if s1_retried else ""
        st.markdown(
            f'<div style="background:#f8fafc; border-left:4px solid {border_col}; padding:8px 12px; margin:6px 0; font-size:0.82rem;">'
            f'<strong>⚡ System 1 Gate:</strong> {s1_status} ({s1_model} · {s1_prob*100:.1f}% sufficiency · {s1_lat:.2f}s{retry_msg})</div>',
            unsafe_allow_html=True
        )

    st.markdown(f"**Confidence:** `{structured_data.get('confidence', 'N/A')}`")
    entities = (structured_data or {}).get("medical_entities", [])
    if entities:
        tags_html = " ".join([f'<span class="entity-tag">{html.escape(str(e))}</span>' for e in entities])
        st.markdown(f"**Entities Identified:** {tags_html}", unsafe_allow_html=True)
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

    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    render_assistant_extras(msg.get("structured_data", {}), msg.get("route", ""), msg.get("metrics", {}), msg.get("sources", []))

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
                        try:
                            data = answer_question(
                                prompt, provider, api_key, docs_to_send, st.session_state.session_id,
                                system1_enabled=system1_enabled,
                                system1_provider=system1_provider,
                                system1_api_key=system1_api_key.strip(),
                            )

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
                        except Exception as e:
                            st.error(f"Error: {e}")

with tab2:
    st.markdown("### View Indexed Documents")
    if docs:
        selected_doc = st.selectbox("Select a document to view", docs)
        file_path_demo = os.path.join("sample_documents", selected_doc)

        found_path = None
        if os.path.exists(file_path_demo):
            found_path = file_path_demo
        else:
            uploads_dir = os.path.join("data", "uploads")
            session_prefix = f"{st.session_state.session_id}_"
            filename_suffix = f"_{selected_doc}"
            if os.path.isdir(uploads_dir):
                matching_paths = [
                    os.path.join(uploads_dir, entry)
                    for entry in os.listdir(uploads_dir)
                    if entry.startswith(session_prefix)
                    and entry.endswith(filename_suffix)
                    and os.path.isfile(os.path.join(uploads_dir, entry))
                ]
                if matching_paths:
                    found_path = max(matching_paths, key=os.path.getmtime)

        if found_path:
            with open(found_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
        else:
            st.error("Document file not found on server disk.")
    else:
        st.info("No documents available. Upload one or load the demo.")
