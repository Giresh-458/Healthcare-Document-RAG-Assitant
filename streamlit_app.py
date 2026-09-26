import streamlit as st
import requests
import os
import base64
import uuid
import html

API_URL = "http://127.0.0.1:8000"

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
    st.caption("Each visitor has an isolated private session. Uploaded files cannot be seen by other users.")

    # ONE-CLICK DEMO LOADER
    if st.button("Load Demo Report (Hemoglobin Lab)", use_container_width=True):
        demo_path = os.path.join("sample_documents", "hemoglobin-report-format.pdf")
        if os.path.exists(demo_path):
            with st.spinner("Indexing demo document..."):
                with open(demo_path, "rb") as f:
                    # We pass "global" session_id for the demo so everyone can see it
                    files = {"file": ("hemoglobin-report-format.pdf", f.read(), "application/pdf")}
                    data = {"session_id": "global"}
                    try:
                        res = requests.post(f"{API_URL}/upload", files=files, data=data)
                        if res.status_code == 200:
                            st.success("Demo report loaded successfully.")
                        else:
                            st.error(f"Upload error: {res.text}")
                    except:
                        st.error("Backend offline. Ensure FastAPI is running on port 8000.")
        else:
            st.error("Demo file not found in sample_documents/ directory.")

    st.caption("Or upload custom clinical PDF:")
    uploaded_file = st.file_uploader("Upload PDF (Max 5MB)", type=["pdf"], label_visibility="collapsed")
    if uploaded_file:
        if uploaded_file.size > 5 * 1024 * 1024:
            st.error("⚠️ File is too large! Please upload a clinical report smaller than 5MB.")
        else:
            if st.button("Upload Document", use_container_width=True):
                with st.spinner("Processing document..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    data = {"session_id": st.session_state.session_id}
                    try:
                        res = requests.post(f"{API_URL}/upload", files=files, data=data)
                        if res.status_code == 200:
                            st.success(f"{uploaded_file.name} uploaded securely to your private session.")
                        else:
                            st.error(f"Upload error: {res.text}")
                    except:
                        st.error("Backend offline.")

    st.markdown("---")
    st.markdown("### Chat Context")
    try:
        # Fetch documents specific to THIS user's session (plus globals)
        docs = requests.get(f"{API_URL}/documents?session_id={st.session_state.session_id}", timeout=3).json().get("documents", [])
        if docs:
            st.caption("Search within specific documents:")
            selected_chat_docs = st.multiselect(
                "Document Filter", 
                docs, 
                default=[], 
                placeholder="All Documents (Click to filter)", 
                label_visibility="collapsed"
            )
            # If nothing is selected, we consider it "All Documents" (None in backend)
            docs_to_send = selected_chat_docs if len(selected_chat_docs) > 0 else None
            
            st.markdown("**Currently Indexed:**")
            for d in docs:
                st.markdown(f'<div class="doc-item">{html.escape(str(d))}</div>', unsafe_allow_html=True)
        else:
            docs_to_send = None
            st.caption("No documents indexed yet.")
    except:
        docs_to_send = None
        st.caption("Backend offline.")

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
            fp = (metrics or {}).get("faithfulness", "—")
            st.markdown(
                f'<div class="metric-card"><div class="label">Faithfulness</div>'
                f'<div class="value high">{fp}</div></div>',
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
                elif not docs_to_send:
                    st.warning("⚠️ Please select at least one document from the 'Document Filter' dropdown in the sidebar to search.")
                elif not api_key:
                    st.error("⚠️ Please provide an API key in the sidebar configuration to proceed.")
                else:
                    with st.spinner("Routing query, retrieving context, generating response..."):
                        payload = {
                            "question": prompt, 
                            "provider": provider, 
                            "api_key": api_key,
                            "document_names": docs_to_send,
                            "session_id": st.session_state.session_id
                        }
                        try:
                            res = requests.post(f"{API_URL}/ask", json=payload, timeout=60)
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
        docs_res = requests.get(f"{API_URL}/documents", timeout=3)
        if docs_res.status_code == 200:
            doc_list = docs_res.json().get("documents", [])
            if doc_list:
                selected_doc = st.selectbox("Select a document to view", doc_list)
                
                # Check locations where the PDF might be stored locally
                pdf_path = os.path.join("data", "uploads", selected_doc)
                if not os.path.exists(pdf_path):
                    pdf_path = os.path.join("sample_documents", selected_doc)
                    
                if os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
                    st.markdown(pdf_display, unsafe_allow_html=True)
                else:
                    st.warning(f"Local file not found for rendering: {selected_doc}")
            else:
                st.info("No documents uploaded yet. Please upload a document or load the demo from the sidebar.")
        else:
            st.error("Could not fetch documents from backend.")
    except:
        st.error("Backend offline. Cannot fetch documents.")
