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
    #MainMenu, footer, header { visibility: hidden; }
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

st.markdown("""
<div class="hero-banner">
    <h1>Healthcare Document Intelligence Platform</h1>
    <p>Clinical report ingestion, adaptive hybrid retrieval, and grounded question answering with source traceability.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="pipeline-bar">
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
    st.markdown("### LLM Configuration")
    provider = st.selectbox("Provider", ["groq", "openai", "gemini"], index=0)
    api_key = st.text_input("API Key", type="password", placeholder="Paste API key here...")

    st.markdown("---")
    st.markdown("### Document Management")
    st.caption("Private Session ID: " + st.session_state.session_id[:8])

    if st.button("Load Demo Report (Hemoglobin Lab)", use_container_width=True):
        demo_path = os.path.join("sample_documents", "hemoglobin-report-format.pdf")
        if os.path.exists(demo_path):
            with st.spinner("Indexing demo document..."):
                pages = process_pdf(demo_path, "hemoglobin-report-format.pdf")
                chunks = chunk_text(pages)
                add_documents(chunks, "global")
                st.success("Demo report loaded successfully.")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
    if uploaded_file:
        if st.button("Upload Document", use_container_width=True):
            with st.spinner("Processing document..."):
                safe_name = uploaded_file.name.replace(" ", "_")
                file_path = f"data/uploads/{st.session_state.session_id}_{safe_name}"
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                pages = process_pdf(file_path, safe_name)
                chunks = chunk_text(pages)
                add_documents(chunks, st.session_state.session_id)
                st.success(f"{safe_name} uploaded securely.")

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
    st.markdown(f"**Confidence:** `{structured_data.get('confidence_score', 'N/A')}`")
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
                if not api_key:
                    st.error("Please provide an API key in the sidebar configuration to proceed.")
                else:
                    with st.spinner("Routing query, retrieving context, generating response..."):
                        try:
                            # Bypass FastAPI and call backend Python functions directly!
                            data = answer_question(prompt, provider, api_key, docs_to_send, st.session_state.session_id)
                            
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
            search_name = f"{st.session_state.session_id}_{selected_doc}"
            user_path = os.path.join("data", "uploads", search_name)
            if os.path.exists(user_path):
                found_path = user_path
                
        if found_path:
            with open(found_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
        else:
            st.error("Document file not found on server disk.")
    else:
        st.info("No documents available. Upload one or load the demo.")
