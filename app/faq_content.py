"""FAQ and Help documentation module for HealthRAG.

Provides beginner-friendly, structured guidance on:
- How to use HealthRAG (System 1 + System 2 workflow)
- Obtaining API keys for all 6 configured providers
- Comparing free vs paid tier allowances and rate limits
- Troubleshooting common errors and connection issues
- Security, privacy, and clinical disclaimers
"""

from typing import Any, Dict, List
import streamlit as st

# Official verified links
OFFICIAL_LINKS = {
    "groq": "https://console.groq.com/",
    "openai": "https://platform.openai.com/api-keys",
    "gemini": "https://aistudio.google.com/apikey",
    "codiv": "https://codiv.ai/",
    "jevmodel": "https://jevmodel.app/",
    "laya": "https://laya.studio/",
}

# Configured providers and exact model IDs
SUPPORTED_PROVIDERS = {
    "system1": [
        {
            "id": "codiv",
            "name": "Codiv / OpenJev",
            "model": "openjev-latest",
            "role": "System 1 Evidence Decision Gate",
            "recommended": True,
            "url": OFFICIAL_LINKS["codiv"],
            "docs_url": "https://codiv.ai/docs/quickstart",
            "free_allowance": (
                "Free public experiment: 100 million input tokens per account (verified Oct 2026). "
                "Quota is tracked on the Codiv dashboard. Returns HTTP 529 when the cluster is at capacity."
            ),
            "paid_details": "Subject to Codiv terms; open experiment pricing subject to future updates.",
            "description": (
                "Hosted open-source OpenJev model served via Jev-compatible wire format. "
                "Evaluates whether retrieved passages contain sufficient clinical facts to answer the question."
            ),
            "how_to_get": [
                "Visit https://codiv.ai/ and register or sign in.",
                "Navigate to your user dashboard or developer API section.",
                "Generate a new API key and copy it.",
                "In HealthRAG sidebar, choose 'Codiv / OpenJev (openjev-latest)' under System 1.",
                "Paste your key into the 'System 1 API Key' input.",
                "Click 'Test System 1 Connection' to verify server connectivity and latency.",
            ],
        },
        {
            "id": "jevmodel",
            "name": "JevModel",
            "model": "typesafe/jev-1.13",
            "role": "System 1 Evidence Decision Gate",
            "recommended": False,
            "url": OFFICIAL_LINKS["jevmodel"],
            "docs_url": "https://jevmodel.app/jev-api",
            "free_allowance": (
                "5 successful free API runs per UTC day after email verification (verified Oct 2026), "
                "subject to service capacity."
            ),
            "paid_details": "Independent hosted API. Higher tiers subject to JevModel commercial plans.",
            "description": (
                "Independent hosted service serving the TypeSafe Jev model using criteria-based binary classification (noul). "
                "Note: Independent service, not official TypeSafe API."
            ),
            "how_to_get": [
                "Visit https://jevmodel.app/ and sign up for an account.",
                "Verify your email address (mandatory to unlock the 5 free runs/day UTC tier).",
                "Navigate to the API Credentials section (see https://jevmodel.app/jev-api).",
                "Copy your API key.",
                "In HealthRAG sidebar, select 'JevModel (typesafe/jev-1.13)' under System 1.",
                "Paste the key into 'System 1 API Key'.",
                "Click 'Test System 1 Connection' to verify.",
            ],
        },
        {
            "id": "laya",
            "name": "Laya Studio",
            "model": "automatic routing (omits model parameter)",
            "role": "System 1 Evidence Decision Gate",
            "recommended": False,
            "url": OFFICIAL_LINKS["laya"],
            "docs_url": "https://laya.studio/docs/api",
            "free_allowance": (
                "Limited trial of 5 free runs per new workspace (verified Oct 2026). "
                "Not a permanent high-volume free tier."
            ),
            "paid_details": "Requires paid workspace after 5 trial runs. Hosted in Switzerland.",
            "description": (
                "Swiss-hosted ModernBERT-based decision model with automatic language routing (Jev wire-compatible). "
                "Operates with zero data retention on Swiss infrastructure."
            ),
            "how_to_get": [
                "Visit https://laya.studio/ and create a new workspace.",
                "Navigate to Workspace Settings -> API Keys (refer to https://laya.studio/docs/api).",
                "Create a new API key (new workspaces receive 5 trial runs).",
                "In HealthRAG sidebar, select 'Laya Studio (laya-latest)' under System 1.",
                "Paste the key into 'System 1 API Key'.",
                "Click 'Test System 1 Connection' to verify.",
            ],
        },
    ],
    "system2": [
        {
            "id": "groq",
            "name": "Groq",
            "model": "openai/gpt-oss-120b",
            "role": "System 2 Answer Generation LLM",
            "recommended": True,
            "url": OFFICIAL_LINKS["groq"],
            "docs_url": "https://console.groq.com/docs",
            "free_allowance": (
                "Free Developer Tier available without credit card (verified Oct 2026). "
                "Rate limits: 30 requests/min (RPM), 1,000 requests/day (RPD), ~8,000 tokens/min (TPM)."
            ),
            "paid_details": (
                "Pay-as-you-go developer tier available for higher volume (~$0.15/1M input tokens, "
                "~$0.60/1M output tokens). Subject to Groq pricing updates."
            ),
            "description": (
                "Provides ultra-fast hardware-accelerated inference for open models. In HealthRAG, Groq executes "
                "the `openai/gpt-oss-120b` open-weight model to synthesize grounded clinical answers and citations."
            ),
            "how_to_get": [
                "Visit https://console.groq.com/ and create a free account (Google, GitHub, or email).",
                "In the left navigation bar, click 'API Keys' (or go directly to console.groq.com/keys).",
                "Click the 'Create API Key' button.",
                "Enter a friendly name (e.g. 'HealthRAG') and click Submit.",
                "Copy your key immediately (starts with 'gsk_'). Groq will not display it again.",
                "In the HealthRAG sidebar under 'System 2: Answer LLM', select 'groq' and paste the key into 'LLM API Key'.",
                "Test by loading the demo document and asking a question in the chat.",
            ],
        },
        {
            "id": "gemini",
            "name": "Google Gemini",
            "model": "gemini-1.5-flash",
            "role": "System 2 Answer Generation LLM",
            "recommended": False,
            "url": OFFICIAL_LINKS["gemini"],
            "docs_url": "https://ai.google.dev/docs",
            "free_allowance": (
                "Free Tier available in Google AI Studio without credit card (verified Oct 2026). "
                "Rate limits: 15 requests/min (RPM), 1,500 requests/day (RPD), 1,000,000 tokens/min (TPM). "
                "Note: In the free tier, submitted data may be used by Google to improve its models."
            ),
            "paid_details": "Paid pay-as-you-go available by attaching a Google Cloud billing account.",
            "description": (
                "Google's lightweight multimodal model optimized for speed and long-context processing. "
                "Generates structured answers, extracts medical entities, and formats citations."
            ),
            "how_to_get": [
                "Visit https://aistudio.google.com/apikey and sign in with your Google account.",
                "Click the blue 'Create API key' button.",
                "Choose to create the key in a new Google Cloud project or an existing project.",
                "Copy your generated API key (starts with 'AIza...').",
                "In HealthRAG sidebar under 'System 2: Answer LLM', select 'gemini'.",
                "Paste your key into 'LLM API Key'.",
                "Test by submitting a question in the chat.",
            ],
        },
        {
            "id": "openai",
            "name": "OpenAI",
            "model": "gpt-4o-mini",
            "role": "System 2 Answer Generation LLM",
            "recommended": False,
            "url": OFFICIAL_LINKS["openai"],
            "docs_url": "https://platform.openai.com/docs",
            "free_allowance": (
                "New accounts historically receive limited promotional credits. "
                "Select ChatGPT Pro ($100+/mo) or Business subscribers receive linked API credits. "
                "IMPORTANT: A standard ChatGPT Plus subscription does NOT include API credits; "
                "API usage requires a funded platform account."
            ),
            "paid_details": (
                "Standard pay-as-you-go (~$0.15/1M input tokens, ~$0.60/1M output tokens). "
                "Requires a funded balance under platform.openai.com/account/billing."
            ),
            "description": (
                "OpenAI's high-speed cost-efficient model. Synthesizes clinical evidence into strict "
                "Pydantic MedicalAnswer structures with confidence rating and clinical warnings."
            ),
            "how_to_get": [
                "Visit https://platform.openai.com/api-keys and log in or create an account.",
                "Ensure your account has billing credits under Settings -> Billing.",
                "Click '+ Create new secret key'.",
                "Name your key (e.g. 'HealthRAG') and click 'Create secret key'.",
                "Copy your key (starts with 'sk-proj-' or 'sk-'). OpenAI will not display it again.",
                "In HealthRAG sidebar under 'System 2: Answer LLM', choose 'openai'.",
                "Paste your key into 'LLM API Key'.",
                "Test by submitting a question in the chat.",
            ],
        },
    ],
}

# 12 Required Troubleshooting FAQs
COMMON_PROBLEMS_FAQS = [
    {
        "q": "What is an API key?",
        "a": (
            "An **API key** (Application Programming Interface key) is a unique alphanumeric passcode that identifies your "
            "developer account to an AI model provider. It acts like a specialized password, permitting HealthRAG to send "
            "text queries and receive model completions on your behalf. HealthRAG never hardcodes keys; you provide your "
            "own keys in the sidebar per session."
        ),
    },
    {
        "q": "Why is my API key rejected?",
        "a": (
            "Common causes for rejected API keys include:\n\n"
            "1. **Extra whitespace:** Accidentally copying spaces or newlines before or after the key string.\n"
            "2. **Provider mismatch:** Pasting a Groq key (`gsk_...`) into the OpenAI field or a System 2 key into the System 1 field.\n"
            "3. **Missing account billing:** OpenAI requires funded credits in your developer balance; a consumer ChatGPT subscription does not fund API requests.\n"
            "4. **Unverified email:** Providers like JevModel require email confirmation before the API key becomes active.\n"
            "5. **Revoked key:** The key was deleted or revoked in the provider's developer console."
        ),
    },
    {
        "q": "Why am I getting a quota or rate-limit error?",
        "a": (
            "A quota or rate-limit error (HTTP 429) occurs when you exceed a provider's allowed request thresholds:\n\n"
            "- **Groq Free:** Limited to 30 requests/minute and 1,000 requests/day.\n"
            "- **Gemini Free:** Limited to 15 requests/minute and 1,500 requests/day.\n"
            "- **JevModel:** Limited to 5 free runs per UTC day.\n"
            "- **Laya Studio:** Limited to 5 free trial runs per new workspace.\n"
            "- **Codiv:** May return HTTP 529 when the public experiment cluster is at capacity.\n\n"
            "**Resolution:** Wait 1 to 2 minutes before retrying, switch to another provider in the sidebar dropdown (e.g. switch from JevModel to Codiv, or OpenAI to Groq), or add prepaid credits in your provider dashboard."
        ),
    },
    {
        "q": "Why does System 1 connection testing fail?",
        "a": (
            "The **Test System 1 Connection** button sends a lightweight probe request to the backend (`POST /system1/test`). "
            "If it fails:\n\n"
            "1. **Missing or invalid key:** Check that you pasted a valid key for the selected System 1 provider.\n"
            "2. **Backend is offline:** Ensure the FastAPI server is running on port 8000.\n"
            "3. **Provider cluster capacity:** Codiv may return HTTP 529 when overloaded; JevModel may return an error if your daily 5 runs have been consumed.\n"
            "4. **Strict Allowlist:** HealthRAG strictly restricts connections to official provider HTTPS endpoints (`ALLOWED_SYSTEM1_ENDPOINTS`) to prevent security leaks."
        ),
    },
    {
        "q": "Why is the backend unavailable?",
        "a": (
            "HealthRAG consists of two components: a FastAPI backend (port 8000) and a Streamlit UI (port 8501).\n\n"
            "- **In local development:** You must start the FastAPI backend (`uvicorn app.main:app --port 8000`) before or alongside Streamlit (`python run.py` starts both automatically).\n"
            "- **On hosted Streamlit Cloud:** If the backend is running on a separate host, verify that the `API_URL` environment variable points to a reachable, active backend instance."
        ),
    },
    {
        "q": "Why is my answer incorrect or missing?",
        "a": (
            "HealthRAG is built to prevent medical hallucinations through grounded retrieval and System 1 gating:\n\n"
            "1. **Evidence not present:** If the uploaded document lacks the requested metric (e.g. asking for potassium when only hemoglobin is in the report), System 1 will flag insufficient evidence and safely abstain.\n"
            "2. **Scanned PDF without OCR:** If the document is an image-only scan, ensure OCR is enabled on the server (`OCR_ENABLED=true` with Tesseract).\n"
            "3. **Patient scope filter:** If a specific patient scope or document filter is selected in the sidebar, chunks outside that scope are excluded from retrieval.\n"
            "4. **Question phrasing:** Try rephrasing your question to match the terminology used in the lab report."
        ),
    },
    {
        "q": "What is the difference between System 1 and System 2?",
        "a": (
            "- **System 1 (Evidence Decision Gate):** A fast, non-autoregressive decision model (OpenJev, JevModel, or Laya) "
            "that strictly evaluates: *'Based only on the retrieved excerpts, is there sufficient relevant information to answer the question?'* "
            "It outputs a calibrated sufficiency probability. If evidence is lacking, it triggers bounded retrieval or safely abstains, preventing the LLM from fabricating data.\n\n"
            "- **System 2 (Answer LLM):** A generative Large Language Model (Groq gpt-oss-120b, OpenAI gpt-4o-mini, or Gemini 1.5 Flash) "
            "that reads the validated evidence and synthesizes a natural-language clinical answer, extracts medical entities, and formats citations."
        ),
    },
    {
        "q": "Can I use HealthRAG without System 1?",
        "a": (
            "**Yes.** In the sidebar, simply uncheck **'Enable System 1 Gate'**. "
            "When unchecked, HealthRAG operates in baseline RAG mode: retrieved evidence chunks are sent directly to the System 2 LLM "
            "without pre-generation sufficiency validation. You only need a System 2 API key when System 1 is disabled."
        ),
    },
    {
        "q": "Does the app diagnose diseases?",
        "a": (
            "**No.** HealthRAG is strictly an information-retrieval and document-reading assistant for clinical and lab reports. "
            "It extracts and explains data already documented in uploaded records. It does **not** provide medical diagnoses, "
            "treatment plans, or clinical decision-making. All responses that discuss medical findings carry an advisory that they require "
            "independent verification by a licensed healthcare professional."
        ),
    },
    {
        "q": "Where can I see the source citations?",
        "a": (
            "Underneath every generated response in the chat, look for the expandable **'Source Citations'** section. "
            "Expanding it reveals source chips indicating the exact document name, upload ID, and page number for each retrieved passage. "
            "You can also click the **'📄 Document Viewer'** tab at the top of the main area to view the full indexed PDF directly."
        ),
    },
    {
        "q": "Are API keys saved?",
        "a": (
            "**No.** API keys are request-scoped. They are stored only in your active browser session memory and sent over secure HTTPS "
            "headers/payloads for the immediate request. Keys are never written to a database, stored on server disk, or printed in logs. "
            "When you refresh or close your browser tab, all session keys are completely discarded."
        ),
    },
    {
        "q": "Is my healthcare data sent to external providers?",
        "a": (
            "**Yes.** When you submit a question, your query string and the specific retrieved text excerpts (chunks) from your uploaded PDF "
            "are transmitted via HTTPS to your chosen external AI providers (System 1 gate and System 2 LLM). Full PDFs and the complete vector database "
            "are not transmitted. Because public and free tiers are not HIPAA-compliant, **never upload real protected health information (PHI)**; "
            "always use synthetic, de-identified, or publicly available sample reports."
        ),
    },
]


def get_faq_data() -> Dict[str, Any]:
    """Returns structured FAQ and provider configuration data for inspection and unit testing."""
    return {
        "links": OFFICIAL_LINKS,
        "providers": SUPPORTED_PROVIDERS,
        "troubleshooting": COMMON_PROBLEMS_FAQS,
    }


def render_security_notice():
    """Renders the standard security and medical privacy advisory."""
    st.markdown(
        """
        <div style="background:#fffbeb; border:1px solid #fde68a; border-left:5px solid #f59e0b; padding:14px 18px; border-radius:10px; margin-bottom:1.2rem; color:#92400e;">
            <div style="font-weight:700; font-size:1.02rem; margin-bottom:6px;">🔒 Security & Medical Privacy Advisory</div>
            <ul style="margin:0; padding-left:1.2rem; font-size:0.88rem; line-height:1.5;">
                <li><strong>Treat API keys like passwords:</strong> Never share your keys publicly, commit them to GitHub, or show them in video recordings or screenshots.</li>
                <li><strong>Synthetic & De-Identified Data Only:</strong> Never upload real Protected Health Information (PHI). Use synthetic or publicly released clinical reports.</li>
                <li><strong>External Data Transmission:</strong> When asking questions, your query and retrieved text excerpts are sent to your selected external AI provider. Public free tiers do NOT offer HIPAA compliance or Business Associate Agreements (BAAs).</li>
                <li><strong>Research Prototype Only:</strong> HealthRAG is an experimental demonstration platform for RAG architectures. It is <strong>not</strong> a certified medical device and does <strong>not</strong> provide primary clinical diagnoses or medical advice.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workflow_guide():
    """Renders the step-by-step usage guide."""
    st.markdown(
        """
        ### 🚀 How to Use HealthRAG (Step-by-Step)

        Follow this simple 9-step workflow to analyze clinical documents:

        1. **Open the Application:** Make sure the backend is active and the Streamlit interface is loaded.
        2. **Upload a Healthcare Report:** In the sidebar under **Document Management**, click **Browse files** to upload a clinical PDF (up to 5MB), or click **Load Demo Report (Hemoglobin Lab)** for an instant pre-loaded sample.
        3. **Choose a System 2 Provider:** Select your preferred answer-generation provider (`groq`, `openai`, or `gemini`). Groq with `openai/gpt-oss-120b` is recommended for high-speed free testing.
        4. **Enter your System 2 API Key:** Paste the corresponding key into the **LLM API Key** input field in the sidebar.
        5. **Configure System 1 (Optional but Recommended):** Keep **Enable System 1 Gate** checked if you want evidence-sufficiency validation before answer generation.
        6. **Select a System 1 Provider:** Choose **Codiv / OpenJev (`openjev-latest`)** [Recommended], **JevModel (`typesafe/jev-1.13`)**, or **Laya Studio**, and paste its API key.
        7. **Test System 1 Connection:** Click the **Test System 1 Connection** button in the sidebar. This confirms network connectivity, latency, and allowance to the gate model.
           > ℹ️ *Note: The connection test verifies System 1 only. System 2 is tested when asking a live question.*
        8. **Ask a Medical Question:** In the chat input at the bottom, type a question whose answer is contained in the uploaded report (e.g., *"What is the patient's hemoglobin value?"* or *"Are there any abnormal findings?"*).
        9. **Review Output & Citations:** Review the grounded answer, clinical confidence score, System 1 decision banner, identified clinical entities, and expand **Source Citations** to inspect referenced pages.

        > 💡 **Baseline Mode:** You can uncheck **Enable System 1 Gate** at any time to run baseline System 2 directly (standard RAG without decision-gate checking).
        """
    )


def render_free_tier_table():
    """Renders the System 1 and System 2 free-tier comparison table."""
    st.markdown(
        """
        ### 💰 Provider Free Allowances & Tier Comparison

        The table below details current free allowances and terms for all configured models.
        """
    )

    st.markdown(
        """
| System Layer | Provider | Model ID in HealthRAG | Free Tier / Trial Allowance | Limitations & Notes | Official Documentation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **System 1 (Gate)** | **Codiv / OpenJev** *(Recommended)* | `openjev-latest` | **100M free input tokens** per account *(verified Oct 2026)* | Public experiment; quota tracked on Codiv dashboard. Returns HTTP 529 when cluster is busy. | [codiv.ai](https://codiv.ai/) |
| **System 1 (Gate)** | **JevModel** | Server-fixed (`typesafe/jev-1.13`) | **5 free runs / day** UTC *(verified Oct 2026)* | Requires email verification. Subject to service capacity. Independent hosted service. | [jevmodel.app](https://jevmodel.app/) |
| **System 1 (Gate)** | **Laya Studio** | Automatic routing (`laya-latest`) | **5 trial runs** per new workspace *(verified Oct 2026)* | Swiss-hosted ModernBERT architecture. Paid workspace needed after trial. | [laya.studio](https://laya.studio/) |
| **System 2 (LLM)** | **Groq** *(Recommended Free)* | `openai/gpt-oss-120b` | **Free Developer Tier** *(no credit card required)* | 30 RPM, 1,000 RPD, ~8K TPM. High speed. Paid tier available for scaling. | [console.groq.com](https://console.groq.com/) |
| **System 2 (LLM)** | **Google Gemini** | `gemini-1.5-flash` | **Free in Google AI Studio** *(no credit card required)* | 15 RPM, 1,500 RPD, 1M TPM. Free-tier data may be reviewed to train Google models. | [aistudio.google.com](https://aistudio.google.com/apikey) |
| **System 2 (LLM)** | **OpenAI** | `gpt-4o-mini` | **Trial credits on new accounts / select Pro plans** | Standard API is pay-as-you-go (~$0.15/1M in, $0.60/1M out). Requires funded billing balance. | [platform.openai.com](https://platform.openai.com/api-keys) |
        """
    )

    st.caption(
        "⚠️ **Disclaimer:** Free tier terms, rate limits, and promotional credit availability are established by each external provider "
        "and are subject to change. None of the providers offer unlimited free usage. Check official links for current pricing."
    )


def render_api_key_guide():
    """Renders expandable instructions for obtaining API keys for each provider."""
    st.markdown("### 🔑 How to Obtain API Keys")
    st.markdown("Expand any provider below for step-by-step instructions on obtaining an API key:")

    st.markdown("#### System 2 — Answer Generation LLMs")
    for p in SUPPORTED_PROVIDERS["system2"]:
        rec_tag = " [Recommended Free]" if p.get("recommended") else ""
        with st.expander(f"🔹 {p['name']} ({p['model']}){rec_tag}"):
            st.markdown(f"**Role:** {p['role']}")
            st.markdown(f"**Model ID in Code:** `{p['model']}`")
            st.markdown(f"**Official Website:** [{p['url']}]({p['url']})")
            st.markdown(f"**Free Allowance:** {p['free_allowance']}")
            st.markdown(f"**Paid Terms:** {p['paid_details']}")
            st.markdown(f"**What it does in HealthRAG:** {p['description']}")
            st.markdown("**Step-by-Step Instructions:**")
            for step in p["how_to_get"]:
                st.markdown(f"- {step}")

    st.markdown("#### System 1 — Evidence Decision Gate Models")
    for p in SUPPORTED_PROVIDERS["system1"]:
        rec_tag = " [Recommended Free]" if p.get("recommended") else ""
        with st.expander(f"⚡ {p['name']} ({p['model']}){rec_tag}"):
            st.markdown(f"**Role:** {p['role']}")
            st.markdown(f"**Model ID in Code:** `{p['model']}`")
            st.markdown(f"**Official Website:** [{p['url']}]({p['url']})")
            st.markdown(f"**Free Allowance:** {p['free_allowance']}")
            st.markdown(f"**Paid Terms:** {p['paid_details']}")
            st.markdown(f"**What it does in HealthRAG:** {p['description']}")
            st.markdown("**Step-by-Step Instructions:**")
            for step in p["how_to_get"]:
                st.markdown(f"- {step}")


def render_troubleshooting():
    """Renders expandable answers for the 12 common problems."""
    st.markdown("### 🛠️ Common Problems & Troubleshooting")
    st.markdown("Click on any question below to see the verified explanation and solution:")

    for item in COMMON_PROBLEMS_FAQS:
        with st.expander(f"❓ {item['q']}"):
            st.markdown(item["a"])


def render_faq_panel():
    """Renders the comprehensive FAQ & Help panel with organized tabs."""
    render_security_notice()

    faq_tabs = st.tabs([
        "🚀 Quickstart Workflow",
        "🔑 Getting API Keys",
        "💰 Free Tier Comparison",
        "🛠️ Troubleshooting & FAQ",
    ])

    with faq_tabs[0]:
        render_workflow_guide()

    with faq_tabs[1]:
        render_api_key_guide()

    with faq_tabs[2]:
        render_free_tier_table()

    with faq_tabs[3]:
        render_troubleshooting()


@st.dialog("❓ HealthRAG — FAQ & Setup Guide", width="large")
def show_faq_dialog():
    """Displays the FAQ and Help section in a clean modal dialog."""
    render_faq_panel()
    st.markdown("---")
    if st.button("✕ Close FAQ", use_container_width=True, key="dlg_close_btn"):
        st.rerun()
