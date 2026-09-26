import time
from langchain_core.messages import HumanMessage, SystemMessage
from app.hybrid_retriever import HybridRetriever
from app.metrics import RagMetrics
from app.query_router import QueryRouter
from app.config import TOP_K
from app.schemas import MedicalAnswer
from typing import Dict, Any

def get_llm(provider: str, api_key: str):
    """Instantiates the correct LLM based on the user's provider choice."""
    if not api_key.strip():
        raise ValueError(f"API key for {provider} is required.")
        
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(api_key=api_key, model_name="openai/gpt-oss-120b", temperature=0)
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-1.5-flash", temperature=0)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(api_key=api_key, model_name="gpt-4o-mini", temperature=0)
    else:
        raise ValueError(f"Unsupported AI provider: {provider}")

def answer_question(question: str, provider: str, api_key: str, document_names: list = None) -> Dict[str, Any]:
    """
    Retrieves relevant chunks and asks the selected AI to answer the question
    based ONLY on the context. Now with basic evaluation metrics.
    """
    start_time = time.time()
    chat = get_llm(provider, api_key)
    
    # 1. Route the Query (Phase 4 Adaptive Router)
    route = QueryRouter.classify(question, chat)
    
    if route == "IRRELEVANT":
        latency = round(time.time() - start_time, 2)
        return {
            "answer": "This question does not appear to be related to the uploaded healthcare documents. Please ask a medical or report-related question.",
            "sources": [],
            "metrics": {"latency_sec": latency, "context_precision": 0, "faithfulness": 0, "retrieved_chunks": 0},
            "route": route
        }
        
    # Dynamically adjust retrieval parameters
    k_value = TOP_K * 3 if route == "AGGREGATION" else TOP_K
    
    # 2. Retrieve relevant chunks using Phase 3 Hybrid Search (BM25 + Dense + RRF)
    relevant_chunks = HybridRetriever.search(question, top_k=k_value, document_names=document_names)

    if not relevant_chunks:
        return {
            "answer": "I could not find any uploaded documents or relevant context.",
            "sources": [],
            "metrics": None
        }

    # 2. Prepare the context
    context_text = "\n\n---\n\n".join(
        [f"Source: {chunk['metadata']['document_name']} (Page {chunk['metadata']['page']})\nContent:\n{chunk['text']}" for chunk in relevant_chunks]
    )

    # 3. Build Prompt
    system_prompt = (
        "You are a healthcare document assistant.\n"
        "Answer the user's question using ONLY the provided document context.\n"
        "Do not invent information.\n"
        "If the answer is not available in the provided context, say: 'I could not find this information in the uploaded document.'\n"
        "Mention the source page when possible.\n"
        "You are not a doctor and should not provide a diagnosis or replace professional medical advice.\n"
    )

    user_prompt = f"Context documents:\n{context_text}\n\nQuestion: {question}"

    # 4. Call dynamically selected AI Provider with Structured Output
    try:
        structured_llm = chat.with_structured_output(MedicalAnswer)
        structured_response = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        
        answer_text = structured_response.answer
        structured_data = {
            "confidence": structured_response.confidence,
            "medical_entities": structured_response.medical_entities,
            "requires_doctor_review": structured_response.requires_doctor_review
        }
    except Exception as e:
        # Graceful fallback if the specific provider model struggles with tool-calling
        response = chat.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        answer_text = response.content
        structured_data = {
            "confidence": "unknown", 
            "medical_entities": [], 
            "requires_doctor_review": True
        }

    # 5. Format sources
    sources_list = []
    # Use a set to avoid duplicate sources if multiple chunks are retrieved from the same page
    seen = set()
    for chunk in relevant_chunks:
        doc = chunk["metadata"]["document_name"]
        page = chunk["metadata"]["page"]
        source_key = (doc, page)
        if source_key not in seen:
            sources_list.append({"document": doc, "page": page})
            seen.add(source_key)

    # 6. Calculate Metrics (Phase 6 feature)
    latency = round(time.time() - start_time, 2)
    metrics = RagMetrics.log_metrics(question, answer_text, relevant_chunks, latency)

    return {
        "answer": answer_text,
        "sources": sources_list,
        "metrics": metrics,
        "route": route,
        "structured_data": structured_data
    }
