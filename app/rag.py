from langchain_core.messages import HumanMessage, SystemMessage
from app.vector_store import search_documents
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

def answer_question(question: str, provider: str, api_key: str) -> Dict[str, Any]:
    """
    Retrieves relevant chunks and asks the selected AI to answer the question
    based ONLY on the context.
    """
    # 1. Retrieve relevant chunks (now uses free local embeddings!)
    relevant_chunks = search_documents(question)

    if not relevant_chunks:
        return {
            "answer": "I could not find any uploaded documents or relevant context.",
            "sources": []
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

    # 4. Call dynamically selected AI Provider
    chat = get_llm(provider, api_key)
    
    response = chat.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

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

    return {
        "answer": response.content,
        "sources": sources_list
    }
