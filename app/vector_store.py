import os
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import TOP_K
from typing import List, Dict

# Initialize ChromaDB client (local persistence)
chroma_client = chromadb.PersistentClient(path="./data/chroma")

# Create or get collection
collection = chroma_client.get_or_create_collection(name="healthcare_documents")

_embeddings_instance = None

def get_embeddings_model():
    """
    Returns the HuggingFace embeddings model instance (cached).
    This runs entirely locally and for free. No API key required!
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings_instance

def add_documents(chunks: List[Dict], session_id: str = "global"):
    """
    Generates local embeddings for chunks and adds them to ChromaDB.
    """
    if not chunks:
        return

    embeddings = get_embeddings_model()
    texts = [chunk["text"] for chunk in chunks]
    
    # Inject session_id into metadata
    metadatas = []
    for chunk in chunks:
        meta = chunk["metadata"].copy()
        meta["session_id"] = session_id
        metadatas.append(meta)
    
    # Create unique IDs for each chunk based on document name, page, and chunk index
    ids = [f"{session_id}_{chunk['metadata']['document_name']}_{chunk['metadata']['page']}_{i}" for i, chunk in enumerate(chunks)]

    # Generate embeddings
    embeds = embeddings.embed_documents(texts)

    # Add to ChromaDB
    collection.add(
        ids=ids,
        embeddings=embeds,
        metadatas=metadatas,
        documents=texts
    )

def search_documents(question: str) -> List[Dict]:
    """
    Searches ChromaDB for the most relevant chunks based on semantic similarity.
    (Note: HybridRetriever now handles search logic primarily)
    """
    embeddings = get_embeddings_model()
    query_embedding = embeddings.embed_query(question)
    
    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K
    )

    retrieved_chunks = []
    if results["documents"] and len(results["documents"][0]) > 0:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            retrieved_chunks.append({
                "text": doc,
                "metadata": meta
            })
    return retrieved_chunks

def delete_document(document_name: str, session_id: str = "global"):
    """
    Deletes all chunks associated with a specific document.
    """
    collection.delete(
        where={"$and": [{"document_name": document_name}, {"session_id": session_id}]}
    )

def list_documents(session_id: str = "global") -> List[str]:
    """
    Returns a list of unique uploaded document names for a given session, 
    plus any global demo documents.
    """
    if session_id == "global":
        results = collection.get(include=["metadatas"])
    else:
        results = collection.get(
            where={"$or": [{"session_id": session_id}, {"session_id": "global"}]},
            include=["metadatas"]
        )
        
    metadatas = results["metadatas"]
    doc_names = set(meta["document_name"] for meta in metadatas)
    return list(doc_names)
