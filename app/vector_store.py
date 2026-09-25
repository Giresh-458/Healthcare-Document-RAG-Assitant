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

def add_documents(chunks: List[Dict]):
    """
    Generates local embeddings for chunks and adds them to ChromaDB.
    """
    if not chunks:
        return

    embeddings = get_embeddings_model()
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    
    # Create unique IDs for each chunk based on document name, page, and chunk index
    ids = [f"{chunk['metadata']['document_name']}_{chunk['metadata']['page']}_{i}" for i, chunk in enumerate(chunks)]

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

def delete_document(document_name: str):
    """
    Deletes all chunks associated with a specific document.
    """
    collection.delete(
        where={"document_name": document_name}
    )

def list_documents() -> List[str]:
    """
    Returns a list of unique uploaded document names.
    """
    results = collection.get(include=["metadatas"])
    metadatas = results["metadatas"]
    doc_names = set(meta["document_name"] for meta in metadatas)
    return list(doc_names)
