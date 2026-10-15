"""Local persistent vector store for the RAG prototype."""
import os
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from typing import Dict, List

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

chroma_client = chromadb.PersistentClient(path="./data/chroma")
collection = chroma_client.get_or_create_collection(name="healthcare_documents")
_embeddings_instance = None


def get_embeddings_model():
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings_instance


def add_documents(chunks: List[Dict], session_id: str = "global"):
    if not chunks:
        return
    embeddings = get_embeddings_model()
    texts = [chunk["text"] for chunk in chunks]
    metadatas = []
    ids = []
    for index, chunk in enumerate(chunks):
        metadata = chunk["metadata"].copy()
        metadata["session_id"] = session_id
        metadatas.append(metadata)
        ids.append(
            f"{session_id}_{metadata['document_name']}_{metadata['page']}_{index}"
        )

    # Make re-uploading the same document idempotent rather than failing on IDs.
    collection.delete(
        where={
            "$and": [
                {"session_id": session_id},
                {"document_name": chunks[0]["metadata"]["document_name"]},
            ]
        }
    )
    vectors = embeddings.embed_documents(texts)
    collection.add(
        ids=ids, embeddings=vectors, metadatas=metadatas, documents=texts
    )


def search_documents(question: str) -> List[Dict]:
    """Simple dense-only utility retained for compatibility."""
    embeddings = get_embeddings_model()
    result = collection.query(
        query_embeddings=[embeddings.embed_query(question)],
        n_results=min(4, max(1, collection.count())),
    )
    return [
        {"text": text, "metadata": metadata}
        for text, metadata in zip(
            (result.get("documents") or [[]])[0],
            (result.get("metadatas") or [[]])[0],
        )
    ]


def delete_document(document_name: str, session_id: str = "global"):
    collection.delete(
        where={
            "$and": [
                {"document_name": document_name},
                {"session_id": session_id},
            ]
        }
    )


def list_documents(session_id: str = "global") -> List[str]:
    results = collection.get(
        where={"session_id": session_id}, include=["metadatas"]
    )
    names = {
        metadata["document_name"]
        for metadata in (results.get("metadatas") or [])
        if metadata and metadata.get("document_name")
    }
    return sorted(names)
