"""Local persistent vector store for the educational RAG prototype."""
import hashlib
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from typing import Dict, List

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

chroma_client = chromadb.PersistentClient(path="./data/chroma")
collection = chroma_client.get_or_create_collection(name="healthcare_documents")
_embeddings_instance = None


def get_embeddings_model():
    """Load the embedding model only once per process."""
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings_instance


def _chunk_id(session_id: str, document_name: str, page: int, index: int) -> str:
    raw = f"{session_id}\0{document_name}\0{page}\0{index}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def add_documents(chunks: List[Dict], session_id: str = "global"):
    """Upsert new vectors before pruning stale chunks from a replaced document."""
    if not chunks:
        return
    document_name = chunks[0]["metadata"]["document_name"]
    if any(c.get("metadata", {}).get("document_name") != document_name for c in chunks):
        raise ValueError("add_documents expects chunks from exactly one document")

    texts, metadatas, ids = [], [], []
    for index, chunk in enumerate(chunks):
        metadata = chunk["metadata"].copy()
        metadata["session_id"] = session_id
        metadatas.append(metadata)
        texts.append(chunk["text"])
        ids.append(_chunk_id(session_id, document_name, int(metadata["page"]), index))

    # Do costly embedding before touching the old index, so embedding failures
    # do not remove the currently indexed version of a document.
    vectors = get_embeddings_model().embed_documents(texts)
    where = {"$and": [{"session_id": session_id}, {"document_name": document_name}]}
    previous_ids = collection.get(where=where, include=[]).get("ids", [])
    collection.upsert(ids=ids, embeddings=vectors, metadatas=metadatas, documents=texts)
    current_ids = set(ids)
    stale_ids = [item for item in previous_ids if item not in current_ids]
    if stale_ids:
        collection.delete(ids=stale_ids)


def search_documents(question: str) -> List[Dict]:
    """Simple dense-only utility retained for compatibility."""
    count = collection.count()
    if count == 0:
        return []
    embeddings = get_embeddings_model()
    result = collection.query(
        query_embeddings=[embeddings.embed_query(question)],
        n_results=min(4, count),
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
        where={"$and": [{"document_name": document_name}, {"session_id": session_id}]}
    )


def list_documents(session_id: str = "global") -> List[str]:
    results = collection.get(where={"session_id": session_id}, include=["metadatas"])
    names = {
        metadata["document_name"]
        for metadata in (results.get("metadatas") or [])
        if metadata and metadata.get("document_name")
    }
    return sorted(names)
