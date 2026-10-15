"""Local persistent vector store for the educational RAG prototype."""
import hashlib
import os
import uuid
from typing import Dict, List, Optional

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import chromadb

chroma_client = chromadb.PersistentClient(path="./data/chroma")
collection = chroma_client.get_or_create_collection(name="healthcare_documents")
_embeddings_instance = None


def get_embeddings_model():
    """Load the optional embedding dependency and model only when retrieval needs it."""
    global _embeddings_instance
    if _embeddings_instance is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings_instance


def _chunk_id(session_id: str, document_id: str, page: int, index: int) -> str:
    raw = f"{session_id}\0{document_id}\0{page}\0{index}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def add_documents(
    chunks: List[Dict],
    session_id: str = "global",
    document_id: Optional[str] = None,
    patient_label: str = "",
):
    """Index one upload under a unique ID; same-name uploads never replace each other."""
    if not chunks:
        return
    document_name = chunks[0]["metadata"]["document_name"]
    if any(c.get("metadata", {}).get("document_name") != document_name for c in chunks):
        raise ValueError("add_documents expects chunks from exactly one document")

    document_id = document_id or uuid.uuid4().hex
    texts, metadatas, ids = [], [], []
    for index, chunk in enumerate(chunks):
        metadata = chunk["metadata"].copy()
        metadata.update({
            "session_id": session_id,
            "document_id": document_id,
            "patient_label": patient_label.strip() or "unspecified",
        })
        metadatas.append(metadata)
        texts.append(chunk["text"])
        ids.append(_chunk_id(session_id, document_id, int(metadata["page"]), index))

    # Embed before writing, so embedding failures do not leave a partial upload.
    vectors = get_embeddings_model().embed_documents(texts)
    collection.upsert(ids=ids, embeddings=vectors, metadatas=metadatas, documents=texts)


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


def delete_document(document_name: str, session_id: str = "global", document_id: Optional[str] = None):
    clauses = [{"document_name": document_name}, {"session_id": session_id}]
    if document_id:
        clauses.append({"document_id": document_id})
    collection.delete(where={"$and": clauses})


def list_document_records(session_id: str = "global") -> List[Dict]:
    """Return one record per uploaded document, including legacy indexed records."""
    results = collection.get(where={"session_id": session_id}, include=["metadatas"])
    records = {}
    for metadata in (results.get("metadatas") or []):
        if not metadata or not metadata.get("document_name"):
            continue
        # Legacy chunks did not have document_id; keep them visible as a legacy record.
        doc_id = metadata.get("document_id") or f"legacy:{metadata['document_name']}"
        records.setdefault(doc_id, {
            "document_id": doc_id,
            "document_name": metadata["document_name"],
            "patient_label": metadata.get("patient_label") or "unspecified",
            "session_id": metadata.get("session_id", session_id),
        })
    return sorted(records.values(), key=lambda item: (item["document_name"].casefold(), item["document_id"]))


def list_documents(session_id: str = "global") -> List[str]:
    """Backward-compatible list of unique filenames (not unique upload records)."""
    return sorted({record["document_name"] for record in list_document_records(session_id)})
