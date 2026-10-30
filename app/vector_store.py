"""Local persistent vector store for the educational RAG prototype."""
import hashlib
import os
import re
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
        import sys
        if "langchain_core.pydantic_v1" not in sys.modules:
            try:
                import pydantic.v1 as pydantic_v1
                sys.modules["langchain_core.pydantic_v1"] = pydantic_v1
            except ImportError:
                pass
        from langchain_huggingface import HuggingFaceEmbeddings
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings_instance


def _chunk_id(session_id: str, document_id: str, page: int, index: int) -> str:
    raw = f"{session_id}\0{document_id}\0{page}\0{index}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _infer_patient_label(chunks: List[Dict], document_name: str) -> str:
    """Infer only explicit synthetic Patient X labels; never infer real names."""
    labels = set()
    for chunk in chunks:
        text = str(chunk.get("text", "") or "")
        for match in re.finditer(r"\bPatient\s+([A-Za-z]|[0-9]{1,3})\b", text, re.I):
            token = match.group(1)
            labels.add(token.upper() if token.isalpha() else token)

    if len(labels) > 1:
        # A mixed-record PDF must not be assigned to whichever patient happens
        # to appear first. Keep it visibly distinct and require explicit scope.
        return "mixed"
    if len(labels) == 1:
        return f"Patient {next(iter(labels))}"

    filename_match = re.search(r"\bpatient[_-]([a-z0-9]+)\b", document_name, re.I)
    if filename_match:
        return f"Patient {filename_match.group(1).upper()}"
    return "unspecified"


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
    inferred_patient_label = _infer_patient_label(chunks, document_name)
    effective_patient_label = patient_label.strip() or inferred_patient_label
    texts, metadatas, ids = [], [], []
    for index, chunk in enumerate(chunks):
        metadata = chunk["metadata"].copy()
        metadata.update({
            "session_id": session_id,
            "document_id": document_id,
            "patient_label": effective_patient_label,
            "report_id": metadata.get("report_id", ""),
            "report_date": metadata.get("report_date", ""),
        })
        metadatas.append(metadata)
        texts.append(chunk["text"])
        ids.append(_chunk_id(session_id, document_id, int(metadata["page"]), index))

    # Embed before writing, so embedding failures do not leave a partial upload.
    vectors = get_embeddings_model().embed_documents(texts)
    collection.upsert(ids=ids, embeddings=vectors, metadatas=metadatas, documents=texts)


def search_documents(question: str, session_id: str = "global") -> List[Dict]:
    """Simple dense-only utility scoped to session_id for compatibility."""
    count = collection.count()
    if count == 0:
        return []
    embeddings = get_embeddings_model()
    result = collection.query(
        query_embeddings=[embeddings.embed_query(question)],
        n_results=min(4, count),
        where={"session_id": session_id},
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
        rec = records.get(doc_id)
        if rec is None:
            records[doc_id] = {
                "document_id": doc_id,
                "document_name": metadata["document_name"],
                "patient_label": metadata.get("patient_label") or "unspecified",
                "session_id": metadata.get("session_id", session_id),
                "report_id": metadata.get("report_id") or "",
                "report_date": metadata.get("report_date") or "",
            }
        else:
            if not rec.get("report_id") and metadata.get("report_id"):
                rec["report_id"] = metadata["report_id"]
            if not rec.get("report_date") and metadata.get("report_date"):
                rec["report_date"] = metadata["report_date"]
    return sorted(records.values(), key=lambda item: (item["document_name"].casefold(), item["document_id"]))


def list_documents(session_id: str = "global") -> List[str]:
    """Backward-compatible list of unique filenames (not unique upload records)."""
    return sorted({record["document_name"] for record in list_document_records(session_id)})
