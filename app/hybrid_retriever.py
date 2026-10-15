"""Hybrid lexical + semantic retrieval with Reciprocal Rank Fusion."""
import hashlib
import re
from collections import OrderedDict
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi

from app.config import TOP_K
from app.vector_store import collection, get_embeddings_model

_BM25_CACHE = OrderedDict()
_MAX_BM25_CACHE_ENTRIES = 8


def _where_clause(document_names=None, session_id="global", document_ids=None, patient_label=None):
    clauses = [{"session_id": session_id}]
    if document_ids:
        clauses.append({"document_id": document_ids[0]} if len(document_ids) == 1 else {"document_id": {"$in": document_ids}})
    elif document_names:
        clauses.append({"document_name": document_names[0]} if len(document_names) == 1 else {"document_name": {"$in": document_names}})
    if patient_label:
        clauses.append({"patient_label": patient_label})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class HybridRetriever:
    """Combine BM25 and dense ranks without comparing their raw scores."""

    @staticmethod
    def get_all_chunks(document_names=None, session_id="global", document_ids=None, patient_label=None):
        where = _where_clause(document_names, session_id, document_ids, patient_label)
        results = collection.get(where=where, include=["documents", "metadatas"])
        documents, metadatas, ids = (
            results.get("documents") or [],
            results.get("metadatas") or [],
            results.get("ids") or [],
        )
        return [
            {"id": doc_id, "text": text, "metadata": metadata}
            for doc_id, text, metadata in zip(ids, documents, metadatas)
        ]

    @staticmethod
    def _get_all_chunks(document_names=None, session_id="global", document_ids=None, patient_label=None):
        return HybridRetriever.get_all_chunks(document_names, session_id, document_ids, patient_label)

    @staticmethod
    def search(
        question: str,
        top_k: int = TOP_K,
        document_names: Optional[List[str]] = None,
        session_id: str = "global",
        candidate_k: Optional[int] = None,
        document_ids: Optional[List[str]] = None,
        patient_label: Optional[str] = None,
    ) -> List[Dict]:
        if not question or not question.strip() or top_k <= 0:
            return []
        chunks = HybridRetriever.get_all_chunks(document_names, session_id, document_ids, patient_label)
        if not chunks:
            return []
        candidate_k = min(max(top_k, candidate_k or top_k * 4), len(chunks))
        where = _where_clause(document_names, session_id, document_ids, patient_label)

        def tokenize(value):
            return re.findall(r"[a-z]+|\d+(?:\.\d+)?", value.lower())

        fingerprint_source = "\n".join(f"{chunk['id']}\0{chunk['text']}" for chunk in chunks)
        fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()
        cached = _BM25_CACHE.get(fingerprint)
        if cached is None:
            corpus = [tokenize(chunk["text"]) for chunk in chunks]
            bm25 = BM25Okapi(corpus)
            _BM25_CACHE[fingerprint] = (bm25, corpus)
            while len(_BM25_CACHE) > _MAX_BM25_CACHE_ENTRIES:
                _BM25_CACHE.popitem(last=False)
        else:
            bm25, corpus = cached
            _BM25_CACHE.move_to_end(fingerprint)

        query_tokens = tokenize(question)
        if not query_tokens:
            return []
        bm25_scores = bm25.get_scores(query_tokens)
        bm25_indices = sorted(range(len(chunks)), key=lambda i: bm25_scores[i], reverse=True)[:candidate_k]

        embeddings = get_embeddings_model()
        dense_results = collection.query(
            query_embeddings=[embeddings.embed_query(question)],
            n_results=candidate_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        dense_ids = (dense_results.get("ids") or [[]])[0]
        dense_ranks = {doc_id: rank for rank, doc_id in enumerate(dense_ids)}
        fused_scores = {}
        chunks_by_id = {chunk["id"]: chunk for chunk in chunks}
        for rank, index in enumerate(bm25_indices):
            doc_id = chunks[index]["id"]
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1 / (60 + rank + 1)
        for doc_id, rank in dense_ranks.items():
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1 / (60 + rank + 1)
        ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
        return [chunks_by_id[doc_id] for doc_id in ranked_ids[:top_k] if doc_id in chunks_by_id]
