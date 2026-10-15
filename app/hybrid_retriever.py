"""Hybrid lexical + semantic retrieval with Reciprocal Rank Fusion.

The lexical corpus is rebuilt from the selected session's indexed chunks per
request. This is simple and correct for a prototype; for large corpora, replace
it with a persistent lexical index and bounded candidate retrieval.
"""
import re
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi

from app.config import TOP_K
from app.vector_store import collection, get_embeddings_model


def _where_clause(document_names: Optional[List[str]], session_id: str):
    """Build a strict session filter; user sessions never inherit other sessions."""
    clauses = [{"session_id": session_id}]
    if document_names:
        clauses.append(
            {"document_name": document_names[0]}
            if len(document_names) == 1
            else {"document_name": {"$in": document_names}}
        )
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


class HybridRetriever:
    """Combine BM25 and dense ranks without comparing their raw scores."""

    @staticmethod
    def _get_all_chunks(
        document_names: Optional[List[str]] = None,
        session_id: str = "global",
    ) -> List[Dict]:
        where = _where_clause(document_names, session_id)
        results = collection.get(where=where, include=["documents", "metadatas"])
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or []
        ids = results.get("ids") or []
        return [
            {"id": doc_id, "text": text, "metadata": metadata}
            for doc_id, text, metadata in zip(ids, documents, metadatas)
        ]

    @staticmethod
    def search(
        question: str,
        top_k: int = TOP_K,
        document_names: Optional[List[str]] = None,
        session_id: str = "global",
        candidate_k: Optional[int] = None,
    ) -> List[Dict]:
        if not question or not question.strip() or top_k <= 0:
            return []

        chunks = HybridRetriever._get_all_chunks(document_names, session_id)
        if not chunks:
            return []

        candidate_k = max(top_k, candidate_k or top_k * 4)
        candidate_k = min(candidate_k, len(chunks))
        where = _where_clause(document_names, session_id)

        # BM25 lexical ranking. Keep punctuation-normalized tokens so common
        # numeric/unit spellings remain discoverable without external services.
        tokenize = lambda value: __import__("re").findall(
            r"[a-zA-Z]+|\d+(?:\.\d+)?|[a-zA-Z]+/ [a-zA-Z]+|[a-zA-Z]+/[a-zA-Z]+",
            value.lower().replace(" ", " "),
        )
        corpus = [tokenize(chunk["text"]) for chunk in chunks]
        query_tokens = tokenize(question)
        if not query_tokens:
            return []
        bm25 = BM25Okapi(corpus)
        bm25_scores = bm25.get_scores(query_tokens)
        bm25_indices = sorted(
            range(len(chunks)), key=lambda i: bm25_scores[i], reverse=True
        )[:candidate_k]

        # Retrieve a bounded dense candidate set, not every chunk in the corpus.
        embeddings = get_embeddings_model()
        query_embedding = embeddings.embed_query(question)
        dense_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        dense_ids = (dense_results.get("ids") or [[]])[0]
        dense_ranks = {doc_id: rank for rank, doc_id in enumerate(dense_ids)}

        # Reciprocal Rank Fusion uses only candidates present in either ranking.
        fused_scores: Dict[str, float] = {}
        chunks_by_id = {chunk["id"]: chunk for chunk in chunks}
        rrf_constant = 60
        for rank, index in enumerate(bm25_indices):
            doc_id = chunks[index]["id"]
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1 / (
                rrf_constant + rank + 1
            )
        for doc_id, rank in dense_ranks.items():
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1 / (
                rrf_constant + rank + 1
            )

        ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
        return [chunks_by_id[doc_id] for doc_id in ranked_ids[:top_k] if doc_id in chunks_by_id]
