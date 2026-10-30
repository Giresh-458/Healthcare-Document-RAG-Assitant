"""Hybrid lexical + semantic retrieval with Reciprocal Rank Fusion."""
import hashlib
import re
import threading
from collections import OrderedDict
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi

from app.config import TOP_K
from app.text_processor import clean_text
from app.vector_store import collection, get_embeddings_model

_BM25_CACHE = OrderedDict()
_BM25_LOCK = threading.Lock()
_MAX_BM25_CACHE_ENTRIES = 8


def _where_clause(
    document_names=None,
    session_id="global",
    document_ids=None,
    patient_label=None,
    report_id=None,
):
    clauses = [{"session_id": session_id}]
    if document_ids is not None:
        if len(document_ids) == 0:
            clauses.append({"document_id": "__NO_DOC_SELECTED__"})
        elif len(document_ids) == 1:
            clauses.append({"document_id": document_ids[0]})
        else:
            clauses.append({"document_id": {"$in": document_ids}})
    elif document_names is not None:
        if len(document_names) == 0:
            clauses.append({"document_name": "__NO_DOC_SELECTED__"})
        elif len(document_names) == 1:
            clauses.append({"document_name": document_names[0]})
        else:
            clauses.append({"document_name": {"$in": document_names}})
    if patient_label:
        clauses.append({"patient_label": patient_label})
    if report_id:
        clauses.append({"report_id": report_id})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class HybridRetriever:
    """Combine BM25 and dense ranks without comparing their raw scores."""

    @classmethod
    def clear_cache(cls):
        """Explicitly clear the BM25 index cache."""
        _BM25_CACHE.clear()

    @staticmethod
    def get_all_chunks(
        document_names=None,
        session_id="global",
        document_ids=None,
        patient_label=None,
        report_id=None,
    ):
        # Fail closed on empty selection
        if document_ids is not None and len(document_ids) == 0:
            return []
        if document_names is not None and len(document_names) == 0:
            return []
        # Missing selection does not silently trigger global retrieval
        if document_ids is None and document_names is None:
            return []

        where = _where_clause(document_names, session_id, document_ids, patient_label, report_id)
        results = collection.get(where=where, include=["documents", "metadatas"])
        documents, metadatas, ids = (
            results.get("documents") or [],
            results.get("metadatas") or [],
            results.get("ids") or [],
        )
        chunks = [
            {"id": doc_id, "text": text, "metadata": metadata}
            for doc_id, text, metadata in zip(ids, documents, metadatas)
        ]

        # Enforce scope boundaries strictly (defense-in-depth against vector store filter leaks)
        if document_ids is not None:
            allowed_ids = set(document_ids)
            chunks = [c for c in chunks if c.get("metadata", {}).get("document_id") in allowed_ids]
        elif document_names is not None:
            allowed_names = set(document_names)
            chunks = [c for c in chunks if c.get("metadata", {}).get("document_name") in allowed_names]

        if patient_label:
            chunks = [c for c in chunks if c.get("metadata", {}).get("patient_label") == patient_label]
        if report_id:
            chunks = [c for c in chunks if c.get("metadata", {}).get("report_id") == report_id]

        return chunks

    @staticmethod
    def _get_all_chunks(
        document_names=None,
        session_id="global",
        document_ids=None,
        patient_label=None,
        report_id=None,
    ):
        return HybridRetriever.get_all_chunks(
            document_names, session_id, document_ids, patient_label, report_id
        )

    @staticmethod
    def search(
        question: str,
        top_k: int = TOP_K,
        document_names: Optional[List[str]] = None,
        session_id: str = "global",
        candidate_k: Optional[int] = None,
        document_ids: Optional[List[str]] = None,
        patient_label: Optional[str] = None,
        report_id: Optional[str] = None,
        patient_query: Optional[str] = None,
        report_query: Optional[str] = None,
    ) -> List[Dict]:
        if not question or not question.strip() or top_k <= 0:
            return []
        # Fail closed on empty selection
        if document_ids is not None and len(document_ids) == 0:
            return []
        if document_names is not None and len(document_names) == 0:
            return []
        # Missing selection does not silently trigger global retrieval
        if document_ids is None and document_names is None:
            return []

        chunks = HybridRetriever.get_all_chunks(
            document_names=document_names,
            session_id=session_id,
            document_ids=document_ids,
            patient_label=patient_label,
            report_id=report_id,
        )
        if not chunks:
            return []

        # Apply question-derived scope to the complete selected corpus BEFORE
        # BM25/dense top-K ranking. Patient/report mentions may live in PDF text
        # or filenames even when upload metadata is absent.
        if patient_query:
            chunks = [
                chunk for chunk in chunks
                if _scope_phrase_matches(chunk.get("metadata", {}).get("patient_label", ""), patient_query)
                or _scope_phrase_matches(chunk.get("text", ""), patient_query)
                or _scope_phrase_matches(chunk.get("metadata", {}).get("document_name", ""), patient_query)
            ]
        if report_query:
            chunks = [
                chunk for chunk in chunks
                if _scope_phrase_matches(chunk.get("metadata", {}).get("report_id", ""), report_query)
                or _scope_phrase_matches(chunk.get("metadata", {}).get("document_name", ""), report_query)
                or _scope_phrase_matches(chunk.get("metadata", {}).get("document_id", ""), report_query)
                or _scope_phrase_matches(chunk.get("text", ""), report_query)
            ]
        if not chunks:
            return []

        candidate_k = min(max(top_k, candidate_k or top_k * 4), len(chunks))
        # Vector-store metadata filtering is safe only when every retained chunk
        # carries the exact matching metadata value. For text-only scope, rely on
        # BM25 over the already-filtered corpus instead of mixing other patients'
        # dense candidates into the ranking.
        patient_metadata_only = (
            not patient_query
            or all(str(c.get("metadata", {}).get("patient_label", "")).casefold() == patient_query.casefold() for c in chunks)
        )
        report_metadata_only = (
            not report_query
            or all(str(c.get("metadata", {}).get("report_id", "")).casefold() == report_query.casefold() for c in chunks)
        )
        dense_filterable = patient_metadata_only and report_metadata_only
        where = _where_clause(
            document_names, session_id, document_ids,
            patient_label or (patient_query if patient_metadata_only and patient_query else None),
            report_id or (report_query if report_metadata_only and report_query else None),
        )

        def tokenize(value):
            return re.findall(r"[a-z]+|\d+(?:\.\d+)?", value.lower())

        # Cache key partitioned strictly by normalized document selection and chunk IDs
        scope_prefix = f"{session_id}|{','.join(sorted(document_ids or []))}|{','.join(sorted(document_names or []))}|{patient_label or ''}|{report_id or ''}|{patient_query or ''}|{report_query or ''}"
        fingerprint_source = scope_prefix + "\n" + "\n".join(f"{chunk['id']}\0{chunk['text']}" for chunk in chunks)
        fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest()
        with _BM25_LOCK:
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
        dense_ids = []
        if dense_filterable:
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
            if doc_id in chunks_by_id:
                fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1 / (60 + rank + 1)
        ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)

        # Deduplicate overlapping chunks while preserving citation metadata
        seen_texts = set()
        deduped_chunks = []
        for doc_id in ranked_ids:
            chunk = chunks_by_id.get(doc_id)
            if not chunk:
                continue
            norm = clean_text(chunk.get("text", "")).lower()
            if norm in seen_texts:
                continue
            seen_texts.add(norm)
            deduped_chunks.append(chunk)
            if len(deduped_chunks) >= top_k:
                break

        return deduped_chunks
