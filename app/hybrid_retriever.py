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


def _scope_phrase_matches(value: str, phrase: str) -> bool:
    """Match a scope phrase across spaces, underscores, or hyphens."""
    if not value or not phrase:
        return False
    tokens = re.findall(r"[A-Za-z0-9]+", phrase)
    if not tokens:
        return False
    pattern = r"(?<![A-Za-z0-9])" + r"[\s_-]*".join(re.escape(token) for token in tokens) + r"(?![A-Za-z0-9])"
    return bool(re.search(pattern, str(value), re.I))


def filter_patient_chunks(chunks: List[Dict], patient_query: Optional[str]) -> List[Dict]:
    """Filter to one patient while retaining unlabelled pages from single-patient uploads."""
    if not patient_query:
        return chunks

    matching_documents = set()
    direct_matches = set()
    identities_by_document = {}

    for index, chunk in enumerate(chunks):
        metadata = chunk.get("metadata", {})
        document_key = metadata.get("document_id") or metadata.get("document_name") or chunk.get("id") or index
        identities = identities_by_document.setdefault(document_key, set())
        chunk_identities = set()

        label = str(metadata.get("patient_label", "") or "").strip()
        if label and label.casefold() != "unspecified":
            chunk_identities.add(label)

        text = str(chunk.get("text", "") or "")
        for match in re.finditer(r"\bPatient[\s_-]+([A-Za-z0-9_-]+)\b", text, re.I):
            token = match.group(1)
            if token.casefold() not in {"id", "name", "record", "history", "details"}:
                chunk_identities.add(f"Patient {token}")
        for match in re.finditer(r"\b((?:Mr\.|Ms\.|Mrs\.|Dr\.)\s+[A-Za-z]+(?:\s+[A-Za-z]+)?)\b", text, re.I):
            chunk_identities.add(match.group(1).strip())

        document_name = str(metadata.get("document_name", "") or "")
        filename_match = re.search(r"\bpatient[_-]([a-z0-9]+)\b", document_name, re.I)
        if filename_match:
            chunk_identities.add(f"Patient {filename_match.group(1)}")

        identities.update(chunk_identities)
        direct = (
            _scope_phrase_matches(label, patient_query)
            or _scope_phrase_matches(text, patient_query)
            or _scope_phrase_matches(document_name, patient_query)
        )
        # Never pass a chunk that explicitly identifies more than one patient
        # (or a different patient) into a single-patient answer.
        if direct and all(_scope_phrase_matches(identity, patient_query) for identity in chunk_identities):
            matching_documents.add(document_key)
            direct_matches.add(chunk.get("id") or index)

    filtered = []
    for index, chunk in enumerate(chunks):
        metadata = chunk.get("metadata", {})
        document_key = metadata.get("document_id") or metadata.get("document_name") or chunk.get("id") or index
        identities = identities_by_document.get(document_key, set())
        document_has_only_target_patient = (
            not identities
            or all(_scope_phrase_matches(identity, patient_query) for identity in identities)
        )
        if document_key in matching_documents and (
            document_has_only_target_patient or (chunk.get("id") or index) in direct_matches
        ):
            filtered.append(chunk)
    return filtered


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

        # Patient labels and report IDs can be present only in PDF text or
        # on a cover page. Do not push these filters into Chroma's per-chunk
        # metadata predicate; load only the selected uploads, then apply
        # upload-aware filters across their complete chunk sets.
        where = _where_clause(document_names, session_id, document_ids)
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
            chunks = filter_patient_chunks(chunks, patient_label)
        if report_id:
            matching_document_keys = {
                c.get("metadata", {}).get("document_id")
                or c.get("metadata", {}).get("document_name")
                or c.get("id")
                for c in chunks
                if _scope_phrase_matches(c.get("metadata", {}).get("report_id", ""), report_id)
                or _scope_phrase_matches(c.get("metadata", {}).get("document_name", ""), report_id)
                or _scope_phrase_matches(c.get("metadata", {}).get("document_id", ""), report_id)
                or _scope_phrase_matches(c.get("text", ""), report_id)
            }
            chunks = [
                c for c in chunks
                if (
                    c.get("metadata", {}).get("document_id")
                    or c.get("metadata", {}).get("document_name")
                    or c.get("id")
                ) in matching_document_keys
                and (
                    not c.get("metadata", {}).get("report_id")
                    or _scope_phrase_matches(c.get("metadata", {}).get("report_id", ""), report_id)
                )
            ]

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
        report_type_query: Optional[str] = None,
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
            chunks = filter_patient_chunks(chunks, patient_query)
        if report_query:
            # Report IDs are extracted per PDF page. Resolve the owning upload
            # from any matching page, then retain the other pages from that same
            # report unless an exact report-ID query encounters a conflicting ID.
            report_descriptors = {"follow-up", "routine", "baseline", "initial", "discharge", "admission"}
            strict_report_id = report_query.casefold() not in report_descriptors
            matching_document_keys = {
                chunk.get("metadata", {}).get("document_id")
                or chunk.get("metadata", {}).get("document_name")
                or chunk.get("id")
                for chunk in chunks
                if _scope_phrase_matches(chunk.get("metadata", {}).get("report_id", ""), report_query)
                or _scope_phrase_matches(chunk.get("metadata", {}).get("document_name", ""), report_query)
                or _scope_phrase_matches(chunk.get("metadata", {}).get("document_id", ""), report_query)
                or _scope_phrase_matches(chunk.get("text", ""), report_query)
            }
            chunks = [
                chunk for chunk in chunks
                if (
                    chunk.get("metadata", {}).get("document_id")
                    or chunk.get("metadata", {}).get("document_name")
                    or chunk.get("id")
                ) in matching_document_keys
                and (
                    not strict_report_id
                    or not chunk.get("metadata", {}).get("report_id")
                    or _scope_phrase_matches(chunk.get("metadata", {}).get("report_id", ""), report_query)
                )
            ]
        if report_type_query:
            matching_document_keys = {
                chunk.get("metadata", {}).get("document_id")
                or chunk.get("metadata", {}).get("document_name")
                or chunk.get("id")
                for chunk in chunks
                if _scope_phrase_matches(chunk.get("metadata", {}).get("document_name", ""), report_type_query)
                or _scope_phrase_matches(chunk.get("text", ""), report_type_query)
            }
            chunks = [
                chunk for chunk in chunks
                if (
                    chunk.get("metadata", {}).get("document_id")
                    or chunk.get("metadata", {}).get("document_name")
                    or chunk.get("id")
                ) in matching_document_keys
            ]
        if not chunks:
            return []

        candidate_k = min(max(top_k, candidate_k or top_k * 4), len(chunks))
        # Vector-store metadata filtering is safe only when every retained chunk
        # carries the exact matching metadata value. For text-only scope, rely on
        # BM25 over the already-filtered corpus instead of mixing other patients'
        # dense candidates into the ranking.
        patient_scope_value = patient_query or patient_label
        report_scope_value = report_query or report_id
        patient_metadata_only = (
            not patient_scope_value
            or all(
                str(c.get("metadata", {}).get("patient_label", "")).casefold()
                == patient_scope_value.casefold()
                for c in chunks
            )
        )
        report_metadata_only = (
            not report_scope_value
            or all(
                str(c.get("metadata", {}).get("report_id", "")).casefold()
                == report_scope_value.casefold()
                for c in chunks
            )
        )
        dense_filterable = patient_metadata_only and report_metadata_only and not report_type_query
        where = _where_clause(
            document_names,
            session_id,
            document_ids,
            patient_scope_value if patient_metadata_only else None,
            report_scope_value if report_metadata_only else None,
        )

        def tokenize(value):
            return re.findall(r"[a-z]+|\d+(?:\.\d+)?", value.lower())

        # Cache key partitioned strictly by normalized document selection and chunk IDs
        scope_prefix = f"{session_id}|{','.join(sorted(document_ids or []))}|{','.join(sorted(document_names or []))}|{patient_label or ''}|{report_id or ''}|{patient_query or ''}|{report_query or ''}|{report_type_query or ''}"
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
