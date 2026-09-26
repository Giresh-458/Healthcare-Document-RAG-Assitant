from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
from app.vector_store import collection, get_embeddings_model
from app.config import TOP_K

class HybridRetriever:
    """
    Implements Hybrid Search by combining Dense (Semantic) Search 
    with BM25 (Keyword) Search using Reciprocal Rank Fusion (RRF).
    """
    
    @staticmethod
    def _get_all_chunks(document_names: Optional[List[str]] = None, session_id: str = "global") -> List[Dict]:
        """Fetches chunks from ChromaDB to build the in-memory BM25 index."""
        
        # Build the session filter
        if session_id == "global":
            session_filter = None
        else:
            session_filter = {"$or": [{"session_id": session_id}, {"session_id": "global"}]}
            
        # Build the document filter
        doc_filter = None
        if document_names:
            if len(document_names) == 1:
                doc_filter = {"document_name": document_names[0]}
            elif len(document_names) > 1:
                doc_filter = {"document_name": {"$in": document_names}}
        
        # Combine filters
        where_clause = None
        if session_filter and doc_filter:
            where_clause = {"$and": [session_filter, doc_filter]}
        elif session_filter:
            where_clause = session_filter
        elif doc_filter:
            where_clause = doc_filter
        
        if where_clause:
            results = collection.get(where=where_clause)
        else:
            results = collection.get()
            
        chunks = []
        if results and results.get("documents"):
            for doc, meta, doc_id in zip(results["documents"], results["metadatas"], results["ids"]):
                chunks.append({
                    "id": doc_id,
                    "text": doc,
                    "metadata": meta
                })
        return chunks

    @staticmethod
    def search(question: str, top_k: int = TOP_K, document_names: Optional[List[str]] = None, session_id: str = "global") -> List[Dict]:
        chunks = HybridRetriever._get_all_chunks(document_names, session_id)
        if not chunks:
            return []
            
        # 1. BM25 (Keyword Search)
        # Excellent for exact medical value matches like "12.5 g/dL"
        tokenized_corpus = [chunk["text"].lower().split() for chunk in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = question.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # 2. Dense Search (Semantic Search via ChromaDB)
        # Excellent for conceptual matches like "high blood sugar" -> "Hyperglycemia"
        embeddings = get_embeddings_model()
        query_embedding = embeddings.embed_query(question)
        
        # Build the session filter
        if session_id == "global":
            session_filter = None
        else:
            session_filter = {"$or": [{"session_id": session_id}, {"session_id": "global"}]}
            
        # Build the document filter
        doc_filter = None
        if document_names:
            if len(document_names) == 1:
                doc_filter = {"document_name": document_names[0]}
            elif len(document_names) > 1:
                doc_filter = {"document_name": {"$in": document_names}}
        
        # Combine filters
        where_clause = None
        if session_filter and doc_filter:
            where_clause = {"$and": [session_filter, doc_filter]}
        elif session_filter:
            where_clause = session_filter
        elif doc_filter:
            where_clause = doc_filter
        
        # Query ALL documents to get global dense ranks for RRF
        dense_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=len(chunks),
            where=where_clause
        )
        
        # Map chunk IDs to their dense rank
        dense_ranks = {}
        if dense_results["ids"] and len(dense_results["ids"][0]) > 0:
            for rank, doc_id in enumerate(dense_results["ids"][0]):
                dense_ranks[doc_id] = rank

        # 3. Reciprocal Rank Fusion (RRF)
        # Sort BM25 by score descending to determine BM25 ranks
        bm25_ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
        
        rrf_scores = {}
        k = 60 # Standard RRF constant
        
        for rank, chunk_idx in enumerate(bm25_ranked_indices):
            chunk_id = chunks[chunk_idx]["id"]
            
            bm25_rank = rank
            # If for some reason not in dense search, assign lowest rank
            dense_rank = dense_ranks.get(chunk_id, len(chunks))
            
            # The RRF formula: 1 / (k + rank)
            rrf_score = (1 / (k + bm25_rank)) + (1 / (k + dense_rank))
            rrf_scores[chunk_idx] = rrf_score
            
        # Sort all chunks by their fused RRF score
        fused_indices = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
        
        # 4. Return Top K (Dynamic based on Router)
        top_k_indices = fused_indices[:top_k]
        return [chunks[i] for i in top_k_indices]
