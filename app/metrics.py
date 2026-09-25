import time
from typing import List, Dict
import json
import os

class RagMetrics:
    @staticmethod
    def calculate_context_precision(question: str, chunks: List[Dict]) -> float:
        """
        Simple heuristic for Context Precision (relevance of retrieved chunks to question).
        In a real RAGAS setup, this uses an LLM. Here we use keyword overlap for speed.
        """
        if not chunks: return 0.0
        question_words = set(question.lower().split())
        score = 0
        for chunk in chunks:
            chunk_words = set(chunk["text"].lower().split())
            overlap = len(question_words.intersection(chunk_words))
            score += overlap / (len(question_words) + 1e-5)
        # Normalize to 0-1 range roughly
        return min(1.0, score / len(chunks) * 2.0)

    @staticmethod
    def calculate_faithfulness(answer: str, chunks: List[Dict]) -> float:
        """
        Calculates Faithfulness (Factual Consistency).
        Normalizes text, strips punctuation, removes conversational stopwords,
        and computes the ratio of factual claim tokens grounded in the context.
        """
        if not chunks or not answer:
            return 0.0
            
        import re
        stopwords = {
            "a", "about", "above", "after", "again", "against", "all", "also", "an", "and", "any", "are",
            "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
            "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further",
            "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
            "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my",
            "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our",
            "ours", "ourselves", "out", "over", "own", "s", "same", "she", "should", "so", "some", "such",
            "t", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these",
            "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we",
            "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would",
            "you", "your", "yours", "yourself", "yourselves",
            # Conversational/scaffolding terms
            "see", "reported", "report", "falls", "stated", "mentioned", "shown", "shows", "page", "document"
        }
        
        context_text = " ".join([c["text"].lower() for c in chunks])
        # Extract alphanumeric tokens, including numbers and units
        answer_tokens = re.findall(r'\b[a-zA-Z0-9]+(?:[.\-][a-zA-Z0-9]+)*\b', answer.lower())
        
        # Filter out stopwords and single characters (unless numeric)
        claim_tokens = [
            t for t in answer_tokens 
            if t not in stopwords and (len(t) > 1 or t.isdigit())
        ]
        
        if not claim_tokens:
            return 1.0
            
        grounded_count = sum(1 for token in claim_tokens if token in context_text)
        return min(1.0, round(grounded_count / len(claim_tokens), 2))
        
    @staticmethod
    def log_metrics(question: str, answer: str, chunks: List[Dict], latency: float):
        precision = RagMetrics.calculate_context_precision(question, chunks)
        faithfulness = RagMetrics.calculate_faithfulness(answer, chunks)
        
        metrics = {
            "timestamp": time.time(),
            "latency_sec": latency,
            "context_precision": round(precision, 2),
            "faithfulness": round(faithfulness, 2),
            "retrieved_chunks": len(chunks)
        }
        
        # Log to file for observability
        os.makedirs("data/metrics", exist_ok=True)
        with open("data/metrics/metrics_log.jsonl", "a") as f:
            f.write(json.dumps(metrics) + "\n")
            
        return metrics
