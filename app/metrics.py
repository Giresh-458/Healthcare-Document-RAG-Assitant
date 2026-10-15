"""Lightweight diagnostic metrics for a prototype RAG pipeline.

These token-overlap scores are heuristics, not validated faithfulness or
context-precision metrics. Use a labelled evaluation set and human review for
quality claims; never interpret these values as clinical safety measures.
"""
import json
import os
import re
import time
from typing import Dict, List


class RagMetrics:
    @staticmethod
    def calculate_context_precision(question: str, chunks: List[Dict]) -> float:
        """Return a rough query-token overlap indicator (not standard precision)."""
        if not chunks:
            return 0.0
        tokens = set(re.findall(r"\w+", question.lower()))
        if not tokens:
            return 0.0
        scores = []
        for chunk in chunks:
            chunk_tokens = set(re.findall(r"\w+", chunk.get("text", "").lower()))
            scores.append(len(tokens & chunk_tokens) / len(tokens))
        return round(sum(scores) / len(scores), 4)

    @staticmethod
    def calculate_faithfulness(answer: str, chunks: List[Dict]) -> float:
        """Return answer-token overlap with retrieved context, not factual verification."""
        if not chunks or not answer:
            return 0.0
        context_tokens = set(
            re.findall(r"[a-zA-Z]+|\d+(?:\.\d+)?", " ".join(c.get("text", "") for c in chunks).lower())
        )
        answer_tokens = re.findall(
            r"[a-zA-Z]+|\d+(?:\.\d+)?", answer.lower()
        )
        if not answer_tokens:
            return 1.0
        return round(sum(t in context_tokens for t in answer_tokens) / len(answer_tokens), 4)

    @staticmethod
    def log_metrics(question: str, answer: str, chunks: List[Dict], latency: float):
        metrics = {
            "timestamp": time.time(),
            "latency_sec": round(max(0.0, latency), 4),
            "query_token_overlap_heuristic": RagMetrics.calculate_context_precision(question, chunks),
            "answer_context_token_overlap_heuristic": RagMetrics.calculate_faithfulness(answer, chunks),
            "retrieved_chunks": len(chunks),
        }
        os.makedirs("data/metrics", exist_ok=True)
        with open("data/metrics/metrics_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")
        return metrics
