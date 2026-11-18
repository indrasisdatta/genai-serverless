"""Reciprocal Rank Fusion for combining search results."""

from typing import List, Dict, Any
import operator


def rrf(
    result_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        result_lists: List of ranked result lists
        k: RRF constant (default 60)
        top_n: Number of results to return

    Returns:
        Fused and re-ranked results
    """
    scores: Dict[int, float] = {}
    by_id: Dict[int, Dict[str, Any]] = {}

    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            doc_id = doc.get("id")
            if doc_id not in scores:
                scores[doc_id] = 0.0
                by_id[doc_id] = doc
            scores[doc_id] += 1.0 / (k + rank)

    # Sort by score descending
    ranked_ids = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)

    # Take top N and attach scores
    result = []
    for doc_id, score in ranked_ids[:top_n]:
        result.append({**by_id[doc_id], "score": score})

    return result