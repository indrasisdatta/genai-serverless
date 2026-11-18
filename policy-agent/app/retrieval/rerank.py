"""Cross-encoder reranking for retrieved documents."""

from functools import lru_cache
from typing import List, Dict, Any

from sentence_transformers import CrossEncoder

from app.config import settings


@lru_cache()
def _get_model() -> CrossEncoder:
    """Get or create the cross-encoder model."""
    return CrossEncoder(settings.rerank_model)


def rerank(
    query: str,
    docs: List[Dict[str, Any]],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    Rerank documents using cross-encoder.

    Args:
        query: Search query
        docs: List of documents to rerank
        top_n: Number of results to return

    Returns:
        Reranked documents with updated scores
    """
    if not docs:
        return []

    model = _get_model()
    pairs = [(query, doc.get("content", "")) for doc in docs]
    scores = model.predict(pairs, show_progress_bar=False)

    # Sort by score descending
    ranked = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        {**doc, "score": float(score)}
        for doc, score in ranked[:top_n]
    ]