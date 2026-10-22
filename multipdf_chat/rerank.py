"""Cross-encoder reranking of retrieved candidates."""

import asyncio
import logging
import time

from sentence_transformers import CrossEncoder
from transformers import AutoConfig

from multipdf_chat.logging_context import get_request_id
from multipdf_chat.retrieval import RetrievedParent

logger = logging.getLogger("api")

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L6-v2"
_MAX_LEN = 512 # tokens; longer inputs are truncated

def load_reranker() -> CrossEncoder:
    """Load once at startup"""
    return CrossEncoder(
        _MODEL_NAME,
        num_labels=1,
        max_length=_MAX_LEN,
    )

def rerank(
    model: CrossEncoder,
    question: str,
    parents: list[RetrievedParent],
    top_n: int = 5,
) -> list[RetrievedParent]:
    if not parents:
        return parents 
     
    request_id = get_request_id()
    t0 = time.perf_counter()

    pairs = [(question, p.content[:1800]) for p in parents]
    scores = model.predict(pairs, show_progress_bar=False)
    ranked = sorted(
        zip(parents, scores),
        key=lambda x: x[1],
        reverse=True
    )
    top = [p for p, _ in ranked[:top_n]]

    logger.info("rerank_completed", extra={
        "request_id": request_id,
        "n_in": len(parents),
        "n_out": len(top),
        "duration_ms": round((time.perf_counter() - t0) * 1000),
        "top_slug_before": parents[0].doc_slug,
        "top_slug_after": top[0].doc_slug if top else None,
    })
    return top

async def rerank_async(model, question, parents, top_n = 5) -> list[RetrievedParent]:
    return await asyncio.to_thread(rerank, model, question, parents, top_n) 