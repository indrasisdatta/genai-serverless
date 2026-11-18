"""Retrieve node - hybrid search (vector + keyword) with RRF fusion and reranking.

Week 1 skeleton: returns an empty doc list so the graph compiles without a
database connection. Week 3 wires in the real pipeline:

    vector top 20 + keyword top 20 -> RRF -> cross-encoder rerank -> top 5

The plumbing mirrors multipdf_chat.retrieval.retrieve_hybrid_async, ported to
the new `chunks` table (see docs/decisions.md once written).
"""

from typing import List

from app.graph.state import GraphState, Chunk
from app.retrieval.vector import vector_search
from app.retrieval.keyword import keyword_search
from app.retrieval.fusion import rrf
from app.retrieval.rerank import rerank


def retrieve(state: GraphState) -> dict:
    """Retrieve relevant documents using hybrid search + rerank."""
    query = state.get("query", state["question"])

    # Week 1 placeholder: no DB session available yet.
    # Week 3 implementation:
    #   vector_results = vector_search(db, query, top_k=20)
    #   keyword_results = keyword_search(db, query, top_k=20)
    #   fused = rrf([vector_results, keyword_results], k=60, top_n=20)
    #   docs = rerank(query, fused, top_n=5)
    docs: List[Chunk] = []

    return {"docs": docs}
