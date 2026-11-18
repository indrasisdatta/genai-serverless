"""Abstain node - returns a polite refusal when no answer can be supported."""

from typing import List

from app.graph.state import GraphState


def abstain(state: GraphState) -> dict:
    """Abstain from answering when retrieval fails."""
    docs = state.get("docs", [])

    # Provide the 2 closest sections as "you may want to check" citations
    citations = [
        {"doc_id": d.get("doc_id", "unknown"), "content": d.get("content", "")[:200]}
        for d in docs[:2]
    ]

    return {
        "answer": (
            "I don't have enough information in the policy documents to answer this confidently. "
            "Please provide more context (e.g., which product or region) or route to a human agent."
        ),
        "citations": citations,
        "status": "abstained",
    }