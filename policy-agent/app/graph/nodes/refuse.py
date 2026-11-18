"""Refuse node - handles out-of-scope questions."""

from app.graph.state import GraphState


def refuse(state: GraphState) -> dict:
    """Refuse to answer out-of-scope questions."""
    return {
        "answer": (
            "I'm sorry, but I can only answer questions about policy documents "
            "in our knowledge base. This question appears to be out of scope."
        ),
        "citations": [],
        "status": "refused",
    }