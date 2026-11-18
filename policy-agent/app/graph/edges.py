"""Conditional edges for the agentic RAG graph."""

from typing import List

from langgraph.types import Send

from app.graph.state import GraphState

MAX_RETRIES = 2
MIN_RELEVANT = 2


def after_grading(state: GraphState) -> str:
    """Route after document grading: generate, rewrite, or abstain."""
    docs = state.get("docs", [])
    retries = state.get("retries", 0)

    if len(docs) >= MIN_RELEVANT:
        return "generate"

    if retries < MAX_RETRIES:
        return "rewrite_query"

    return "abstain"


def after_groundedness(state: GraphState) -> str:
    """Route after groundedness check: end, regenerate, or abstain."""
    grounded = state.get("grounded", False)
    retries = state.get("retries", 0)

    if grounded:
        return "end"

    if retries < MAX_RETRIES:
        return "generate"

    return "abstain"


def route_after_router(state: GraphState) -> str:
    """Route after the router node."""
    route = state.get("route", "simple")

    if route == "multi_hop":
        return "decompose"
    if route == "out_of_scope":
        return "refuse"
    return "retrieve"


def fan_out(state: GraphState) -> List[Send]:
    """Fan out to parallel sub-RAG branches for multi-hop questions."""
    sub_questions = state.get("sub_questions", [])
    return [
        Send(
            "sub_rag",
            {"question": q, "query": q, "retries": 0},
        )
        for q in sub_questions[:3]  # Cap at 3 sub-questions to protect rate limit
    ]
