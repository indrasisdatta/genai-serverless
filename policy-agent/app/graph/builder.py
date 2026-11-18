"""Graph builder - compiles the agentic RAG StateGraph."""

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from app.graph.state import GraphState
from app.graph.nodes.router import router
from app.graph.nodes.decompose import decompose
from app.graph.nodes.retrieve import retrieve
from app.graph.nodes.grade_docs import grade_docs
from app.graph.nodes.rewrite_query import rewrite_query
from app.graph.nodes.generate import generate
from app.graph.nodes.groundedness import groundedness
from app.graph.nodes.abstain import abstain
from app.graph.nodes.refuse import refuse
from app.graph.edges import (
    after_grading,
    after_groundedness,
    route_after_router,
    fan_out,
)


def build_graph(checkpointer=None):
    """
    Build and compile the agentic RAG graph.

    Args:
        checkpointer: Optional checkpointer for persistence (e.g., AsyncPostgresSaver)

    Returns:
        Compiled LangGraph application
    """
    g = StateGraph(GraphState)

    # Nodes
    g.add_node("router", router)
    g.add_node("decompose", decompose)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_docs", grade_docs)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("generate", generate)
    g.add_node("groundedness", groundedness)
    g.add_node("abstain", abstain)
    g.add_node("refuse", refuse)

    # Edges
    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router",
        route_after_router,
        {
            "retrieve": "retrieve",
            "decompose": "decompose",
            "refuse": "refuse",
        },
    )
    g.add_edge("decompose", "retrieve")  # In production: use fan_out Send
    g.add_edge("retrieve", "grade_docs")
    g.add_conditional_edges(
        "grade_docs",
        after_grading,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "abstain": "abstain",
        },
    )
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("generate", "groundedness")
    g.add_conditional_edges(
        "groundedness",
        after_groundedness,
        {
            "end": END,
            "generate": "generate",
            "abstain": "abstain",
        },
    )
    g.add_edge("abstain", END)
    g.add_edge("refuse", END)

    return g.compile(checkpointer=checkpointer)