"""Graph state definition for the agentic RAG pipeline."""

from typing import TypedDict, Annotated
import operator


class Chunk(TypedDict):
    """A retrieved chunk from the graph state."""

    id: int
    doc_id: str
    section: str
    content: str
    score: float


class GraphState(TypedDict, total=False):
    """
    State shared across all nodes in the agentic RAG graph.

    Each field holds data produced or consumed by specific nodes.
    Nodes should return only the keys they change.
    """

    question: str  # Original user question
    query: str  # Possibly rewritten query used for retrieval
    route: str  # Route decision: simple | multi_hop | out_of_scope
    sub_questions: list[str]  # Sub-questions for multi-hop
    docs: list[Chunk]  # Retrieved documents
    sub_answers: Annotated[list[dict], operator.add]  # Filled in parallel branches
    answer: str  # Final answer
    citations: list[dict]  # Source citations
    retries: int  # Retry counter for CRAG loop
    grounded: bool  # Whether answer is grounded in retrieved docs
    status: str  # answered | abstained | refused