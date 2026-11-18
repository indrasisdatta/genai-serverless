"""Router node - classifies user questions into routes."""

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from app.graph.state import GraphState
from app.llm.factory import get_llm


class RouteDecision(BaseModel):
    route: str = Field(
        description="One of: simple, multi_hop, out_of_scope"
    )
    reason: str = Field(description="Short justification for the route")


PROMPT = """Classify the user's question into one of three routes:

- "simple": The question can be answered from a single policy document chunk.
- "multi_hop": The question requires combining information from multiple chunks or documents.
- "out_of_scope": The question is not about policy information that could be answered from our corpus.

Question: {question}

Respond with only the route and a brief reason."""


def router(state: GraphState) -> dict:
    """Route the user question to a processing path."""
    llm = get_llm("fast")
    prompt = ChatPromptTemplate.from_template(PROMPT)

    structured_llm = llm.with_structured_output(RouteDecision)
    result = structured_llm.invoke(
        prompt.format(question=state["question"])
    )

    return {
        "route": result.route,
        "query": state["question"],
    }