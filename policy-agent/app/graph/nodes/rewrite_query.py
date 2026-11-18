"""Query rewrite node - rewrites query when retrieval fails."""

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from app.graph.state import GraphState
from app.llm.factory import get_llm


class RewrittenQuery(BaseModel):
    query: str = Field(description="Rewritten query")
    reason: str = Field(description="Why the query was rewritten")


PROMPT = """The original retrieval did not return relevant documents. Rewrite the question
into a better search query that might find relevant information.

Original question: {question}
Previous query: {query}

Return a rewritten query and a brief reason."""


def rewrite_query(state: GraphState) -> dict:
    """Rewrite the query for better retrieval."""
    llm = get_llm("fast")
    prompt = ChatPromptTemplate.from_template(PROMPT)

    structured_llm = llm.with_structured_output(RewrittenQuery)
    result = structured_llm.invoke(
        prompt.format(
            question=state.get("question", ""),
            query=state.get("query", state["question"]),
        )
    )

    retries = state.get("retries", 0) + 1

    return {
        "query": result.query,
        "retries": retries,
    }