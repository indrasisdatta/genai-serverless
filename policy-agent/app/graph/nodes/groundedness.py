"""Groundedness node - checks if the answer is supported by retrieved docs."""

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from app.graph.state import GraphState
from app.llm.factory import get_llm


class GroundednessCheck(BaseModel):
    grounded: bool = Field(description="Whether every claim is supported by the context")
    unsupported_claims: list[str] = Field(description="Claims not supported by context")


PROMPT = """Check if every claim in the answer is supported by the provided context.

Context:
{context}

Answer:
{answer}

Identify any claims that are NOT supported by the context."""


def groundedness(state: GraphState) -> dict:
    """Check if the answer is grounded in the retrieved context."""
    docs = state.get("docs", [])
    answer = state.get("answer", "")
    query = state.get("query", state["question"])

    context = "\n\n".join(
        f"[{i}] {d.get('content', '')}" for i, d in enumerate(docs)
    )

    llm = get_llm("fast")
    prompt = ChatPromptTemplate.from_template(PROMPT)

    structured_llm = llm.with_structured_output(GroundednessCheck)
    result = structured_llm.invoke(
        prompt.format(
            context=context,
            answer=answer,
        )
    )

    return {"grounded": result.grounded}