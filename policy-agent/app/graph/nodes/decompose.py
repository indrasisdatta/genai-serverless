"""Decompose node - breaks multi-hop questions into sub-questions."""

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from app.graph.state import GraphState
from app.llm.factory import get_llm


class Decomposition(BaseModel):
    sub_questions: list[str] = Field(
        description="List of sub-questions, max 3 items"
    )
    reason: str = Field(description="Justification for decomposition")


PROMPT = """Break this question into at most 3 simpler sub-questions that can be answered independently,
then combined to answer the original question.

Question: {question}

Respond with the sub-questions and a brief reason."""


def decompose(state: GraphState) -> dict:
    """Break a multi-hop question into sub-questions."""
    llm = get_llm("fast")
    prompt = ChatPromptTemplate.from_template(PROMPT)

    structured_llm = llm.with_structured_output(Decomposition)
    result = structured_llm.invoke(
        prompt.format(question=state["question"])
    )

    # Cap at 3 sub-questions to protect rate limits
    sub_questions = result.sub_questions[:3]

    return {
        "sub_questions": sub_questions,
        "query": state["question"],
    }