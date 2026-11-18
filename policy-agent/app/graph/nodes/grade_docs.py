"""Document grading node - CRAG self-correction loop."""

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from typing import List

from app.graph.state import GraphState
from app.llm.factory import get_llm


class ChunkGrade(BaseModel):
    index: int = Field(description="Index of the chunk in the list")
    relevant: bool = Field(description="Whether the chunk is relevant to the question")


class Grades(BaseModel):
    grades: List[ChunkGrade] = Field(description="One entry per chunk")


PROMPT = """Grade whether each retrieved chunk is relevant to answering the question.

Question: {question}

Chunks:
{chunks}

For each chunk, decide if it contains information relevant to answering the question."""


def grade_docs(state: GraphState) -> dict:
    """Grade all retrieved chunks in a single LLM call."""
    docs = state.get("docs", [])
    query = state.get("query", state["question"])

    if not docs:
        return {"docs": []}

    numbered = "\n\n".join(
        f"[{i}] {d.get('content', '')}" for i, d in enumerate(docs)
    )

    llm = get_llm("fast")
    prompt = ChatPromptTemplate.from_template(PROMPT)

    structured_llm = llm.with_structured_output(Grades)
    result = structured_llm.invoke(
        prompt.format(question=query, chunks=numbered)
    )

    keep = {g.index for g in result.grades if g.relevant}
    kept_docs = [d for i, d in enumerate(docs) if i in keep]

    return {"docs": kept_docs}