"""Generate node - produces the final answer from retrieved chunks."""

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from app.graph.state import GraphState
from app.llm.factory import get_llm


class Generation(BaseModel):
    answer: str = Field(description="The final answer to the question")
    citations: list[str] = Field(description="List of source document references")


PROMPT = """Answer the question using ONLY the provided context.
If the answer cannot be determined from the context, say so.

Context:
{context}

Question: {question}

Answer:

Citations:
{citations}"""


def generate(state: GraphState) -> dict:
    """Generate an answer from retrieved documents."""
    docs = state.get("docs", [])
    query = state.get("query", state["question"])

    context = "\n\n".join(
        f"[{i}] {d.get('content', '')}" for i, d in enumerate(docs)
    )
    citations = [d.get("doc_id", "unknown") for d in docs]

    llm = get_llm("strong")
    prompt = ChatPromptTemplate.from_template(PROMPT)

    structured_llm = llm.with_structured_output(Generation)
    result = structured_llm.invoke(
        prompt.format(
            context=context,
            question=query,
            citations=", ".join(citations),
        )
    )

    return {
        "answer": result.answer,
        "citations": [
            {"doc_id": d.get("doc_id", "unknown"), "content": d.get("content", "")[:200]}
            for d in docs
        ],
    }