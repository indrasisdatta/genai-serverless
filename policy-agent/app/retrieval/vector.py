"""Vector search using pgvector."""

from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.retrieval.embeddings import get_embedder


class Chunk:
    """Retrieved chunk from vector search."""

    def __init__(
        self,
        id: int,
        doc_id: str,
        section: Optional[str],
        content: str,
        score: float,
    ):
        self.id = id
        self.doc_id = doc_id
        self.section = section
        self.content = content
        self.score = score


def vector_search(
    db: Session,
    query: str,
    top_k: int = 10,
) -> List[Chunk]:
    """
    Search chunks using vector similarity.

    Args:
        db: Database session
        query: Search query
        top_k: Number of results to return

    Returns:
        List of Chunks ordered by similarity
    """
    embedder = get_embedder()
    query_embedding = embedder.embed_query(query)

    result = db.execute(
        text("""
            SELECT id, doc_id, section, content,
                   embedding <=> CAST(:embedding AS vector) AS distance
            FROM chunks
            ORDER BY distance
            LIMIT :limit
        """),
        {
            "embedding": str(query_embedding),
            "limit": top_k,
        },
    )

    rows = result.fetchall()
    return [
        Chunk(
            id=row.id,
            doc_id=row.doc_id,
            section=row.section,
            content=row.content,
            score=row.distance,
        )
        for row in rows
    ]