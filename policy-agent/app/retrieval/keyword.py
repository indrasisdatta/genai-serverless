"""Keyword search using Postgres full-text search."""

from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text


class Chunk:
    """Retrieved chunk from keyword search."""

    def __init__(
        self,
        id: int,
        doc_id: str,
        section: Optional[str],
        content: str,
        rank: float,
    ):
        self.id = id
        self.doc_id = doc_id
        self.section = section
        self.content = content
        self.rank = rank


def keyword_search(
    db: Session,
    query: str,
    top_k: int = 10,
) -> List[Chunk]:
    """
    Search chunks using Postgres full-text search.

    Args:
        db: Database session
        query: Search query
        top_k: Number of results to return

    Returns:
        List of Chunks ordered by rank
    """
    result = db.execute(
        text("""
            SELECT id, doc_id, section, content,
                   ts_rank(tsv, websearch_to_tsquery('english', :query)) AS rank
            FROM chunks
            WHERE tsv @@ websearch_to_tsquery('english', :query)
            ORDER BY rank DESC
            LIMIT :limit
        """),
        {
            "query": query,
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
            rank=row.rank,
        )
        for row in rows
    ]