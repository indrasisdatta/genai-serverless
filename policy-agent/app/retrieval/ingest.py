"""Document ingestion - chunk, embed, and upsert."""

from typing import Optional, List
from sqlalchemy.orm import Session
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.retrieval.embeddings import get_embedder
from app.config import settings


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[str]:
    """Split text into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def ingest_document(
    db: Session,
    doc_id: str,
    title: str,
    section: str,
    content: str,
) -> List[str]:
    """
    Chunk, embed, and store document content.

    Args:
        db: Database session
        doc_id: Document identifier
        title: Document title
        section: Section path
        content: Raw text content

    Returns:
        List of chunk IDs created
    """
    embedder = get_embedder()
    chunks = chunk_text(content)

    # Embed all chunks
    embeddings = embedder.embed_documents(chunks)

    chunk_ids = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_id = f"{doc_id}_{section}_{i}"
        chunk_ids.append(chunk_id)

        # Store in database (implementation depends on your schema)
        # This is a placeholder for the actual DB insert
        db.execute(
            """
            INSERT INTO chunks (id, doc_id, section, content, embedding)
            VALUES (:id, :doc_id, :section, :content, :embedding)
            """,
            {
                "id": chunk_id,
                "doc_id": doc_id,
                "section": section,
                "content": chunk,
                "embedding": str(embedding),
            },
        )

    db.commit()
    return chunk_ids