"""Retrieval extracted so both the API and the eval harness call the same code path."""
from dataclasses import dataclass
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
from multipdf_chat.logging_context import get_request_id
import time 
import logging

@dataclass 
class RetrievedParent:
    parent_id: str 
    doc_slug: str 
    doc_title: str 
    section_path: Optional[str]
    content: str 
    distance: float 
    source_url: Optional[str] = None 
    authority_tier: Optional[int] = None 

logger = logging.getLogger("api")

def retrieve(
    db: Session,
    embeddings,
    user_question: str,
    product_line: Optional[str] = None,
    k: int = 10
) -> list[RetrievedParent]:

    """
    Step 1 baseline: uses the current unfiltered/ungrouped query so we can
    record where we're starting from. The `product_line` argument is
    accepted but ignored here - step 2 wires it into the SQL and replaces
    the body with the CTE from ReArchitecture.md #3.
    """
    request_id = get_request_id()
    logger.info("retrieval_started", extra={
        "request_id": request_id,
        "k": k,
        "product_line": product_line,
        "question_len": len(user_question),
    })
    t_embed_start = time.perf_counter()
    query_embedding = embeddings.embed_query(user_question)
    embed_ms = round((time.perf_counter() - t_embed_start) * 1000)
    
    # WHERE metadata ->> 'session_id' = :session_id 

    # Find nearest child chunks - order by cosine distance between stored embeddings and query embedding
    t_sql_start = time.perf_counter()
    result = db.execute(
        text("""
            SELECT 
                parent_id,
                embedding <=> CAST(:embedding AS vector) AS distance 
            FROM child_chunks                 
            ORDER BY distance
            LIMIT :limit
        """),
        {
            # "session_id": session_id,
            "embedding": str(query_embedding),
            "limit": k
        }
    )
    child_rows = result.fetchall() 

    if not child_rows: 
       return []

    # Store unique parent ids 
    parent_ids: list[str] = []
    dist_by_parent: dict[str, float] = {}

    for row in child_rows:
        pid = str(row.parent_id)
        if pid not in dist_by_parent:
            dist_by_parent[pid] = row.distance
            parent_ids.append(pid)
        if len(parent_ids) >= k: 
            break

    if not parent_ids: 
        return []

    # Fetch parent documents 
    parents = db.execute(
        text("""
            SELECT 
                p.id AS parent_id, 
                d.slug AS doc_slug,
                d.title AS doc_title,
                d.source_url,
                d.authority_tier,
                p.section_path, 
                p.content 
            FROM 
                parent_documents p JOIN documents d ON p.doc_id = d.doc_id 
            WHERE p.id = ANY(CAST(:ids AS uuid[]))
        """),
        {"ids": parent_ids}
    ).mappings().all()

    by_id = { str(p['parent_id']): p for p in parents }

    ordered =  [ by_id[pid] for pid in parent_ids if pid in by_id ]

    sql_ms = round((time.perf_counter() - t_sql_start) * 1000)

    logger.info("retrieval_completed", extra={
        "request_id": request_id,
        "n_parents": len(parents),
        "min_distance": min(dist_by_parent.values()) if dist_by_parent else None,
        "max_distance": max(dist_by_parent.values()) if dist_by_parent else None,
        "embed_ms": embed_ms,
        "sql_ms": sql_ms,
        "duration_ms": embed_ms + sql_ms,
        "docs_returned": [p.doc_slug for p in parents[:3]],
    })

    return [
        RetrievedParent(
            parent_id=str(p['parent_id']), 
            doc_slug=p['doc_slug'],
            doc_title=p['doc_title'],  
            section_path=p['section_path'], 
            content=p['content'],
            distance=dist_by_parent[str(p['parent_id'])], 
            source_url=p['source_url'], 
            authority_tier=p['authority_tier']
        )
        for p in ordered
    ]

    # docs = [
    #     Document(
    #         page_content = row[1],
    #         metadata = {
    #             "id": str(row[0]),
    #             "doc_id": row[2],
    #             "section_path": row[3]
    #             # "session_id": session_id
    #         }
    #     )
    #     for row in result.fetchall()
    # ]