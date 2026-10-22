"""In-memory BM25 index over child chunks. Rebuilt on FastAPI startup."""
from dataclasses import dataclass, field
import re
from typing import Optional
from rank_bm25 import BM25Okapi
from sqlalchemy import text
from sqlalchemy.orm import Session

_TOKEN_RE = re.compile(r"\w+")

print(_TOKEN_RE)

def tokenize(s: str) -> list[str]:
    """
    Example:
    Input: Late Payment fee is $9 under section 11
    Output: ['late', 'payment', 'fee', 'is', '9', 'under', 'section', '11']
    """
    return _TOKEN_RE.findall(s.lower())

@dataclass 
class BM25Index:
    bm25: BM25Okapi
    parent_ids: list[str] = field(default_factory=list)
    product_lines: list[str] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)

def build_bm25(db: Session) -> BM25Index:
    rows = db.execute(text("""
        SELECT 
            c.parent_id, c.content, c.product_line, d.status 
        FROM child_chunks c 
        JOIN documents d 
        ON d.doc_id = c.doc_id 
    """)).mappings().all()

    corpus = [tokenize(r['content']) for r in rows]

    return BM25Index(
        bm25=BM25Okapi(corpus),
        parent_ids=[str(r['parent_id']) for r in rows], 
        product_lines=[str(r['product_line']) for r in rows], 
        statuses=[str(r['status']) for r in rows] 
    )

def bm25_search(
    idx: BM25Index,
    question: str, 
    product_line: Optional[str],
    k: int
): 
    """
    Return parent_ids ranked by BM25 score, after filter + dedup.
    tokenize(question) converts the question into lowercase words.
    get_scores() calculates a relevance score for every indexed document.
    """
    scores = idx.bm25.get_scores(tokenize(question))
    ranked = sorted(
        range(len(scores)), # [0, 1, 2]
        key=lambda i: scores[i], # sort based on score
        reverse=True
    )
    seen: set[str] = set() 
    out: list[str] = []

    for i in ranked:    
        if scores[i] <= 0 or idx.statuses[i] != 'active':
            continue 
        if product_line and product_line not in idx.product_lines[i]:
            continue 
        pid = idx.parent_ids[i]
        if pid in seen:
            continue 
        seen.add(pid)
        out.append(pid)
        if len(out) >= k:
            break 

    return out 


