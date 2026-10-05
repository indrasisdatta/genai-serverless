# Evaluation & Improvement Roadmap

One step at a time. Each step is scoped to ~2–3 days of focused work: read the
concepts, ship the change, re-run the golden set, record the delta, commit the
numbers.

________________________________________

## Guiding principles

Baseline before improvement. No change ships until the previous baseline
is recorded. Otherwise you can't say what the change bought.

One variable at a time. BM25 + reranking + a query rewrite all at once
means you learn nothing when the score moves.

Retrieval and generation are separate axes. Score them independently.
A recall drop and a faithfulness drop look identical in user-facing output.

Numbers get committed. Every step ends with a `baselines/step_XX.json`
or similar, checked into the repo alongside the code that produced it.

Learning is a first-class output. Each step names the concepts to
internalise before starting. Skipping the reading turns the step into

________________________________________

## Current state (2026-09-21)

Ingestion: [multipdf_chat/api/create_file_embeddings_handler.py](multipdf_chat/api/create_file_embeddings_handler.py) walks a slug list, reads the PDF from `storage_path`, parent-splits with `RecursiveCharacterTextSplitter(2000/200)`, child-splits with `SemanticChunker`, stamps identity, batch-embeds children, bulk-inserts.

Retrieval (non-stream): `user_input` in [helper.py:401](multipdf_chat/helper.py#L401) — filters on `metadata->>'session_id'`, which is the leftover FAISS path. Does not match the query in [ReArchitecture.md §3](Notes/RAG_Workflow_Docs/ReArchitecture.md).

Retrieval (stream): `stream_user_input` in [helper.py:471](multipdf_chat/helper.py#L471) — no `product_line` filter, no `status = 'active'`, no `GROUP BY parent_id`, no join to `documents`. Also missing from the payload: `product_line` is not on `UserQuery` ([models/userQuery.py](multipdf_chat/models/userQuery.py)).

No evaluation harness. No `golden_v1.jsonl` on disk yet.

No structured logs around retrieval. `db.execute()` is sync inside `async def stream_user_input` — blocks the event loop.

________________________________________

## Step 1 — Freeze the golden set and build the eval harness

Duration: 2–3 days

Ships: [eval/golden_v1.jsonl](eval/golden_v1.jsonl) (already drafted), `multipdf_chat/retrieval.py`, `eval/run_golden.py`, `baselines/step_01_current.json`

### Concepts to learn first

Retrieval metrics. Recall@k, MRR, nDCG@k. Understand what each one is
sensitive to and why "top-1 hit" and "top-5 hit" tell you different things.

Generation metrics. Exact-match string checks (`must_include` /
`must_not_include`) vs. LLM-as-judge for faithfulness. Know the failure
modes of each (string checks miss paraphrase; judges are noisy and biased).

The `must_not_include` idea. Why the wireless-vs-fios-vs-DPA fee example
in ReArchitecture.md §4 is a scored failure, not just a code review note.

Suggested reading (~2 hrs): the "Evaluating RAG" chapter of any modern RAG
guide (RAGAS docs are decent), and skim the pgvector benchmarking blogs for
how people report numbers.

### The golden set

A first-cut set is committed at [eval/golden_v1.jsonl](eval/golden_v1.jsonl)
with 28 items covering:

| Category | Item ids | What it tests |
|----------|----------|---------------|
| Basic doc lookups | gd-001, gd-002 | product_line routes to the right document |
| Late-fee disambiguation | gd-003…gd-005, gd-007, gd-008, gd-023 | the $7/$9/$5 case from ReArchitecture.md §4 — the load-bearing test for stamping + filtering |
| No product context | gd-006 | must refuse, not merge three fee schedules (needs step 5) |
| Returned payment fee | gd-009 | $30 wireless fee from db.sql seed note |
| Arbitration | gd-010, gd-011 | needs PDF verification for section refs |
| Fios equipment return | gd-012, gd-013 | return window + 12-item exclusion list |
| IPI-only items | gd-014…gd-018 | ETF, activation fee, surcharges — the mixed-authority doc is the ONLY source |
| Device Payment state provisions | gd-019, gd-020, gd-028 | DC/SD/IL specifics |
| Fios billing dispute | gd-021 | gd-051 in the arch doc — the 30-vs-60-day contradiction |
| Cross-product leakage | gd-022 | Fios exclusion list must NOT surface on a wireless ticket |
| Fios rate paraphrase | gd-024 | 18%/yr vs 1.5%/month — same number, different framing |
| Insufficient evidence | gd-025, gd-026, gd-027 | not in corpus / competitor / external process — must refuse |

Items flagged `"_verify": true` need the actual PDF read before their
`must_include` figures can be trusted.** Do this pass first: open each PDF,
verify each figure or section number, fill in the `must_include` values,
remove the `_verify` flag. Committing scores against unverified items is
worse than not scoring — it feels rigorous and isn't.

### The retrieval refactor (do this first)

Retrieval currently lives inside `stream_user_input`. Extract it so the
harness and the API call the same function.

Create `multipdf_chat/retrieval.py`:

```python
"""Retrieval extracted so both the API and the eval harness call the same code path."""
from dataclasses import dataclass
from typing import Optional
from sqlalchemy import text
from sqlalchemy.orm import Session

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

def retrieve(
    db: Session,
    embeddings,
    question: str,
    product_line: Optional[str] = None,
    k: int = 10,
) -> list[RetrievedParent]:
    """
    Step 1 baseline: uses the current unfiltered/ungrouped query so we can
    record where we're starting from. The `product_line` argument is
    accepted but ignored here — step 2 wires it into the SQL and replaces
    the body with the CTE from ReArchitecture.md §3.
    """
    query_vec = embeddings.embed_query(question)
    # Oversample children, then dedup to parents.
    child_rows = db.execute(
        text("""
SELECT c.parent_id,
c.embedding <=> CAST(:embedding AS vector) AS distance
FROM child_chunks c
ORDER BY distance
LIMIT :limit
"""),
        {"embedding": str(query_vec), "limit": k * 3},
    ).fetchall()
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
    parents = db.execute(
        text("""
SELECT p.id AS parent_id,
d.slug AS doc_slug,
d.title AS doc_title,
d.source_url,
d.authority_tier,
p.section_path,
p.content
FROM parent_documents p
JOIN documents d ON d.doc_id = p.doc_id
WHERE p.id = ANY(:ids)
"""),
        {"ids": parent_ids},
    ).mappings().all()
    by_id = {str(p["parent_id"]): p for p in parents}
    ordered = [by_id[pid] for pid in parent_ids if pid in by_id]
    return [
        RetrievedParent(
            parent_id=str(p["parent_id"]),
            doc_slug=p["doc_slug"],
            doc_title=p["doc_title"],
            section_path=p["section_path"],
            content=p["content"],
            distance=dist_by_parent[str(p["parent_id"])],
            source_url=p["source_url"],
            authority_tier=p["authority_tier"],
        )
        for p in ordered
    ]
```

Then update [helper.py:471](multipdf_chat/helper.py#L471)'s
`stream_user_input` to call it:

```python
from multipdf_chat.retrieval import retrieve
async def stream_user_input(request: Request, user_question, session_id, product_line=None):
    embeddings = request.app.state.embeddings
    db = request.app.state.db()
    try:
        parents = retrieve(db, embeddings, user_question, product_line, k=10)
        if not parents:
            yield "No relevant data found"
            return
        docs = [
            Document(
                page_content=p.content,
                metadata={
                    "id": p.parent_id,
                    "doc_slug": p.doc_slug,
                    "section_path": p.section_path,
                },
            )
            for p in parents
        ]
        handler = StreamingHandler()
        chain = get_conversational_chain(streaming=True, callbacks=[handler])
        asyncio.create_task(
            chain.ainvoke(
                {"input_documents": docs, "question": user_question},
                config={"callbacks": [handler]},
            )
        )
        buffer = ""
        while True:
            token = await handler.queue.get()
            if token is None:
                break
            if not token or token.isspace():
                continue
            buffer += token
            if len(buffer) >= 20:
                yield buffer
                buffer = ""
        if buffer:
            yield buffer
    finally:
        db.close()
```

### The harness

Create `eval/run_golden.py`:

```python
"""
Golden-set evaluation harness.
Usage:
python -m eval.run_golden --golden eval/golden_v1.jsonl --out baselines/step_01_current.json
# retrieval-only (skip LLM, useful while iterating on ranking)
python -m eval.run_golden --golden eval/golden_v1.jsonl --out baselines/retrieval_only.json --skip-generation
"""
import argparse
import json
import subprocess
import time
from pathlib import Path
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from multipdf_chat.db import SessionLocal
from multipdf_chat.helper import get_conversational_chain
from multipdf_chat.retrieval import retrieve

def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"

def score_retrieval(item, retrieved, k_values=(1, 5, 10)):
    """Recall@k, MRR, section_hit@5. Gold-empty items score None (handled in summary)."""
    gold_docs = set(item.get("gold_doc_ids") or [])
    gold_sections = set(item.get("gold_sections") or [])
    retrieved_docs = [r.doc_slug for r in retrieved]
    retrieved_sections = [r.section_path for r in retrieved]
    scores = {}
    if not gold_docs:
        # Insufficient-evidence items: retrieval isn't scored on recall.
        for k in k_values:
            scores[f"recall@{k}"] = None
            scores["mrr"] = None
    else:
        for k in k_values:
            scores[f"recall@{k}"] = 1.0 if gold_docs & set(retrieved_docs[:k]) else 0.0
        mrr = 0.0
        for rank, doc in enumerate(retrieved_docs, start=1):
            if doc in gold_docs:
                mrr = 1.0 / rank
                break
        scores["mrr"] = mrr
    if gold_sections:
        top5 = [s for s in retrieved_sections[:5] if s]
        hit = any(g in s for g in gold_sections for s in top5)
        scores["section_hit@5"] = 1.0 if hit else 0.0
    else:
        scores["section_hit@5"] = None
    return scores

def score_generation(item, answer_text: str):
    """String checks. Case-insensitive. Empty must-lists score as passing."""
    ans_lower = answer_text.lower()
    must_include = item.get("must_include") or []
    must_not_include = item.get("must_not_include") or []
    include_hits = [s for s in must_include if s.lower() in ans_lower]
    exclude_hits = [s for s in must_not_include if s.lower() in ans_lower]
    return {
        "must_include_pass": len(include_hits) == len(must_include),
        "must_include_hit_rate": (
            len(include_hits) / len(must_include) if must_include else None
        ),
        "must_not_include_pass": len(exclude_hits) == 0,
        "must_not_include_violations": exclude_hits,
    }

def generate_answer(retrieved, question: str) -> str:
    if not retrieved:
        return ""
    docs = [
        Document(
            page_content=r.content,
            metadata={"doc_slug": r.doc_slug, "section_path": r.section_path},
        )
        for r in retrieved[:5]
    ]
    chain = get_conversational_chain(streaming=False)
    resp = chain(
        {"input_documents": docs, "question": question},
        return_only_outputs=True,
    )
    return resp["output_text"]

def summarise(results):
    def _mean(section, key):
        vals = [r[section][key] for r in results if r[section].get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None
    def _pass_rate(key):
        vals = [r["generation"][key] for r in results if r["generation"].get(key) is not None]
        return round(sum(1 for v in vals if v) / len(vals), 3) if vals else None
    return {
        "n_items": len(results),
        "recall@1": _mean("retrieval", "recall@1"),
        "recall@5": _mean("retrieval", "recall@5"),
        "recall@10": _mean("retrieval", "recall@10"),
        "mrr": _mean("retrieval", "mrr"),
        "section_hit@5": _mean("retrieval", "section_hit@5"),
        "must_include_pass_rate": _pass_rate("must_include_pass"),
        "must_not_include_pass_rate": _pass_rate("must_not_include_pass"),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--skip-generation", action="store_true")
    args = ap.parse_args()
    items = [
        json.loads(line)
        for line in Path(args.golden).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("//")
    ]
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    db = SessionLocal()
    results = []
    t0 = time.time()
    try:
        for item in items:
            r_start = time.time()
            retrieved = retrieve(
                db, embeddings, item["question"], item.get("product_line"), k=args.k
            )
            r_ms = round((time.time() - r_start) * 1000)
            r_scores = score_retrieval(item, retrieved)
            if args.skip_generation:
                answer, g_scores, g_ms = "", {}, 0
            else:
                g_start = time.time()
                answer = generate_answer(retrieved, item["question"])
                g_ms = round((time.time() - g_start) * 1000)
            g_scores = score_generation(item, answer)
            results.append({
                "id": item["id"],
                "question": item["question"],
                "product_line": item.get("product_line"),
                "retrieved_docs": [r.doc_slug for r in retrieved[:5]],
                "retrieved_sections": [r.section_path for r in retrieved[:5]],
                "answer": answer,
                "retrieval": r_scores,
                "generation": g_scores,
                "timing_ms": {"retrieval": r_ms, "generation": g_ms},
            })
    finally:
        db.close()
    out = {
        "git_sha": git_sha(),
        "golden_set": args.golden,
        "k": args.k,
        "total_seconds": round(time.time() - t0, 1),
        "summary": summarise(results),
        "per_item": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["summary"], indent=2))

if __name__ == "__main__":
    main()
```

### Tasks

- Verify the `_verify: true` items in the golden set. Open each
  flagged PDF, fill in `must_include` figures, remove the `_verify` flag.
- Add or drop items as you learn what's actually in the docs.
- Extract retrieval into `multipdf_chat/retrieval.py` (code above).
- Wire `stream_user_input` to call `retrieve()` so the API and harness
  share one code path.
- Add `product_line: Optional[str]` to `UserQuery`
  ([models/userQuery.py](multipdf_chat/models/userQuery.py)) — accept it
  from the client even though step 1's SQL ignores it. Step 2 turns it on.
- Build `eval/run_golden.py` (code above).
- Run it:

```bash
python -m eval.run_golden \
  --golden eval/golden_v1.jsonl \
  --out baselines/step_01_current.json
```

- Read the results. Expect the current-state numbers to be poor —
  especially the disambiguation items (gd-003…gd-008, gd-022), because
  there's no `product_line` filter yet. That's the point: step 2 will
  fix them, and you'll see it in the diff.
- Commit `baselines/step_01_current.json`. Include the summary in the
  commit message so future-you can diff without opening the file.

### How to use this in day-to-day work

- Before any retrieval or generation change: run the harness. That's
  your before-number.
- After the change: run it again. Compare summaries. Compare per-item
  results for regressions — an average that stayed flat can hide one item
  going from pass to fail and another going the other way.
- When adding a golden item: run the harness immediately. If the new
  item passes with no code change, ask whether the item is actually testing
  something (often it's testing a case that already worked).
- `--skip-generation`: the LLM call dominates runtime. Use this mode
  when iterating on retrieval-only changes (steps 2, 6, 7, 8).

### Done when

- `python -m eval.run_golden` runs end-to-end and prints a summary.
- `baselines/step_01_current.json` is committed.
- Every `_verify: true` flag is gone from the golden set.
- You can explain, without checking notes, the difference between `recall@k`
  and `MRR`.

________________________________________

## Step 2 — Bring retrieval in line with the architecture doc
Duration: 2–3 days
Ships: rewritten retrieval SQL, `product_line` on the API payload, `baselines/step_02_arch_aligned.json`
The current retrieval is behind the design. Before adding anything new, close
the gap. This is the change most likely to move the score, because right now
a Fios ticket can retrieve wireless chunks.
### Concepts to learn first
pgvector operators. `<=>` (cosine distance), `<#>` (negative inner
product), `<->` (L2). Which opclass each needs.
Postgres array overlap `&&`. Why it's the right predicate here instead
of `= ANY`.
`GROUP BY` + `MIN` for parent aggregation. Why "top 10 children" is not
"top 10 parents", and why sending duplicates to the LLM burns context.
The new `retrieve()` body
Replace the body of `retrieve()` in `multipdf_chat/retrieval.py` (from step 1)
with the CTE from [ReArchitecture.md §3](Notes/RAG_Workflow_Docs/ReArchitecture.md):
```python
def retrieve(
db: Session,
embeddings,
question: str,
product_line: Optional[str] = None,
k: int = 10,
) -> list[RetrievedParent]:
"""
Aligned with ReArchitecture.md §3:
filter by status = 'active' and product_line overlap
GROUP BY parent_id, MIN(distance) so sibling children collapse
single SQL statement, one JOIN back to documents for citation fields
"""
query_vec = embeddings.embed_query(question)
rows = db.execute(
text("""
WITH nearest AS (
SELECT c.parent_id,
MIN(c.embedding <=> CAST(:embedding AS vector)) AS distance
FROM child_chunks c
JOIN documents d ON d.doc_id = c.doc_id
WHERE d.status = 'active'
AND (:product_line IS NULL
OR d.product_line && ARRAY[:product_line]::text[])
GROUP BY c.parent_id
ORDER BY distance
LIMIT :k
)
SELECT d.slug AS doc_slug,
d.title AS doc_title,
d.source_url,
d.authority_tier,
p.id AS parent_id,
p.section_path,
p.content,
n.distance
FROM nearest n
JOIN parent_documents p ON p.id = n.parent_id
JOIN documents d ON d.doc_id = p.doc_id
ORDER BY n.distance
"""),
{
"embedding": str(query_vec),
"product_line": product_line,
"k": k,
},
).mappings().all()
return [
RetrievedParent(
parent_id=str(r["parent_id"]),
doc_slug=r["doc_slug"],
doc_title=r["doc_title"],
section_path=r["section_path"],
content=r["content"],
distance=r["distance"],
source_url=r["source_url"],
authority_tier=r["authority_tier"],
)
for r in rows
]
```
API surface changes
Update the request payload to accept `product_line`:
```python
multipdf_chat/models/userQuery.py
from typing import Optional
from pydantic import BaseModel

class UserQuery(BaseModel):
user_question: str
session_id: Optional[str] = None
product_line: Optional[str] = None # new: routed from the ticket record
```
And thread it through the endpoint:
```python
multipdf_chat/main.py
@app.post('/chat/stream')
def streamUserQuery(userQuery: UserQuery, request: Request):
logger.info(f"User query: {userQuery}")
return StreamingResponse(
stream_user_input(
request,
userQuery.user_question,
userQuery.session_id,
userQuery.product_line,
),
media_type="text/plain",
)
```
Delete `user_input` from [helper.py:401](multipdf_chat/helper.py#L401) and
the `/user_query` endpoint that calls it — the streaming path is the only
one worth maintaining. The `metadata->>'session_id'` filter in that
function is a FAISS leftover; keeping it will silently cost you retrieval.
Manual smoke test before running the harness
```bash
curl -N -X POST http://localhost:8000/chat/stream \
-H "Content-Type: application/json" \
-d '{"user_question":"what is my late fee?","product_line":"fios_internet"}'
```
You should see `$9` and no mention of `$7` or `$5`. If you don't, the filter
isn't being applied — check EXPLAIN on the query with `\set VERBOSITY verbose`
in `psql`.
### Tasks

- Update [models/userQuery.py](multipdf_chat/models/userQuery.py) (code above).
- Update `/chat/stream` in [main.py](multipdf_chat/main.py) to pass
  `product_line` (code above).
- Replace `retrieve()` body with the aligned SQL (code above).
- Update `stream_user_input`'s signature to `(request, question,
  session_id, product_line=None)`.
- Delete `user_input` and the `/user_query` endpoint. Grep for callers first.
- Manual smoke test with curl (above).
- Run the harness:
```bash
python -m eval.run_golden \
--golden eval/golden_v1.jsonl \
--out baselines/step_02_arch_aligned.json
```
Diff the summary against `baselines/step_01_current.json`. Expect large
jumps on gd-003…gd-008, gd-022, gd-023.
### Done when
Retrieval SQL matches the doc, verbatim.
Manual curl test shows product_line routing works.
Golden set numbers are recorded and diffed in the commit message.
You can explain why `GROUP BY c.parent_id` exists.
________________________________________
## Step 3 — Structured logs around retrieval, without breaking streaming
Duration: 2 days
Ships: structured log lines at each retrieval stage, no regression in streaming latency
### Concepts to learn first
Why logging inside a token loop is a footgun. Even a formatted
`logger.info` per token adds meaningful latency at 100+ tokens/sec, and
buffering / flushing behaviour can serialise tokens that were supposed to
interleave with the network write.
`extra=` vs f-strings for structured logs. The `pythonjsonlogger`
formatter you already use in [main.py:47](multipdf_chat/main.py#L47) reads
`extra` fields — putting values in `extra` makes them queryable; putting
them in the message string does not.
Request-scoped correlation IDs. You already have `X-Request-ID`
middleware in [main.py:61](multipdf_chat/main.py#L61). Understand how to
propagate it into deeper functions (contextvars is the clean way).
Request-scoped context
Create `multipdf_chat/logging_context.py`:
```python
"""Per-request context propagated via contextvars — safe under asyncio."""
import contextvars
from typing import Optional
_request_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "request_id", default=None
)

def set_request_id(rid: str) -> None:
    _request_id.set(rid)

def get_request_id() -> Optional[str]:
    return _request_id.get()
```
Update the middleware in [main.py:61](multipdf_chat/main.py#L61) to publish
it:
```python
from multipdf_chat.logging_context import set_request_id
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    set_request_id(request_id) # new: any downstream code can now read it
    start_time = time.time()
    logger.info("request_started", extra={
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
    })
    # ... rest unchanged ...
```
Instrumented retrieval
Add log lines at boundaries only — never inside a token loop. Update
`retrieve()`:
```python
multipdf_chat/retrieval.py
import logging
import time
from multipdf_chat.logging_context import get_request_id
logger = logging.getLogger("api")

def retrieve(db, embeddings, question, product_line=None, k=10):
    request_id = get_request_id()
    logger.info("retrieval_started", extra={
        "request_id": request_id,
        "k": k,
        "product_line": product_line,
        "question_len": len(question),
    })
    t_embed_start = time.perf_counter()
    query_vec = embeddings.embed_query(question)
    embed_ms = round((time.perf_counter() - t_embed_start) * 1000)
    t_sql_start = time.perf_counter()
    rows = db.execute(text(""" ... same CTE as step 2 ... """),
        {"embedding": str(query_vec),
         "product_line": product_line,
         "k": k}).mappings().all()
    sql_ms = round((time.perf_counter() - t_sql_start) * 1000)
    parents = [RetrievedParent(...) for r in rows] # as before
    logger.info("retrieval_completed", extra={
        "request_id": request_id,
        "n_parents": len(parents),
        "min_distance": round(parents[0].distance, 4) if parents else None,
        "max_distance": round(parents[-1].distance, 4) if parents else None,
        "embed_ms": embed_ms,
        "sql_ms": sql_ms,
        "duration_ms": embed_ms + sql_ms,
        "docs_returned": [p.doc_slug for p in parents[:3]],
    })
    return parents
```
Update `stream_user_input` to bracket generation (log outside the loop):
```python
multipdf_chat/helper.py
from multipdf_chat.logging_context import get_request_id
async def stream_user_input(request, user_question, session_id, product_line=None):
    request_id = get_request_id()
    embeddings = request.app.state.embeddings
    db = request.app.state.db()
    try:
        parents = retrieve(db, embeddings, user_question, product_line, k=10)
        if not parents:
            logger.info("generation_skipped", extra={
                "request_id": request_id, "reason": "no_parents",
            })
            yield "No relevant data found"
            return
        docs = [Document(page_content=p.content,
                         metadata={"id": p.parent_id,
                                   "doc_slug": p.doc_slug,
                                   "section_path": p.section_path})
                for p in parents]
        logger.info("generation_started", extra={
            "request_id": request_id, "n_context_docs": len(docs),
        })
        gen_start = time.perf_counter()
        handler = StreamingHandler()
        chain = get_conversational_chain(streaming=True, callbacks=[handler])
        asyncio.create_task(chain.ainvoke(
            {"input_documents": docs, "question": user_question},
            config={"callbacks": [handler]},
        ))
        token_count = 0
        buffer = ""
        while True:
            token = await handler.queue.get()
            if token is None:
                break
            if not token or token.isspace():
                continue
            token_count += 1 # counter only — do NOT log per token
            buffer += token
            if len(buffer) >= 20:
                yield buffer
                buffer = ""
        if buffer:
            yield buffer
        logger.info("generation_completed", extra={
            "request_id": request_id,
            "duration_ms": round((time.perf_counter() - gen_start) * 1000),
            "tokens": token_count,
        })
    finally:
        db.close()
```
Latency benchmark
Create `eval/bench_stream_latency.py`:
```python
"""
Measure first-token and total-response latency of /chat/stream.
Usage:
python -m eval.bench_stream_latency --n 20 --url http://localhost:8000/chat/stream
"""
import argparse
import asyncio
import statistics
import time
import httpx

async def one_request(client: httpx.AsyncClient, url: str, payload: dict) -> dict:
    t0 = time.perf_counter()
    first_token_at = None
    total_bytes = 0
    async with client.stream("POST", url, json=payload, timeout=60.0) as r:
        async for chunk in r.aiter_bytes():
            if not chunk:
                continue
            if first_token_at is None:
                first_token_at = time.perf_counter()
            total_bytes += len(chunk)
    t1 = time.perf_counter()
    return {
        "first_token_ms": round((first_token_at - t0) * 1000) if first_token_at else None,
        "total_ms": round((t1 - t0) * 1000),
        "bytes": total_bytes,
    }

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/chat/stream")
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    payload = {
        "user_question": "What is my late fee?",
        "session_id": "bench",
        "product_line": "mobile_postpaid",
    }
    async with httpx.AsyncClient() as client:
        # Warm up (first request loads model / opens pool)
        await one_request(client, args.url, payload)
        results = []
        for _ in range(args.n):
            results.append(await one_request(client, args.url, payload))
        first = [r["first_token_ms"] for r in results if r["first_token_ms"] is not None]
        total = [r["total_ms"] for r in results]
        print(f"n = {args.n}")
        print(f"first_token: p50={statistics.median(first):.0f} ms "
              f"p95={statistics.quantiles(first, n=20)[-1]:.0f} ms")
        print(f"total: p50={statistics.median(total):.0f} ms "
              f"p95={statistics.quantiles(total, n=20)[-1]:.0f} ms")

if __name__ == "__main__":
    asyncio.run(main())
```
### Tasks

- Create `multipdf_chat/logging_context.py` (code above).
- Update the middleware in [main.py](multipdf_chat/main.py) to call
  `set_request_id` (code above).
- Add `retrieval_started` / `retrieval_completed` / `generation_started` /
  `generation_completed` log lines.
- Baseline the latency:
```bash
python -m eval.bench_stream_latency --n 20
```
Save the number.
Apply the log changes. Re-run the benchmark. Confirm first-token latency
is within 5% of the pre-change number.
Re-run the golden set — scores should not move. Record
`baselines/step_03_logged.json` anyway (proves harness stability across
no-op edits).
### Done when
Every retrieval invocation produces a queryable JSON log line with
distance, chunk count, and duration.
`request_id` appears in every retrieval and generation log line.
First-token latency is within 5% of the pre-change measurement.
________________________________________
## Step 4 — Fix async: stop blocking the event loop
Duration: 2–3 days
Ships: async DB path, async embedding path, latency measurements
### Concepts to learn first
What "blocking the event loop" actually means. A sync `db.execute()`
inside `async def` stalls every other request on that worker until it
completes. For a single-user local demo this is invisible; for concurrency
benchmarks it's catastrophic.
`asyncpg` vs `psycopg` async vs SQLAlchemy 2.x async. Understand the
three layers and pick one deliberately.
`asyncio.to_thread` as an escape hatch. For CPU-bound work
(embeddings on CPU), `to_thread` is often the right answer rather than
rewriting the library. Know when each applies.
Async engine, alongside the sync one
Do not convert ingestion — the sync path in
[create_file_embeddings_handler.py](multipdf_chat/api/create_file_embeddings_handler.py)
is a batch job, not on the hot path. Only the streaming query needs async.
Extend `multipdf_chat/db.py`:
```python
whatever you already have for sync:
engine = create_engine(DATABASE_URL, ...)
SessionLocal = sessionmaker(bind=engine, ...)
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
asyncpg is the driver; the URL prefix picks it
ASYNC_DATABASE_URL = DATABASE_URL.replace(
    "postgresql://", "postgresql+asyncpg://", 1
)
async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    pool_size=10,
    pool_pre_ping=True,
)
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
```
Register in lifespan ([main.py:25](multipdf_chat/main.py#L25)):
```python
from multipdf_chat.db import SessionLocal, AsyncSessionLocal
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.db = SessionLocal # existing, for ingestion
        app.state.async_db = AsyncSessionLocal # new, for queries
        app.state.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        yield
    finally:
        app.state.db = None
        app.state.async_db = None
        app.state.embeddings = None
```
Async `retrieve()`
Add alongside the sync version in `multipdf_chat/retrieval.py`:
```python
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession

async def retrieve_async(
    db: AsyncSession,
    embeddings,
    question: str,
    product_line: Optional[str] = None,
    k: int = 10,
) -> list[RetrievedParent]:
    """Async twin of retrieve(). Same SQL, same shape, non-blocking."""
    request_id = get_request_id()
    logger.info("retrieval_started", extra={
        "request_id": request_id, "k": k,
        "product_line": product_line, "question_len": len(question),
    })
    # embed_query is CPU-bound sync work — push it off the event loop.
    t_embed = time.perf_counter()
    query_vec = await asyncio.to_thread(embeddings.embed_query, question)
    embed_ms = round((time.perf_counter() - t_embed) * 1000)
    t_sql = time.perf_counter()
    result = await db.execute(
        text("""
        WITH nearest AS (
            SELECT c.parent_id,
            MIN(c.embedding <=> CAST(:embedding AS vector)) AS distance
            FROM child_chunks c
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE d.status = 'active'
            AND (:product_line IS NULL
            OR d.product_line && ARRAY[:product_line]::text[])
            GROUP BY c.parent_id
            ORDER BY distance
            LIMIT :k
        )
        SELECT d.slug AS doc_slug, d.title AS doc_title,
        d.source_url, d.authority_tier,
        p.id AS parent_id, p.section_path, p.content, n.distance
        FROM nearest n
        JOIN parent_documents p ON p.id = n.parent_id
        JOIN documents d ON d.doc_id = p.doc_id
        ORDER BY n.distance
        """),
        {"embedding": str(query_vec),
         "product_line": product_line,
         "k": k},
    )
    rows = result.mappings().all()
    sql_ms = round((time.perf_counter() - t_sql) * 1000)
    parents = [
        RetrievedParent(
            parent_id=str(r["parent_id"]),
            doc_slug=r["doc_slug"],
            doc_title=r["doc_title"],
            section_path=r["section_path"],
            content=r["content"],
            distance=r["distance"],
            source_url=r["source_url"],
            authority_tier=r["authority_tier"],
        )
        for r in rows
    ]
    logger.info("retrieval_completed", extra={
        "request_id": request_id, "n_parents": len(parents),
        "min_distance": round(parents[0].distance, 4) if parents else None,
        "embed_ms": embed_ms, "sql_ms": sql_ms,
        "duration_ms": embed_ms + sql_ms,
    })
    return parents
```
Rewire `stream_user_input`
Replace the sync `db = request.app.state.db()` with the async context
manager:
```python
async def stream_user_input(request, user_question, session_id, product_line=None):
    request_id = get_request_id()
    embeddings = request.app.state.embeddings
    async with request.app.state.async_db() as db:
        parents = await retrieve_async(db, embeddings, user_question,
                                       product_line, k=10)
        if not parents:
            yield "No relevant data found"
            return
        docs = [Document(page_content=p.content,
                         metadata={"id": p.parent_id,
                                   "doc_slug": p.doc_slug,
                                   "section_path": p.section_path})
                for p in parents]
        handler = StreamingHandler()
        chain = get_conversational_chain(streaming=True, callbacks=[handler])
        asyncio.create_task(chain.ainvoke(
            {"input_documents": docs, "question": user_question},
            config={"callbacks": [handler]},
        ))
        buffer = ""
        while True:
            token = await handler.queue.get()
            if token is None:
                break
            if not token or token.isspace():
                continue
            buffer += token
            if len(buffer) >= 20:
                yield buffer
                buffer = ""
        if buffer:
            yield buffer
```
Note: the `finally: db.close()` is gone — `async with` handles it.
Concurrency benchmark
Create `eval/bench_concurrent.py`:
```python
"""
Fire N /chat/stream requests concurrently, measure first-token latency for each.
Usage:
python -m eval.bench_concurrent --concurrency 20 --url http://localhost:8000/chat/stream
"""
import argparse
import asyncio
import statistics
import time
import httpx

async def one(client, url, payload) -> float:
    t0 = time.perf_counter()
    first_at = None
    async with client.stream("POST", url, json=payload, timeout=60.0) as r:
        async for chunk in r.aiter_bytes():
            if chunk and first_at is None:
                first_at = time.perf_counter()
            break # only need first-token
    return round(((first_at or time.perf_counter()) - t0) * 1000)

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/chat/stream")
    ap.add_argument("--concurrency", type=int, default=20)
    args = ap.parse_args()
    payload = {
        "user_question": "What is my late fee?",
        "product_line": "mobile_postpaid",
    }
    async with httpx.AsyncClient() as client:
        # Warm up
        await one(client, args.url, payload)
        t0 = time.perf_counter()
        tasks = [one(client, args.url, payload) for _ in range(args.concurrency)]
        latencies = await asyncio.gather(tasks)
        wall = round((time.perf_counter() - t0) * 1000)
        latencies.sort()
        print(f"concurrency = {args.concurrency}")
        print(f"wall clock: {wall} ms")
        print(f"first_token p50: {statistics.median(latencies):.0f} ms")
        print(f"first_token p95: {statistics.quantiles(latencies, n=20)[-1]:.0f} ms")
        print(f"first_token max: {max(latencies)} ms")

if __name__ == "__main__":
    asyncio.run(main())
```
### Tasks

- Extend `multipdf_chat/db.py` with the async engine + sessionmaker.
- Register `async_db` in the lifespan.
- Add `retrieve_async()` alongside `retrieve()`.
- Rewrite `stream_user_input` to use `async with` + `retrieve_async` +
  the async embedding path via `asyncio.to_thread`.
- Baseline first:
```bash
python -m eval.bench_concurrent --concurrency 20
```
Save the numbers. Do this before the async rewrite, on the current
sync-inside-async code.
Apply the async changes. Rebench. Expect a meaningful drop in p95
(typically 30–60% at concurrency 20). If it doesn't move, profile —
probably the Groq LLM API is now the bottleneck, and that's a finding
worth writing down.
Re-run the golden set: `baselines/step_04_async.json`. Should be flat
on scores; if it isn't, the async rewrite has a bug.
### Done when
No sync `db.execute` runs inside `async def` on the streaming path.
Concurrency benchmark shows measurable improvement (or you've documented
why it didn't).
Golden-set scores match step 3 within noise.
________________________________________
## Step 5 — Insufficient-evidence and conflicting-sources routing
Duration: 2 days
Ships: two new response modes, calibrated threshold, `baselines/step_05_evidence.json`
Right now the pipeline always answers. The doc (§3, "evidence check") calls
for `insufficient_evidence` and `conflicting_sources` outcomes. These are
worth doing before reranking because they change what "correct" means: an
"I don't know" that routes to a human is a correct outcome for gd-006, not
a failure.
### Concepts to learn first
Distance-threshold routing. Why "top result distance > 0.4" is a
useful signal for insufficient evidence, and why the threshold is
corpus-specific (must be measured, not guessed).
Multi-source disagreement. How to detect "the top 3 parents come from
different `doc_id`s with overlapping `product_line` but differ on a key
Simple heuristics beat LLM judges here.
Routing module
Create `multipdf_chat/routing.py`:
```python
"""Post-retrieval routing: answer / refuse / flag for human."""
import re
from dataclasses import dataclass, field
from typing import Literal, Optional
from multipdf_chat.retrieval import RetrievedParent
Calibrated per step 5 tasks. See eval/calibrate_threshold.py.
DEFAULT_DISTANCE_THRESHOLD = 0.40
_MONEY_RE = re.compile(r"$\s?\d{1,4}(?:.\d{2})?")
_PCT_RE = re.compile(r"\d+(?:.\d+)?\s%")

Status = Literal["answered", "insufficient_evidence", "conflicting_sources"]

@dataclass
class RoutingDecision:
    status: Status
    reason: str
    parents: list[RetrievedParent] = field(default_factory=list)

def _figures(text: str) -> frozenset[str]:
    """Extract dollar amounts and percentages as normalised strings."""
    money = {m.replace(" ", "") for m in _MONEY_RE.findall(text)}
    pct = {p.replace(" ", "") for p in _PCT_RE.findall(text)}
    return frozenset(money | pct)

def _figures_conflict(parents: list[RetrievedParent], top_n: int = 3) -> Optional[str]:
    """
    Return a reason string if the top-N parents from different docs disagree
    on key figures. Returns None if there's no conflict or not enough
    distinct docs to detect one.
    """
    seen_docs: set[str] = set()
    per_doc: list[tuple[str, frozenset[str]]] = []
    for p in parents[:top_n]:
        if p.doc_slug in seen_docs:
            continue
        seen_docs.add(p.doc_slug)
        figs = _figures(p.content)
        if figs:
            per_doc.append((p.doc_slug, figs))
    if len(per_doc) < 2:
        return None
    all_figs = set().union((f for _, f in per_doc))
    shared = set.intersection((set(f) for _, f in per_doc))
    # If <50% of figures are shared across docs, treat as conflict.
    if all_figs and len(shared) / len(all_figs) < 0.5:
        docs = ", ".join(slug for slug, _ in per_doc)
        return f"figures diverge across {docs}"
    return None

def route(
    parents: list[RetrievedParent],
    product_line: Optional[str],
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> RoutingDecision:
    if not parents:
        return RoutingDecision("insufficient_evidence", "no candidates returned")
    if parents[0].distance > distance_threshold:
        return RoutingDecision(
            "insufficient_evidence",
            f"top distance {parents[0].distance:.3f} > threshold {distance_threshold}",
            parents,
        )
    # Only look for conflicts when the caller didn't pin a product_line —
    # with a filter in place, the doc set is already narrowed.
    if product_line is None:
        conflict = _figures_conflict(parents)
        if conflict:
            return RoutingDecision("conflicting_sources", conflict, parents)
    return RoutingDecision("answered", "", parents)
```
Threshold calibration
Create `eval/calibrate_threshold.py`:
```python
"""
Sweep distance thresholds against the golden set. Print a confusion matrix
per threshold so you can pick the operating point deliberately.
Insufficient-evidence golden items should refuse; answerable items should
not. The right threshold minimises 'wrongly refused' without letting too
many 'missed refusals' through.
"""
import json
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from multipdf_chat.db import SessionLocal
from multipdf_chat.retrieval import retrieve

THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]

def main():
    items = [
        json.loads(l)
        for l in Path("eval/golden_v1.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    db = SessionLocal()
    per_item = []
    for item in items:
        parents = retrieve(
            db, embeddings, item["question"], item.get("product_line"), k=1
        )
        per_item.append({
            "id": item["id"],
            "expected": item.get("expected_status", "answered"),
            "top_distance": parents[0].distance if parents else 1.0,
            "top_doc": parents[0].doc_slug if parents else None,
        })
    db.close()
    print(f"{'thr':>6} {'correct_refuse':>15} {'missed_refuse':>15} "
          f"{'wrong_refuse':>13} {'correct_answer':>15}")
    for thr in THRESHOLDS:
        tp = sum(1 for d in per_item
                 if d["expected"] == "insufficient_evidence" and d["top_distance"] > thr)
        fn = sum(1 for d in per_item
                 if d["expected"] == "insufficient_evidence" and d["top_distance"] <= thr)
        fp = sum(1 for d in per_item
                 if d["expected"] == "answered" and d["top_distance"] > thr)
        tn = sum(1 for d in per_item
                 if d["expected"] == "answered" and d["top_distance"] <= thr)
        print(f"{thr:>6.2f} {tp:>15} {fn:>15} {fp:>13} {tn:>15}")
    # Diagnostic: list borderline items so you can eyeball them
    print("
borderline items (top_distance between 0.30 and 0.50):")
    for d in sorted(per_item, key=lambda x: x["top_distance"]):
        if 0.30 <= d["top_distance"] <= 0.50:
            print(f" {d['id']:12} exp={d['expected']:22} "
                  f"d={d['top_distance']:.3f} {d['top_doc']}")

if __name__ == "__main__":
    main()
```
Wire routing into the stream
Update `stream_user_input`:
```python
import json as _json
from multipdf_chat.routing import route
async def stream_user_input(request, user_question, session_id, product_line=None):
    embeddings = request.app.state.embeddings
    async with request.app.state.async_db() as db:
        parents = await retrieve_async(db, embeddings, user_question,
                                        product_line, k=10)
        decision = route(parents, product_line)
        if decision.status == "insufficient_evidence":
            yield _json.dumps({
                "status": "insufficient_evidence",
                "reason": decision.reason,
                "message": (
                    "I don't have enough information in the policy documents "
                    "to answer this confidently. Please provide more context "
                    "(e.g., which product) or route to a human agent."
                ),
            })
            return
        if decision.status == "conflicting_sources":
            yield _json.dumps({
                "status": "conflicting_sources",
                "reason": decision.reason,
                "candidate_docs": sorted({p.doc_slug for p in parents[:3]}),
                "message": (
                    "The policies differ across products. Please clarify "
                    "which product this ticket is about."
                ),
            })
            return
        # ... normal generation path (unchanged from step 4) ...
```
"The policies differ across products. Please clarify "
"which product this ticket is about."
),
})
return
# ... normal generation path (unchanged from step 4) ...
```
Update the harness to score refusals
Extend `score_generation` in `eval/run_golden.py`:
```python
def score_generation(item, answer_text: str):
    """String checks plus status routing."""
    ans_lower = answer_text.lower()
    expected_status = item.get("expected_status", "answered")
    # A refusal from the router will be a JSON blob starting with a status field
    refused = '"status": "insufficient_evidence"' in answer_text \
        or '"status": "conflicting_sources"' in answer_text
    if expected_status == "insufficient_evidence":
        return {
            "routing_pass": refused,
            "must_include_pass": None, # not scored on refusals
            "must_not_include_pass": (
                not any(s.lower() in ans_lower for s in item.get("must_not_include") or [])
            ),
        }
    must_include = item.get("must_include") or []
    must_not_include = item.get("must_not_include") or []
    include_hits = [s for s in must_include if s.lower() in ans_lower]
    exclude_hits = [s for s in must_not_include if s.lower() in ans_lower]
    return {
        "routing_pass": not refused, # answerable items must NOT refuse
        "must_include_pass": len(include_hits) == len(must_include),
        "must_not_include_pass": len(exclude_hits) == 0,
        "must_not_include_violations": exclude_hits,
    }
```
Add `routing_pass_rate` to the summary via `_pass_rate("routing_pass")`.
### Tasks

- Create `multipdf_chat/routing.py` (code above).
- Create `eval/calibrate_threshold.py`. Run it. Pick a threshold — probably
  0.35–0.45 for MiniLM-L6-v2 on this corpus. Update
  `DEFAULT_DISTANCE_THRESHOLD` in `routing.py`.
- Wire `route()` into `stream_user_input`.
- Extend `score_generation` in the harness to score `routing_pass`.
- Add 2 more insufficient-evidence items to the golden set if the calibration
  shows you have fewer than 3.
- Run the harness, record `baselines/step_05_evidence.json`.
- gd-006 should now show `routing_pass = true`. Verify that answerable
  items are not being wrongly refused — check `routing_pass_rate` on
  items where `expected_status = "answered"`.

### Done when
gd-006 returns a refusal, not a merged answer.
The threshold was chosen from measurement, not guessed.
`routing_pass_rate` appears in the summary; commit message diffs it
against step 4.
________________________________________
## Step 6 — Hybrid retrieval: BM25 alongside vectors
Duration: 3 days
Ships: BM25 index, hybrid scoring, `baselines/step_06_hybrid.json`
### Concepts to learn first
Why BM25 catches what dense retrieval misses. Exact identifiers
("section 11.1.12", "$9", model numbers) are where BM25 wins. Synonyms
and paraphrase are where dense wins. Hybrid is not "better on average" —
it's "better on the tail cases dense fails on."
Postgres `tsvector` + GIN, or a separate `rank_bm25` in Python.
Trade-offs: `tsvector` is in-DB (one query), `rank_bm25` is flexible but
needs the whole corpus in memory. For 800 chunks, in-memory is fine.
Score fusion. Reciprocal Rank Fusion (RRF) vs. weighted score
RRF has one fewer knob and is usually the right default.
BM25 index (in-memory)
`pip install rank_bm25`. For 800 chunks it's ~10 MB in memory; fine to
rebuild on startup.
Create `multipdf_chat/bm25.py`:
```python
"""In-memory BM25 index over child chunks. Rebuilt on FastAPI startup."""
import re
from dataclasses import dataclass, field
from typing import Optional
from rank_bm25 import BM25Okapi
from sqlalchemy import text
from sqlalchemy.orm import Session
_TOKEN_RE = re.compile(r"\w+")

def tokenize(s: str) -> list[str]:
    return _TOKEN_RE.findall(s.lower())

@dataclass
class BM25Index:
    bm25: BM25Okapi
    parent_ids: list[str] = field(default_factory=list)
    product_lines: list[list[str]] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)

def build_bm25(db: Session) -> BM25Index:
    rows = db.execute(text("""
        SELECT c.parent_id,
               c.content,
               d.product_line,
               d.status
        FROM child_chunks c
        JOIN documents d ON d.doc_id = c.doc_id
    """)).mappings().all()
    corpus = [tokenize(r["content"]) for r in rows]
    return BM25Index(
        bm25=BM25Okapi(corpus),
        parent_ids=[str(r["parent_id"]) for r in rows],
        product_lines=[list(r["product_line"]) for r in rows],
        statuses=[r["status"] for r in rows],
    )

def bm25_search(
    idx: BM25Index,
    question: str,
    product_line: Optional[str],
    k: int,
) -> list[str]:
    """Return parent_ids ranked by BM25 score, after filter + dedup."""
    scores = idx.bm25.get_scores(tokenize(question))
    # argsort descending
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    seen: set[str] = set()
    out: list[str] = []
    for i in ranked:
        if scores[i] <= 0:
            continue
        if idx.statuses[i] != "active":
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
```
Build it in the lifespan:
```python
main.py
from multipdf_chat.bm25 import build_bm25
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.db = SessionLocal
        app.state.async_db = AsyncSessionLocal
        app.state.embeddings = HuggingFaceEmbeddings(...)
        # Build BM25 once at startup (few seconds for ~800 chunks)
        with SessionLocal() as db:
            app.state.bm25 = build_bm25(db)
            logger.info("bm25_built", extra={
                "n_chunks": len(app.state.bm25.parent_ids),
            })
        yield
    finally:
        app.state.db = None
        app.state.async_db = None
        app.state.embeddings = None
        app.state.bm25 = None
```
Hybrid retrieval with RRF
Add to `multipdf_chat/retrieval.py`:
```python
from multipdf_chat.bm25 import BM25Index, bm25_search

def _rrf(rankings: list[list[str]], k_rrf: int = 60) -> list[str]:
    """Reciprocal Rank Fusion. Returns fused parent_ids in descending order."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, pid in enumerate(ranking, start=1):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k_rrf + rank)
    return [pid for pid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]

async def retrieve_hybrid_async(
    db: AsyncSession,
    embeddings,
    bm25_idx: BM25Index,
    question: str,
    product_line: Optional[str] = None,
    k: int = 10,
    fetch_multiplier: int = 3,
) -> list[RetrievedParent]:
    """
    Vector top-N + BM25 top-N, fused with RRF, then final top-k.
    We oversample by `fetch_multiplier` to give RRF room to reorder.
    """
    request_id = get_request_id()
    n = k * fetch_multiplier
    # Vector arm
    vector_parents = await retrieve_async(
        db, embeddings, question, product_line, k=n
    )
    vec_ranking = [p.parent_id for p in vector_parents]
    # BM25 arm (sync, but tiny — no need to thread it out)
    bm25_ranking = bm25_search(bm25_idx, question, product_line, k=n)
    # Fuse
    fused = _rrf([vec_ranking, bm25_ranking])[:k]
    # Materialise: reuse the parents we already fetched, look up the rest.
    by_id: dict[str, RetrievedParent] = {p.parent_id: p for p in vector_parents}
    missing = [pid for pid in fused if pid not in by_id]
    if missing:
        result = await db.execute(
            text("""
            SELECT p.id AS parent_id, d.slug AS doc_slug,
            d.title AS doc_title, d.source_url,
            d.authority_tier, p.section_path, p.content
            FROM parent_documents p
            JOIN documents d ON d.doc_id = p.doc_id
            WHERE p.id = ANY(:ids)
            AND d.status = 'active'
            """),
            {"ids": missing},
        )
        for r in result.mappings().all():
            by_id[str(r["parent_id"])] = RetrievedParent(
                parent_id=str(r["parent_id"]),
                doc_slug=r["doc_slug"],
                doc_title=r["doc_title"],
                section_path=r["section_path"],
                content=r["content"],
                distance=float("nan"), # BM25-only hit; no vector distance
                source_url=r["source_url"],
                authority_tier=r["authority_tier"],
            )
    ordered = [by_id[pid] for pid in fused if pid in by_id]
    logger.info("retrieval_fused", extra={
        "request_id": request_id,
        "vector_top1": vec_ranking[0] if vec_ranking else None,
        "bm25_top1": bm25_ranking[0] if bm25_ranking else None,
        "fused_top1_source": (
            "both" if vec_ranking and bm25_ranking and vec_ranking[0] == bm25_ranking[0]
            else "vector" if fused and fused[0] in vec_ranking[:3]
            else "bm25"
        ),
        "n_fused": len(ordered),
    })
    return ordered
```
Update `stream_user_input` to call it:
```python
parents = await retrieve_hybrid_async(
db, embeddings, request.app.state.bm25,
user_question, product_line, k=10,
)
```
Extend the harness to A/B test
Add a `--retrieval` flag to `eval/run_golden.py` so you can compare arms
without editing code:
```python
ap.add_argument("--retrieval", choices=["vector", "bm25", "hybrid"], default="hybrid")
then, replacing the retrieve() call:
if args.retrieval == "vector":
    retrieved = retrieve(db, embeddings, item["question"], item.get("product_line"), k=args.k)
elif args.retrieval == "bm25":
    from multipdf_chat.bm25 import build_bm25, bm25_search
    idx = getattr(main, "_bm25_cache", None) or build_bm25(db)
    main._bm25_cache = idx
    pids = bm25_search(idx, item["question"], item.get("product_line"), k=args.k)
    # fetch parents by id — copy the SQL from retrieve_hybrid_async
    ...
else: # hybrid
    ...
```
Run all three, save `baselines/step_06_{vector,bm25,hybrid}.json`. The
per-item diff between them is the most informative artifact of this step.
### Tasks

- `pip install rank_bm25`, add to `requirements.txt`.
- Create `multipdf_chat/bm25.py` (code above).
- Build the index in the FastAPI lifespan.
- Add `retrieve_hybrid_async` + `_rrf` to `multipdf_chat/retrieval.py`.
- Rewire `stream_user_input` to call it.
- Extend the harness with `--retrieval {vector,bm25,hybrid}`.
- Run all three:
```bash
for m in vector bm25 hybrid; do
python -m eval.run_golden --golden eval/golden_v1.jsonl \
--out baselines/step_06_$m.json --retrieval $m
done
```
Read the per-item diffs. Expect: gd-004 and gd-024 (which mention
`18%`, `1.5%`, section `11.1.12` — exact strings) benefit most from
Paraphrase items (gd-007, gd-008) should stay ~equal or improve
slightly with hybrid. Any regression is a finding worth investigating.
### Done when
Hybrid retrieval is default; vector-only and bm25-only are available
behind a flag.
Three baselines exist and are diffed in the commit message.
You can name at least 2 golden items BM25 fixed and (if any) 1 it broke.
________________________________________
## Step 7 — Cross-encoder reranking on the top-N
Duration: 3 days
Ships: reranker on the retrieval path, latency budget, `baselines/step_07_rerank.json`
### Concepts to learn first
Bi-encoder vs cross-encoder. Why cross-encoders are more accurate but
can't be pre-indexed, so they only run on the top-N candidates.
`ms-marco-MiniLM-L-6-v2` or similar. Small enough to run on CPU;
understand its latency envelope.
The latency budget. Rerank adds ~50–200ms depending on N. If your
first-token target is 500ms, you have to spend that budget deliberately.
Reranker module
`pip install sentence-transformers` (already a transitive dep via
`langchain-huggingface`, but pin it).
Create `multipdf_chat/rerank.py`:
```python
"""Cross-encoder reranking of retrieved candidates."""
import asyncio
import logging
import time
from typing import Optional
from sentence_transformers import CrossEncoder
from multipdf_chat.logging_context import get_request_id
from multipdf_chat.retrieval import RetrievedParent
logger = logging.getLogger("api")
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_MAX_LEN = 512 # tokens; longer inputs are truncated

def load_reranker() -> CrossEncoder:
    """Load once at startup. ~90 MB on disk, ~200 MB resident."""
    return CrossEncoder(_MODEL_NAME, max_length=_MAX_LEN)

def rerank(
    model: CrossEncoder,
    question: str,
    parents: list[RetrievedParent],
    top_n: int = 5,
) -> list[RetrievedParent]:
    if not parents:
        return parents
    request_id = get_request_id()
    t0 = time.perf_counter()
    # CrossEncoder tokenises internally; character truncation here is only
    # to keep the pair well under the model's token limit for long parents.
    pairs = [(question, p.content[:1800]) for p in parents]
    scores = model.predict(pairs, show_progress_bar=False)
    ranked = sorted(
        zip(parents, scores),
        key=lambda x: x[1],
        reverse=True,
    )
    top = [p for p, _ in ranked[:top_n]]
    logger.info("rerank_completed", extra={
        "request_id": request_id,
        "n_in": len(parents),
        "n_out": len(top),
        "duration_ms": round((time.perf_counter() - t0) * 1000),
        "top_slug_before": parents[0].doc_slug,
        "top_slug_after": top[0].doc_slug if top else None,
    })
    return top

async def rerank_async(model, question, parents, top_n=5) -> list[RetrievedParent]:
    """CrossEncoder is CPU-bound sync — push it off the event loop."""
    return await asyncio.to_thread(rerank, model, question, parents, top_n)
```
Load in the lifespan:
```python
from multipdf_chat.rerank import load_reranker
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # ... existing setup ...
        app.state.reranker = load_reranker()
        logger.info("reranker_loaded", extra={"model": "ms-marco-MiniLM-L-6-v2"})
        yield
    finally:
        # ...
        app.state.reranker = None
```
Wire into the pipeline
Update `stream_user_input`:
```python
from multipdf_chat.rerank import rerank_async
...
parents = await retrieve_hybrid_async(
    db, embeddings, request.app.state.bm25,
    user_question, product_line, k=20, # oversample for the reranker
)
if parents and request.app.state.reranker is not None:
    parents = await rerank_async(
        request.app.state.reranker, user_question, parents, top_n=5,
    )
decision = route(parents, product_line)
... unchanged ...
```
Extend the harness
Add a `--rerank` flag:
```python
ap.add_argument("--rerank", action="store_true")
Once, before the loop:
    reranker = load_reranker() if args.rerank else None
In the loop, after retrieval:
    if reranker:
        retrieved = rerank(reranker, item["question"], retrieved, top_n=5)
```
Latency budget check
Before/after `bench_stream_latency.py` runs — the reranker adds cost. Budget:
N candidates
Typical CPU add
5
~30 ms
10
~60 ms
20
~120 ms
40
~250 ms
If your first-token target is 500 ms and your current baseline is 350 ms,
you have ~150 ms of headroom — top_n from 20 candidates fits, from 40
doesn't.
### Tasks

- Create `multipdf_chat/rerank.py` (code above).
- Load in lifespan, plumb into `stream_user_input`.
- Baseline latency:
```bash
python -m eval.bench_stream_latency --n 20
```
Run the harness both ways:
```bash
python -m eval.run_golden --out baselines/step_07_no_rerank.json
python -m eval.run_golden --out baselines/step_07_rerank.json --rerank
```
Compare per-item. Rerank usually moves `mrr` and `section_hit@5` more
than `recall@10` — it reorders, not adds.
Rebench latency. Confirm you're within budget.
If reranking regresses on any item, look at the raw scores — often the
cross-encoder was confidently wrong on a paraphrase; that's a known
MS-MARCO training-domain limitation.
### Done when
Reranker is on by default with a `--rerank`/env flag to disable.
Latency measurement shows the added cost is within your budget.
Per-item diffs are recorded and understood — you can name what the
reranker helped and what it didn't.
________________________________________
## Step 8 — HNSW index and re-measure approximation cost
Duration: 2 days
Ships: HNSW index, exact-vs-approximate comparison
Only worth doing once the corpus grows past low tens of thousands of
children (per ReArchitecture.md §5). If you're still on the five-PDF corpus,
skip this and go to step 9. When you do this step, the point isn't "add
HNSW" — it's "measure what approximation costs against the frozen baseline."
### Concepts to learn first
HNSW vs IVFFlat. Why the doc rules IVFFlat out at this size, and why
HNSW is worth waiting for larger corpora.
`ef_search` at query time. The one knob that trades latency for
Understand how to sweep it.
The migration
Create `migrations/002_hnsw.sql`:
```sql
-- HNSW on child_chunks.embedding for cosine distance.
-- CONCURRENTLY so we don't block writes during creation. Note: this cannot
-- run inside a transaction block, so apply it with psql, not through your
-- normal migration runner if that wraps everything in BEGIN.
--
-- m = 16, ef_construction = 64 are pgvector defaults; increase if you want
-- higher recall at the cost of build time and memory.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_child_embedding_hnsw
ON child_chunks
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
-- After building, check size + stats:
-- SELECT pg_size_pretty(pg_relation_size('idx_child_embedding_hnsw'));
-- EXPLAIN (ANALYZE, BUFFERS) SELECT ... ORDER BY embedding <=> '[...]' LIMIT 10;
```
Apply:
```bash
psql "$DATABASE_URL" -f migrations/002_hnsw.sql
```
The sweep script
Create `eval/sweep_ef_search.py`:
```python
"""
For each ef_search value, run the retrieval-only path against the golden
set. Record recall@5, mrr, and p50/p95 retrieval latency.
The exact baseline (no index, sequential scan) is your ground truth. Any
recall gap between exact and a given ef_search value IS the approximation
cost.
Usage:
python -m eval.sweep_ef_search
"""
import json
import time
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from sqlalchemy import text
from multipdf_chat.db import SessionLocal
from multipdf_chat.retrieval import retrieve

EF_VALUES = [10, 40, 100, 200, 400]
GOLDEN = "eval/golden_v1.jsonl"
OUT = "baselines/step_08_hnsw_sweep.json"

def _load_golden():
    return [
        json.loads(l)
        for l in Path(GOLDEN).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

def _run(db, embeddings, items, k=5):
    latencies = []
    hits = 0
    scored = 0
    mrr_sum = 0.0
    for item in items:
        gold = set(item.get("gold_doc_ids") or [])
        if not gold:
            continue # skip insufficient-evidence items
        t0 = time.perf_counter()
        parents = retrieve(
            db, embeddings, item["question"], item.get("product_line"), k=k
        )
        latencies.append((time.perf_counter() - t0) * 1000)
        docs = [p.doc_slug for p in parents]
        if gold & set(docs):
            hits += 1
        for rank, d in enumerate(docs, start=1):
            if d in gold:
                mrr_sum += 1.0 / rank
                break
        scored += 1
    latencies.sort()
    return {
        "n_scored": scored,
        "recall@5": round(hits / scored, 3) if scored else None,
        "mrr": round(mrr_sum / scored, 3) if scored else None,
        "p50_ms": round(latencies[len(latencies) // 2]),
        "p95_ms": round(latencies[int(len(latencies) * 0.95)]),
    }

def main():
    items = _load_golden()
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    results = []
    # Exact baseline: force a sequential scan for ground truth.
    with SessionLocal() as db:
        db.execute(text("SET enable_indexscan = off"))
        db.execute(text("SET enable_bitmapscan = off"))
        results.append({"mode": "exact", _run(db, embeddings, items)})
    # HNSW at each ef_search.
    for ef in EF_VALUES:
        with SessionLocal() as db:
            db.execute(text(f"SET hnsw.ef_search = {ef}"))
            results.append({"mode": f"hnsw_ef={ef}", _run(db, embeddings, items)})
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text(json.dumps(results, indent=2))
    print(f"{'mode':>15} {'recall@5':>10} {'mrr':>7} {'p50':>6} {'p95':>6}")
    for r in results:
        print(f"{r['mode']:>15} {r['recall@5']:>10} {r['mrr']:>7} "
              f"{r['p50_ms']:>4} ms {r['p95_ms']:>4} ms")

if __name__ == "__main__":
    main()
```
Making the operating point permanent
Once you've picked (say) `ef_search = 100`, set it per-session at connection
open time. In `multipdf_chat/db.py`:
```python
from sqlalchemy import event
HNSW_EF_SEARCH = 100
@event.listens_for(async_engine.sync_engine, "connect")
def _set_hnsw_ef_search(dbapi_conn, connection_record):
    cur = dbapi_conn.cursor()
    cur.execute(f"SET hnsw.ef_search = {HNSW_EF_SEARCH}")
    cur.close()
```
(The `sync_engine` attribute exists on the async engine — this is the
correct hook target for asyncpg connections too.)
### Tasks

- Apply `migrations/002_hnsw.sql`. Verify with
  `\d child_chunks` in psql.
- Run the sweep. Read the output.
- Pick the smallest `ef_search` where `recall@5` is within ~2% of exact,
  and where p95 is meaningfully lower than exact. Often 100.
- Wire the connect-event hook so every session gets it automatically.
- Re-run the full golden set through the harness (with reranker + hybrid).
- Record `baselines/step_08_hnsw.json`. Should match step 7 within noise
  on retrieval scores, with lower latency.
- Done when
Sweep JSON is committed to the repo.
Chosen `ef_search` is applied on every DB connection.
Query latency at chosen `ef_search` is < exact scan latency for the
current corpus size, and recall gap is within 2 points.
________________________________________
## Step 9 — LLM-as-judge for faithfulness and answer quality
Duration: 3 days
Ships: judge harness, faithfulness/relevance scores per item
String checks (`must_include`) are cheap and precise but blind to
paraphrase. Add a judge to catch answers that are technically correct but
unsupported by the retrieved context.
### Concepts to learn first
Faithfulness vs relevance vs answer-correctness. Three separate
dimensions; a judge should score each independently.
Judge model choice. Use a different model than the generation
model when possible, to avoid the "grading its own homework" bias.
Judge noise. Run each judgment 3× and take the median. Otherwise
scores wobble ±5% between runs and you'll chase phantoms.
Judge module
Create `eval/judge.py`:
```python
"""LLM-as-judge for faithfulness, relevance, completeness."""
import json
import os
import re
import statistics
from typing import Optional
from langchain_groq import ChatGroq

Deliberately a different model than generation (which uses gpt-oss-20b via
Groq). Using the same model to grade itself inflates scores.
JUDGE_MODEL = "llama-3.3-70b-versatile"
JUDGE_PROMPT = """You are grading an answer produced by a retrieval-augmented Q&A system \
for a policy document corpus.
Question:
{question}
Retrieved context (all snippets the system had access to):
________________________________________
{context}
________________________________________
Answer produced by the system:
{answer}
Score each dimension on a 0.0 to 1.0 scale (one decimal place):
faithfulness: is every factual claim in the answer supported by the
retrieved context? 1.0 = every claim is directly supported.
0.0 = the answer invents facts not in the context.
relevance: does the answer address the question that was asked?
1.0 = directly and specifically answers the question.
0.0 = the answer is off-topic or evades the question.
completeness: does the answer include the information a user needs?
1.0 = fully complete for the question. 0.0 = missing critical info.
Return ONLY a JSON object, no preamble:
{{"faithfulness": 0.X, "relevance": 0.X, "completeness": 0.X, "notes": "one sentence"}}
"""

_JSON_RE = re.compile(r"{[^{}]}", re.DOTALL)

def _extract_json(raw: str) -> Optional[dict]:
    """Grab the first JSON object in the response. LLMs sometimes wrap it."""
    m = _JSON_RE.search(raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None

def _one_run(model, question: str, context: str, answer: str) -> Optional[dict]:
    prompt = JUDGE_PROMPT.format(
        question=question,
        context=context[:8000], # cap to keep judge cost sane
        answer=answer,
    )
    raw = model.invoke(prompt).content
    return _extract_json(raw)

def judge_answer(
    question: str,
    context: str,
    answer: str,
    n_runs: int = 3,
) -> dict:
    """
    Runs the judge n_runs times, takes the per-dimension median. Judges are
    noisy at temperature 0 (yes really) — median across runs beats any
    single call. Returns Nones on total failure so scoring code can skip.
    """
    if not answer.strip():
        return {"faithfulness": None, "relevance": None,
                "completeness": None, "runs": 0}
    model = ChatGroq(
        model=JUDGE_MODEL,
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0,
    )
    runs = []
    for _ in range(n_runs):
        parsed = _one_run(model, question, context, answer)
        if parsed and all(k in parsed for k in ("faithfulness", "relevance", "completeness")):
            runs.append(parsed)
    if not runs:
        return {"faithfulness": None, "relevance": None,
                "completeness": None, "runs": 0}
    def _med(k):
        vals = [float(r[k]) for r in runs if isinstance(r.get(k), (int, float))]
        return round(statistics.median(vals), 2) if vals else None
    return {
        "faithfulness": _med("faithfulness"),
        "relevance": _med("relevance"),
        "completeness": _med("completeness"),
        "runs": len(runs),
        "notes": [r.get("notes", "") for r in runs],
    }
```
Wire into the harness
Update `eval/run_golden.py`:
```python
from eval.judge import judge_answer
argparse:
ap.add_argument("--judge", action="store_true",
help="Run LLM-as-judge on each answer (slow, uses Groq API)")
ap.add_argument("--judge-runs", type=int, default=3)
In the loop, after generate_answer:
if args.judge and answer:
    context = "
".join(
        f"[{r.doc_slug} | {r.section_path or 'no-section'}]
{r.content}"
        for r in retrieved[:5]
    )
    judge_scores = judge_answer(
        item["question"], context, answer, n_runs=args.judge_runs,
    )
else:
    judge_scores = {}
In the per-item result:
results.append({
    # ... existing fields ...
    "judge": judge_scores,
})
```
Extend `summarise()`:
```python
def summarise(results):
    # ... existing means ...
    def _judge_mean(key):
        vals = [r["judge"].get(key) for r in results
                if r.get("judge") and r["judge"].get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None
    return {
        # ... existing keys ...
        "faithfulness_mean": _judge_mean("faithfulness"),
        "relevance_mean": _judge_mean("relevance"),
        "completeness_mean": _judge_mean("completeness"),
    }
```
Cross-check script
Find items where string checks and the judge disagree — those are the
interesting ones:
```python
eval/find_disagreements.py
"""
Load a baselines/.json produced with --judge and print items where the
string checks say pass but faithfulness < 0.7 (lucky-string-match), or
string checks say fail but faithfulness > 0.8 (paraphrase we missed).
"""
import argparse
import json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    args = ap.parse_args()
    data = json.loads(Path(args.baseline).read_text())
    for r in data["per_item"]:
        gen = r.get("generation") or {}
        judge = r.get("judge") or {}
        faith = judge.get("faithfulness")
        if faith is None:
            continue
        strong_string_pass = (
            gen.get("must_include_pass") and gen.get("must_not_include_pass")
        )
        if strong_string_pass and faith < 0.7:
            print(f"[LUCKY-MATCH] {r['id']}: string pass, faithfulness={faith}")
            print(f" Q: {r['question']}")
            print(f" A: {r['answer'][:200]}")
            print()
        if not strong_string_pass and faith > 0.8:
            print(f"[MISSED-PARAPHRASE] {r['id']}: string fail, faithfulness={faith}")
            print(f" Q: {r['question']}")
            print(f" A: {r['answer'][:200]}")
            print()

if __name__ == "__main__":
    main()
```
### Tasks

- Create `eval/judge.py`.
- Extend `eval/run_golden.py` with `--judge` / `--judge-runs`.
- Run with the judge:
```bash
python -m eval.run_golden \
--golden eval/golden_v1.jsonl \
--out baselines/step_09_judged.json \
--judge --judge-runs 3
```
Expect this run to be ~10× slower than un-judged. Only run judged
evaluations at end-of-step, not during iteration.
Run `python -m eval.find_disagreements --baseline baselines/step_09_judged.json`.
Investigate at least one of each disagreement type. Add a note to the
golden item explaining what the true failure mode was.
If a golden item's string checks were wrong (e.g., `must_include: ["$9"]`
but the correct paraphrase is "nine dollars"), fix the item and re-run.
### Done when
Golden set produces both string-check and judge scores.
You have at least one investigated case where string and judge scores
disagreed and you understood why.
Golden items whose string checks were overly strict are relaxed.
________________________________________
## What's deliberately not here
Query rewriting / HyDE. Adds a whole LLM call per query. Come back to
this only if the golden set says specific question phrasings are failing.
Multi-hop / agentic retrieval. Wrong shape for this corpus (policy
lookup, not multi-doc reasoning).
Fine-tuning the embedding model. Enormous cost, tiny expected win at
this scale. Revisit at 10x the corpus.
Everything above earns its place by moving a number on the golden set.
Nothing that doesn't should ship.