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

- pgvector operators. `<=>` (cosine distance), `<#>` (negative inner
  product), `<->` (L2). Which opclass each needs.
- Postgres array overlap `&&`. Why it's the right predicate here instead
  of `= ANY`.
- `GROUP BY` + `MIN` for parent aggregation. Why "top 10 children" is not
  "top 10 parents", and why sending duplicates to the LLM burns context.

### Tasks

- Add `product_line: Optional[str]` to `UserQuery`
  ([models/userQuery.py](multipdf_chat/models/userQuery.py)) and thread it
  into `stream_user_input` and the non-stream path.
- Replace the SQL in `stream_user_input` with the CTE from ReArchitecture.md
  §3 (filter → group → limit → join). Same for `user_input` — or delete
  `user_input` if you're moving fully to the streaming path.
- Delete the `metadata->>'session_id'` filter in `user_input`. It's a FAISS
  leftover; `ingest_session_id` is provenance, not a query predicate
  (ReArchitecture.md §3, "No session filter").
- Re-run the golden set. Record `baselines/step_02_arch_aligned.json`.
- Diff against step 1 in the commit message.

### Done when

- Retrieval SQL matches the doc, verbatim.
- Golden set numbers are recorded and diffed.
- You can explain why `GROUP BY c.parent_id` exists.

________________________________________

## Step 3 — Structured logs around retrieval, without breaking streaming

Duration: 2 days

Ships: structured log lines at each retrieval stage, no regression in streaming latency

### Concepts to learn first

- Why logging inside a token loop is a footgun. Even a formatted
  `logger.info` per token adds meaningful latency at 100+ tokens/sec, and
  buffering / flushing behaviour can serialise tokens that were supposed to
  interleave with the network write.
- `extra=` vs f-strings for structured logs. The `pythonjsonlogger`
  formatter you already use in [main.py:47](multipdf_chat/main.py#L47) reads
  `extra` fields — putting values in `extra` makes them queryable; putting
  them in the message string does not.
- Request-scoped correlation IDs. You already have `X-Request-ID`
  middleware in [main.py:61](multipdf_chat/main.py#L61). Understand how to
  propagate it into deeper functions (contextvars is the clean way).

### Tasks

- Add log lines at each retrieval boundary — `retrieval_started`,
  `retrieval_completed` (with `k`, `n_parents`, `min_distance`, `duration_ms`,
  `product_line`), `generation_started`, `generation_completed`. All in
  `extra=`, none inside the token loop.
- Propagate `request_id` from the middleware into `stream_user_input` via a
  `contextvars.ContextVar` or an explicit function argument.
- Measure streaming latency before and after with a tiny script that
  times first-token and last-token from `/chat/stream`. Confirm no

- Re-run the golden set (should be a no-op on scores — this is a
  correctness/observability change, not a quality change). Record anyway;
  it proves the harness is stable across no-op edits.

### Done when

- Every retrieval invocation produces a queryable JSON log line with
  distance, chunk count, and duration.
- First-token latency is within 5% of the pre-change measurement.

________________________________________

## Step 4 — Fix async: stop blocking the event loop

Duration: 2–3 days

Ships: async DB path, async embedding path, latency measurements

### Concepts to learn first

- What "blocking the event loop" actually means. A sync `db.execute()`
  inside `async def` stalls every other request on that worker until it
  For a single-user local demo this is invisible; for concurrency
  benchmarks it's catastrophic.
- `asyncpg` vs `psycopg` async vs SQLAlchemy 2.x async. Understand the
  three layers and pick one deliberately.
- `asyncio.to_thread` as an escape hatch. For CPU-bound work
  (embeddings on CPU), `to_thread` is often the right answer rather than
  rewriting the library. Know when each applies.

### Tasks

- Convert `stream_user_input`'s DB calls to async — either SQLAlchemy async
  engine (`create_async_engine`) or drop to `asyncpg`. Start with the
  streaming path only; leave ingestion sync.
- `embeddings.embed_query()` is sync CPU work — wrap in
  `asyncio.to_thread(...)` rather than pretending it's async.
- Benchmark: 20 concurrent `/chat/stream` requests before and after. Report
  p50 and p95 first-token latency.
- Re-run golden set. Record `baselines/step_04_async.json`. Should be
  no-change on scores; it's a concurrency fix, not a quality fix.

### Done when

- No `def db.execute` runs inside `async def` on the streaming path.
- Concurrency benchmark shows measurable improvement (or you've documented
  why it didn't — often it's the LLM API that's the bottleneck, not the DB).

________________________________________

## Step 5 — Insufficient-evidence and conflicting-sources routing

Duration: 2 days

Ships: two new response modes, golden-set items for both

Right now the pipeline always answers. The doc (§3, "evidence check") calls
for `insufficient_evidence` and `conflicting_sources` outcomes. These are
worth doing before reranking because they change what "correct" means: an
"I don't know" that routes to a human is a correct outcome for gd-006, not
a failure.

### Concepts to learn first

- Distance-threshold routing. Why "top result distance > 0.4" is a
  useful signal for insufficient evidence, and why the threshold is
  corpus-specific (must be measured, not guessed).
- Multi-source disagreement. How to detect "the top 3 parents come from
  different `doc_id`s with overlapping `product_line` but differ on a key
  Simple heuristics beat LLM judges here.

### Tasks

- Add a distance-threshold check after retrieval. If `min_distance >
  threshold` (calibrated from the golden set), return
  `{"status": "insufficient_evidence", ...}` without calling the LLM.
- Add golden items whose correct answer is `insufficient_evidence` or a
- Update the harness to score these.
- For conflicting sources: detect at prompt-assembly time by comparing
  figures across the top-k parents. Simple string-diff of numbers is
  enough for v1.
- Record `baselines/step_05_evidence.json`.

### Done when

- gd-006 (no product context) returns a refusal, not a merged answer.
- Golden set has at least 3 insufficient-evidence items and 1 conflict
  item, all scoring correctly.

________________________________________

## Step 6 — Hybrid retrieval: BM25 alongside vectors

Duration: 3 days

Ships: BM25 index, hybrid scoring, `baselines/step_06_hybrid.json`

### Concepts to learn first

- Why BM25 catches what dense retrieval misses. Exact identifiers
  ("section 11.1.12", "$9", model numbers) are where BM25 wins. Synonyms
  and paraphrase are where dense wins. Hybrid is not "better on average" —
  it's "better on the tail cases dense fails on."
- Postgres `tsvector` + GIN, or a separate `rank_bm25` in Python.
  Trade-offs: `tsvector` is in-DB (one query), `rank_bm25` is flexible but
  needs the whole corpus in memory. For 800 chunks, in-memory is fine.
- Score fusion. Reciprocal Rank Fusion (RRF) vs. weighted score
  RRF has one fewer knob and is usually the right default.

### Tasks

- Add BM25 over `child_chunks.content` — either a `tsvector` column with a
  GIN index, or `rank_bm25.BM25Okapi` built lazily on startup.
- Wire BM25 into retrieval: pull top-k from each, combine with RRF
  (`score = 1 / (60 + rank_bm25) + 1 / (60 + rank_vec)`).
- Re-run golden set. Compare per-item to step 5 — which items moved?
  Which regressed? Regressions are the interesting cases.
- Record `baselines/step_06_hybrid.json`.

### Done when

- Hybrid scoring is on by default with a flag to disable.
- You can name at least 2 golden items BM25 fixed and (if any) 1 it broke.

________________________________________

## Step 7 — Cross-encoder reranking on the top-N

Duration: 3 days

Ships: reranker on the retrieval path, latency budget, `baselines/step_07_rerank.json`

### Concepts to learn first

- Bi-encoder vs cross-encoder. Why cross-encoders are more accurate but
  can't be pre-indexed, so they only run on the top-N candidates.
- `ms-marco-MiniLM-L-6-v2` or similar. Small enough to run on CPU;
  understand its latency envelope.
- The latency budget. Rerank adds ~50–200ms depending on N. If your
  first-token target is 500ms, you have to spend that budget deliberately.

### Tasks

- Add a `sentence-transformers` cross-encoder. Rerank the top-20 from
  step 6 down to top-5.
- Log the reranker's `duration_ms` (see step 3 — reuse the log schema).
- Re-run golden set. Compare — reranking usually helps `nDCG` more than
  `recall@k`, because it reorders rather than adds.
- Record `baselines/step_07_rerank.json`.

### Done when

- Reranker is on by default with a disable flag.
- Latency measurement shows the added cost is within your budget.

________________________________________

## Step 8 — HNSW index and re-measure approximation cost

Duration: 2 days

Ships: HNSW index, exact-vs-approximate comparison

Only worth doing once the corpus grows past low tens of thousands of
children (per ReArchitecture.md §5). If you're still on the five-PDF corpus,
skip this and go to step 9. When you do this step, the point isn't "add
HNSW" — it's "measure what approximation costs against the frozen baseline."

### Concepts to learn first

- HNSW vs IVFFlat. Why the doc rules IVFFlat out at this size, and why
  HNSW is worth waiting for larger corpora.
- `ef_search` at query time. The one knob that trades latency for
  Understand how to sweep it.

### Tasks

- `CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)`.
- Sweep `ef_search` at values 10, 40, 100, 200. For each, run the golden
- Plot recall vs. latency.
- Pick the point that keeps recall within (say) 2% of exact and commit
  that as the operating point.

### Done when

- You have a recall/latency curve committed to the repo.
- Query latency at chosen `ef_search` is < exact scan latency for the
  current corpus size.

________________________________________

## Step 9 — LLM-as-judge for faithfulness and answer quality

Duration: 3 days

Ships: judge harness, faithfulness/relevance scores per item

String checks (`must_include`) are cheap and precise but blind to
paraphrase. Add a judge to catch answers that are technically correct but
unsupported by the retrieved context.

### Concepts to learn first

- Faithfulness vs relevance vs answer-correctness. Three separate
  dimensions; a judge should score each independently.
- Judge model choice. Use a different model than the generation
  model when possible, to avoid the "grading its own homework" bias.
- Judge noise. Run each judgment 3× and take the median. Otherwise
  scores wobble ±5% between runs and you'll chase phantoms.

### Tasks

- Add a judge step to `eval/run_golden.py`. For each generated answer:
  score `faithfulness` (0-1), `relevance` (0-1), `completeness` (0-1)
  against the retrieved context.
- Cross-check: any item where `must_include` passes but `faithfulness <
  0.7` is a hallucination-with-lucky-string-match. Investigate.
- Record `baselines/step_09_judged.json`.

### Done when

- Golden set produces both string-check and judge scores.
- You have at least one investigated case where the two disagreed and you
  understood why.

________________________________________

## What's deliberately not here

- Query rewriting / HyDE. Adds a whole LLM call per query. Come back to
  this only if the golden set says specific question phrasings are failing.
- Multi-hop / agentic retrieval. Wrong shape for this corpus (policy
  lookup, not multi-doc reasoning).
- Fine-tuning the embedding model. Enormous cost, tiny expected win at
  this scale. Revisit at 10x the corpus.

Everything above earns its place by moving a number on the golden set.
Nothing that doesn't should ship.
