# AI Systems Engineer Notes - RAG, Vector DB

## Where RAG Actually Fails

```
[ User Query ]
    ↓    
[ Query Encoder ]
    ↓
[ Vector Index ]   ⚠ stale if pipeline broken
    ↓
[ Access Filter ]  ⚠ skip this → data leaks across tenants
    ↓
[ Re-ranker ]      ⚠ missing → low-relevance docs pass through
    ↓
[ LLM Prompt ]     ⚠ bad chunks → hallucination / refusal
    ↓
[ Response ]
```

## AI Systems Engineer Role

- **System spec design** - inputs, outputs, fallbacks, uncertainty
- **Retrieval as a system** - chunking, freshness, access control
- **Agent orchestration** - state, retries, idempotency, contracts
- **Evaluation** - golden sets, graders, regression suites
- **Observability** - tracing, drift, cost attribution
- **Safety** - prompt injection, data leakage, policy enforcement

## Project: Permission-Aware RAG Service

**Goal**: Multi-tenant RAG with access control enforced at query time

**Components**:
- Ingestion + indexing pipeline (insert metadata for access role)
- Query time authorization
- Evaluation harness + regression rules

## What You Need to Build

- **Python for AI systems** - debug, profile, tooling
- **LLM APIs** - structured outputs, schemas, tool calls, errors
- **RAG fundamentals** - freshness, access control, relevance
- **Vector DBs** - hybrid search when embeddings aren't enough
- **Evaluation frameworks** - offline + online, acceptance criteria
- **Agent patterns** - ReAct, planner/executor (know when NOT to use)
- **Reliability patterns** - timeouts, retries, circuit breakers
- **Deployment + cost control** - batching, budgets, model selection

## AI Systems Interviews: Four Axes

### What Actually Gets Tested

1. **System design** - architecture, boundaries, trust zones
2. **Tradeoffs** - quality vs latency vs cost vs risk
3. **Debugging retrieval** - tool prompt + model data
4. **Measurement** - prove it works and stays working

### Sample Question

**"Should you cache LLM responses in a RAG system?"**

**Strong answer** covers:
- Query distribution analysis
- Hit rate analysis
- Semantic caching thresholds
- Layer-specific TTLs
- Freshness constraints

**Weak answer**: "It depends, sometimes it helps."


## Production RAG Architecture

What ships reliably requires seven layers of intentional engineering:

### Layer 1 - Chunking Strategy
| Level | Approach |
|-------|----------|
| **Tutorial** | Character split, fixed size |
| **Production** | Semantic chunking with document-aware boundaries; metadata tagging per chunk (doc_id, section) |

### Layer 2 - Query Understanding
| Level | Approach |
|-------|----------|
| **Tutorial** | Raw query → embedding |
| **Production** | Intent classification → query rewrite/expansion → routing (single vs multi-corpus) before embedding |

### Layer 3 - Hybrid Retrieval
| Level | Approach |
|-------|----------|
| **Tutorial** | Dense vector search only (cosine similarity) |
| **Production** | Dense (semantic) + sparse (BM25/keyword) in parallel; Reciprocal Rank Fusion (RRF) to merge results |

### Layer 4 - Reranking
| Level | Approach |
|-------|----------|
| **Tutorial** | Cosine distance IS the final ranking signal |
| **Production** | Cross-encoder reranker over top-20 candidates; relevance ≠ embedding proximity |

### Layer 5 - Context Filtering & Fallback
| Level | Approach |
|-------|----------|
| **Tutorial** | Always pass top-k to LLM regardless of results |
| **Production** | Minimum relevance threshold gate; explicit fallback when retrieval confidence is low |

### Layer 6 - Retrieval Evaluation
| Level | Approach |
|-------|----------|
| **Tutorial** | Evaluate answer quality (did the answer match?) |
| **Production** | Evaluate retrieval quality separately: precision@1, recall@k, MRR on held-out queries |

### Layer 7 - Monitoring & Observability
| Level | Approach |
|-------|----------|
| **Tutorial** | None |
| **Production** | Log every retrieval (query, chunks, scores, latency); alert on precision degradation and corpus drift |

---

## Failure Modes by Layer

| Missing Layer | Consequence | Example |
|---------------|-------------|---------|
| **1** | Relevance ceiling set at ingestion; cannot improve downstream | Chunk a 200-page manual into 5KB sentence-level chunks → lose cross-topic semantic relationships forever |
| **2** | Informal queries degrade retrieval; coverage drops unpredictably | User asks "how do I fix my thing?" → not expanded to synonyms → returns nothing, even though docs exist |
| **3** | Keyword queries miss documents in corpus; low recall | Search for "vehicle" with dense embeddings only → miss docs talking about "car," "automobile," "transport" |
| **4** | Wrong chunk at rank 1 served as context; hallucination | Query "best practices for security" → top result mentions "security" in passing → LLM generates incorrect advice |
| **5** | LLM hallucinates into retrieval failures; no graceful degradation | No relevant chunks found for query → LLM makes up an answer instead of saying "information not available" |
| **6** | You find out from users, not metrics; reactive vs proactive | Retrieval quality degrades 15% → no alerts → customer reports "answers got worse" in support ticket |
| **7** | Corpus drift and query shift are invisible; silent degradation | 1000 new docs added with different terminology → retrieval scores stay same but precision drops 20% → undetected |

# Vector DB Basics + Cost/Latency/Quality Tradeoffs
> Fills the gap: your other notes cover *when* to pick pgvector vs Pinecone, not *how they work* internally. This is the missing 30 min + the 15 min tradeoff synthesis.

---

# PART A — Vector DB Fundamentals

## 1. What a vector DB actually stores
A row = `[vector (N floats), metadata (JSON), id]`. The "database" part is really an **index structure over the vectors** that makes similarity search sub-linear (avoid comparing the query to all N vectors).

## 2. Distance metrics (pick one, be consistent)
| Metric | Formula intuition | When |
|---|---|---|
| **Cosine similarity** | angle between vectors, ignores magnitude | Default for text embeddings (OpenAI/Cohere models are trained for this) |
| **Dot product** | cosine × magnitude | Use if embeddings are already normalized (faster, same ranking as cosine) |
| **Euclidean (L2)** | straight-line distance | Image embeddings, some clustering use cases |

> Interview trap: "Which distance metric?" → answer must match what the embedding model was trained/optimized for. Mismatched metric silently degrades retrieval quality — no error, just worse ranking.

## 3. ANN index types (the core mechanism)
Exact nearest-neighbor search is O(n) — too slow at scale. Approximate Nearest Neighbor (ANN) trades a small accuracy loss for massive speed gains.

| Index | How it works | Tradeoff |
|---|---|---|
| **Flat / brute-force** | Compare query to every vector | 100% recall, slow, fine for <10k vectors |
| **IVFFlat** (Inverted File) | Cluster vectors into N buckets (k-means); search only nearest buckets | Fast, but recall depends on `nprobe` (buckets searched); needs a training step |
| **HNSW** (Hierarchical Navigable Small World) | Multi-layer graph — skip-list-like traversal to nearest neighbors | Best recall/speed tradeoff at scale; higher memory + slower index *build*; the current default for most production vector DBs |
| **IVF-PQ** | IVF + Product Quantization (compress vectors) | Lower memory footprint (10-100x), some recall loss — for billions of vectors |

**Key knobs:**
- `nprobe` / `ef_search` — how many candidates to check at query time. Higher = better recall, higher latency.
- Index build time vs query time — HNSW is slower to build/update than IVFFlat, which matters for high-write workloads.

## 4. pgvector — mechanics
Postgres extension; vectors live as a column type in an ordinary table.

```sql
CREATE EXTENSION vector;

CREATE TABLE chunks (
    id bigserial PRIMARY KEY,
    content text,
    tenant_id text,
    embedding vector(1536)
);

-- HNSW index (pgvector >= 0.5.0)
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- or IVFFlat
CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Query with metadata filter (native SQL WHERE — this is pgvector's big advantage)
SELECT content, 1 - (embedding <=> query_vec) AS similarity
FROM chunks
WHERE tenant_id = 'acme'
ORDER BY embedding <=> query_vec
LIMIT 4;
```

- `<=>` = cosine distance operator, `<->` = L2, `<#>` = negative inner product
- **Strength:** metadata filtering is just SQL — joins, row-level security, transactions, all native. One less service to run.
- **Weakness:** scales worse than purpose-built vector DBs past tens of millions of vectors; you're bound by Postgres's read replicas/sharding story.

## 5. Pinecone — mechanics
Managed, purpose-built vector DB. No SQL — everything is API calls.

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="...")
pc.create_index(
    name="handbook",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),  # serverless = pay per use
)

index = pc.Index("handbook")
index.upsert(vectors=[("id1", [0.1, 0.2, ...], {"tenant_id": "acme"})], namespace="acme")

index.query(
    vector=query_vec, top_k=4,
    filter={"tenant_id": {"$eq": "acme"}},
    namespace="acme",
)
```

- **Namespaces** — logical partitions within an index (common multi-tenancy pattern: 1 namespace per tenant, cheaper than 1 index per tenant).
- **Serverless vs Pod-based** — serverless auto-scales, pay-per-query/storage (good default in 2026); pods = fixed capacity, predictable cost at very high steady QPS.
- **Strength:** purpose-built ANN performance at huge scale (billions of vectors), managed ops, hybrid search built-in (sparse-dense).
- **Weakness:** yet another service/vendor, cost scales with usage, metadata filters are less expressive than SQL.

## 6. Quick comparison table

| | pgvector | Pinecone | Chroma | MongoDB Atlas Vector |
|---|---|---|---|---|
| **Best for** | Already on Postgres; need SQL joins/RLS | High QPS, huge scale, managed | Local dev/prototyping | Already on MongoDB |
| **Index types** | IVFFlat, HNSW | HNSW (managed) | HNSW | HNSW |
| **Metadata filter** | Native SQL | API filter syntax | API filter | Native Mongo query |
| **Ops burden** | Low (one DB) | Zero (managed) | Low (embedded/local) | Low (one DB) |
| **Scale ceiling** | 10s of millions | Billions | Dev-scale only | 10s of millions |
| **Multi-tenancy** | Row-level (`WHERE tenant_id=`) | Namespaces | Collections | Row-level |

> **Rule of thumb (from your other notes' Q9):** start with whatever DB you already run (Postgres → pgvector, Mongo → Atlas). Migrate to Pinecone/Weaviate only when you have *data* proving you've hit throughput/scale limits — don't pre-optimize.

---

# PART B — Cost / Latency / Quality Tradeoffs (the 3 you need)

The three axes are in constant tension — pulling one lever almost always costs you on another. Structure your answer around **where in the pipeline** the tradeoff lives:

## 1. Retrieval depth (top-k)
| Lever | Cost ↑↓ | Latency ↑↓ | Quality ↑↓ |
|---|---|---|---|
| Increase k (more chunks retrieved) | ↑ (more tokens to LLM) | ↑ (more to embed/rerank/prompt) | ↑ recall, but risk of "lost in the middle" hurting precision past a point |
| Decrease k | ↓ | ↓ | ↓ recall — may miss the answer entirely |

**Answer pattern:** "I don't maximize recall blindly — I tune k against a golden set, since past ~6-8 chunks, generation quality often *drops* due to lost-in-the-middle even though recall keeps climbing."

## 2. Reranking
| Lever | Cost/Latency | Quality |
|---|---|---|
| Add cross-encoder reranker | +100-300ms, +compute cost per query | Biggest single quality jump after hybrid search (per `Prod-RAG-2.md` benchmarks) |
| Skip reranker | Faster, cheaper | Fine for low-stakes queries; risky for high-stakes (legal/financial/policy) |

**Answer pattern:** tiered — skip reranking for casual/FAQ queries, apply it for anything touching money, legal terms, or safety.

## 3. Model tier selection
| Lever | Cost | Latency | Quality |
|---|---|---|---|
| Frontier model (GPT-4o, Claude Sonnet) | High $/token | Higher | Best reasoning/faithfulness |
| Mid-tier (mini/Haiku) | Low | Lower | Fine for straightforward extraction/summarization |
| Route by complexity | Optimized | Optimized | Requires an upfront classifier — adds a small latency/cost tax to save more downstream |

## 4. Caching
| Lever | Cost | Latency | Quality risk |
|---|---|---|---|
| Semantic cache (embed + threshold match) | Big $ savings on repeat/similar queries | Near-zero latency on hit | Stale answers if underlying docs changed — needs TTL tied to source freshness |
| No cache | Full cost every time | Full latency every time | Always fresh |

## 5. Embedding model choice
| Lever | Cost | Latency | Quality |
|---|---|---|---|
| Larger embedding model (e.g. `text-embedding-3-large`) | Higher $, more storage (more dims) | Slightly slower search | Better semantic separation — matters most for domain-heavy text (legal/clinical/financial) |
| Smaller model | Cheaper, faster | Faster | Fine for general English; weak on jargon-heavy domains (see the 0.87-cosine-but-40%-wrong example in `Prod-RAG-2.md`) |

## 6. The one-liner synthesis (use this to open your answer)
> "Every RAG lever — k, reranking, model tier, caching, embedding size — is a cost/latency/quality dial, not a free win. The job of a systems engineer is to tune each dial against a **measured golden set per use case**, not by intuition, because the 'right' setting for a low-stakes FAQ bot is wrong for a policy/compliance assistant."

---

# Where to focus your 90 minutes

| Time | Task | Primary source(s) |
|---|---|---|
| **45 min — RAG pipeline E2E** | `Prod-RAG.md` (7-layer table + failure modes) as the skeleton → `Prod-RAG-2.md` Part B (failure chain, "73%/80%" stat) for the *why it breaks* narrative → `Eval-RAG-failure.md` as your one fully worked concrete example to walk through if asked "tell me about a RAG failure you've seen" |
| **30 min — Vector DB basics** | **This new file**, Parts 1–6 — focus especially on distance metrics + HNSW vs IVFFlat (mechanics, not just names) and the pgvector-vs-Pinecone comparison table, since that's the part your existing notes only touched at the decision level |
| **15 min — 3 tradeoffs** | **This new file, Part B** — memorize the one-liner synthesis and pick 3 concrete levers (I'd go: **k/retrieval depth**, **reranking**, **model tier**) since those map cleanly to cost/latency/quality and you already have supporting detail in `AIEngineeringLifeCycle.md` Q5 and Q11 to back them up with real answers |

Your differentiator angle, per the "not happy-path tutorials" goal: lead with the **Eval-RAG-failure.md case study** (faithful-but-wrong answer, high faithfulness score masking a retrieval bug) — it's a concrete, non-generic story that shows you understand *why* metrics can lie, which is exactly the kind of thing that separates a tutorial-level answer from a systems-engineer answer.


<invoke name="present_files">
<parameter name="filepaths">["/mnt/user-data/outputs/VectorDB-Basics-and-Tradeoffs.md"]
