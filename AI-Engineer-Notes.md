# AI Systems Engineer Notes

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