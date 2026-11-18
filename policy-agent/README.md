# Policy RAG → Agentic RAG → Multi-Agent (LangGraph + ChatGroq)

**Owner:** Indrasis
**Duration:** 10 weeks + Week 0 cleanup
**Runs on:** Local machine (FastAPI + PostgreSQL/pgvector + React/Vite)
**LLM:** ChatGroq only (langchain-groq)
**Embeddings / reranker:** HuggingFace models, run locally on CPU
**Codebase:** your existing multipdf_chat FastAPI app (reviewed below)
**Package layout:** `agentic_rag/` replaces the old `app/` layout

---

## 0. How This Plan Works

One app, three stages. Each stage reuses the previous one, so nothing is thrown away:

| Stage | Weeks | What you build | What you learn |
|-------|-------|----------------|----------------|
| A. Port | 0–1 | Existing LangChain pipeline rebuilt as a LangGraph graph, same behaviour | StateGraph, nodes, dependency injection, testing |
| B. Agentic RAG | 2–5 | CRAG loop, hybrid search, rerank, conversation memory, SSE streaming, eval gate | Conditional edges, loops, persistence, streaming, evals |
| C. Multi-agent | 6–10 | Supervisor + Policy QA agent + Catalog agent + Comparison agent + Verifier, HITL approvals | Production multi-agent architecture |

The Policy QA graph from Stage B becomes one agent inside the Stage C system. That is the main learning point: in production, agents are composed from smaller, tested graphs.

---

## 1. Review of the Current Code

### 1.1 What you already have (keep it)

| Piece | Where | Keep? |
|-------|-------|-------|
| FastAPI app with lifespan, CORS, request-ID logging middleware, JSON logs | main.py, helper.setup_logging | Yes |
| SQLAlchemy SessionLocal on app.state.db | db.py, main.py | Yes |
| HF embeddings all-MiniLM-L6-v2 (384-dim) on app.state.embeddings | main.py | Yes, no re-embedding needed |
| Parent–child chunking (2000-char parents, semantic child chunks) | helper.generate_embedding | Yes, it's a good design |
| Title / product-line stamping on chunks | helper.generate_embedding | Yes |
| Section extraction from headings | helper.find_section | Yes |
| Tables documents, parent_documents, child_chunks | DB | Yes, with 2 new columns (Week 3) |
| ChatGroq(model="openai/gpt-oss-20b") | helper.get_conversational_chain | Yes, becomes the "fast" model |
| srv.agentic_rag.api router already mounted | main.py | New code goes in srv/agentic_rag/ |

### 1.2 What gets replaced

| Current | Replaced by | Why |
|---------|-------------|-----|
| load_qa_chain(..., chain_type="stuff") | LangGraph generate node with a ChatPromptTemplate | Legacy chain API; no control over citations |
| StreamingHandler + asyncio.Queue + create_task | graph.astream(stream_mode=["messages", ...]) | LangGraph streams tokens natively; the manual task is never cancelled when the client disconnects |
| text/plain stream | SSE with typed events (node, token, done, error, interrupt) | UI can show progress, citations and errors separately |
| Ingestion functions that take request: Request | Ingestion service that takes (session_factory, embeddings) | So agents and scripts can call it without FastAPI |
| FAISS / S3 helpers | Remove after the port | pgvector is the single store |
| app/ layout (multi-module) | agentic_rag/ package (§4 structure) | Single-package, request-free ingestion for agents |

### 1.3 Bugs to fix in Week 0 (before porting)

Fix these first, otherwise your baseline numbers are wrong and you'll be comparing the new graph against a broken pipeline.

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | helper.stream_user_input | `if not token or token.isspace(): continue` drops whitespace tokens, so words run together in the stream ("Thepolicy covers") | Only skip None/empty string; keep spaces |
| 2 | helper.stream_user_input | `asyncio.create_task(chain.ainvoke(...))` keeps running (and spending Groq quota) if the client disconnects | Temporary fix: keep the task handle and task.cancel() in finally. Permanent fix: Week 4 SSE endpoint |
| 3 | helper.user_input (/user_query) | Filters metadata ->> 'session_id', but generate_embedding writes ingest_session_id, so documents ingested via /create_embeddings are never found | Drop the session filter (same as the stream path) or filter by doc_id |
| 4 | helper.user_input | Passes a list[str] to load_qa_chain, which expects Document objects | Wrap rows in Document(page_content=...) (or skip, since Week 1 replaces it) |
| 5 | upload.upload_handler | Calls generate_embedding(request, raw_text, session_id), but that function needs (request, raw_text, doc_row, ingest_session_id) | Call generate_embedding_upload(request, raw_text, session_id) |
| 6 | upload.lambda_handler | get_text_chunks(raw_text) is missing type and request; generate_embedding(chunks, session_id) has the wrong signature | Delete it (no AWS in this plan) |
| 7 | create_file_embeddings_handler | 'message': e puts an exception object in the response; FastAPI can't serialise it, so any failure becomes a 500 | 'message': str(e) |
| 8 | generate_embedding | Re-running /create_embeddings for the same document inserts duplicate parents and children; retrieval then returns duplicates and old versions | Week 3: is_current flag, supersede old rows in the same transaction |
| 9 | stream_user_input | `WHERE id = ANY(:ids)` returns parents in arbitrary order, losing the vector-search ranking | Re-order in Python by the original parent_ids order |
| 10 | get_pdf_texts | `text += page.extract_text()` fails when a page returns None (scanned page) | `page.extract_text() or ""` (as get_pdf_text already does) |
| 11 | stream_user_input | Sync DB and embedding calls inside an async generator block the event loop for every request | LangGraph runs sync nodes in a thread pool (Week 4) |
| 12 | StreamingHandler.on_llm_error | Error text is pushed into the answer stream as if it were content | Typed error SSE event (Week 4) |

Small clean-ups while you're there: remove unused imports (from email import message, from requests import session, from requests import Session, watchtower, boto3 once S3 is gone).

---

## 2. Stack

| Layer | Choice | Notes |
|-------|--------|-------|
| Orchestration | langgraph | |
| Persistence | langgraph-checkpoint-postgres + psycopg[binary,pool] | Same Postgres instance; its own tables |
| LLM | langchain-groq (ChatGroq) | Two tiers: fast + strong |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Already in use; 384-dim; keep, no re-embedding |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Small, fast on CPU. Try BAAI/bge-reranker-base later and measure |
| Keyword search | Postgres full-text (tsvector, GIN index) | No extra service |
| API | FastAPI + SSE | Agentic_rag package (§4 structure) |
| UI | Your React/Vite app (localhost:5173) | Switch to SSE parsing |
| Tracing | Langfuse (free cloud tier or self-hosted) | Week 8 |
| Evals | Custom harness first, then RAGAS or DeepEval | Week 5 |
| MCP | langchain-mcp-adapters | For multi-agent phase (§7) |

### 2.1 Groq models

You already use openai/gpt-oss-20b. Use two tiers, configured in .env:

| Tier | Model | Used for |
|------|-------|----------|
| fast | openai/gpt-oss-20b | Supervisor, graders, rewrites, verifier, catalog agent |
| strong | openai/gpt-oss-120b | Final answers, comparison tables, eval judge |

Groq retires models regularly. Confirm both IDs with client.models.list() before starting (Week 0, Step 4) and keep IDs only in .env.

### 2.2 Living with free-tier rate limits

- Fast model for every high-frequency call (grading, routing, verification).
- Grade all retrieved passages in one call, not one call per passage.
- InMemoryRateLimiter on every ChatGroq instance so you throttle before Groq returns 429.
- max_retries on ChatGroq for transient 429s.
- strong.with_fallbacks([fast]) for answer generation: if the 120B model is rate-limited, answer with the 20B model instead of failing.
- SQLite LLM cache during eval runs only.
- Budgets in code: max retries, max sub-questions, max supervisor turns, max tool calls.

### 2.3 Library versions

Your current imports (langchain.chains.question_answering, langchain.text_splitter, langchain.retrievers, langchain.docstore) are from the LangChain 0.x layout. In LangChain 1.x these legacy APIs moved to separate packages (langchain-classic, langchain-text-splitters). Options:

- **Recommended:** keep your current versions working, add langgraph, langgraph-checkpoint-postgres, psycopg[binary,pool], upgrade langchain-groq, and let pip resolve. New code only imports from langchain_core, langgraph, langchain_groq, langchain_huggingface, so it's unaffected.
- After Week 1, when the legacy chains are gone, upgrade everything and fix the splitter imports (from langchain_text_splitters import RecursiveCharacterTextSplitter).
- Pin with `pip freeze > requirements.lock` once it works.

---

## 3. Target Architecture (end of Week 10)

### 3.1 Multi-agent graph

```
                      ┌──────────┐
   user message ──────▶│ condense │  follow-up → standalone question
                      └────┬─────┘
                           ▼
                    ┌────────────┐   Route(next, task, reason)
            ┌──────▶│ supervisor │──────────────────────────────┐
            │       └─────┬──────┘                              │
            │   ┌─────────┼─────────────┐                       │ next = "finish"
            │   ▼         ▼             ▼                       ▼
            │ policy_qa  catalog    comparison              ┌─────────┐
            │ (agentic   (tool loop: (plan → Send →         │ compose │ strong model
            │  RAG       list/find    policy_qa per         └────┬────┘
            │  subgraph) docs, embed   target → table)            ▼
            │            job w/ HITL)                       ┌─────────┐
            │   │         │            │                   │ verify  │ code checks + LLM
            └───┴─────────┴────────────┘                   └────┬────┘
              AgentResult appended to state                pass │      │ fail (once)
                                                             ▼      └──▶ compose
                                                            ┌──────────┐
                                                            │ finalize │──▶ END
                                                            └──────────┘
```

### 3.2 Policy QA subgraph (built in Weeks 1–3, reused as an agent)

```
START → retrieve → grade ──relevant──▶ generate → check ──grounded──▶ finalize → END
                │                   ▲          │
                │ none relevant     │          │ not grounded (1 retry)
                ▼                 └──────────┘
              rewrite → retrieve         │ still not grounded
                │ retries exhausted       ▼
                └────────────────▶ abstain → END
```

No web-search fallback: for policy questions, "not found in the policy documents" is the correct answer when retrieval fails.

### 3.3 Why these agents (interview answer)

Each agent exists because it has a different toolset or failure mode, not because multi-agent is fashionable:

| Agent | Type | Tools / data | Why separate |
|-------|------|--------------|--------------|
| Policy QA | Deterministic workflow (subgraph) | Hybrid retrieval over chunks | Needs grading/retry loop; fully testable |
| Catalog | Tool-calling loop (ReAct) | Read-only SQL tools on documents; one write tool (embedding job) behind approval | Metadata questions ("which documents cover prepaid?") that RAG answers badly; only agent allowed to write |
| Comparison | Workflow with Send fan-out | Calls Policy QA once per target | Parallel, structured output (table) |
| Verifier | Single LLM call + code checks | Final draft + pooled sources | Independent check; a different prompt catches what the generator misses |
| Supervisor | Router | Agent registry | Picks which agent(s) a question needs |

Honest trade-off to state in interviews: the comparison and verifier steps could be plain nodes. They're modelled as agents with contracts so they can be tested, traced and swapped independently, and so the system can grow (e.g. a billing agent with its own tools) without changing the supervisor.

---

## 4. Folder Structure

```text
multipdf_chat/                     # existing app (unchanged except Week 0 fixes)
srv/
└── agentic_rag/
    ├── __init__.py
    ├── api.py                     # router already mounted in main.py
    ├── config.py
    ├── llm.py                     # ChatGroq factory
    ├── prompts.py                 # prompt file loader
    ├── prompts/
    │   ├── qa_grade.txt
    │   ├── qa_rewrite.txt
    │   ├── qa_generate.txt
    │   ├── qa_groundedness.txt
    │   ├── condense.txt
    │   ├── supervisor.txt
    │   ├── catalog.txt
    │   ├── compare_plan.txt
    │   ├── compare_table.txt
    │   └── verify.txt
    ├── retrieval/
    │   ├── __init__.py
    │   ├── retriever.py           # vector + keyword + parents
    │   ├── fusion.py              # RRF
    │   └── rerank.py
    ├── ingestion/
    │   ├── __init__.py
    │   └── service.py             # request-free ingestion (moved from helper.py)
    ├── qa/
    │   ├── __init__.py
    │   ├── state.py
    │   ├── nodes.py
    │   ├── edges.py
    │   └── graph.py
    ├── chat/
    │   ├── __init__.py
    │   └── graph.py               # Week 4 single-agent chat graph
    ├── agents/
    │   ├── __init__.py
    │   ├── contracts.py           # AgentResult, AgentSpec, registry
    │   ├── state.py               # AppState
    │   ├── wrapper.py             # as_agent(): timeouts, error isolation, events
    │   ├── supervisor.py
    │   ├── catalog.py
    │   ├── comparison.py
    │   ├── compose.py
    │   ├── verifier.py
    │   └── graph.py               # top-level multi-agent graph
    └── observability.py
evals/
├── golden.jsonl                   # your 70 items + labels
├── smoke.jsonl                    # 15-item subset
├── routing.jsonl                  # Week 9
├── verifier_seeded.jsonl          # Week 9
├── run_evals.py
├── thresholds.yaml
└── results/
tests/
├── unit/
└── integration/
```

---

## 5. Week 0: Clean-up and Baseline (3–4 days)

### Step 1: Fix the bugs in 1.3

Commit each fix separately with a clear message. You'll reference these in interviews ("found and fixed a streaming bug that merged words…").

### Step 2: Install new packages

```bash
pip install -U langgraph langgraph-checkpoint-postgres "psycopg[binary,pool]" langchain-groq pydantic-settings sentence-transformers pytest pytest-asyncio
pip freeze > requirements.lock
```

### Step 3: .env additions

```
GROQ_API_KEY=gsk_...
FAST_MODEL=openai/gpt-oss-20b
STRONG_MODEL=openai/gpt-oss-120b
GROQ_RPS=0.5
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
MAX_QA_RETRIES=2
MAX_SUPERVISOR_TURNS=5
```

### Step 4: Confirm Groq model IDs

```python
# scripts/list_models.py
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
for m in Groq(api_key=os.environ["GROQ_API_KEY"]).models.list().data:
    print(m.id)
```

### Step 5: Smoke-test structured output on both models

Graders and the supervisor depend on it.

```python
from pydantic import BaseModel
from langchain_groq import ChatGroq

class Verdict(BaseModel):
    relevant: bool
    reason: str

for model in ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
    llm = ChatGroq(model=model, temperature=0)
    print(model, llm.with_structured_output(Verdict).invoke(
        "Question: Do I pay extra in Canada? Passage: Roaming charges apply outside the US. "
        "Is the passage relevant?"))
```

If one fails, try `with_structured_output(Verdict, method="json_schema")` or `method="json_mode"` and note which method works for which model. Put the working method in config.py.

### Step 6: Move ingestion out of the request layer

`generate_embedding` reads `request.app.state`. The Catalog agent (Week 7) needs to run ingestion without an HTTP request, so move the body into a service:

```python
# srv/agentic_rag/ingestion/service.py
import json, uuid
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from langchain_core.embeddings import Embeddings
from multipdf_chat.helper import get_pdf_text, find_section, get_parent_child_splitters_for

FIND_DOCS = text("""
    SELECT doc_id, slug, title, product_line, storage_path
    FROM documents WHERE slug = ANY(:slugs)
""")

class IngestionService:
    def __init__(self, session_factory: sessionmaker, embeddings: Embeddings):
        self.session_factory = session_factory
        self.embeddings = embeddings

    def ingest_slugs(self, slugs: list[str]) -> list[dict]:
        with self.session_factory() as db:
            docs = db.execute(FIND_DOCS, {"slugs": slugs}).mappings().all()
        results = []
        for doc in docs:
            try:
                raw = get_pdf_text(doc["storage_path"])
                self.ingest_document(raw, doc, ingest_session_id=str(uuid.uuid4()))
                results.append({"slug": doc["slug"], "status": "success"})
            except Exception as e:
                results.append({"slug": doc["slug"], "status": "failed", "message": str(e)})
        found = {d["slug"] for d in docs}
        results += [{"slug": s, "status": "failed", "message": "not found"} for s in slugs if s not in found]
        return results

    def ingest_document(self, raw_text: str, doc_row, ingest_session_id: str) -> None:
        # Move the body of helper.generate_embedding here, replacing
        #   request.app.state.embeddings -> self.embeddings
        #   request.app.state.db()       -> self.session_factory()
        ...
In helper.py, change get_parent_child_splitters(request) to accept embeddings directly (keep a thin wrapper for the old call sites). Then make create_file_embeddings_handler call IngestionService(...).ingest_slugs(...). Same behaviour, no FastAPI dependency.
```

### Step 7: Record the baseline

Run your 70 golden questions through the fixed `/chat/stream` endpoint and save answers to `evals/results/v0_baseline.jsonl`. Every later week is compared against this file.

### Week 0 done when

- [ ] All 12 bugs fixed and committed
- [ ] Both Groq models confirmed, structured-output method recorded
- [ ] IngestionService used by /create_embeddings with identical results
- [ ] v0_baseline.jsonl saved
---