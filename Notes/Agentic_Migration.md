# Agentic AI Upskilling Plan (Local, Free Models)

**Owner:** Indrasis  
**Duration:** 10 weeks (Week 0 setup + 8 weeks core + 2 weeks multi-agent)  
**Runs on:** Local machine with Docker. No cloud.  
**Models:** Groq (LLMs, free tier) + HuggingFace (embeddings and reranker, run locally)  
**Weekly budget assumed:** ~10–12 hours

---

## 0. What Changed From the AWS Plan

| Before | Now |
|--------|-----|
| Amazon Bedrock | Groq API (free tier) for LLMs, Ollama as fallback |
| Bedrock embeddings / rerank | HuggingFace sentence-transformers models, run locally on CPU |
| RDS PostgreSQL | PostgreSQL + pgvector in Docker |
| ECS Fargate | docker compose up on your laptop |
| Bedrock Guardrails | Prompt-injection classifier + Presidio for PII |
| CloudWatch | Langfuse (self-hosted or free cloud tier) |

Everything else (LangGraph design, evals, streaming, multi-agent) stays the same. That's the part interviewers care about; the cloud layer can be added later.

---

## 1. Goal

Rebuild the LangChain Policy RAG as an Agentic RAG service in LangGraph, running locally with free models, backed by evals and tracing. Then build a multi-agent Auto-Triage app using MCP tools.

Final interview line:

> "I rebuilt a fixed RAG pipeline as an agentic LangGraph service with routing, multi-hop decomposition, retrieval grading and groundedness checks. It streams to a React UI, persists conversations in Postgres, and every change runs against a 70-item golden dataset. Faithfulness went from X to Y, and out-of-scope hallucinations dropped to Z."

Measure everything. The numbers are the differentiator.

---

## 2. Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Orchestration | LangGraph | Target skill |
| LLM | Groq via langchain-groq | Free tier, very fast |
| LLM fallback | Ollama (already set up) | When Groq rate-limits you |
| Embeddings | BAAI/bge-small-en-v1.5 (384-dim) or BAAI/bge-base-en-v1.5 (768-dim) | Free, local, strong retrieval quality |
| Reranker | BAAI/bge-reranker-base (cross-encoder) | Free, local |
| Vector + keyword + checkpoints | PostgreSQL 16 + pgvector (Docker) | One DB for everything |
| API | FastAPI + SSE | Streaming to UI |
| UI | React (Vite) | Your strength |
| Evals | RAGAS or DeepEval (pick one) | Most-requested eval tools in JDs |
| Tracing | Langfuse | Free, self-hostable |
| PII | Microsoft Presidio | Free, local |
| MCP | langchain-mcp-adapters | For multi-agent phase |

### Groq model notes (check before you start)

Groq's model list changes often. Per a third-party summary dated August 2026, the public production text models were openai/gpt-oss-20b and openai/gpt-oss-120b, with Qwen models in preview, and the older Llama 3.x models were being withdrawn from self-serve. Verify with the live list (Week 0, Step 5) and never hardcode model IDs; keep them in .env.

Suggested split:

| Role | Model (verify) | Reason |
|------|----------------|--------|
| Router, graders, query rewrite | openai/gpt-oss-20b | Fast, cheap on rate limits, simple decisions |
| Answer generation | openai/gpt-oss-120b | Better quality for final answers |
| Eval judge | openai/gpt-oss-120b | Judge should be at least as strong as the generator |

---

## 3. Designing Around Free-Tier Limits

Free tiers have per-minute and per-day request and token limits. This shapes the design, and it's a good interview topic in itself ("how did you control LLM cost and rate limits?").

- Use the small model for every grading/routing call. These are the most frequent calls.
- Client-side rate limiting. Use LangChain's InMemoryRateLimiter so you throttle before Groq returns 429s.
- Retries with backoff. ChatGroq(max_retries=...) handles transient 429s.
- Cache LLM calls during development. Use LangChain's SQLite cache so re-running the same eval item doesn't cost requests.
- Batch graders. Grade all retrieved chunks in one LLM call (return a list of scores), not one call per chunk. This is a 5x reduction in calls.
- Eval tiers. A 15-item smoke set on every change; the full 70 items only before you merge or once a day.
- Provider fallback. LLM_PROVIDER=ollama switches everything to local models when you hit the daily cap.

Check your actual limits on the Groq console (Settings → Limits) and note them in README.md.

---

## 4. Week 0: Setup (2–3 days)

### Step 1: Accounts and keys

Create a Groq account, generate an API key.
Create a HuggingFace account (needed only for gated models; BGE models are open).
Optional: Langfuse cloud account (free tier), or plan to self-host in Week 7.

### Step 2: Repo and Python environment

```bash
mkdir policy-agent && cd policy-agent
git init
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

requirements.txt (pin exact versions once it works with pip freeze):

```text
langgraph
langgraph-checkpoint-postgres
langchain
langchain-core
langchain-groq
langchain-ollama
langchain-huggingface
langchain-community
sentence-transformers
psycopg[binary,pool]
pgvector
fastapi
uvicorn[standard]
pydantic-settings
python-dotenv
langfuse
ragas    # or deepeval
presidio-analyzer
presidio-anonymizer
pytest
pytest-asyncio
```

```bash
pip install -r requirements.txt
```

### Step 3: Postgres + pgvector in Docker

docker-compose.yml:

```yaml
services:
  db:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: app
      POSTGRES_PASSWORD: app
      POSTGRES_DB: policy
    ports: ["5432:5432"]
    volumes: ["pgdata:/var/lib/postgresql/data"]
volumes:
  pgdata:
```

```bash
docker compose up -d db
docker compose exec db psql -U app -d policy -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

### Step 4: .env

```
GROQ_API_KEY=gsk_...
LLM_PROVIDER=groq           # groq | ollama
FAST_MODEL=openai/gpt-oss-20b   # verify in Step 5
STRONG_MODEL=openai/gpt-oss-120b  # verify in Step 5
OLLAMA_MODEL=llama3.1:8b
EMBED_MODEL=BAAI/bge-small-en-v1.5
EMBED_DIM=384
RERANK_MODEL=BAAI/bge-reranker-base
DATABASE_URL=postgresql://app:app@localhost:5432/policy
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=
```

Add .env to .gitignore immediately.

### Step 5: Verify Groq models

scripts/list_models.py:

```python
import os
from groq import Groq   # installed with langchain-groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])
for m in client.models.list().data:
    print(m.id)
```

Update FAST_MODEL / STRONG_MODEL in .env from the output.

### Step 6: Smoke test structured output

Graders depend on structured output. Test it once on your chosen models:

```python
from pydantic import BaseModel
from langchain_groq import ChatGroq

class Verdict(BaseModel):
    relevant: bool
    reason: str

llm = ChatGroq(model="<FAST_MODEL>", temperature=0)
print(llm.with_structured_output(Verdict).invoke(
    "Is 'roaming charges apply outside the US' relevant to 'Do I pay extra in Canada?'"
))
```

If it fails, try with_structured_output(Verdict, method="json_mode") and note which method works.

Week 0 done when

- [ ] docker compose up -d db works and pgvector extension exists
- [ ] Model list printed, .env updated
- [ ] Structured output works on the fast model

---

## 5. Project Structure

```text
policy-agent/
├── app/
│   ├── config.py              # pydantic-settings, reads .env
│   ├── llm/
│   │   └── factory.py         # get_llm("fast" | "strong")
│   ├── retrieval/
│   │   ├── embeddings.py      # HF embeddings (singleton)
│   │   ├── ingest.py          # chunk + embed + upsert
│   │   ├── vector.py          # pgvector search
│   │   ├── keyword.py         # Postgres full-text search
│   │   ├── fusion.py          # reciprocal rank fusion
│   │   └── rerank.py          # cross-encoder reranker
│   ├── graph/
│   │   ├── state.py
│   │   ├── nodes/
│   │   │   ├── router.py
│   │   │   ├── decompose.py
│   │   │   ├── retrieve.py
│   │   │   ├── grade_docs.py
│   │   │   ├── rewrite_query.py
│   │   │   ├── generate.py
│   │   │   ├── groundedness.py
│   │   │   ├── abstain.py
│   │   │   └── refuse.py
│   │   ├── edges.py
│   │   └── builder.py
│   ├── guards/
│   │   ├── injection.py
│   │   └── pii.py
│   ├── prompts/               # one .txt per prompt, versioned in git
│   ├── api/
│   │   ├── main.py            # FastAPI app + lifespan
│   │   └── routes.py          # /chat (SSE), /threads/{id}, /resume
│   └── observability.py       # Langfuse handler
├── evals/
│   ├── golden.jsonl
│   ├── smoke.jsonl            # 15-item subset
│   ├── run_evals.py
│   ├── thresholds.yaml
│   └── results/               # one JSON per run_id, committed
├── tests/
│   ├── unit/
│   └── integration/
├── frontend/                  # React + Vite
├── scripts/
├── docker-compose.yml
├── Makefile
└── README.md
```

---

## 6. Week-by-Week Plan

### Week 1: Port to LangGraph + local embeddings

Learn

- StateGraph, nodes, edges, START / END, compile(), invoke()
- State as TypedDict; reducers with Annotated[list, operator.add]

#### Step 1: Config and LLM factory

app/config.py:

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    groq_api_key: str = ""
    llm_provider: str = "groq"
    fast_model: str
    strong_model: str
    ollama_model: str = "llama3.1:8b"
    embed_model: str
    embed_dim: int
    rerank_model: str
    database_url: str

    class Config:
        env_file = ".env"

settings = Settings()
```

app/llm/factory.py:

```python
from functools import lru_cache
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from app.config import settings

# Tune to your actual Groq limits (requests per second)
_limiter = InMemoryRateLimiter(requests_per_second=0.5, max_bucket_size=5)

@lru_cache
def get_llm(tier: str = "fast"):
    if settings.llm_provider == "ollama":
        return ChatOllama(model=settings.ollama_model, temperature=0)
    model = settings.fast_model if tier == "fast" else settings.strong_model
    return ChatGroq(model=model, temperature=0, max_retries=3,
                    rate_limiter=_limiter)
```

#### Step 2: Re-embed the corpus with HuggingFace

The new embedding model has a different dimension from your current one, so create a fresh table instead of mixing vectors.

```sql
CREATE TABLE chunks (
  id     BIGSERIAL PRIMARY KEY,
  doc_id TEXT NOT NULL,
  version INT NOT NULL,
  is_current BOOLEAN NOT NULL DEFAULT TRUE, -- supersede-don't-delete
  section TEXT,
  content TEXT NOT NULL,
  embedding vector(384) NOT NULL,
  tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON chunks USING gin (tsv);
CREATE INDEX ON chunks (is_current);
```

app/retrieval/embeddings.py:

```python
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import settings

@lru_cache
def get_embedder():
    return HuggingFaceEmbeddings(
        model_name=settings.embed_model,
        encode_kwargs={"normalize_embeddings": True},
    )
```

BGE models retrieve better when queries get an instruction prefix (check the model card for the exact text). Apply it to queries only, not documents.

Port your existing chunking and versioning logic into ingest.py and run it once.

#### Step 3: Minimal graph (parity with old pipeline)

app/graph/state.py:

```python
from typing import TypedDict, Annotated
import operator

class Chunk(TypedDict):
    id: int
    doc_id: str
    section: str
    content: str
    score: float

class GraphState(TypedDict, total=False):
    question: str
    query: str                    # possibly rewritten
    route: str
    sub_questions: list[str]
    docs: list[Chunk]
    sub_answers: Annotated[list[dict], operator.add]  # filled in parallel
    answer: str
    citations: list[dict]
    retries: int
    grounded: bool
    status: str                   # answered | abstained | refused
```

app/graph/builder.py (Week 1 version):

```python
from langgraph.graph import StateGraph, START, END
from app.graph.state import GraphState
from app.graph.nodes.retrieve import retrieve
from app.graph.nodes.generate import generate

def build_graph(checkpointer=None):
    g = StateGraph(GraphState)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile(checkpointer=checkpointer)
```

Each node is a plain function: def retrieve(state: GraphState) -> dict. It returns only the keys it changes.

#### Step 4: Parity check

Run 10 golden items through old pipeline and new graph; compare answers side by side in a small table.

#### Step 5: Unit tests

Test each node by calling it directly with a hand-built state and a fake LLM (langchain_core.language_models.fake_chat_models.GenericFakeChatModel).

Done when

- [ ] python -m scripts.ask "question" returns an answer via the graph
- [ ] LLM_PROVIDER=ollama works with no code changes
- [ ] Parity table recorded in notes/week1.md
- [ ] One unit test per node passes

---

### Week 2: CRAG self-correction loop

Learn

- CRAG and Self-RAG papers (method sections only)
- Conditional edges, loop guards, recursion_limit

Design reminder: no web-search fallback. For policy questions the fallback is rewrite → re-retrieve → abstain.

#### Step 1: Batched document grader

app/graph/nodes/grade_docs.py:

```python
from pydantic import BaseModel, Field
from app.llm.factory import get_llm

class ChunkGrade(BaseModel):
    index: int
    relevant: bool

class Grades(BaseModel):
    grades: list[ChunkGrade] = Field(description="One entry per chunk")

PROMPT = open("app/prompts/grade_docs.txt").read()

def grade_docs(state):
    docs = state["docs"]
    numbered = "\n\n".join(f"[{i}] {d['content']}" for i, d in enumerate(docs))
    result = get_llm("fast").with_structured_output(Grades).invoke(
        PROMPT.format(question=state["query"], chunks=numbered)
    )
    keep = {g.index for g in result.grades if g.relevant}
    return {"docs": [d for i, d in enumerate(docs) if i in keep]}
```

One call grades all chunks. That matters on a free tier.

#### Step 2: Rewrite node

Takes question + why retrieval failed, returns a new query, increments retries.

#### Step 3: Routing after grading

app/graph/edges.py:

```python
MAX_RETRIES = 2
MIN_RELEVANT = 2

def after_grading(state) -> str:
    if len(state.get("docs", [])) >= MIN_RELEVANT:
        return "generate"
    if state.get("retries", 0) < MAX_RETRIES:
        return "rewrite_query"
    return "abstain"
```

#### Step 4: Groundedness check

Ask the fast model whether every claim in the answer is supported by the given chunks. If not, regenerate once with a stricter prompt; if it fails again, abstain.

#### Step 5: Abstain node

Return a fixed message plus the 2 closest sections as "you may want to check" citations. Set status="abstained".

#### Step 6: Wire it up

```python
g.add_edge(START, "retrieve")
g.add_edge("retrieve", "grade_docs")
g.add_conditional_edges("grade_docs", after_grading,
    {"generate": "generate", "rewrite_query": "rewrite_query", "abstain": "abstain"})
g.add_edge("rewrite_query", "retrieve")
g.add_edge("generate", "groundedness")
g.add_conditional_edges("groundedness", after_groundedness,
    {"end": END, "generate": "generate", "abstain": "abstain"})
g.add_edge("abstain", END)
```

Always invoke with config={"recursion_limit": 15} as a second safety net.

#### Step 7: Graph image

graph.get_graph().draw_mermaid() → paste into README.

Done when

- [ ] Out-of-corpus golden items abstain instead of hallucinating
- [ ] A test proves no path exceeds MAX_RETRIES
- [ ] Rough baseline numbers recorded (correct / wrong / abstained counts on 70 items)

---

### Week 3: Hybrid search + reranking

Learn

- Postgres full-text: tsvector, websearch_to_tsquery, ts_rank
- Reciprocal Rank Fusion
- Bi-encoder vs cross-encoder

#### Step 1: Vector search

```sql
SELECT id, doc_id, section, content
FROM chunks
WHERE is_current
ORDER BY embedding <=> %(qvec)s::vector
LIMIT 20;
```

#### Step 2: Keyword search

```sql
SELECT id, doc_id, section, content,
    ts_rank(tsv, websearch_to_tsquery('english', %(q)s)) AS rank
FROM chunks
WHERE is_current AND tsv @@ websearch_to_tsquery('english', %(q)s)
ORDER BY rank DESC
LIMIT 20;
```

#### Step 3: RRF

app/retrieval/fusion.py:

```python
def rrf(result_lists: list[list[dict]], k: int = 60, top_n: int = 20) -> list[dict]:
    scores, by_id = {}, {}
    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            scores[doc["id"]] = scores.get(doc["id"], 0) + 1 / (k + rank)
            by_id[doc["id"]] = doc
    ranked = sorted(scores, key=scores.get, reverse=True)[:top_n]
    return [{**by_id[i], "score": scores[i]} for i in ranked]
```

#### Step 4: Rerank

app/retrieval/rerank.py:

```python
from functools import lru_cache
from sentence_transformers import CrossEncoder
from app.config import settings

@lru_cache
def _model():
    return CrossEncoder(settings.rerank_model)

def rerank(query: str, docs: list[dict], top_n: int = 5) -> list[dict]:
    if not docs:
        return []
    scores = _model().predict([(query, d["content"]) for d in docs])
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [{**d, "score": float(s)} for d, s in ranked[:top_n]]
```

The retrieve node now does: vector top 20 + keyword top 20 → RRF → rerank → top 5.

#### Step 5: Chunking check

Open 20 random chunks. Do they keep the section heading and policy ID? If not, prepend "{doc_title} > {section}: " to each chunk's text before embedding.

#### Step 6: Measure

Retrieval-only metric, no LLM needed: for each golden item, is the expected source section in the top 5? Compare vector-only vs hybrid vs hybrid+rerank.

Done when

- [ ] Hit-rate@5 table for the three retrieval variants
- [ ] Exact-term queries (plan names, policy IDs) retrieve correctly
- [ ] Reranker latency on CPU measured (note it; it's a trade-off talking point)

---

### Week 4: Router, decomposition, persistence, human-in-the-loop

Learn

- Send for map-reduce fan-out
- Checkpointers, thread_id, get_state, get_state_history
- interrupt() and Command(resume=...)

#### Step 1: Router

Structured output with route: Literal["simple", "multi_hop", "out_of_scope"]. Label the route on every golden item so you can measure router accuracy later.

#### Step 2: Decompose + parallel sub-RAG with Send

```python
from langgraph.types import Send

def fan_out(state):
    return [Send("sub_rag", {"question": q, "query": q, "retries": 0})
            for q in state["sub_questions"]]

g.add_conditional_edges("decompose", fan_out, ["sub_rag"])
g.add_edge("sub_rag", "synthesize")
```

sub_rag is the Week 2 CRAG loop compiled as a subgraph. It returns {"sub_answers": [{"q": ..., "answer": ..., "citations": [...]}]}; the operator.add reducer merges results from parallel branches. synthesize combines them into one cited answer.

Cap sub-questions at 3 to protect your rate limit.

#### Step 3: Postgres checkpointer

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async with AsyncPostgresSaver.from_conn_string(settings.database_url) as cp:
    await cp.setup()   # creates checkpoint tables, run once
    graph = build_graph(checkpointer=cp)
    config = {"configurable": {"thread_id": "user-123-conv-1"},
              "recursion_limit": 25}
    result = await graph.ainvoke({"question": "..."}, config)
```

#### Step 4: Human-in-the-loop

Use case: groundedness failed twice. Instead of abstaining immediately, ask the user to narrow the question.

```python
from langgraph.types import interrupt

def clarify(state):
    reply = interrupt({"message": "I couldn't find a well-supported answer. "
        "Can you narrow it down (plan, region, date)?"})
    return {"question": f"{state['question']} ({reply})", "retries": 0}
```

Resume from outside:

```python
from langgraph.types import Command
await graph.ainvoke(Command(resume="Unlimited plan, Canada"), config)
```

Done when

- [ ] Multi-hop golden items answered with citations from each sub-question
- [ ] Kill and restart the process; the same thread_id continues the conversation
- [ ] Interrupt → resume works from a script

---

### Week 5: FastAPI streaming + React UI

Learn

- graph.astream(..., stream_mode=["updates", "messages"])
- FastAPI StreamingResponse, SSE format, lifespan events

#### Step 1: App lifespan (one checkpointer, one compiled graph)

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncPostgresSaver.from_conn_string(settings.database_url) as cp:
        await cp.setup()
        app.state.graph = build_graph(checkpointer=cp)
        yield

app = FastAPI(lifespan=lifespan)
```

#### Step 2: SSE endpoint

```python
import json
from fastapi import Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

class ChatIn(BaseModel):
    thread_id: str
    question: str

def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

@app.post("/chat")
async def chat(body: ChatIn, request: Request):
    graph = request.app.state.graph
    config = {"configurable": {"thread_id": body.thread_id}, "recursion_limit": 25}

    async def events():
        async for mode, chunk in graph.astream(
            {"question": body.question, "retries": 0}, config,
            stream_mode=["updates", "messages"],
        ):
            if await request.is_disconnected():
                break
            if mode == "updates":
                for node in chunk:
                    yield sse("node", {"node": node})
            elif mode == "messages":
                msg, meta = chunk
                if meta.get("langgraph_node") == "generate" and msg.content:
                    yield sse("token", {"text": msg.content})
        state = await graph.aget_state(config)
        yield sse("done", {"citations": state.values.get("citations", []),
                           "status": state.values.get("status")})

    return StreamingResponse(events(), media_type="text/event-stream")
```

Also add GET /threads/{id} (history from aget_state) and POST /resume (sends Command(resume=...)). If the stream ends on an interrupt, emit an interrupt event with the message.

#### Step 3: React UI (keep it small)

EventSource doesn't support POST, so use fetch + ReadableStream reader and parse SSE lines.
Show live node progress: "Routing → Retrieving → Grading → Writing".
Stream tokens into the answer bubble.
Citation cards (doc, section, snippet).
Clarification prompt when an interrupt event arrives.
Thumbs up/down (stored later as Langfuse scores).

Done when

- [ ] Answers stream in the browser with visible node progress
- [ ] Refreshing the page and loading the same thread shows history
- [ ] Clarify flow works end-to-end in the UI

---

### Week 6: Evals as a gate

Learn

- RAGAS or DeepEval metric definitions: faithfulness, answer correctness/relevancy, context precision, context recall
- LLM-as-judge pitfalls (judge bias, same-model judging)

#### Step 1: Extend the golden dataset

Each line in golden.jsonl:

```json
{"id": "q017", "question": "...", "expected_answer": "...",
 "expected_sources": ["doc_12#roaming"], "route": "multi_hop",
 "category": "multi_hop", "should_abstain": false}
```

Make sure you have all four categories: simple, multi_hop, out_of_scope, adversarial (prompt injection planted inside a test document). Pick 15 items across categories for smoke.jsonl.

#### Step 2: Eval runner (evals/run_evals.py)

Generate a run_id (timestamp + git commit hash).
Run each item through the graph; store answer, citations, route, status, latency, token usage.
Compute:

- No-LLM metrics: hit-rate@5, router accuracy, abstention precision/recall, p50/p95 latency, tokens per query
- LLM-judge metrics (RAGAS/DeepEval with Groq strong model as judge): faithfulness, answer correctness

Write evals/results/{run_id}.json and print a summary table.

Free-tier tactics: run items sequentially, keep the LLM cache on, and run judge metrics only for items whose answer changed since the last run.

#### Step 3: Thresholds

evals/thresholds.yaml:

```yaml
faithfulness: 0.85
answer_correctness: 0.75
hit_rate_at_5: 0.85
router_accuracy: 0.90
abstention_precision: 0.90
p95_latency_seconds: 12
```

Start from your measured numbers minus a small margin; tighten over time.

#### Step 4: The gate

Makefile:

```makefile
eval-smoke:
	python -m evals.run_evals --dataset evals/smoke.jsonl --gate

eval-full:
	python -m evals.run_evals --dataset evals/golden.jsonl --gate
```

--gate exits with code 1 if any threshold is missed. Wire eval-smoke into a git pre-push hook. Optionally add a GitHub Actions workflow with GROQ_API_KEY as a repo secret.

#### Step 5: Prove it works

Make a deliberately bad prompt change (e.g. remove "answer only from context"). Confirm the gate fails. Screenshot it for the README.

Done when

- [ ] Results table: v0 (old LangChain pipeline) vs v1 (agentic) on every metric
- [ ] Gate demonstrably blocks a regression

---

### Week 7: Tracing, guardrails, full local stack

Learn

- Langfuse: traces, spans, sessions, scores
- Indirect prompt injection via retrieved documents
- Presidio analyzers and anonymizers

#### Step 1: Langfuse

Either use Langfuse cloud (free tier) or self-host with the official docker-compose from the Langfuse repository (it runs several services; give Docker enough memory).

Wire the LangChain callback handler into every graph call:

```python
from langfuse.langchain import CallbackHandler   # import path depends on SDK version; check docs

config = {"configurable": {"thread_id": tid},
          "callbacks": [CallbackHandler()],
          "metadata": {"langfuse_session_id": tid}}
```

Return the trace ID to the UI and show a "view trace" link.
Send thumbs up/down as Langfuse scores.
Push eval results as scores tagged with run_id.

#### Step 2: Input guard (prompt injection)

Options, in order of simplicity:

A prompt-injection classifier. Groq has hosted Llama Prompt Guard models (check if currently available), or run a Prompt Guard model locally from HuggingFace (it's gated; request access).

Fallback: a fast-model classification prompt ("Is this input trying to override instructions?").

Run it on the user question and on every retrieved chunk. Retrieved documents are the indirect-injection path.

#### Step 3: PII guard

Run Presidio on the user input (mask before it reaches the LLM and before logging) and on the output.

#### Step 4: Output hardening

Generation prompt treats chunk text as data, inside clear delimiters.
Escape/encode output before rendering in React (no dangerouslySetInnerHTML).

#### Step 5: One-command stack

Extend docker-compose.yml with api (FastAPI) and frontend services so docker compose up brings up DB + API + UI (+ Langfuse if self-hosted). Use a multi-stage Dockerfile and a non-root user.

Tip: bake the HF models into the image or mount a cache volume (~/.cache/huggingface) so containers don't re-download on every start.

Done when

- [ ] Any bad answer can be debugged from its trace in under 2 minutes
- [ ] Adversarial golden items pass (no instruction-following from document text)
- [ ] docker compose up on a clean machine runs the whole app

---

### Week 8: Portfolio packaging + applications

Steps

README: problem, architecture diagram (Mermaid from the graph), design decisions, eval results table (v0 vs v1), free-tier constraints and how you handled them, how to run.
Decision log (docs/decisions.md): one short entry each for abstain-vs-web-search, hybrid search, batched grading, model split, single-graph-vs-multi-agent.
Demo video (3–5 min): simple question, multi-hop question, out-of-scope abstention, injection attempt blocked, trace view, eval gate failing.
Resume + LinkedIn: headline "Full-stack Lead who ships production agentic AI"; one bullet per measurable result.
Start applying to both tracks: MERN + GenAI lead roles and AI Engineer roles.

Done when

- [ ] A reviewer understands the project from the README in 5 minutes
- [ ] First batch of applications sent

---

### Weeks 9–10: Multi-agent Auto-Triage

Why multi-agent fits here: each agent has a different toolset and expertise. That's the real justification for splitting.

Data warning: don't connect company Jira/GitLab/Kibana to a personal project. Use sample data:

Logs: a local Elasticsearch/OpenSearch container loaded with generated or public sample logs, or plain JSON log files behind a small MCP server
Code changes: a public GitHub repo you own with seeded "bad commits"
Tickets: a JSON file of fake past incidents behind a small MCP server

Writing these small MCP servers is itself good practice and a talking point.

```text
                 ┌──────────────┐
incident ───────▶│  supervisor  │◀──────────────┐
                 └──────┬───────┘               │
       ┌───────────────┼───────────────┐         │
       ▼               ▼               ▼         │
   logs_agent   changes_agent   tickets_agent┘
               │
               ▼
       synthesizer → RCA report
               │
               ▼
   interrupt() → human approves → "post" ticket
```

#### Week 9: Agents + supervisor

Learn

- Supervisor vs handoff (Command(goto=...)) vs hierarchical patterns
- Subgraphs; shared vs private state
- create_react_agent from langgraph.prebuilt for each worker
- MultiServerMCPClient from langchain-mcp-adapters

Steps

Build three tiny MCP servers (logs, changes, tickets).
Load tools per agent; each agent only gets its own server's tools (least privilege).

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
client = MultiServerMCPClient({
    "logs": {"command": "python", "args": ["mcp_servers/logs.py"], "transport": "stdio"},
})
log_tools = await client.get_tools()
```

Build each worker with create_react_agent(get_llm("fast"), tools=...).
Supervisor node with structured output: next: Literal["logs_agent", "changes_agent", "tickets_agent", "synthesize"] plus a short reason.
Hard caps in state: max 6 supervisor turns, max 4 tool calls per worker.

#### Week 10: HITL, evals, packaging

Steps

Synthesizer outputs a structured RCA: hypothesis, evidence (with tool-call references), confidence, suggested fix.
interrupt() before the "post ticket" tool; resume only on approval. Write a test proving no write happens without approval.
Agent evals on 15 seeded incidents (you know the true root cause because you seeded it):

- Root cause correct?
- Tool-call accuracy and unnecessary calls
- Loops hit the cap?
- Turns and tokens per incident

Reuse the Week 5 API and UI pattern; show agent hand-offs live in the UI.
README + demo video.

Done when

- [ ] Correct root cause on at least 12 of 15 seeded incidents (or an honest number with failure analysis)
- [ ] Write action impossible without approval (tested)

---

## 7. Learning Resources

- LangGraph official docs: concepts and how-to guides for state, persistence, interrupts, streaming, Send, subgraphs, multi-agent. Primary source; APIs move faster than blog posts.
- LangChain Academy: Intro to LangGraph (free): structured walkthrough of the core concepts.
- Groq docs: supported models, rate limits, structured outputs, deprecations page.
- Sentence-Transformers docs + BGE model cards: embedding/reranker usage, query instruction prefix.
- pgvector README: index types and operators.
- RAGAS or DeepEval docs: metric definitions.
- Langfuse docs: LangChain integration, scores, self-hosting.
- OWASP Top 10 for LLM Applications: turn each item into a golden-set test case.
- CRAG and Self-RAG papers: method sections.

Rule: every concept gets written into the project the same week. No standalone notebooks.

---

## 8. Interview Talking Points

Prepare a 3–4 sentence answer for each, with your own numbers:

- Fixed pipeline vs agentic: what it gained and what it cost (latency, LLM calls per query)
- Why abstain instead of web-search fallback for policy questions
- Hybrid search + rerank: which queries improved, and the reranker's latency cost on CPU
- How you prevent infinite loops and runaway calls
- Running on free-tier limits: batching, model split, caching, fallback provider
- How evals gate changes; a regression the gate caught
- Using a separate, stronger judge model and why
- Indirect prompt injection via retrieved documents: test cases and defences
- When multi-agent is justified and when it's over-engineering
- How you'd move this to the cloud (only the infra layer changes: managed Postgres, container service, managed models)

Habit to practice: for every design answer, volunteer how you verified it ("…and I measured it with X; result was Y") without being asked.

---

## 9. Weekly Rhythm

| Day | Activity | Time |
|-----|----------|------|
| Mon–Tue | Learn the week's concepts | 1.5 h/day |
| Wed–Thu | Build | 1.5 h/day |
| Sat | Long build block + tests | 3–4 h |
| Sun | Run evals, update results, write a short note or LinkedIn post | 1 h |

---

## 10. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Groq rate limits block progress | Batched graders, cache, smoke set, LLM_PROVIDER=ollama fallback |
| Groq model deprecated mid-plan | Model IDs only in .env; re-run list_models.py; re-run evals after switching |
| Library API changes | Pin versions after Week 0; follow official docs |
| Scope creep | Don't start next week until "done when" is met |
| Company data in personal repo | Public policy docs and seeded sample data only |
| Falling behind | Weeks 9–10 are the buffer; Weeks 1–8 alone make a strong portfolio |

---

## 11. Progress Tracker

| Week | Focus | Status | Key metric |
|------|-------|--------|------------|
| 0 | Setup, model check | ☐ | Structured output works |
| 1 | LangGraph port + HF embeddings | ☐ | Parity on 10 items |
| 2 | CRAG loop | ☐ | Out-of-corpus abstention |
| 3 | Hybrid search + rerank | ☐ | Hit-rate@5 by variant |
| 4 | Router, Send, persistence, HITL | ☐ | Multi-hop correctness |
| 5 | FastAPI SSE + React UI | ☐ | Streaming works end-to-end |
| 6 | Evals gate | ☐ | v0 vs v1 table |
| 7 | Tracing + guardrails + compose stack | ☐ | Injection tests pass |
| 8 | Portfolio + applications | ☐ | README + demo |
| 9 | Multi-agent supervisor | ☐ | Tool-call accuracy |
| 10 | HITL + agent evals | ☐ | RCA correct on 15 incidents |