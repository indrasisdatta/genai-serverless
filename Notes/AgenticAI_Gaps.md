# Agentic AI Upskilling Plan (AWS Track)

**Owner:** Indrasis  
**Duration:** ~10 weeks (8 weeks core + 2 weeks multi-agent project)  
**Target roles:** MERN Lead + GenAI, Full-stack AI Engineer, Agentic AI Engineer (Senior/Lead)  
**Cloud:** AWS  
**Weekly budget assumed:** ~10–12 hours (weekday evenings + one weekend block)

---

## 1. Goal

Convert the existing LangChain fixed-pipeline Policy RAG into a production-grade Agentic RAG service built on LangGraph, deployed on AWS with evals, tracing and guardrails. Then build a real multi-agent app (Auto-Triage) on top of the existing MCP tooling layer.

By the end, I should be able to say in an interview:

> "I built an agentic RAG service with routing, multi-hop decomposition, retrieval grading and groundedness checks in LangGraph. It runs on ECS Fargate with Bedrock, persists state in Postgres, streams to a React UI, and every change is gated by an eval suite on a 70-item golden dataset. Here are the before/after numbers."

The last sentence (the numbers) is the part that differentiates the profile. Measure everything.

---

## 2. Baseline vs Gaps

### Already have

| Area | Evidence |
|------|----------|
| Python, FastAPI | Policy RAG, Root Cause Analyzer |
| Vector search | PostgreSQL + pgvector, IVFFlat vs HNSW understanding |
| RAG fundamentals | LangChain fixed pipeline, versioning (supersede-don't-delete) |
| Eval dataset | 70-item golden dataset |
| Tool integration | MCP layer (Jira, GitLab, Kibana, Figma) |
| Security awareness | RAG attack reference, OWASP LLM Top 10 |
| Frontend | 12 years MERN / React |
| Local models | Ollama-served models |

### Gaps (priority order)

| Gap | Why it matters |
|-----|----------------|
| 1 LangGraph production patterns | Named in most Agentic AI postings |
| 2 Persistence + human-in-the-loop | Separates real apps from notebook demos |
| 3 Serving a graph (FastAPI + SSE streaming) | Needed for any real UI |
| 4 Hybrid search + reranking | Expected in "production RAG" requirements |
| 5 Evals as a CI gate (RAGAS / DeepEval) | Fastest-rising requirement in JDs |
| 6 Tracing (Langfuse / LangSmith) | "AI observability" in JDs |
| 7 AWS AI deployment (Bedrock, ECS, Guardrails) | Cloud deployment is baseline |
| 8 Multi-agent architecture | Senior agentic roles |

---

## 3. Key Design Decision: CRAG vs Agentic RAG

**Decision:** Build Agentic RAG, with a CRAG-style self-correction loop as its first component.

- **CRAG** = grade retrieved docs → if weak, correct retrieval before answering.
- **Agentic RAG** = graph decides the path (route, decompose, retry, abstain). CRAG is one loop inside it.

**Domain adaptation (important):** Standard CRAG falls back to web search. For internal policy questions this is wrong; a web answer is worse than no answer. Fallback here is:

1. Rewrite query → re-retrieve (max 2 retries)
2. Still weak → abstain ("Not found in policy documents") with the closest sections cited

Not multi-agent. The policy assistant is a single graph. Multi-agent is reserved for Auto-Triage (Phase 4), where agents have genuinely different toolsets.

---

## 4. Target Architecture

### 4.1 Graph (final state after Phase 2)

```text
┌──────────────┐
question ───────▶│ router │
└──────┬───────┘
┌────────────────┼──────────────────┐
▼ ▼ ▼
[simple] [multi_hop] [out_of_scope]
│ │ │
│ decompose → Send() ▼
│ (parallel sub-RAG) refuse → END
│ │
▼ ▼
retrieve ◀──── rewrite_query ◀──┐
│ │ weak & retries < 2
▼ │
grade_docs ─────────────────────┘
│ relevant │ weak & retries exhausted
▼ ▼
generate abstain → END
│
▼
check_groundedness ── not grounded (1 retry) ──▶ generate
│ grounded
▼
END (answer + citations)
```

### 4.2 AWS deployment

```text
React UI (S3 + CloudFront)
│ SSE
▼
ALB ──▶ ECS Fargate (FastAPI + LangGraph)
│ │ │
│ │ └──▶ Langfuse (self-hosted on ECS, or cloud)
│ └──▶ Amazon Bedrock (LLM, embeddings, rerank, Guardrails)
└──▶ RDS / Aurora PostgreSQL (pgvector + full-text + checkpointer)
Secrets Manager (DB creds, keys) · CloudWatch (logs, metrics) · ECR (images)
```

**Why ECS Fargate over Lambda:** agent runs are long, multi-step and streamed; Lambda timeouts and response-streaming constraints make it a poor fit. EKS is overkill for one service. (Good interview trade-off to articulate.)

**Why build retrieval instead of Bedrock Knowledge Bases:** control over chunking, document versioning, hybrid search and grading. Know that Knowledge Bases and Bedrock AgentCore exist and when you would use them (speed to market, smaller team, no custom retrieval needs).

---

## 5. Project Structure

```text
policy_agent/
├── app/
│ ├── api/
│ │ ├── routes.py # /chat (SSE), /threads/{id}, /health
│ │ └── schemas.py # request/response Pydantic models
│ ├── graph/
│ │ ├── state.py # GraphState (TypedDict) + reducers
│ │ ├── nodes/
│ │ │ ├── router.py
│ │ │ ├── decompose.py
│ │ │ ├── retrieve.py
│ │ │ ├── grade_docs.py
│ │ │ ├── rewrite_query.py
│ │ │ ├── generate.py
│ │ │ ├── groundedness.py
│ │ │ └── abstain.py
│ │ ├── edges.py # routing functions only
│ │ └── builder.py # build + compile with checkpointer
│ ├── retrieval/
│ │ ├── vector.py # pgvector search
│ │ ├── keyword.py # Postgres full-text (tsvector)
│ │ ├── fusion.py # reciprocal rank fusion
│ │ └── rerank.py # Bedrock rerank / cross-encoder
│ ├── llm/
│ │ └── factory.py # get_llm(): Ollama | Bedrock via config
│ ├── prompts/ # versioned .md/.txt prompt files
│ ├── observability/
│ │ └── tracing.py # Langfuse callback handler setup
│ └── config.py # pydantic-settings, env-driven
├── evals/
│ ├── golden_dataset.jsonl # existing 70 items + new categories
│ ├── run_evals.py # batch runner, writes run_id + results
│ └── thresholds.yaml # CI gate thresholds
├── tests/
│ ├── unit/ # nodes tested with fake LLM
│ └── integration/ # full graph against test DB
├── frontend/ # React chat UI
├── infra/ # Terraform or CDK (optional stretch)
├── Dockerfile
├── docker-compose.yml # app + postgres + langfuse
└── .github/workflows/ or .gitlab-ci.yml
```

---

## 6. Week-by-Week Plan

### Phase 1: LangGraph Foundations + CRAG Loop (Weeks 1–3)

#### Week 1: Port to LangGraph, project skeleton

**Learn**
- StateGraph, nodes, edges, conditional edges, `START`/`END`
- State design: TypedDict vs Pydantic, reducers (`Annotated[list, operator.add]`)
- `with_structured_output` for graders
- `recursion_limit` and why loop guards matter

**Build**
- Project skeleton (Section 5), `config.py`, `llm/factory.py` (Ollama + `ChatBedrockConverse`)
- Port existing LangChain pipeline into a linear graph: `retrieve → generate → END`
- Confirm parity: same answers as the old pipeline on 10 golden items

**Done when**
- [ ] Graph runs end-to-end via a CLI script
- [ ] Switching `LLM_PROVIDER=bedrock` works without code changes
- [ ] Each node is a plain function with a unit test

#### Week 2: CRAG self-correction loop

**Learn**
- CRAG and Self-RAG papers (read the method sections only)
- Grader prompt design; binary vs graded scores

**Build**
- `grade_docs` node (Pydantic output: `relevant: bool`, `reason: str` per chunk)
- `rewrite_query` node + `retry_count` in state; max 2 retries
- `abstain` node returning closest citations
- `check_groundedness` node (answer claims vs retrieved context), 1 regenerate retry

**Done when**
- [ ] Out-of-corpus questions abstain instead of hallucinating
- [ ] No path can loop more than the configured limit (test it)
- [ ] Baseline metrics captured on the 70-item set (see Week 5 metric list; a rough manual pass is fine now)

#### Week 3: Hybrid search + reranking

**Learn**
- Postgres full-text search (`tsvector`, `ts_rank`, GIN index)
- Reciprocal Rank Fusion
- Rerankers: cross-encoder vs Bedrock rerank models

**Build**
- `retrieval/keyword.py`, `fusion.py`, `rerank.py`
- Retrieve top 20 from each → RRF → rerank → top 5
- Structure-aware chunking check: are section headers and policy IDs preserved in chunks?

**Done when**
- [ ] Before/after comparison on golden set: context recall and answer correctness
- [ ] Exact-term queries (policy IDs, plan names) now retrieve correctly

---

### Phase 2: Agentic Routing + Real App Concerns (Weeks 4–5)

#### Week 4: Router, decomposition, persistence, HITL

**Learn**
- `Send` API (map-reduce) for parallel sub-questions
- Checkpointers: `PostgresSaver`, `thread_id`, state history, replay
- `interrupt()` + `Command(resume=...)`

**Build**
- `router` node (simple / multi_hop / out_of_scope) with structured output
- `decompose` node + `Send` fan-out + merge step
- `PostgresSaver` on the existing Postgres instance
- HITL use case: when groundedness fails twice, interrupt and ask the user to confirm/narrow the question

**Done when**
- [ ] Multi-hop questions from golden set answered with citations from each hop
- [ ] Conversation survives an app restart (same `thread_id`)
- [ ] Interrupt → resume works via API

#### Week 5: Serving, streaming, evals as a CI gate

**Learn**
- FastAPI SSE (`StreamingResponse`), LangGraph `astream` with `stream_mode` (`updates`, `messages`)
- RAGAS or DeepEval (pick one, go deep)

**Build: API**
- `POST /chat` streams node progress events + tokens + final citations
- `GET /threads/{id}` returns history from checkpointer
- Timeouts per LLM call; client-disconnect cancellation

**Build: evals**
- Extend golden set with category tags: simple, multi-hop, out-of-scope, adversarial (prompt injection in docs)
- Metrics to track:
  - Faithfulness, answer correctness, context precision, context recall
  - Router accuracy (vs labelled route)
  - Abstention precision (abstained only when it should)
  - p50/p95 latency, tokens per query, cost per query
- `thresholds.yaml` + CI job that fails the pipeline on regression

**Done when**
- [ ] Results table: v0 (old LangChain pipeline) vs v1 (agentic) on all metrics
- [ ] CI blocks a merge that drops faithfulness below threshold (prove it with a deliberately bad prompt change)

---

### Phase 3: Observability, Guardrails, AWS Deployment (Weeks 6–8)

#### Week 6: Tracing + guardrails

**Learn**
- Langfuse tracing concepts: traces, spans, scores, sessions
- Bedrock Guardrails: content filters, denied topics, PII redaction, contextual grounding check
- Indirect prompt injection via retrieved docs

**Build**
- Langfuse callback wired into every graph run; `trace_id` returned to the UI
- Push eval scores into Langfuse as trace scores
- Bedrock Guardrails on input and output
- Retrieved-content sanitisation step + injection test cases in golden set

**Done when**
- [ ] Any bad answer can be debugged from its trace in under 2 minutes
- [ ] Injection test cases pass (no instruction-following from retrieved text)

#### Week 7: Containerise + deploy to AWS

**Learn**
- ECS Fargate: task definitions, services, ALB, health checks
- RDS/Aurora PostgreSQL with pgvector extension
- IAM roles for Bedrock access (no static keys), Secrets Manager

**Build**
- Dockerfile (multi-stage, non-root user), `docker-compose.yml` for local
- Push to ECR, deploy ECS service behind ALB
- RDS PostgreSQL with pgvector; run ingestion job against it
- CloudWatch logs + a basic dashboard (latency, error rate, tokens)

**Stretch:** Terraform or CDK for the infra

**Done when**
- [ ] Public (auth-protected) endpoint works end-to-end
- [ ] Cost per 100 queries measured and documented
- [ ] Ollama vs Bedrock comparison recorded (quality, latency, cost)

#### Week 8: React UI + portfolio packaging

**Build**
- React chat UI: SSE streaming, node-progress indicator ("grading documents…"), citation cards, trace link, thumbs up/down feedback → Langfuse score
- README with architecture diagram, design decisions, eval results table, trade-offs
- 3–5 minute demo video
- Resume + LinkedIn update: "Full-stack Lead who ships production agentic AI"

**Done when**
- [ ] A reviewer can understand the project from the README in 5 minutes
- [ ] Applications start going out (both MERN+GenAI and AI Engineer tracks)

---

### Phase 4: Multi-Agent Auto-Triage (Weeks 9–10)

**Why multi-agent fits here:** each agent has a distinct toolset and expertise (logs vs code changes vs ticket history). That's the justification for splitting.

```text
┌──────────────┐
incident ─────▶│ supervisor │◀─────────────────┐
└──────┬───────┘ │
┌───────────────┼───────────────┐ │
▼ ▼ ▼ │
kibana_agent gitlab_agent jira_agent ─────┘ (results back)
(log queries) (recent MRs, (similar past
deploys) incidents)
│
▼
synthesizer → RCA report
│
▼
interrupt() → human approves → post to Jira
```

#### Week 9: Agents + supervisor

**Learn**
- Supervisor vs handoff (`Command(goto=...)`) vs hierarchical patterns; when each fits
- Subgraphs; per-agent state vs shared state
- `langchain-mcp-adapters` to load existing MCP tools
- Tool permission scoping (least privilege per agent)

**Build**
- Three agents as subgraphs, each with only its own MCP tools
- Supervisor routes and decides when evidence is sufficient
- Hard caps: max supervisor turns, max tool calls per agent

#### Week 10: HITL write-back, evals, packaging

**Build**
- Synthesizer produces structured RCA (hypothesis, evidence, confidence, suggested fix)
- `interrupt()` before any Jira write; resume on approval
- Agent evals on 15–20 historical incidents: tool-call accuracy, unnecessary calls, loop detection, goal completion, time-to-RCA vs manual
- Deploy alongside the policy agent on ECS; README + demo

**Done when**
- [ ] No write action can happen without human approval (tested)
- [ ] Measured time-to-RCA vs manual triage baseline

---

## 7. AWS Services Cheat Sheet

| Need | Service | Notes |
|------|---------|-------|
| LLM | Amazon Bedrock | Use `ChatBedrockConverse`; check model availability in your region |
| Embeddings | Bedrock embedding models | Re-embed corpus if switching from local embeddings; keep dimension in config |
| Reranking | Bedrock rerank models or self-hosted cross-encoder | Compare cost vs quality |
| Guardrails | Bedrock Guardrails | Input + output; contextual grounding check |
| Vector + keyword + checkpointer | RDS / Aurora PostgreSQL | pgvector + tsvector in one DB |
| Compute | ECS Fargate | Long-running, streaming workloads |
| Images | ECR | |
| Secrets | Secrets Manager | IAM task roles for Bedrock; no static keys |
| Logs/metrics | CloudWatch | Dashboard: latency, errors, tokens |
| Frontend | S3 + CloudFront | Managed alternatives (know, don't use) |
| Bedrock Knowledge Bases, Bedrock AgentCore | Be ready to explain when you'd choose them |

---

## 8. Learning Resources

- LangGraph official docs: concepts (state, persistence, HITL, streaming) and how-to guides. Primary source; prefer over blog posts since APIs change.
- LangChain Academy: Intro to LangGraph (free course): good structured pass over core concepts.
- CRAG and Self-RAG papers: read the method sections only.
- RAGAS or DeepEval docs: metric definitions, especially faithfulness and context precision/recall.
- Langfuse docs: LangChain/LangGraph integration and scores.
- AWS docs: Bedrock Converse API, Bedrock Guardrails, ECS Fargate getting started, RDS pgvector.
- OWASP Top 10 for LLM Applications: map each item to a test case in the golden set.
- Rule: for every concept, write the code in the project the same week. No standalone notebook exercises.

---

## 9. Interview Talking Points to Prepare

Write a 3–4 sentence answer for each, backed by numbers from your own evals:

- Why agentic RAG over the fixed pipeline, and what it cost (latency, tokens) vs what it gained
- Why abstain instead of web-search fallback for policy questions
- Hybrid search: which query types improved and by how much
- How you prevent infinite loops and runaway costs
- How evals gate deployments; a regression you caught
- Ollama vs Bedrock: quality, latency, cost, data residency trade-offs
- ECS Fargate vs Lambda vs EKS for agent workloads
- Why build retrieval instead of Bedrock Knowledge Bases
- When multi-agent is justified, and when it's over-engineering
- How you defend against indirect prompt injection and tool misuse
- Habit to practice: for every design answer, volunteer the verification step ("…and I measured it by X, result was Y") without waiting to be asked.

---

## 10. Weekly Rhythm

| Day | Activity | Time |
|-----|----------|------|
| Mon–Tue | Learn the week's concepts (docs, one course module) | 1.5 h/day |
| Wed–Thu | Build | 1.5 h/day |
| Sat | Long build block + tests | 3–4 h |
| Sun | Run evals, update results table, write 1 short note / LinkedIn post | 1 h |

---

## 11. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Scope creep (adding features before measuring) | Each week has a "done when" list; don't start the next week until it's met |
| LangGraph API changes | Pin versions in `requirements.txt`; follow official docs, not old tutorials |
| AWS cost surprises | Billing alarm on day 1; stop ECS tasks/RDS when idle; use smallest instance sizes |
| Company data in personal project | Use only public Verizon policy documents; no internal data |
| Falling behind schedule | Phase 4 is the buffer; Phases 1–3 alone make a strong portfolio |

---

## 12. Progress Tracker

| Week | Focus | Status | Key metric |
|------|-------|--------|------------|
| 1 | LangGraph port + skeleton | ☐ | Parity on 10 items |
| 2 | CRAG loop | ☐ | Abstention on out-of-corpus |
| 3 | Hybrid search + rerank | ☐ | Context recall Δ |
| 4 | Router, Send, persistence, HITL | ☐ | Multi-hop correctness |
| 5 | Streaming API + eval CI gate | ☐ | v0 vs v1 table |
| 6 | Tracing + guardrails | ☐ | Injection tests pass |
| 7 | AWS deployment | ☐ | Cost / 100 queries |
| 8 | React UI + portfolio | ☐ | README + demo done |
| 9 | Multi-agent supervisor | ☐ | Tool-call accuracy |
| 10 | HITL write-back + agent evals | ☐ | Time-to-RCA vs manual |

---