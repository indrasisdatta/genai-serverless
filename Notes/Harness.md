# Harness Engineering in AI — Interview Notes
> The discipline of building the *environment* around an AI agent, not just the agent itself.

---

## The Core Equation

```
Agent = Model + Harness
```

The **model** does the reasoning. The **harness** does everything else.

> Think of it like a race car driver (model) vs the pit crew, track rules, safety systems, and telemetry (harness). The driver's talent matters, but without the environment around them, they crash.

---

## What Problem Does a Harness Solve?

LLMs are **probabilistic**, not deterministic. They:
- Don't reliably follow instructions 100% of the time
- Don't know your codebase or team conventions
- Can't reliably evaluate their own output
- Drift architecturally over time when used in long-running agents

A harness wraps the model with **deterministic controls** so the overall system becomes reliable, even if the model itself isn't perfectly consistent.

> Key insight from OpenAI's Codex team: *"Our most difficult challenges now center on designing environments, feedback loops, and control systems."* — not the model itself.

---

## The Three Pillars (Your Understanding Was Right)

### 1. Context
What the agent *knows* going into a task.

- `AGENTS.md` / `CLAUDE.md` files — project rules, conventions, architecture decisions
- Codebase maps — "here's how this repo is structured"
- Reference docs, how-to guides, examples
- The goal: give the agent **a map, not a 1,000-page instruction manual**

### 2. Constraints (Linters + Guardrails)
Deterministic rules that make bad outputs *impossible*, not just unlikely.

- Custom linters that enforce naming conventions, file size limits, logging standards
- Type checkers, structural tests (e.g. ArchUnit for module boundary violations)
- CI gates — agent's output fails CI if it breaks a constraint
- The key: linter error messages are written as **remediation instructions** so the agent can self-correct when it reads the error

> "In a human-first workflow, these rules feel pedantic. With agents, they become multipliers — once encoded, they apply everywhere at once." — OpenAI

### 3. Feedback Loops (Sensors)
Observing what the agent *did* and correcting it.

- Pre-commit / pre-push hooks that run linters before output is committed
- Automated test runs after each agent action
- "LLM as judge" — a second AI model reviews the first agent's output
- Observability: traces, logs of every tool call and reasoning step (OpenTelemetry, Langfuse, Arize Phoenix)

---

## Two Types of Controls

| Type | What it is | Speed | Reliable? | Examples |
|------|-----------|-------|-----------|---------|
| **Computational** | Deterministic, run by CPU | Milliseconds | ✅ Always | Linters, type checkers, tests, ArchUnit |
| **Inferential** | Semantic, run by LLM | Seconds–minutes | ⚠️ Probabilistic | AI code review, LLM-as-judge, AGENTS.md rules |

You need both. Computational alone = fast but misses semantic issues. Inferential alone = catches meaning but non-deterministic and expensive.

---

## Feedforward vs Feedback

These are the two directions a harness works in:

**Feedforward (Guides)** — steer the agent *before* it acts
- AGENTS.md with coding conventions
- Bootstrap scripts, project templates
- Reference documentation injected into context

**Feedback (Sensors)** — correct the agent *after* it acts
- Linters that run after code is generated
- Tests that catch behavioral regressions
- Review agents that score the output

> Both are required. Feedforward-only = agent never learns what actually failed. Feedback-only = agent keeps making the same mistakes.

---

## Your Question: PreTool/PostTool Hooks + MCP Security Proxy

**Yes, you're on the right track — but here's the precise mapping:**

| What you said | Harness term | What it actually does |
|---------------|-------------|----------------------|
| PreTool hook | Feedforward control (computational) | Intercepts before a tool call — validate inputs, check permissions, enforce rate limits |
| PostTool hook | Feedback sensor (computational) | Intercepts after a tool call — validate outputs, log results, trigger self-correction |
| MCP security proxy | Guardrail / permission boundary | Acts as a gateway that enforces what tools/resources the agent is allowed to touch at all |

So your design is correct. The full picture for an agentic workflow with a harness looks like:

```
User Request
    ↓
[Context Injection] ← AGENTS.md, project docs, conventions
    ↓
[PreTool Hook] ← validate intent, check permissions, rate limit
    ↓
Model + Tool Calls
    ↓
[PostTool Hook] ← validate output, run linters, check schema
    ↓
[MCP Security Proxy] ← enforces what tools/resources are reachable
    ↓
[Feedback Sensor] ← tests, CI, LLM-as-judge scoring
    ↓
Output (or self-correction loop)
```

The MCP security proxy is specifically the **permission boundary layer** — it's not inside the model's reasoning, it's a deterministic gate that enforces what the agent is structurally *allowed* to do.

---

## The Plan-Execute-Verify (PEV) Pattern

A common harness architecture for agentic workflows:

1. **Plan** — agent decomposes the task into explicit steps
2. **Execute** — agent works through each step
3. **Verify** — output is checked against the plan AND external quality criteria (linters, tests, review agent)

This prevents the agent from "solving in one pass" and producing plausible-sounding but incorrect results.

---

## Harness Engineering vs Context Engineering vs Prompt Engineering

| Era | Focus | What you build |
|-----|-------|---------------|
| Prompt Engineering (2022–24) | Craft the right instruction | Better prompts |
| Context Engineering (2025) | Curate what the agent knows | RAG pipelines, memory, AGENTS.md |
| **Harness Engineering (2026)** | **Design the whole environment** | Constraints, feedback loops, hooks, guardrails |

They're not replacements — each layer builds on the previous one. Harness engineering is the outermost, most mature layer.

---

## GitHub Copilot Analogy (How to Explain in an Interview)

> "GitHub Copilot helps you write code — that's the model. But Copilot Workspace and similar tools have a harness around them: the context of your repo, the constraints of your language/linter setup, and feedback from your test suite. When Stripe built their internal coding agents, they added pre-push hooks as feedback sensors and blueprints as feedforward guides — that system around the model is the harness."

Real-world examples of teams building harnesses:
- **OpenAI** — custom linters with remediation instructions in error messages, structural tests, garbage collection agents
- **Stripe** — pre-push hooks as feedback sensors, "blueprints" as feedforward guides
- **LangChain** — changed only the harness (not the model), and their coding agent jumped from 52.8% → 66.5% on benchmarks

---

## How to Answer "What is a Harness?" in an Interview

> "A harness is everything around an AI model that makes it reliable in production. The formula is: Agent = Model + Harness. The harness has three main parts: context (what the agent knows), constraints (deterministic rules like linters and tests that prevent bad outputs), and feedback loops (sensors that let the agent self-correct). In an agentic workflow, this translates to things like PreTool/PostTool hooks, custom linters with remediation instructions, CI gates, and security proxies that enforce permission boundaries — especially in MCP-based architectures."

---

## TL;DR

- **Harness** = the environment built around an agent to make it reliable
- **Context** = what it knows (AGENTS.md, docs, codebase maps)
- **Constraints** = deterministic rules (linters, type checkers, CI gates)
- **Feedback loops** = sensors that catch errors and trigger self-correction
- **PreTool/PostTool hooks** = where you implement feedforward and feedback at the tool-call level
- **MCP security proxy** = the permission boundary layer — deterministically enforces what the agent can touch
- More constraints = more reliability (counter-intuitive but proven)
- The model is not the bottleneck. The harness is.

---
*Notes based on Martin Fowler / Thoughtworks, OpenAI Codex team, and LangChain harness research — 2026*