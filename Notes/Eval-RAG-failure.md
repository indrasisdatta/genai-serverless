# Case Study — Verizon Product Terms & Policy RAG (Production Issue → Fix)

> **Note on sourcing:** Verizon has not published an official engineering postmortem of this kind. This is a **representative** walkthrough modelled on real, well-documented failure modes for RAG over telecom terms/policy documents. The data setup it assumes (scraping Verizon terms-and-conditions + user guides, recursive chunking with LangChain) mirrors a real public community project, [VerAIzon](https://github.com/Chirayu-Tripathi/VerAIzon), which built a Verizon customer-service RAG bot over exactly these sources. Numbers below are illustrative.

---

## 1. The product

An internal/customer-facing assistant answering questions about Verizon product terms & policy docs: plan terms (Unlimited Welcome / Plus / Ultimate), Auto Pay & paper-free billing discounts, device-payment agreements, early-termination/upgrade fees, and international/roaming policy. Stack: PDF/HTML terms → chunks → embeddings → vector store → LLM (LCEL chain in LangChain).

## 2. The production issue

Support tickets and a QA sweep showed the bot **quoting the right number for the wrong plan.**

> *"What's the Auto Pay discount on **Unlimited Welcome**?"* → bot answered with the discount and eligibility wording for **Unlimited Ultimate**.

Two compounding root causes:

1. **Plan-tier conflation (retrieval/precision).** "Unlimited Welcome / Plus / Ultimate" clauses are near-identical in wording, so their embeddings sit almost on top of each other. The dense retriever pulled a *similar* chunk, not the *correct-plan* chunk.
2. **Table-splitting (chunking).** The fee/discount tables were split by a naive character splitter, so the dollar figure ended up in a chunk separated from its plan-name heading — the chunk had no plan identity at all.

**The dangerous part:** the answer was a *real Verizon figure*, so **faithfulness/groundedness stayed high (~0.95)** and hallucination checks didn't flag it. A faithful-but-wrong answer is the classic signature of a **retrieval** problem, not a generation one.

## 3. Metrics tracked

| Metric | What it catches | Before | After |
|---|---|---:|---:|
| Context Precision (RAGAS) | retrieved noise vs. signal | 0.62 | 0.89 |
| Context Recall (RAGAS) | did we fetch all needed evidence | 0.70 | 0.93 |
| Faithfulness (RAGAS) | hallucination | 0.95 | 0.97 |
| Answer Correctness (RAGAS) | matches ground truth | 0.66 | 0.92 |
| Hit@4 / MRR | is right chunk retrieved & ranked high | 0.74 / 0.58 | 0.96 / 0.88 |
| **Plan-attribution accuracy** *(domain)* | answer cites the *right plan* | 0.71 | 0.96 |
| **Numeric/fee exactness** *(domain)* | $ amounts match exactly | 0.64 | 0.95 |
| p95 latency | production SLA budget | 1.9 s | 2.4 s |

The two domain metrics (plan-attribution + numeric exactness) were the real production KPIs — they map directly to customer harm (quoting the wrong fee). Faithfulness alone would have said "all good."

## 4. How production scores were tracked

- **Golden set + CI gate.** ~150 curated Q&A pairs from the actual terms docs, each labelled with `ground_truth`, `expected_plan`, and `expected_amounts`. Run on every deploy; **block the release** if any metric drops below threshold (e.g., plan-attribution < 0.93).
- **Online sampling.** Every prod request logs `{query, retrieved_chunks, answer}`. An async job scores a 5–10% sample with the same scorers and pushes to a dashboard (LangSmith / Arize Phoenix / Grafana), with alerts on regression.
- **Primary KPIs on the dashboard:** plan-attribution accuracy and numeric exactness, plus context precision/recall and p95 latency.

---

## 5. The fix (LangChain)

> Import paths shift across LangChain/RAGAS versions — pin them. These reflect recent (`langchain` 0.3.x / `ragas` 0.2.x) layouts.

### 5.1 Root-cause fix — structure-aware chunking + plan identity + metadata
Keep fee tables whole, and stamp every chunk with its plan so the embedding *carries* plan identity.

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

headers = [
    ("#",   "doc"),        # "Verizon Wireless Customer Agreement"
    ("##",  "plan_name"),  # "Unlimited Welcome"
    ("###", "section"),    # "Auto Pay & paper-free billing discount"
]
md_splitter   = MarkdownHeaderTextSplitter(headers, strip_headers=False)
char_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)

def build_chunks(markdown_text, effective_date, version):
    chunks = []
    for sec in md_splitter.split_text(markdown_text):
        plan    = sec.metadata.get("plan_name", "general")
        section = sec.metadata.get("section", "")
        # only sub-split long prose; keep short fee tables intact
        parts = char_splitter.split_documents([sec]) if len(sec.page_content) > 800 else [sec]
        for p in parts:
            p.page_content = f"[Plan: {plan}] [Section: {section}]\n{p.page_content}"  # plan identity in the vector
            p.metadata.update({
                "plan_name": plan, "section": section,
                "effective_date": effective_date, "version": version,
            })
            chunks.append(p)
    return chunks
```

### 5.2 Hybrid retrieval + metadata filter
BM25 catches the exact plan-name token that dense embeddings blur; metadata filter pins the right plan.

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings(model="text-embedding-3-large"))

def hybrid_retriever(plan_filter=None):
    sk = {"k": 10}
    if plan_filter:                       # e.g. detected from the query
        sk["filter"] = {"plan_name": plan_filter}
    dense  = vectorstore.as_retriever(search_kwargs=sk)
    sparse = BM25Retriever.from_documents(chunks); sparse.k = 10
    return EnsembleRetriever(retrievers=[dense, sparse], weights=[0.5, 0.5])
```
*(For automatic plan detection, a `SelfQueryRetriever` with an `AttributeInfo` for `plan_name` can derive the filter from the query.)*

### 5.3 Cross-encoder reranker
Push the exact-plan chunk to the top before generation.

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

reranker  = CrossEncoderReranker(model=HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base"), top_n=4)
retriever = ContextualCompressionRetriever(base_compressor=reranker, base_retriever=hybrid_retriever())
```

### 5.4 Grounded generation prompt + chain
Force the model to name the plan + effective date, quote amounts exactly, and refuse when the plan isn't matched.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

SYSTEM = (
    "Answer questions about Verizon product terms using ONLY the provided context. "
    "Always name the exact plan and effective date, and quote fees/amounts exactly as written. "
    "If the context does not clearly cover the plan in the question, say you are not certain and "
    "ask the user to confirm their plan. Cite the source section."
)
prompt = ChatPromptTemplate.from_messages([("system", SYSTEM),
                                           ("human", "Question: {question}\n\nContext:\n{context}")])
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def format_docs(docs):
    return "\n\n---\n\n".join(
        f"(plan={d.metadata['plan_name']}, eff={d.metadata['effective_date']})\n{d.page_content}" for d in docs)

chain = ({"context": retriever | format_docs, "question": RunnablePassthrough()}
         | prompt | llm | StrOutputParser())
```

### 5.5 Eval harness — domain scorers + RAGAS (CI gate)
The plain-Python scorers below are version-proof and are the primary prod KPIs.

```python
import re

PLANS = ["Unlimited Welcome", "Unlimited Plus", "Unlimited Ultimate"]

def extract_plan(text):
    hits = [p for p in PLANS if p.lower() in text.lower()]
    return hits[0] if hits else None

def plan_attribution_correct(answer, expected_plan):
    return extract_plan(answer) == expected_plan

def numeric_exact(answer, expected_amounts):
    found = set(re.findall(r"\$\d+(?:\.\d{2})?", answer))
    return all(a in found for a in expected_amounts)
```

```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (faithfulness, answer_relevancy,
                           context_precision, context_recall, answer_correctness)

rows = [{
    "question":     ex["q"],
    "answer":       chain.invoke(ex["q"]),
    "contexts":     [d.page_content for d in retriever.invoke(ex["q"])],
    "ground_truth": ex["gt"],
} for ex in golden_set]

scores = evaluate(Dataset.from_list(rows),
                  metrics=[context_precision, context_recall,
                           faithfulness, answer_relevancy, answer_correctness])

# CI gate: block deploy on regression
assert scores["context_precision"] >= 0.85
assert scores["answer_correctness"] >= 0.88
# plus domain gates computed from plan_attribution_correct / numeric_exact over golden_set
```
*(RAGAS ≥ 0.2 also supports `EvaluationDataset` / `SingleTurnSample`; the `Dataset` form above still works in the compatibility path. Pin your version.)*

---

## 6. Takeaways
- **Faithful but wrong ⇒ look upstream at retrieval**, not the LLM.
- For policy/terms docs, **chunk by structure and keep tables intact**, and put the entity (plan) identity *inside* the embedded text + metadata.
- Embeddings blur near-identical entity names — **hybrid search + metadata filter + reranker** is what separates "Welcome" from "Ultimate."
- Generic metrics (faithfulness) miss real harm. Add **domain KPIs** (plan-attribution accuracy, numeric exactness) and gate deploys on them in CI, plus sample-score live traffic.