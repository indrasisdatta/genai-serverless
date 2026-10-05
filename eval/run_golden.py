"""
Golden-set evaluation harness.
Usage:
python -m eval.run_golden --golden eval/golden_v1.jsonl --out baselines/step_01_current.json
# retrieval-only (skip LLM, useful while iterating on ranking)
python -m eval.run_golden --golden eval/golden_v1.jsonl --out baselines/retrieval_only.json --skip-generation

Steps:
1. Read the golden dataset containing questions and expected answers/sources.
2. For each question, retrieve the top-K relevant chunks and evaluate retrieval quality.
3. Generate an answer from the retrieved chunks and evaluate the generated answer.
4. Collect scores and timings for all questions, calculate a summary, and save the results as JSON.
"""
import subprocess
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from multipdf_chat.db import SessionLocal
from multipdf_chat.helper import get_conversational_chain
from multipdf_chat.retrieval import retrieve
import argparse
import json
from pathlib import Path 
import time 


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"

def score_retrieval(item, retrieved, k_values=(1,5,10)):
    """Calculate Recall@K, MRR, section_hit@5"""
    gold_docs = set(item.get("gold_doc_ids") or [])
    gold_sections = set(item.get("gold_sections") or [])
    retrieved_docs = [r.doc_slug for r in retrieved]
    retrieved_sections = [r.section_path for r in retrieved]
    scores = {}
    # No gold document = insufficient evidence.
    if not gold_docs:
        for k in k_values:
            scores[f"recall@{k}"] = None
            scores["mrr"] = None
    else:
        # Recall@K / Hit@K
        for k in k_values:
            top_k_docs = set(retrieved_docs[:k])
            # If there's intersection then set 1, otherwise 0
            scores[f"recall@{k}"] = (
                1.0 if gold_docs & top_k_docs else 0.0
            )
        # Mean Reciprocal Rank - 1/1, 1/2, 1/3
        mrr = 0.0
        for rank, doc in enumerate(retrieved_docs, start=1):
            if doc in gold_docs:
                mrr = 1.0 / rank
                break
        scores["mrr"] = mrr
    return scores

    # Did we retrieve the expected section within top 5?
    hit = False
    if gold_sections:
        top5_sections = [section  for section in retrieved_sections[:5] if section]        
        for gold_section in gold_sections: 
            for retrieved_section in top5_sections:
                if gold_section in retrieved_section:
                    hit = True 
                    break 
            if hit:
                break
        scores["section_hit@5"] = 1.0 if hit else 0.0
    else:
        scores["section_hit@5"] = None

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
        "must_not_include_hit_rate": (
            len(exclude_hits) / len(must_not_include) if must_not_include else None
        ),
    }

def generate_answer(retrieved, question: str) -> str:
    if not retrieved:
        return ""
    docs = [
        Document(
            page_content=r.content,
            metadata={
                "doc_slug": r.doc_slug,
                "section_path": r.section_path
            }
        )
        for r in retrieved[:5]
    ]
    chain = get_conversational_chain(streaming=False)
    resp = chain(
        {"input_documents": docs, "question": question},
        return_only_outputs=True
    )
    return resp['output_text']

def summarize(results):

    def _mean(section, key):
        vals = [
            r[section][key] 
            for r in results 
            if r[section].get(key) is not None
        ]
        return round(sum(vals) / len(vals), 3) if vals else None

    def _pass_rate(key):
        vals = [
            r["generation"][key] 
            for r in results 
            if r["generation"].get(key) is not None
        ]
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
        "summary": summarize(results),
        "per_item": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out["summary"], indent=2))

if __name__ == "__main__":
    main()