"""
Sweep distance thresholds against the golden set. Report a confusion matrix
and expected cost per threshold so the operating point is a deliberate
business decision, not a guess.
FP A wrongly-refused answerable item frustrates a user (cost 1).
FN A missed refusal (hallucinated answer on an out-of-scope item) is worse
(cost 5 by default) — tune per your incident history.

total business cost = fn * COST_MISSED_REFUSE + fp * COST_WRONG_REFUSE)

"""

import json
import os
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings

from multipdf_chat.db import SessionLocal
from multipdf_chat.retrieval import retrieve


THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
# The AI says "I don't know" to an answerable question (False Positive). 
# Frustrating for users, but low risk.
COST_WRONG_REFUSE = float(os.getenv("COST_WRONG_REFUSE", "1"))
# The AI attempts to answer an unanswerable or out-of-scope question (False Negative). 
# Highly risky because it leads to hallucinations.
COST_MISSED_REFUSE = float(os.getenv("COST_MISSED_REFUSE", "5"))

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
            db, embeddings, item["question"], item.get("product_line"), k=2
        )
        per_item.append({
            "id": item["id"],
            "question": item["question"],
            "expected": item.get("expected_status", "answered"),
            "top_distance": parents[0].distance if parents else 1.0,
            "margin": (parents[1].distance - parents[0].distance) if len(parents) > 1 else 1.0,
            "top_doc": parents[0].doc_slug if parents else None,
            "top_chunk": parents[0].content if parents else None,
        })
    db.close()

    print(f"{'thr':>6} {'correct_refuse':>15} {'missed_refuse':>14} "
          f"{'wrong_refuse':>13} {'correct_answer':>15} {'exp_cost':>10}")

    for thr in THRESHOLDS:
        tp = sum(1 for d in per_item
             if d["expected"] == "insufficient_evidence" and d["top_distance"] > thr)
        fn = sum(1 for d in per_item
             if d["expected"] == "insufficient_evidence" and d["top_distance"] <= thr)
        fp = sum(1 for d in per_item
             if d["expected"] == "answered" and d["top_distance"] > thr)
        tn = sum(1 for d in per_item
                if d["expected"] == "answered" and d["top_distance"] <= thr)
        cost = fn * COST_MISSED_REFUSE + fp * COST_WRONG_REFUSE     

        print(f"{thr:>6.2f} {tp:>15} {fn:>14} {fp:>13} {tn:>15} {cost:>10.1f}")

    # Diagnostic: borderline items for human review
    print("\nborderline items (top_distance between 0.30 and 0.50):")
    for d in sorted(per_item, key=lambda x: x["top_distance"]):
        if 0.30 <= d["top_distance"] <= 0.50:
            print("\n" + "=" * 80)
            print(f"ID:       {d['id']}")
            print(f"Question: {d['question']}")
            print(f"Expected: {d['expected']}")
            print(f"Distance: {d['top_distance']:.3f}")
            print(f"Margin:   {d['margin']:.3f}")
            print(f"Document: {d['top_doc']}")
            print("\nRetrieved chunk:")
            print(d["top_chunk"] or "[No chunk text available]")
            # print(f" {d['id']:12} exp={d['expected']:22} "
                # f"d={d['top_distance']:.3f} m={d['margin']:.3f} {d['top_doc']}")


if __name__ == "__main__":
    main()

    