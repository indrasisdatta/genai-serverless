"""Eval runner - runs golden dataset through the graph and computes metrics."""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import yaml


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load JSONL dataset."""
    items = []
    with open(path) as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def load_thresholds(path: str) -> Dict[str, float]:
    """Load evaluation thresholds from YAML."""
    with open(path) as f:
        return yaml.safe_load(f)


def run_evals(
    dataset_path: str,
    thresholds_path: str = "evals/thresholds.yaml",
    gate: bool = False,
) -> Dict[str, Any]:
    """
    Run evaluations on the dataset.

    This is a skeleton - in production you'd invoke the graph here.
    """
    dataset = load_dataset(dataset_path)
    thresholds = load_thresholds(thresholds_path)

    results = {
        "run_id": datetime.utcnow().isoformat() + "Z",
        "dataset": dataset_path,
        "num_items": len(dataset),
        "metrics": {},
        "passed": True,
    }

    # Placeholder metrics - replace with actual graph runs
    results["metrics"] = {
        "faithfulness": 0.0,
        "answer_correctness": 0.0,
        "hit_rate_at_5": 0.0,
        "router_accuracy": 0.0,
        "abstention_precision": 0.0,
        "p95_latency_seconds": 0.0,
    }

    # Check thresholds
    if gate:
        for metric, threshold in thresholds.items():
            value = results["metrics"].get(metric, 0)
            if value < threshold:
                results["passed"] = False
                print(f"FAIL: {metric} = {value:.3f} < {threshold:.3f}")
            else:
                print(f"PASS: {metric} = {value:.3f} >= {threshold:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run evaluations")
    parser.add_argument("--dataset", default="evals/golden.jsonl", help="Path to dataset")
    parser.add_argument("--thresholds", default="evals/thresholds.yaml", help="Path to thresholds")
    parser.add_argument("--gate", action="store_true", help="Exit with error if thresholds not met")
    parser.add_argument("--output", default=None, help="Output file for results")

    args = parser.parse_args()

    results = run_evals(args.dataset, args.thresholds, args.gate)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)

    if args.gate and not results["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()