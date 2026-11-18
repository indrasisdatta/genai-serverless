#!/usr/bin/env python3
"""Simple ask script for testing the graph."""

import sys
from app.graph.builder import build_graph


def main():
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is the late fee?"

    graph = build_graph()

    result = graph.invoke(
        {"question": question, "retries": 0},
        config={"recursion_limit": 15},
    )

    print(f"Question: {question}")
    print(f"Answer: {result.get('answer', 'N/A')}")
    print(f"Status: {result.get('status', 'N/A')}")
    print(f"Citations: {result.get('citations', [])}")


if __name__ == "__main__":
    main()