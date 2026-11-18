"""Golden dataset for evaluation.

Each line is a JSON object with evaluation fields.
"""

GOLDEN_EXAMPLES = [
    {
        "id": "q001",
        "question": "What is the late fee for Verizon Wireless postpaid plans?",
        "expected_answer": "The late fee is $7 for Verizon Wireless postpaid plans.",
        "expected_sources": ["doc_vzw_ca_2025#late_fee"],
        "route": "simple",
        "category": "simple",
        "should_abstain": False,
    },
    {
        "id": "q002",
        "question": "What are the early termination fees for mobile postpaid and prepaid plans?",
        "expected_answer": "Mobile postpaid has an ETF of up to $350, while prepaid has no ETF.",
        "expected_sources": ["doc_vzw_ca_2025#etf", "doc_vzw_ipi_2025#etf"],
        "route": "multi_hop",
        "category": "multi_hop",
        "should_abstain": False,
    },
    {
        "id": "q003",
        "question": "How do I cook a perfect steak?",
        "expected_answer": "I'm sorry, but I can only answer questions about policy documents in our knowledge base.",
        "expected_sources": [],
        "route": "out_of_scope",
        "category": "out_of_scope",
        "should_abstain": False,
    },
    {
        "id": "q004",
        "question": "What is the return window for Fios equipment?",
        "expected_answer": "Fios equipment must be returned within 30 days of service cancellation.",
        "expected_sources": ["doc_fios_return_2022#return_window"],
        "route": "simple",
        "category": "simple",
        "should_abstain": False,
    },
    {
        "id": "q005",
        "question": "Compare the late fee policies across all Verizon products.",
        "expected_answer": "Wireless postpaid: $7 late fee. Wireless prepaid: no late fee. Fios: $9 late fee. Device financing: varies by state.",
        "expected_sources": ["doc_vzw_ca_2025#late_fee", "doc_fios_ca_home#late_fee", "doc_vzw_dpa_2026#late_fee"],
        "route": "multi_hop",
        "category": "multi_hop",
        "should_abstain": False,
    },
]


def main():
    import json
    import sys

    output_path = sys.argv[1] if len(sys.argv) > 1 else "evals/golden.jsonl"

    with open(output_path, "w") as f:
        for item in GOLDEN_EXAMPLES:
            f.write(json.dumps(item) + "\n")

    print(f"Wrote {len(GOLDEN_EXAMPLES)} items to {output_path}")


if __name__ == "__main__":
    main()