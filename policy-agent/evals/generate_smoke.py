"""Smoke test dataset - 15 items for quick validation."""

SMOKE_ITEMS = [
    "What is the late fee for Verizon Wireless postpaid plans?",
    "What are the early termination fees for mobile postpaid and prepaid plans?",
    "How do I cook a perfect steak?",
    "What is the return window for Fios equipment?",
    "Compare the late fee policies across all Verizon products.",
    "What is the activation fee for a new line?",
    "Is there a restocking fee for returned devices?",
    "What is the dispute window for billing errors?",
    "What are the returned payment fees?",
    "Does the device payment agreement have a late fee?",
    "What is the ETF for a mobile prepaid plan?",
    "What happens if I don't return Fios equipment?",
    "What is the interest rate on device financing?",
    "Can I dispute a late fee?",
    "What is the cancellation fee for home phone service?",
]


def main():
    import json
    import sys

    # Load full golden set and extract smoke subset
    golden_path = sys.argv[1] if len(sys.argv) > 1 else "evals/golden.jsonl"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "evals/smoke.jsonl"

    # For now just write the smoke questions
    with open(output_path, "w") as f:
        for q in SMOKE_ITEMS:
            f.write(json.dumps({"question": q}) + "\n")

    print(f"Wrote {len(SMOKE_ITEMS)} smoke items to {output_path}")


if __name__ == "__main__":
    main()