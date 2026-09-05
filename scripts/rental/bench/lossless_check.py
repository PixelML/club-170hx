#!/usr/bin/env python3
"""20-prompt greedy (temperature=0) determinism/lossless sanity check with
speculative decoding (MTP) enabled. Each prompt is run twice; PASS requires
byte-identical output both times (greedy + spec should be exactly lossless).

Usage:
    python3 lossless_check.py --base-url http://127.0.0.1:18098 --model glm-5.3-flash \
        --out /workspace/receipts/tp4-record
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from common import Client, base_parser, now, save  # noqa: E402

PROMPTS = [
    "Summarize the plot of Hamlet in three sentences.",
    "Write a haiku about the ocean.",
    "What is the capital of Mongolia?",
    "Explain photosynthesis to a five-year-old.",
    "List the first ten prime numbers.",
    "Write a SQL query that selects the top 5 customers by total order value.",
    "Translate 'good morning' into French, Spanish, and Japanese.",
    "What year did the Berlin Wall fall?",
    "Write a regex that matches a US phone number.",
    "Explain the difference between TCP and UDP.",
    "Write a function in Python that reverses a linked list.",
    "What is the boiling point of water at sea level in Celsius?",
    "Give three synonyms for 'happy'.",
    "Explain what a hash table is.",
    "Write a short limerick about a cat.",
    "What is the chemical formula for table salt?",
    "Describe the water cycle in two sentences.",
    "Write a bash one-liner that counts lines in a file.",
    "What is the speed of light in a vacuum, in meters per second?",
    "Name the planets of the solar system in order from the sun.",
]


def main():
    p = base_parser(__doc__)
    args = p.parse_args()
    client = Client(args.base_url, args.model)

    reps = []
    n_match = 0
    for i, prompt in enumerate(PROMPTS):
        r1 = client.completion(prompt, 128, temperature=0, seed=42)
        r2 = client.completion(prompt, 128, temperature=0, seed=42)
        match = r1.get("ok") and r2.get("ok") and r1.get("text") == r2.get("text")
        n_match += int(match)
        reps.append({
            "prompt": prompt, "match": match,
            "text1_head": r1.get("text_head"), "text2_head": r2.get("text_head"),
            "ok1": r1.get("ok"), "ok2": r2.get("ok"),
        })
        print(f"[{i:02d}] match={match}", flush=True)

    rec = {
        "phase": "lossless_check", "utc": now(), "n_prompts": len(PROMPTS), "n_match": n_match,
        "verdict": "PASS" if n_match == len(PROMPTS) else "FAIL", "reps": reps,
    }
    save(args.out, "lossless_check.json", rec)
    print(rec["verdict"], f"{n_match}/{len(PROMPTS)} identical")


if __name__ == "__main__":
    main()
