#!/usr/bin/env python3
"""Driver: runs prefill ladder, then needle + decode + vision using the largest passing length."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import longctx_bench as B

OUT = os.environ.get("DSV4_OUT", ".")
GOLDEN_FIXTURES = os.environ.get("DSV4_GOLDEN_FIXTURES", "./golden/image_fixtures.json")

def largest_pass(prefill_rec):
    passed = [l["target_prompt_tokens"] for l in prefill_rec["levels"] if l["status"] == "PASS"]
    return max(passed) if passed else None

def main():
    print("tokenize endpoint present:", B.has_tokenize(), flush=True)
    prefill_rec = B.prefill_ladder()
    largest = largest_pass(prefill_rec)
    print("largest passing prefill length:", largest, flush=True)
    if largest is None:
        print("no prefill length passed; aborting downstream phases", flush=True)
        return
    needle_lengths = sorted(set([l for l in [32000, 131000, largest] if l <= largest]))
    B.needle_test(needle_lengths)
    c1_lengths = sorted(set([l for l in [131000, largest] if l <= largest]))
    B.decode_longctx(c1_lengths, 32000 if largest >= 32000 else largest)
    if largest >= 131000:
        image_fixtures = json.load(open(GOLDEN_FIXTURES))
        grad = next(f for f in image_fixtures if f["id"] == "img03_gradient_red_green")
        B.vision_longctx(131000, grad["data_url"])
    else:
        print(f"largest passing length {largest} < 131000; skipping vision_longctx per protocol step 4", flush=True)

if __name__ == "__main__":
    main()
