#!/usr/bin/env python3
"""c=8 aggregate stability check: 3 rounds of 8 concurrent requests.

Usage:
    python3 c8_stability.py --base-url http://127.0.0.1:18098 --model glm-5.3-flash \
        --out /workspace/receipts/tp4-record --concurrency 8 --rounds 3
"""
import json
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
from common import Client, base_parser, build_fixture, save  # noqa: E402

PROMPT_TOKENS = 2900
OUT_TOKENS = 256


def one_request(client, i, round_tag):
    nonce = f"[{round_tag}-{i}-{uuid.uuid4().hex}] "
    text, _ = build_fixture(client, PROMPT_TOKENS, nonce)
    t0 = time.perf_counter()
    r = client.completion(text, OUT_TOKENS, temperature=0.7, ignore_eos=True)
    dt = time.perf_counter() - t0
    if r["ok"]:
        return {"ok": True, "wall_s": round(dt, 3), "completion_tokens": r["usage"]["completion_tokens"]}
    return {"ok": False, "wall_s": round(dt, 3), "error": r.get("error")}


def main():
    p = base_parser(__doc__)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--rounds", type=int, default=3)
    args = p.parse_args()
    client = Client(args.base_url, args.model)

    results = []
    for rnd in range(1, args.rounds + 1):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            res = list(ex.map(lambda i: one_request(client, i, f"r{rnd}"), range(args.concurrency)))
        wall = time.perf_counter() - t0
        ok = [r for r in res if r["ok"]]
        toks = sum(r["completion_tokens"] for r in ok)
        rec = {
            "round": rnd, "concurrency": args.concurrency, "succeeded": len(ok), "failed": args.concurrency - len(ok),
            "wall_s": round(wall, 3), "aggregate_tok_s": round(toks / wall, 2) if wall > 0 else None,
            "errors": [r.get("error") for r in res if not r["ok"]],
        }
        results.append(rec)
        print(json.dumps(rec), flush=True)

    save(args.out, f"c{args.concurrency}_stability.json", results)


if __name__ == "__main__":
    main()
