#!/usr/bin/env python3
"""Prefill-tok/s context-length sweep: 4k / 16k / 64k tokens, 3 reps each.

Usage:
    python3 context_sweep.py --base-url http://127.0.0.1:18098 --model glm-5.3-flash \
        --out /workspace/receipts/tp4-record --lengths 4096 16384 65536 --reps 3
"""
import sys
import os
import uuid

sys.path.insert(0, os.path.dirname(__file__))
from common import Client, base_parser, build_fixture, now, save  # noqa: E402


def main():
    p = base_parser(__doc__)
    p.add_argument("--lengths", type=int, nargs="+", default=[4096, 16384, 65536])
    p.add_argument("--reps", type=int, default=3)
    args = p.parse_args()
    client = Client(args.base_url, args.model)

    cells = []
    for target in args.lengths:
        reps = []
        for i in range(args.reps):
            prefix = f"[ctx-{target}-{uuid.uuid4().hex}] "
            text, exact = build_fixture(client, target, prefix)
            r = client.completion(text, 1)
            pt = r.get("usage", {}).get("prompt_tokens")
            r["prefill_tok_s"] = round(pt / r["wall_s"], 2) if r["ok"] and pt else None
            r["fixture_tokens_exact"] = exact
            reps.append(r)
            print(f"ctx={target} rep {i}", r.get("usage"), r["wall_s"], r.get("prefill_tok_s"), flush=True)
        ok = [x for x in reps if x["ok"]]
        cells.append({
            "target_tokens": target, "reps": reps,
            "median_prefill_tok_s": sorted(x["prefill_tok_s"] for x in ok)[len(ok) // 2] if ok else None,
        })

    rec = {"phase": "context_sweep", "utc": now(), "cells": cells}
    save(args.out, "context_sweep.json", rec)


if __name__ == "__main__":
    main()
