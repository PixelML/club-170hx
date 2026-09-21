#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Analyse the cross-check receipt: is the HTTP path faithful to the engine, and
what does batching do to the numbers?

`crosscheck.json` holds, per fixture, the label logprobs read twice: in-process
with vLLM's LLM.generate (all fixtures in one batch) and over the HTTP
completions path. Two of the fixtures are *the same prompt* as another fixture
(`choice-billing-rot2` repeats `choice-billing`, `noul-angry-2` repeats
`noul-angry`), which turns the receipt into a batching probe as well:

  * HTTP vs in-process, same prompt            -> transport fidelity
  * in-process, duplicate prompts in one batch -> batch-position sensitivity

  python analyze_crosscheck.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECEIPTS = HERE.parent / "receipts"

DUPLICATES = {
    "choice-billing": "choice-billing-rot2",
    "noul-angry": "noul-angry-2",
}


def max_delta(a, b):
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    return max(abs(a[k] - b[k]) for k in keys)


def main():
    with open(RECEIPTS / "crosscheck.json") as f:
        cc = json.load(f)
    rows = {r["case"]: r for r in cc["rows"]}

    transport = []
    for case, r in rows.items():
        transport.append({
            "case": case,
            "labels": len(r["shared_label_ids"]),
            "max_abs_delta_http_vs_inprocess": r["max_abs_delta"],
            "inprocess_only": r["local_only"],
            "http_only": r["http_only"],
        })

    batching = []
    for first, second in DUPLICATES.items():
        a, b = rows[first], rows[second]
        batching.append({
            "prompt": first,
            "duplicate_fixture": second,
            "same_prompt": a["shared_label_ids"] == b["shared_label_ids"],
            "max_abs_delta_inprocess_pair": max_delta(a["local_logprobs"],
                                                      b["local_logprobs"]),
            "max_abs_delta_http_pair": max_delta(a["http_logprobs"],
                                                 b["http_logprobs"]),
            "inprocess_logprobs": [a["local_logprobs"], b["local_logprobs"]],
            "http_logprobs": [a["http_logprobs"], b["http_logprobs"]],
        })

    exact = [r for r in transport if r["max_abs_delta_http_vs_inprocess"] == 0.0]
    out = {
        "method": "derived from crosscheck.json: the same label-masked reads taken "
                  "in-process (one batch of five prompts) and over HTTP, plus a "
                  "duplicate-prompt pair per question type",
        "transport_fidelity": {
            "note": "HTTP vs in-process for the same prompt. The HTTP path is the "
                    "one the endpoint serves (one read per request); the "
                    "in-process run placed all five fixtures in a single batch.",
            "rows": transport,
            "exact_matches": len(exact),
            "of": len(transport),
        },
        "batch_position_sensitivity": {
            "note": "Two fixtures carry the same prompt. In-process they sit at "
                    "different batch positions and do not agree; over HTTP they "
                    "are separate requests and do. This is the engine's "
                    "batch-composition dependence, not a transport defect.",
            "rows": batching,
        },
    }
    (RECEIPTS / "crosscheck-analysis.json").write_text(
        json.dumps(out, indent=1) + "\n")
    print("wrote", RECEIPTS / "crosscheck-analysis.json")
    for row in transport:
        print(f"  transport {row['case']:22s} labels={row['labels']} "
              f"max|delta|={row['max_abs_delta_http_vs_inprocess']}")
    for row in batching:
        print(f"  batch {row['prompt']:22s} in-process pair "
              f"{row['max_abs_delta_inprocess_pair']}, http pair "
              f"{row['max_abs_delta_http_pair']}")


if __name__ == "__main__":
    main()
