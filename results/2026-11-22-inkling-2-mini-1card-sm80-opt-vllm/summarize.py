"""Load every receipt of this experiment into plain Python structures (no third-party deps).

Parameterized via ROUNDS so adding Round 4 (once Track 4 lands) requires only appending one
spec dict to ROUNDS (or placing receipts/round4.json) and rerunning make_chart.py + make_notebook.py.
"""
from __future__ import annotations

import csv
import json
import re
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE / "receipts"

HARDWARE_STOCK = "stock CMP 170HX 64 GiB @ 140 W TDP, 1,140 MHz SM / 1,728 MHz HBM2e, 1,600.5 GB/s D2D copy BW"
HARDWARE_PR90_OFFSET = "CMP 170HX 64 GiB @ 140 W TDP (+200 MHz SM offset @ 1,410 MHz, 1,597 GB/s D2D copy BW, 32K ctx)"

# Declarative milestone registry driving make_chart.py, make_notebook.py, and summary.csv.
# To add Round 4 later: append a Round-4 entry here (or drop receipts/round4-spec.json) and re-run.
ROUNDS: list[dict] = [
    {
        "id": "baseline",
        "label": "Baseline\n(PR #90,\nk=7)",
        "short": "Baseline (`ctl-01/02`, `k=7`)",
        "color": "#7f8c8d",
        "ledger_runs": ["ctl-01-base", "ctl-02-base"],
        "peak_runs": ["ctl-00-cold", "ctl-01-base", "ctl-02-base"],
        "cold_startup_run": "ctl-00-cold",
        "long_131k_run": None,
        "long_231k_run": None,
        "pr90_file": "pr90-dspark7-stock.jsonl",
        "pr90_c16_ref_note": "510.9* (+200 MHz card, 32K)",
        "gsm8k_file": None,
        "gsm8k_fallback": (48, 50),
        "gates_file": None,
        "metrics_file": None,
    },
    {
        "id": "round1",
        "label": "Round 1\n(CUDA align\n+ 49K head)",
        "short": "Round 1 (`cand-10`, `k=7`)",
        "color": "#5b8def",
        "ledger_runs": ["cand-10-combo49k-b4096"],
        "peak_runs": ["cand-10-combo49k-b4096", "ctl-12-combo49k", "ctl-13-combo49k"],
        "cold_startup_run": "cand-10-combo49k-b4096",
        "long_131k_run": "winner-long-131k",
        "long_231k_run": "winner-long-231k",
        "pr90_file": "pr90-r1best.jsonl",
        "gsm8k_file": "gsm8k-r1best.jsonl",
        "gates_file": "r1-gates-summary.json",
        "metrics_file": "r1-winner-metrics.txt",
    },
    {
        "id": "round2",
        "label": "Round 2\n(ll_gemm +\nmega + k=4)",
        "short": "Round 2 (`cand-16`, `k=4`)",
        "color": "#f2a33a",
        "ledger_runs": ["cand-16-r2combo-k4", "ctl-20-r2combo-k4", "ctl-21-r2combo-k4"],
        "peak_runs": ["cand-16-r2combo-k4", "ctl-20-r2combo-k4", "ctl-21-r2combo-k4"],
        "cold_startup_run": "cand-16-r2combo-k4",
        "long_131k_run": "r2-winner-long-131k",
        "long_231k_run": "r2-winner-long-231k",
        "pr90_file": "pr90-r2best.jsonl",
        "gsm8k_file": "gsm8k-r2best.jsonl",
        "gates_file": "r2-gates-summary.json",
        "metrics_file": "r2-winner-metrics.txt",
    },
    {
        "id": "round3",
        "label": "Round 3 Best\n(v4 CUDA +\nBlock + dyn-k)",
        "short": "Round 3 Best (`cand-25`, unified)",
        "color": "#1f9d55",
        "ledger_runs": ["cand-25-r3v3-exact", "cand-25b-r3v3-exact", "cand-25c-r3v3-exact"],
        "peak_runs": [
            "cand-25-r3v3-exact",
            "cand-25b-r3v3-exact",
            "cand-25c-r3v3-exact",
            "ctl-24-r3-dynk",
            "ctl-25-r3-dynk",
            "ctl-26-r3-dynk",
            "ctl-27-r3-dynk",
        ],
        "cold_startup_run": "cand-25b-r3v3-exact",
        "long_131k_run": "r3-winner-long-131k",
        "long_231k_run": "r3-winner-long-231k",
        "pr90_file": "pr90-r3best.jsonl",
        "gsm8k_file": "gsm8k-r3best.jsonl",
        "gates_file": "r3-gates-summary.json",
        "metrics_file": "r3-winner-metrics.txt",
    },
]

# Optional Round-4 and Round-5 auto-hooks:
if (R / "round4-spec.json").is_file():
    ROUNDS.append(json.loads((R / "round4-spec.json").read_text(encoding="utf-8")))
if (R / "round5-spec.json").is_file():
    ROUNDS.append(json.loads((R / "round5-spec.json").read_text(encoding="utf-8")))

# PR #90 published reference on the +200 MHz SM offset card @ 32K context (for side-by-side context)
PR90_PUBLISHED_OFFSET_CARD = {
    "code_t0": 166.9,
    "struct_t0": 189.1,
    "prose_t0": 94.0,
    "c16_t1": 510.9,
    "prefill_23k": 4700,
    "gsm8k_100": "99/100",
}

CEILING_SUMMARY = {
    "c1_mixed_4k": 238.0,
    "c1_code_4k": 298.0,
    "c16_4k": 15181.0,
    "prefill_4k": 5731.0,
    "prefill_32k": 5026.0,
    "long_131k_prefill": 3534.0,
    "long_131k_decode": 185.0,
    "long_231k_prefill": 2532.0,
    "long_231k_decode": 150.0,
    "kv_tokens_cap": 563878,
}


def ledger_rows() -> list[dict]:
    return [json.loads(line) for line in open(R / "ledger.jsonl", encoding="utf-8") if line.strip()]


def ledger_by_name() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in ledger_rows():
        out.setdefault(row["name"], []).append(row)
    return out


def pr90_summary(filename: str | None) -> dict:
    if not filename or not (R / filename).is_file():
        return {}
    rows = [json.loads(line) for line in open(R / filename, encoding="utf-8") if line.strip()]
    out: dict = {}
    for case in ("code", "struct", "prose"):
        t0 = [r["decode_tok_s"] for r in rows if r["suite"] == "c1" and r["case"] == case and r.get("temperature") == 0.0]
        t1 = [r["decode_tok_s"] for r in rows if r["suite"] == "c1" and r["case"] == case and r.get("temperature") == 1.0]
        both = t0 + t1
        ttfts = [r["ttft_s"] for r in rows if r["suite"] == "c1" and r["case"] == case and r.get("ttft_s") is not None]
        out[case] = {
            "t0": round(st.median(t0), 2) if t0 else None,
            "t1": round(st.median(t1), 2) if t1 else None,
            "pooled": round(st.median(both), 2) if both else None,
            "min": round(min(both), 2) if both else None,
            "max": round(max(both), 2) if both else None,
            "ttft_s": round(st.median(ttfts), 3) if ttfts else None,
        }
    c16 = [r["decode_tok_s"] for r in rows if r["suite"] == "conc" and r["case"] == "c16"]
    out["c16"] = round(st.median(c16), 2) if c16 else None
    prefill = {}
    for case in sorted({r["case"] for r in rows if r["suite"] == "prefill"}):
        sub = [r for r in rows if r["suite"] == "prefill" and r["case"] == case]
        prefill[case] = {
            "prompt_tokens": sub[0]["prompt_tokens"],
            "prefill_tok_s": round(st.median([x["prefill_tok_s"] for x in sub]), 1),
            "ttft_s": round(st.median([x["ttft_s"] for x in sub]), 3),
        }
    out["prefill"] = prefill
    return out


def gsm8k_summary(filename: str | None, fallback: tuple[int, int] | None = None) -> dict:
    if filename and (R / filename).is_file():
        rs = [json.loads(line) for line in open(R / filename, encoding="utf-8") if line.strip()]
        ok = sum(1 for r in rs if r["correct"])
        wrong = sorted(r["idx"] for r in rs if not r["correct"])
        return {"ok": ok, "n": len(rs), "pct": round(100.0 * ok / len(rs), 1), "wrong": wrong}
    if fallback:
        ok, n = fallback
        return {"ok": ok, "n": n, "pct": round(100.0 * ok / n, 1), "wrong": []}
    return {"ok": None, "n": None, "pct": None, "wrong": []}


def gates_summary(filename: str | None) -> dict | None:
    if not filename or not (R / filename).is_file():
        return None
    return json.loads((R / filename).read_text(encoding="utf-8"))["gates"]


def spec_metrics(filename: str | None) -> dict | None:
    if not filename or not (R / filename).is_file():
        return None
    m: dict[str, float] = {}
    for line in open(R / filename, encoding="utf-8"):
        k = re.match(r"vllm:(\w+)\{([^}]*)\} ([0-9.e+-]+)", line)
        if k and not k.group(1).endswith("_created"):
            pos = re.search(r'position="(\d+)"', k.group(2))
            key = k.group(1) + (f"[{pos.group(1)}]" if pos else "")
            m[key] = float(k.group(3))
    drafts = int(m.get("spec_decode_num_drafts_total", 0))
    draft_toks = int(m.get("spec_decode_num_draft_tokens_total", 0))
    accepted = int(m.get("spec_decode_num_accepted_tokens_total", 0))
    cond_pos = []
    for i in range(7):
        acc_i = m.get(f"spec_decode_num_accepted_tokens_per_pos_total[{i}]")
        ver_i = m.get(f"spec_decode_num_verified_tokens_per_pos_total[{i}]", drafts)
        if acc_i is not None and ver_i and ver_i > 0:
            cond_pos.append(round(acc_i / ver_i, 3))
    return {
        "drafts": drafts,
        "draft_tokens": draft_toks,
        "accepted_tokens": accepted,
        "accept_rate_pct": round(100.0 * accepted / draft_toks, 2) if draft_toks else None,
        "accepted_per_step": round(accepted / drafts, 3) if drafts else None,
        "emitted_per_step": round(1.0 + accepted / drafts, 3) if drafts else None,
        "cond_pos": cond_pos,
    }


def phase_profile(filename: str) -> dict:
    return json.loads((R / filename).read_text(encoding="utf-8"))["median"]


def scoreboard() -> list[dict]:
    by_name = ledger_by_name()
    out: list[dict] = []
    base_c1 = None
    base_c16 = None
    for spec in ROUNDS:
        runs = [r for name in spec["ledger_runs"] for r in by_name.get(name, [])]
        peaks = [r for name in spec["peak_runs"] for r in by_name.get(name, [])]
        c1_vals = [r["c1_decode"] for r in runs if r.get("c1_decode") is not None]
        c1_peaks = [r["c1_decode"] for r in peaks if r.get("c1_decode") is not None]
        c16_vals = [r["c16_aggregate"] for r in runs if r.get("c16_aggregate") is not None]
        c16_peaks = [r["c16_aggregate"] for r in peaks if r.get("c16_aggregate") is not None]
        p4k_vals = [r["prefill_4096"] for r in runs if r.get("prefill_4096") is not None]
        p32k_vals = [r["prefill_32768"] for r in runs if r.get("prefill_32768") is not None]
        ttft_vals = [r["ttft_short_ms"] for r in runs if r.get("ttft_short_ms") is not None]
        start_vals = [r["startup_s"] for r in runs if r.get("startup_s") is not None]
        best_c1_run = max(runs, key=lambda x: x.get("c1_decode", 0))

        c1_med = round(st.median(c1_vals), 2)
        c1_pk = round(max(c1_peaks), 2)
        c16_med = round(st.median(c16_vals), 2)
        c16_pk = round(max(c16_peaks), 2)
        if base_c1 is None:
            base_c1, base_c16 = c1_med, c16_med

        l131 = by_name[spec["long_131k_run"]][0]["long"] if spec.get("long_131k_run") else None
        l231 = by_name[spec["long_231k_run"]][0]["long"] if spec.get("long_231k_run") else None
        gates = gates_summary(spec.get("gates_file"))
        soak = gates["5_concurrent_long_soak"] if gates and "5_concurrent_long_soak" in gates else None

        out.append(
            {
                **spec,
                "c1_med": c1_med,
                "c1_peak": c1_pk,
                "c1_gain_pct": round(100.0 * (c1_med / base_c1 - 1.0), 1),
                "c1_peak_gain_pct": round(100.0 * (c1_pk / base_c1 - 1.0), 1),
                "c1_all": [round(v, 1) for v in best_c1_run["c1_all"]],
                "c16_med": c16_med,
                "c16_peak": c16_pk,
                "c16_gain_pct": round(100.0 * (c16_med / base_c16 - 1.0), 1),
                "c16_peak_gain_pct": round(100.0 * (c16_pk / base_c16 - 1.0), 1),
                "prefill_4k": round(st.median(p4k_vals), 1),
                "prefill_4k_peak": round(max(r["prefill_4096"] for r in peaks if r.get("prefill_4096")), 1),
                "prefill_32k": round(st.median(p32k_vals), 1),
                "prefill_32k_peak": round(max(r["prefill_32768"] for r in peaks if r.get("prefill_32768")), 1),
                "ttft_ms": round(st.median(ttft_vals), 1),
                "ttft_best_ms": round(min(ttft_vals), 1),
                "startup_s": round(by_name[spec["cold_startup_run"]][0]["startup_s"], 1)
                if spec.get("cold_startup_run") == "ctl-00-cold"
                else round(st.median(start_vals), 1),
                "startup_best_s": round(min(start_vals), 1),
                "long_131k": l131,
                "long_231k": l231,
                "soak": soak,
                "gates": gates,
                "pr90": pr90_summary(spec.get("pr90_file")),
                "gsm8k": gsm8k_summary(spec.get("gsm8k_file"), spec.get("gsm8k_fallback")),
                "spec_metrics": spec_metrics(spec.get("metrics_file")),
            }
        )
    return out


def write_summary_csv(out_path: Path | None = None) -> Path:
    dest = out_path or (R / "summary.csv")
    sb = scoreboard()
    with dest.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(
            [
                "round_id",
                "label",
                "c1_median_tok_s",
                "c1_peak_tok_s",
                "c16_median_tok_s",
                "c16_peak_tok_s",
                "prefill_4k_tok_s",
                "prefill_32k_tok_s",
                "ttft_short_ms",
                "startup_s",
                "pr90_code_pooled_tok_s",
                "pr90_code_t0_tok_s",
                "pr90_prose_pooled_tok_s",
                "pr90_struct_pooled_tok_s",
                "pr90_c16_tok_s",
                "long_131k_prefill_tok_s",
                "long_131k_decode_tok_s",
                "long_131k_needles",
                "long_231k_prefill_tok_s",
                "long_231k_decode_tok_s",
                "long_231k_needles",
                "soak_2x168k_kv_tokens",
                "gsm8k_score",
            ]
        )
        for r in sb:
            p = r["pr90"]
            w.writerow(
                [
                    r["id"],
                    r["short"],
                    r["c1_med"],
                    r["c1_peak"],
                    r["c16_med"],
                    r["c16_peak"],
                    r["prefill_4k"],
                    r["prefill_32k"],
                    r["ttft_ms"],
                    r["startup_s"],
                    p.get("code", {}).get("pooled", ""),
                    p.get("code", {}).get("t0", ""),
                    p.get("prose", {}).get("pooled", ""),
                    p.get("struct", {}).get("pooled", ""),
                    p.get("c16") or "",
                    (r["long_131k"] or {}).get("prefill", ""),
                    (r["long_131k"] or {}).get("decode", ""),
                    (r["long_131k"] or {}).get("retrieval", ""),
                    (r["long_231k"] or {}).get("prefill", ""),
                    (r["long_231k"] or {}).get("decode", ""),
                    (r["long_231k"] or {}).get("retrieval", ""),
                    (r["soak"] or {}).get("total_prompt_tokens", ""),
                    f"{r['gsm8k']['ok']}/{r['gsm8k']['n']}" if r["gsm8k"]["ok"] is not None else "",
                ]
            )
    return dest


if __name__ == "__main__":
    p = write_summary_csv()
    print("Wrote", p)
