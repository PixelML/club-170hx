# Build summary.json and the chart from the raw receipts in ../raw. Every number in the notebook comes from here.
import csv, json, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
RAW = os.path.join(RES, "raw")
CHART = os.path.join(RES, "..", "..", "assets", "charts", "2026-10-10-qwen3.8-flash-next-megakernel-1card.png")
PROMPTS = ["code", "chat", "story", "math"]


def gpu_stats(name, busy_util=50):
    rows = list(csv.reader(open(os.path.join(RAW, name))))[1:]
    num = lambda s: float(re.sub(r"[^0-9.]", "", s) or 0)
    busy = [r for r in rows if num(r[7]) >= busy_util]
    return {
        "samples": len(rows), "busy_samples": len(busy),
        "busy_mean_power_w": round(statistics.mean(num(r[1]) for r in busy), 1) if busy else None,
        "peak_power_w": max(num(r[1]) for r in rows),
        "power_limit_w": num(rows[0][2]),
        "peak_core_c": max(num(r[5]) for r in rows),
        "peak_mem_c": max(num(r[6]) for r in rows),
        "sm_clock_mhz_busy_median": statistics.median(num(r[3]) for r in busy) if busy else None,
        "mem_clock_mhz": num(rows[0][4]),
    }


# megakernel: 4 prompts x 3 reps, greedy, up to 512 tokens
mk = [json.loads(l) for l in open(os.path.join(RAW, "fnx_bench.jsonl"))]
mk_by = {p: [r["decode_tok_s"] for r in mk if r["name"] == p] for p in PROMPTS}
mk_gen = {p: [r["gen_tokens"] for r in mk if r["name"] == p][0] for p in PROMPTS}
mk_prefill = {p: statistics.median(r["prompt_tokens"] / r["prefill_s"] for r in mk if r["name"] == p) for p in PROMPTS}
load_s = float(re.search(r"loaded in ([0-9.]+) s", open(os.path.join(RAW, "fnx_bench.log")).read()).group(1))

# llama.cpp llama-server: 3 rounds of the same 4 prompts, rounds 1-2 warm
ll = {}
for p in PROMPTS:
    ll[p] = [json.load(open(os.path.join(RAW, f"ref512_r{r}", f"{p}.json")))["timings"]["predicted_per_second"] for r in range(3)]
ll_prefill = {p: statistics.median(json.load(open(os.path.join(RAW, f"ref512_r{r}", f"{p}.json")))["timings"]["prompt_per_second"]
                                   for r in (1, 2)) for p in PROMPTS}
bench_md = open(os.path.join(RAW, "llama-bench.md")).read()
pp512 = float(re.search(r"pp512 \|\s+([0-9.]+)", bench_md).group(1))
tg128 = float(re.search(r"tg128 \|\s+([0-9.]+)", bench_md).group(1))

# correctness: teacher-forced on llama.cpp's greedy output
score = {}
cur = None
for line in open(os.path.join(RAW, "score512.txt")):
    if line.startswith("== "):
        cur = line[3:].strip()
    m = re.match(r"score: (\d+) / (\d+)", line)
    if m:
        score[cur] = {"agree": int(m.group(1)), "total": int(m.group(2))}
margins = json.load(open(os.path.join(RAW, "margins.json")))
miss_gaps = [g for p in margins.values() for g in p["miss_logprob_gap"].values()]

# per-phase time of one decode step (profile run, block 0 timestamps at each grid barrier)
names = {1109: "PLE projections", 1122: "PLE pending combine", 1124: "PLE gate + conv", 1130: "HC mix pre (attn)",
         1132: "HC mix post (attn)", 1135: "GDN projections", 1137: "GDN core", 1141: "attention projections",
         1143: "attention core", 1146: "mixer out projection", 1150: "HC mix pre (ffn)", 1152: "HC mix post (ffn)",
         1154: "router + shared expert", 1156: "experts gate/up", 1158: "experts down", 1162: "head HC pre",
         1164: "head HC post", 1166: "lm_head + argmax"}
prof_txt = open(os.path.join(RAW, "profile.log")).read()
step_ms = float(re.search(r"profile over \d+ steps: ([0-9.]+) ms/step", prof_txt).group(1))
phases = []
for m in re.finditer(r"line\s+(\d+)\s+x(\d+)\s+([0-9.]+) us/step\s+\(([0-9.]+) us each\)", prof_txt):
    phases.append({"phase": names[int(m.group(1))], "count": int(m.group(2)), "us_per_step": float(m.group(3)), "us_each": float(m.group(4))})
barrier_us = float(re.search(r"barrier: ([0-9.]+) us each", open(os.path.join(RAW, "barrier.txt")).read()).group(1))

med = lambda v: round(statistics.median(v), 2)
summary = {
    "megakernel": {
        "decode_tok_s_median_by_prompt": {p: med(mk_by[p]) for p in PROMPTS},
        "decode_tok_s_all_runs": mk_by,
        "decode_tok_s_median_all": med([v for p in PROMPTS for v in mk_by[p]]),
        "gen_tokens": mk_gen,
        "prompt_tok_s_token_by_token": {p: round(v, 1) for p, v in mk_prefill.items()},
        "load_s_from_cache": load_s,
        "gpu": gpu_stats("fnx_gpu.csv"),
        "step_ms_profiled": step_ms,
        "grid_barrier_us": barrier_us,
        "barriers_per_step": sum(p["count"] for p in phases),
        "phases": phases,
    },
    "llamacpp": {
        "server_decode_tok_s_by_prompt_rounds": ll,
        "server_decode_tok_s_median_warm": {p: med(ll[p][1:]) for p in PROMPTS},
        "server_decode_tok_s_median_all_warm": med([v for p in PROMPTS for v in ll[p][1:]]),
        "server_prompt_tok_s_median_warm": {p: round(v, 1) for p, v in ll_prefill.items()},
        "llama_bench_pp512": pp512, "llama_bench_tg128": tg128,
        "gpu": gpu_stats("llama_gpu.csv"),
    },
    "correctness": {
        "teacher_forced": score,
        "agree_total": sum(s["agree"] for s in score.values()),
        "positions_total": sum(s["total"] for s in score.values()),
        "miss_logprob_gap_max": round(max(miss_gaps), 4),
        "miss_logprob_gap_median": round(statistics.median(miss_gaps), 4),
        "median_gap_all_positions": {p: round(v["median_gap_all"], 3) for p, v in margins.items()},
    },
}
summary["speedup_decode_median"] = round(summary["megakernel"]["decode_tok_s_median_all"] /
                                         summary["llamacpp"]["server_decode_tok_s_median_all_warm"], 3)
json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1)

if "--chart" in sys.argv:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.35]})
    x = range(len(PROMPTS))
    mkv = [summary["megakernel"]["decode_tok_s_median_by_prompt"][p] for p in PROMPTS]
    llv = [summary["llamacpp"]["server_decode_tok_s_median_warm"][p] for p in PROMPTS]
    a.bar([i - 0.2 for i in x], llv, 0.4, label="llama.cpp llama-server", color="#9aa5b1")
    a.bar([i + 0.2 for i in x], mkv, 0.4, label="megakernel (this work)", color="#2b6cb0")
    for i in x:
        a.text(i - 0.2, llv[i] + 1, f"{llv[i]:.1f}", ha="center", fontsize=9)
        a.text(i + 0.2, mkv[i] + 1, f"{mkv[i]:.1f}", ha="center", fontsize=9)
    a.set_xticks(list(x)); a.set_xticklabels(PROMPTS)
    a.set_ylabel("decode, batch 1, greedy (tok/s)"); a.set_ylim(0, max(mkv) * 1.35)
    a.set_title("Decode on 1x CMP 170HX, same GGUF")
    a.legend(loc="upper center", ncol=2, fontsize=9, frameon=False)
    a.spines[["top", "right"]].set_visible(False)
    agg = {}
    for p in phases:
        k = p["phase"].replace(" (attn)", "").replace(" (ffn)", "")
        agg[k] = agg.get(k, 0) + p["us_per_step"] / 1000
    items = sorted(agg.items(), key=lambda kv: kv[1])
    b.barh([k for k, _ in items], [v for _, v in items], color="#2b6cb0")
    for i, (_, v) in enumerate(items):
        b.text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=8)
    b.set_xlabel("ms per decode step"); b.tick_params(axis="y", labelsize=8)
    b.set_title(f"Where one {step_ms:.1f} ms step goes")
    b.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(CHART, dpi=150)
    print("chart", CHART)
print(json.dumps({k: summary[k] for k in ("speedup_decode_median",)}),
      summary["megakernel"]["decode_tok_s_median_all"], summary["llamacpp"]["server_decode_tok_s_median_all_warm"])
