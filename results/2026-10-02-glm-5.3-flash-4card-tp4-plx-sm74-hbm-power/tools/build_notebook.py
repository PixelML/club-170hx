#!/usr/bin/env python3
"""Build notebooks/<NAME>.ipynb for the 74-SM / HBM / power-cap follow-up. Every number is read from receipts."""
import hashlib
import json
import os

REPO = os.environ.get("REPO", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
EXP = "2026-10-02-glm-5.3-flash-4card-tp4-plx-sm74-hbm-power"
PRIOR = "2026-10-01-glm-5.3-flash-morrowmake-1.6.0-4card-tp4-plx"
NAME = "2026-10-02-glm-5.3-flash-4card-tp4-sm74-hbm-power-vllm"


def _cid(prefix, src):
    return prefix + hashlib.sha1(src.encode()).hexdigest()[:8]


def md(src): return {"cell_type": "markdown", "id": _cid("m", src), "metadata": {}, "source": src.splitlines(keepends=True)}
def code(src): return {"cell_type": "code", "id": _cid("c", src), "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}


cells = []
cells.append(md(f"""# GLM-5.3-Flash TP4 on 4× CMP 170HX — 74 SMs, equalised HBM clock, power-cap sweep

| Metric | Value |
|---|---:|
| Decode, 1 user, structured / code / prose (150 W, HBM 1,728 MHz on all cards) | **418.0 / 309.9 / 214.1 tok/s** |
| Decode, 8 users aggregate, structured (150 W) | **816.5 tok/s** |
| Same at the new 140 W default | 412.7 / 305.9 / 211.6 · 814.2 tok/s |
| Best tokens per watt (8 users, GPU boards only) | 1.62 tok/s/W at 110 W (vs 1.40 at 150 W) |
| Prefill / TTFT | untested in this run (see the 2026-10-01 notebook) |

![decode vs power cap](../assets/charts/{NAME}.png)

```bash
git clone https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe && cd glm53-flash-cmp170hx-recipe && git checkout v1.6.0
```

Evidence: [`results/{EXP}/`](../results/{EXP}/README.md) · card background: [CMP 170HX understanding](../docs/CMP170HX-UNDERSTANDING.md)"""))

cells.append(code(f"""# --- Status cell ---
import os
EXPERIMENT = "{EXP}"
RESULTS_DIR = "../results/" + EXPERIMENT
RECEIPTS = RESULTS_DIR + "/receipts"
PRIOR = "../results/{PRIOR}/receipts"   # 70-SM, 180 W run from 2026-10-01
LIVE = False  # True sends the final request to the endpoint in GLM_URL (an OpenAI-compatible /v1 base)
ENDPOINT_ENV_VAR = "GLM_URL"
print("LIVE =", LIVE, "| receipts:", RECEIPTS)"""))

cells.append(code("""# --- Helpers: receipts, tables, chart style ---
import json, csv, statistics
from IPython.display import display, Markdown
import matplotlib
import matplotlib.pyplot as plt

def receipt(base, *parts):
    with open("/".join([base, *parts])) as f:
        return json.load(f)

def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---:" if i else "---" for i in range(len(header))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    display(Markdown("\\n".join(out)))

S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
matplotlib.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 2,
})

def label_bars(ax, bars, fmt="{:.0f}"):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(b.get_height()), va="bottom", ha="center", color=INK2, fontsize=7)

PROMPTS = ("structured", "coding", "prose")

def matrix(base, run):
    d = {}
    for c in (1, 8):
        for p in PROMPTS:
            r = receipt(base, run, f"decode-{p}-c{c}.json")
            d[(p, c)] = {"tok": r["tok_s_median"] if c == 1 else r["aggregate_tok_s_median"],
                         "step": r["decode_ms_per_draft_step_median"], "acc": r["accepted_per_step_median"]}
    return d

def num(x):
    return float("".join(ch for ch in x if ch.isdigit() or ch == ".") or 0)

def telemetry(run):
    rows = [r for r in csv.reader(open(f"{RECEIPTS}/{run}/telemetry.csv"), skipinitialspace=True) if len(r) >= 8 and r[1].strip().isdigit()]
    busy = [r for r in rows if num(r[7]) > 50]
    out = {"busy_w": 0.0, "core": {}, "hbm": {}, "sm_p50": 0}
    for i in range(4):
        p = [num(r[4]) for r in busy if int(r[1]) == i]
        out["busy_w"] += sum(p) / len(p) if p else 0
        out["core"][i] = max(num(r[2]) for r in rows if int(r[1]) == i)
        out["hbm"][i] = max(num(r[3]) for r in rows if int(r[1]) == i)
    clk = sorted(num(r[5]) for r in busy)
    out["sm_p50"] = clk[len(clk) // 2] if clk else 0
    return out"""))

cells.append(md("""## 1. TL;DR

- **Measured:** two of the four cards ship a VBIOS (`92.00.67.00.01`) that runs HBM at NDIV 54 = 1,458 MHz; the other two (`92.00.6D.00.0A`) run NDIV 64 = 1,728 MHz, same board part number, same timings. TP4 waits for the slowest card. After a hot 170tune gate (12/12 sweeps, peak HBM 75 °C) the two slow cards run NDIV 64: decode step 19.5 → 18.4 ms, single-user +5.8 to +6.3%, eight-user +3.1 to +3.8%, same draft acceptance.
- **Measured:** 70 → 74 SMs (cmpunlocker `6c442ee`) shows no clear decode gain; the comparison crosses a power-cap change (180 W → 150 W), so it is not a clean A/B.
- **Measured:** power-cap sweep on the live server. 140 W keeps 98.7–99.7% of 150 W throughput for 8% less GPU power; 110 W gives the most tokens per watt (1.62 vs 1.40) at 13–22% lower throughput; 100 W loses on both; 165 W adds 0.1–1.1% and runs HBM at 82–83 °C. The host's default is now 140 W.
- **Measured:** sm80vllm copy drafts: +37% on an edit reply that repeats the prompt (byte-identical), +28% on a second edit task (reply differs), −0.3 to −2.2% on general decode.
- **Measured:** SM VF offset +200 at a 1,410 MHz ceiling, 140 W: 415.0 / 308.3 / 213.1 tok/s single-user and 816.3 at eight users vs 412.7 / 305.9 / 211.6 and 814.2 without it (+0.6%), 6 W less GPU power: within noise. The offset only applies on the two cards with the 300 W VBIOS; on the 250 W VBIOS cards NVML exposes a VF offset range of [0..0] and silently refuses it."""))

cells.append(code("""pins = {
    "recipe": "Morrowmake/glm53-flash-cmp170hx-recipe @ a242b4fc53724bc5216f416fbbfaf7bf318b9f2c (v1.6.0); --gpus all -> --device nvidia.com/gpu=all (LXC/CDI)",
    "engine image (stock runs)": "ghcr.io/morrowmake/vllm-cmp170hx@sha256:cc26c8abb63953a37c6c1861cadc9188e99c1cceab403600b5822740f3a82ae0 (Morrowmake/vllm-cmp170hx @ 3a2bf16dae8b97f5ff2c7e9bc5809d24545e6340; the recipe's default IMAGE)",
    "copy-drafts image": "ghcr.io/pixelml/sm80vllm@sha256:f0dbb483b000f85ac66adb6f190c11d1af395914df67a0d9b7e54dfe2258fa86 (tf-learnings-c93c274c8 = PixelML/sm80vllm @ c93c274c86543d513c67243859480aeec46bc951 overlay on @ 127c6f0761b2c81d2c0ba7e2d8a5cf71e3247903, image sha256:704cbb8841be7c99bd1d5d6a8136fc118ca22cfc85457ef4a669a07ef957e8ff); build recipe: PixelML/sm80vllm docker/cmp170hx/",
    "target": "canada-quant/GLM-5.3-Flash-W4A16-MTP @ 5723f4d02af36366c23ace8668866ca7775855c1",
    "drafter": "incoai/GLM-5.3-Flash-DFlash2 @ bf582e4eacc1810f76656d1811693ff6c6737d2a (CC BY-NC-ND 4.0, benchmark only)",
    "driver": "NVIDIA open 615.71.09 + PixelML/cmpunlocker tag cmp170hx-plx-p2p-2026-10-02 @ 6ca4da72f078553d37089ee741cd128aa7804504 (upstream 6c442ee +4 SM, PR #60 HBM PLMs, rebar-serialize, 4 static-BAR1 P2P patches); ./install.sh --profile=8gb --no-gen2-service --no-iommu",
    "HBM / SM tuning": "cachenetics/170tune @ 5eb4775063b6cc055607ee5affecd210b13f0156; build deps: CUDA nvcc + cudart/cublas/nvml dev headers (cuda-nvcc-13-2, cuda-cudart-dev-13-2, libcublas-dev-13-2, cuda-nvml-dev-13-2)",
    "host": "Proxmox VE 9.2.2, kernel 7.0.2-6-pve; server in a privileged LXC sharing the host driver",
    "topology": "TP4, P2P off + replicated embedding, 4 × Gen2 x16, 2 cards per PLX PEX 8747, one CPU socket",
}
table(["pin", "value"], pins.items())"""))

cells.append(md("""## 2. Results (measured)

Decode matrix: MiaAI-Lab `bench_decode.py` @ 943912cd, T=0, 400 tokens (the harness sends `enable_thinking: false`, but this GLM-5.3-Flash chat template has no such switch: reasoning streams in `content` and counts as output tokens), median of 5, at 1 and 8 users. Busy power = mean board power while utilisation > 50%, summed over four cards. Tokens per watt = eight-user structured aggregate ÷ busy power."""))

cells.append(code("""RUNS = [
    ("sm74-150w", "74 SM, mixed HBM", 150),
    ("sm74-hbm64-165w", "74 SM, HBM equal", 165),
    ("sm74-hbm64-150w", "74 SM, HBM equal", 150),
    ("sm74-hbm64-140w", "74 SM, HBM equal", 140),
    ("sm74-hbm64-130w", "74 SM, HBM equal", 130),
    ("sm74-hbm64-120w", "74 SM, HBM equal", 120),
    ("sm74-hbm64-110w", "74 SM, HBM equal", 110),
    ("sm74-hbm64-100w", "74 SM, HBM equal", 100),
]
M = {k: matrix(RECEIPTS, k) for k, _, _ in RUNS}
T = {k: telemetry(k) for k, _, _ in RUNS}
M["prior"] = matrix(PRIOR, "tp4-p2p-off-repembed")
rows = []
for k, lab, w in RUNS:
    m, t = M[k], T[k]
    rows.append([lab, f"{w} W", " / ".join(f"{m[(p, 1)]['tok']:.1f}" for p in PROMPTS), " / ".join(f"{m[(p, 8)]['tok']:.1f}" for p in PROMPTS),
                 f"{m[('structured', 1)]['step']:.2f}", f"{t['busy_w']:.0f}", f"{m[('structured', 8)]['tok'] / t['busy_w']:.2f}", f"{t['sm_p50']:.0f}"])
m = M["prior"]
rows.append(["*70 SM, mixed HBM (2026-10-01)*", "180 W", " / ".join(f"{m[(p, 1)]['tok']:.1f}" for p in PROMPTS), " / ".join(f"{m[(p, 8)]['tok']:.1f}" for p in PROMPTS), f"{m[('structured', 1)]['step']:.2f}", "—", "—", "—"])
table(["run", "cap", "1 user tok/s (struct / code / prose)", "8 users tok/s", "ms/step (1u struct)", "busy GPU W", "8u tok/s per W", "busy SM clock p50 (MHz)"], rows)
print("accepted per step, 1 user structured:", sorted({M[k][("structured", 1)]["acc"] for k, _, _ in RUNS} | {M["prior"][("structured", 1)]["acc"]}))"""))

cells.append(md("""### 2.1 Headline: decode and efficiency vs power cap

Left: single-user and eight-user structured decode per cap (HBM equalised). Right: eight-user tokens per watt. Exported to `assets/charts/` as the notebook's headline chart."""))

cells.append(code(f"""sweep = [r for r in RUNS if r[0].startswith("sm74-hbm64")]
caps = [w for _, _, w in sweep]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for c, col in ((1, S[0]), (8, S[1])):
    ys = [M[k][("structured", c)]["tok"] for k, _, _ in sweep]
    axes[0].plot(caps, ys, marker="o", color=col, label=f"{{c}} user{{'s' if c > 1 else ''}}")
    for x, y in zip(caps, ys):
        axes[0].text(x, y + 8, f"{{y:.0f}}", ha="center", color=INK2, fontsize=7)
axes[0].axvline(140, color=INK2, ls="--", lw=1); axes[0].text(140, 300, " 140 W default", color=INK2, fontsize=8)
axes[0].set_xlabel("power cap per card (W)"); axes[0].set_ylabel("structured decode (tokens per second)")
axes[0].set_title("Decode vs power cap (HBM 1,728 MHz on all cards)", loc="left"); axes[0].legend(fontsize=8, loc="center left"); axes[0].invert_xaxis()
eff = [M[k][("structured", 8)]["tok"] / T[k]["busy_w"] for k, _, _ in sweep]
bars = axes[1].bar([str(w) for w in caps], eff, color=[S[2] if e == max(eff) else S[0] for e in eff], width=0.6)
label_bars(axes[1], bars, "{{:.2f}}")
axes[1].set_xlabel("power cap per card (W)"); axes[1].set_ylabel("8-user tokens per second per GPU watt")
axes[1].set_title("Efficiency (GPU boards only; host power not included)", loc="left"); axes[1].set_ylim(1.0, 1.75)
fig.suptitle("GLM-5.3-Flash W4A16 + DFlash2, TP4, 4× CMP 170HX behind 2 PLX switches, 74 SMs", color=INK, x=0.01, ha="left")
fig.tight_layout()
os.makedirs("../assets/charts", exist_ok=True)
fig.savefig("../assets/charts/{NAME}.png", dpi=150, bbox_inches="tight"); fig.savefig("../assets/charts/{NAME}.svg", bbox_inches="tight")
plt.show()"""))

cells.append(md("""### 2.2 What the HBM clock and the extra SMs did (150 W, one change at a time)

Three runs: 70 SM / mixed HBM at 180 W (2026-10-01), 74 SM / mixed HBM at 150 W, 74 SM / equal HBM at 150 W. The first step also changes the power cap, so read it as context, not as an A/B."""))

cells.append(code("""steps = [("prior", "70 SM, mixed HBM, 180 W"), ("sm74-150w", "74 SM, mixed HBM, 150 W"), ("sm74-hbm64-150w", "74 SM, HBM equal, 150 W")]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.0))
for ax, c, title in ((axes[0], 1, "1 user (stream tok/s)"), (axes[1], 8, "8 users (aggregate tok/s)")):
    w = 0.26
    for i, (k, lab) in enumerate(steps):
        bars = ax.bar([j + (i - 1) * w for j in range(3)], [M[k][(p, c)]["tok"] for p in PROMPTS], w * 0.92, color=S[i], label=lab)
        label_bars(ax, bars)
    ax.set_xticks(range(3), PROMPTS); ax.set_title(title, loc="left"); ax.set_ylabel("tokens per second")
axes[0].legend(fontsize=8, loc="upper right")
fig.tight_layout(); plt.show()
base, new = M["sm74-150w"], M["sm74-hbm64-150w"]
table(["prompt", "users", "mixed HBM", "HBM equal", "change", "ms/step mixed → equal"],
      [[p, c, f"{base[(p, c)]['tok']:.1f}", f"{new[(p, c)]['tok']:.1f}", f"{(new[(p, c)]['tok'] / base[(p, c)]['tok'] - 1) * 100:+.1f}%",
        f"{base[(p, c)]['step']:.2f} → {new[(p, c)]['step']:.2f}"] for c in (1, 8) for p in PROMPTS])"""))

cells.append(md("""### 2.3 Step time vs core clock across the sweep

Below 140 W the busy core clock drops and the decode step lengthens with it; between 140 and 165 W the core clock barely moves and neither does decode."""))

cells.append(code("""fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
clk = [T[k]["sm_p50"] for k, _, _ in sweep]
axes[0].plot(caps, clk, marker="o", color=S[0])
for x, y in zip(caps, clk):
    axes[0].text(x, y + 10, f"{y:.0f}", ha="center", color=INK2, fontsize=7)
axes[0].set_xlabel("power cap per card (W)"); axes[0].set_ylabel("median busy SM clock (MHz)"); axes[0].invert_xaxis()
axes[0].set_title("Core clock under load", loc="left")
st = [M[k][("structured", 1)]["step"] for k, _, _ in sweep]
axes[1].plot(caps, st, marker="o", color=S[1])
for x, y in zip(caps, st):
    axes[1].text(x, y + 0.2, f"{y:.1f}", ha="center", color=INK2, fontsize=7)
axes[1].set_xlabel("power cap per card (W)"); axes[1].set_ylabel("ms per decode step (1 user, structured)"); axes[1].invert_xaxis()
axes[1].set_title("Decode step time (lower is faster)", loc="left")
fig.tight_layout(); plt.show()"""))

cells.append(md("""### 2.4 Temperatures per run

Peak core and HBM temperature per card (1 s samples). Runs were back to back on a warm server, highest cap first, so heat carries over from one run to the next; read the trend, not single values. The 140 W run was the first of the sweep and started from an idle, cool server, which is why it reads cooler than the 130 W run after it. The 165 W run crossed this repository's 80 °C core stop rule on one card (flagged, kept)."""))

cells.append(code("""rows = []
for k, lab, w in RUNS:
    t = T[k]
    flag = "above 80 °C core rule" if max(t["core"].values()) >= 80 else ""
    rows.append([f"{lab}, {w} W", ", ".join(f"{t['core'][i]:.0f}" for i in range(4)), ", ".join(f"{t['hbm'][i]:.0f}" for i in range(4)), flag])
table(["run", "peak core °C (gpu0..3)", "peak HBM °C (gpu0..3)", "note"], rows)"""))

cells.append(md("""### 2.5 sm80vllm copy drafts (A/B, one boot each)

Image `pixelml/sm80vllm:tf-learnings-c93c274c8`, `VLLM_GLM5_COPY_DRAFTS=0/1`, 70-SM driver, mixed HBM, 150 W cap (run the evening before the SM and HBM changes). Copy drafts propose the next tokens from a matching span of the prompt or the reply so far; they only pay when the reply repeats text."""))

cells.append(code("""CD = {v: matrix(RECEIPTS, f"copy-drafts-{v}") for v in ("off", "on")}
ED = {v: receipt(RECEIPTS, f"copy-drafts-{v}", "edit.json") for v in ("off", "on")}
table(["copy drafts", "1 user struct / code / prose", "8 users struct / code / prose", "edit: rename tok/s (sha)", "edit: comment tok/s (sha)"],
      [[v, " / ".join(f"{CD[v][(p, 1)]['tok']:.1f}" for p in PROMPTS), " / ".join(f"{CD[v][(p, 8)]['tok']:.1f}" for p in PROMPTS),
        f"{ED[v]['rename']['tok_s_median']:.1f} ({ED[v]['rename']['shas'][0]})", f"{ED[v]['docstring']['tok_s_median']:.1f} ({ED[v]['docstring']['shas'][0]})"] for v in ("off", "on")])
labels = ["rename", "comment edit"] + [f"{p} 1u" for p in PROMPTS]
off = [ED["off"]["rename"]["tok_s_median"], ED["off"]["docstring"]["tok_s_median"]] + [CD["off"][(p, 1)]["tok"] for p in PROMPTS]
on = [ED["on"]["rename"]["tok_s_median"], ED["on"]["docstring"]["tok_s_median"]] + [CD["on"][(p, 1)]["tok"] for p in PROMPTS]
fig, ax = plt.subplots(figsize=(10, 3.8))
w = 0.38
label_bars(ax, ax.bar([i - w / 2 for i in range(5)], off, w * 0.92, color=S[0], label="copy drafts off"))
label_bars(ax, ax.bar([i + w / 2 for i in range(5)], on, w * 0.92, color=S[1], label="copy drafts on"))
ax.set_xticks(range(5), labels); ax.set_ylabel("decode tokens per second (1 user)"); ax.legend(fontsize=8)
ax.set_title("Copy drafts help replies that repeat the prompt; general decode is unchanged to −2%", loc="left")
fig.tight_layout(); plt.show()
print(open(f"{RECEIPTS}/copy-drafts-on/server-key-lines.txt").read().splitlines()[-1])"""))

cells.append(md("""### 2.6 SM VF offset (+200 MHz at a 1,410 MHz ceiling), 140 W

The offset shifts the voltage/frequency curve so a clock is reached at a lower voltage; the risk is silent wrong results when hot, so each card was gated hot first (`170tune -i N gate 200 1410 12`). All four gates reported a pass (12/12 sweeps + compute; peak HBM 75 / 76 / 70 / 76 °C; GPU 2 gated at `GATE_TEMP=70`, its serving range).

**Measured caveat:** on the two cards with the 250 W VBIOS (`92.00.67.00.01`, stock HBM NDIV 54) NVML reports the GPC VF offset range as [0..+0] MHz; `nvmlDeviceSetGpcClkVfOffset` returns *Unknown Error* and reads back +0. Their gate therefore ran at offset 0 (stock voltage, 1,410 MHz ceiling), and 170tune still reported a pass. Only the two 300 W VBIOS cards (`92.00.6D.00.0A`, range ±1000 MHz) take the undervolt, and only they keep it persisted.

The bench below is the 140 W matrix with the offset on GPU 0 and GPU 3 only, compared with the 140 W run without it."""))

cells.append(code("""def per_card(run):
    rows = [r for r in csv.reader(open(f"{RECEIPTS}/{run}/telemetry.csv"), skipinitialspace=True) if len(r) >= 8 and r[1].strip().isdigit()]
    out = []
    for i in range(4):
        busy = [r for r in rows if int(r[1]) == i and num(r[7]) > 50]
        clk = sorted(num(r[5]) for r in busy)
        out.append((sum(num(r[4]) for r in busy) / len(busy), clk[len(clk) // 2], max(num(r[3]) for r in rows if int(r[1]) == i)))
    return out
OFF = "sm74-hbm64-off200-140w"
M[OFF], T[OFF] = matrix(RECEIPTS, OFF), telemetry(OFF)
pair = [("sm74-hbm64-140w", "140 W, no offset"), (OFF, "140 W, +200 @ 1,410 (GPU 0/3)")]
table(["run", "1 user tok/s (struct / code / prose)", "8 users tok/s", "ms/step (1u struct)", "busy GPU W"],
      [[lab, " / ".join(f"{M[k][(p, 1)]['tok']:.1f}" for p in PROMPTS), " / ".join(f"{M[k][(p, 8)]['tok']:.1f}" for p in PROMPTS),
        f"{M[k][('structured', 1)]['step']:.2f}", f"{T[k]['busy_w']:.0f}"] for k, lab in pair])
a, b = M["sm74-hbm64-140w"][("structured", 1)]["tok"], M[OFF][("structured", 1)]["tok"]
print(f"structured 1-user change with the offset: {(b / a - 1) * 100:+.1f}%  |  busy GPU power {T[OFF]['busy_w'] - T['sm74-hbm64-140w']['busy_w']:+.0f} W")
table(["card", "busy power W (no offset → offset)", "busy SM clock p50 MHz (no offset → offset)", "peak HBM °C with offset"],
      [[f"gpu{i}", f"{x[0]:.0f} → {y[0]:.0f}", f"{x[1]:.0f} → {y[1]:.0f}", f"{y[2]:.0f}"] for i, (x, y) in enumerate(zip(per_card("sm74-hbm64-140w"), per_card(OFF)))])"""))

cells.append(md("""Reading: the two undervolted cards reach the 1,410 MHz ceiling at lower power, but the two cards that could not take the offset stay power-limited near 1,290 MHz, and TP4 runs at the slowest card's pace. Net: +0.6% and −6 W, within noise.

**Coherence probe (measured):** `coherence.json` in this run reads `coherent: false`. The failing item is the `cmp` probe ("is 9.9 greater than 9.11"): its token budget ended while the model was still writing its reasoning preamble, before the answer. A direct request with `max_tokens` 800 answered "9.9" twice (67 tokens each, `finish_reason=stop`). This is a probe-budget artifact, not output corruption."""))

print_probe = """c = receipt(RECEIPTS, OFF, "coherence.json")["coherence"]
for k in ("paris", "cmp", "sky"):
    print(f"{k:5}  ok={c[k]['ok']!s:5}  ...{c[k]['text'][-90:]!r}")"""
cells.append(code(print_probe))

cells.append(md(f"""## 3. Reproduce

Hardware: 4 × CMP 170HX exposing 65,536 MiB each, forced-air cooling, ~200 GB disk for weights. Every pin below is in the table in section 1.

```bash
# 1. Driver: NVIDIA open 615.71.09 + the exact patch stack (GPL-2.0 fork, tagged)
git clone https://github.com/PixelML/cmpunlocker && cd cmpunlocker
git checkout cmp170hx-plx-p2p-2026-10-02          # = 6ca4da72f078553d37089ee741cd128aa7804504
sudo ./install.sh --profile=8gb --no-gen2-service --no-iommu    # then cold-boot; expect 74 SM, 64 GB, Gen2 x16
# static-BAR1 P2P behind PLX switches is optional here (the served config runs P2P off): platform/plx-static-bar1/README.md

# 2. Serve: recipe at the pinned commit, complete .env from receipts/<run>/env.txt with the image pinned by digest
git clone https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe && cd glm53-flash-cmp170hx-recipe
git checkout a242b4fc53724bc5216f416fbbfaf7bf318b9f2c
cat > .env <<'EOF'
IMAGE=ghcr.io/morrowmake/vllm-cmp170hx@sha256:cc26c8abb63953a37c6c1861cadc9188e99c1cceab403600b5822740f3a82ae0
VLLM_GLM5_REPLICATED_EMBED=1
LAYOUT=tp4
MODELS_DIR=/path/to/models
SPEC_MODE=dflash
VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE=0
HOST=0.0.0.0
PORT=8030
EOF
./install.sh && ./download.sh && ./start.sh      # cold start ~13 min on this host

# 3. Tools: 170tune at the pinned commit (its gate needs CUDA nvcc; compiler packages only, no driver packages)
sudo apt-get install --no-install-recommends cuda-nvcc-13-2 cuda-cudart-dev-13-2 libcublas-dev-13-2 cuda-nvml-dev-13-2
git clone https://github.com/cachenetics/170tune && cd 170tune && git checkout 5eb4775063b6cc055607ee5affecd210b13f0156 && sudo ./install.sh

# 4. HBM: read each card's stock NDIV (x 27 MHz; 54 = 1,458 MHz, 64 = 1,728 MHz); gate + persist 64 on cards at 54
for i in 0 1 2 3; do sudo 170tune -i $i snapshot-stock; done
./stop.sh                                        # gates need the cards to themselves; one card at a time
sudo GATE_TEMP=75 GATE_SOAK_MAX=600 170tune -i 1 hbm-gate --ndiv 64 --sweeps 12
sudo 170tune -i 1 persist save --ndiv 64 && sudo 170tune -i 1 persist enable

# 5. SM offset (optional, measured within noise): check the range first; [0..0] means the VBIOS refuses it
sudo nvml_oc -i 0                                # "GPC clock VF offset allowed range" must not be [0 .. +0]
sudo GATE_TEMP=75 GATE_SOAK_MAX=600 170tune -i 0 gate 200 1410 12
sudo 170tune -i 0 persist save --offset 200 --clk 1410   # cards that also run NDIV 64: add --ndiv 64

# 6. Power cap (live, no restart) and benches
sudo nvidia-smi -pl 140                          # 170tune's recover/crash paths reset to 250 W: re-apply after
./start.sh && bash run_decode.sh                 # one cap; or: bash sweep.sh (150 140 130 120 110 100 W)

# 7. Copy-drafts A/B: the same .env with these two lines changed, one boot each
#    IMAGE=ghcr.io/pixelml/sm80vllm@sha256:f0dbb483b000f85ac66adb6f190c11d1af395914df67a0d9b7e54dfe2258fa86
#    VLLM_GLM5_COPY_DRAFTS=0   (then =1)  -> python3 bench_edit.py --runs 3 --out receipts/edit.json
```

Set `GATE_TEMP` to the card's real serving HBM peak: a card that never reaches the target waits out the full soak before every sweep. Harness: [`run_decode.sh`](../results/{EXP}/run_decode.sh), [`sweep.sh`](../results/{EXP}/sweep.sh), [`bench_edit.py`](../results/{EXP}/bench_edit.py). The sm80vllm images can also be rebuilt from `docker/cmp170hx/` in [PixelML/sm80vllm](https://github.com/PixelML/sm80vllm) (pinned CUDA base digest and full commits)."""))

cells.append(md("""## Try your own prompt

Edit `PROMPT`. The served template has no thinking switch (only `reasoning_effort` low/high/max, default max), so the reply starts with reasoning text. With `LIVE = False` this prints the recorded structured run at 150 W (HBM equalised) instead."""))

cells.append(code("""PROMPT = "Count from 1 to 200. Output only the numbers, separated by spaces. No other text."
if LIVE:
    import urllib.request
    url = os.environ[ENDPOINT_ENV_VAR]
    # No thinking switch: this GLM-5.3-Flash template has no enable_thinking; reasoning streams in content.
    body = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 400, "temperature": 0}
    req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    resp = json.load(urllib.request.urlopen(req, timeout=600))
    print(resp["choices"][0]["message"].get("content")); print(resp["usage"])
else:
    run = receipt(RECEIPTS, "sm74-hbm64-150w", "decode-structured-c1.json")["runs"][0]
    print(run["text_head"][:400]); print({k: run[k] for k in ("completion_tokens", "prompt_tokens", "tok_s", "ttft_s")})"""))

cells.append(md("""<details><summary>Appendix</summary>

- **nvidia-smi does not show the new HBM clock.** After `170tune` raises NDIV live, `nvidia-smi --query-gpu=clocks.mem` keeps reporting the value cached at driver load (1,458 MHz on the re-clocked cards). `170tune -i N status` reads the PLL. Telemetry in these receipts therefore shows 1,458 MHz for two cards in every run.
- **Gate temperature.** 170tune's default `GATE_TEMP` is 60 °C. This host serves at HBM 72–75 °C, so the HBM gates ran at 75 °C. The coolest card never reached 75 °C (or 72 °C) during the SM-offset gate and each sweep waited out the 600 s soak; it was re-run at 70 °C, its measured serving range.
- **Power-limit side effects.** 170tune soaks at 300 W and its `recover` / crash-revert paths set 250 W. A systemd drop-in re-applies the host cap after 170tune's services.
- **NVML silently refuses the offset on the 250 W VBIOS.** `nvidia-smi`/NVML report a GPC VF offset range of [0..0] on those cards, the set call fails, and 170tune's gate still passes because it gates whatever the card actually runs. Check `nvml_oc` (the range query) before trusting an offset gate.\n- **Not measured:** prefill and long-context at 74 SM / equal HBM, and quality suites.
- **Licences.** Recipe MIT; drafter CC BY-NC-ND 4.0 (benchmark only); MiaAI-Lab bench AGPL-3.0, referenced by commit, not vendored; `bench_edit.py` uses Python's own `textwrap.py` (PSF licence) as the edit input.

</details>"""))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
path = REPO + "/notebooks/" + NAME + ".ipynb"
with open(path, "w") as f:
    json.dump(nb, f, indent=1)
print("notebook written:", path, len(cells), "cells")
