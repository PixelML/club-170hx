#!/usr/bin/env python3
"""Render the nine charts of the 2026-10-01 4-card BAR1-P2P notebook from the receipts.

  python3 results/2026-10-01-cmp170hx-4card-bar1-p2p-plx/tools/charts.py   (from the repo root)

Writes assets/charts/<EXP>-<n>-<slug>.png and .svg. Colours: categorical slots 1-3 of the
reference palette (validated light-mode, all pairs); P2P = blue, host-staged = orange,
third series = aqua (always direct-labelled, it is below 3:1 on the surface).
"""
import collections
import json
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

EXP = "2026-10-01-cmp170hx-4card-bar1-p2p-plx"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.dirname(HERE)
REC = os.path.join(RES, "receipts")
OUT = os.path.join(RES, "..", "..", "assets", "charts")

P2P, STAGED, THIRD = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9.5,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.fontsize": 8.5, "font.size": 9, "lines.linewidth": 2, "lines.markersize": 5,
})


def load(*p):
    with open(os.path.join(REC, *p)) as f:
        return json.load(f)


def jsonl(*p):
    with open(os.path.join(REC, *p)) as f:
        return [json.loads(l) for l in f if l.strip().startswith("{")]


def save(fig, n, slug):
    os.makedirs(OUT, exist_ok=True)
    base = os.path.join(OUT, f"{EXP}-{n}-{slug}")
    fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
    fig.savefig(base + ".svg", bbox_inches="tight")
    plt.close(fig)
    return base + ".png"


def human(b):
    return f"{b >> 20} MiB" if b >= 1 << 20 else f"{b >> 10} KiB"


# ---------------------------------------------------------------- data
def latency(name, pairs=None):
    """median µs per size for peer rows (optionally a subset of pairs), and d2h."""
    d = load(name)
    peer, d2h = collections.defaultdict(list), {}
    for r in d["rows"]:
        if r["kind"] == "peer" and (pairs is None or (r["src"], r["dst"]) in pairs):
            if not (r["bytes"] == 4096 and r["median_us"] > 100):  # first-touch outlier on one pair
                peer[r["bytes"]].append(r["median_us"])
        elif r["kind"] == "d2h":
            d2h[r["bytes"]] = r["median_us"]
    us = {b: statistics.median(v) for b, v in sorted(peer.items())}
    return us, d2h


def nccl(name):
    return {r["bytes"]: r for r in jsonl("nccl", name)}


def qwen(run):
    cases = collections.defaultdict(list)
    for d in jsonl("qwen", run, "suite.jsonl"):
        if d.get("type") == "sample":
            cases[d["case"]].append(d)
    s = {k: (statistics.median(x["decode_tok_s"] for x in v), statistics.median(x["prefill_tok_s"] for x in v))
         for k, v in cases.items()}
    conc = {l["concurrency"]: l["aggregate_tok_s"] for l in load("qwen", run, "conc.json")["levels"]}
    return s, conc


def glm(run, phase):
    return load("glm", run, "receipts", f"decode-{phase}.json")


# ---------------------------------------------------------------- 1. topology
def chart_topology():
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    fig.subplots_adjust(wspace=0.12)
    for ax, mode in zip(axes, ("staged", "p2p")):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 7)
        ax.axis("off")
        box = lambda x, y, w, h, t, c=BASE, fc="white", fs=8.5: (  # noqa: E731
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", lw=1.2, ec=c, fc=fc)),
            ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, color=INK))
        box(3.2, 5.9, 3.6, 0.8, "CPU2 (Broadwell Xeon) + host DRAM")
        box(0.7, 3.9, 3.2, 0.8, "PLX PEX 8747 switch A")
        box(6.1, 3.9, 3.2, 0.8, "PLX PEX 8747 switch B")
        for i, x in enumerate((0.2, 2.4, 5.6, 7.8)):
            box(x, 1.4, 2.0, 0.9, f"GPU{i}\n64 GiB HBM2e", c=P2P if mode == "p2p" else STAGED, fc="#f4f8fd" if mode == "p2p" else "#fdf3ee")
        for x0, x1 in ((2.3, 4.4), (7.7, 5.6)):
            ax.plot([x0, x1 if x0 < 5 else x1], [4.7, 5.9], color=BASE, lw=3)
        for xs, xd in ((1.2, 1.6), (3.4, 3.0), (6.6, 7.0), (8.8, 8.4)):
            ax.plot([xs, xd], [2.3, 3.9], color=BASE, lw=3)
        ax.text(5.0, 5.35, "Gen3 x16", ha="center", fontsize=7.5, color=MUTED)
        ax.text(5.0, 3.0, "Gen2 x16 per card", ha="center", fontsize=7.5, color=MUTED)
        if mode == "staged":
            ax.annotate("", xy=(4.6, 5.95), xytext=(1.0, 2.35), arrowprops=dict(arrowstyle="-|>", color=STAGED, lw=1.8, connectionstyle="arc3,rad=-0.25"))
            ax.annotate("", xy=(3.2, 2.35), xytext=(5.4, 5.95), arrowprops=dict(arrowstyle="-|>", color=STAGED, lw=1.8, ls="--", connectionstyle="arc3,rad=-0.25"))
            ax.set_title("Stock driver: GPU→GPU copies staged through host memory", loc="left")
            ax.text(0.2, 0.5, "cuDeviceCanAccessPeer = 0 on all pairs · 2 PCIe trips + a DRAM bounce per copy", fontsize=8, color=INK2)
        else:
            ax.annotate("", xy=(2.9, 2.35), xytext=(1.3, 2.35), arrowprops=dict(arrowstyle="-|>", color=P2P, lw=2, connectionstyle="arc3,rad=-0.6"))
            ax.annotate("", xy=(6.3, 2.35), xytext=(1.5, 2.4), arrowprops=dict(arrowstyle="-|>", color=P2P, lw=2, connectionstyle="arc3,rad=-0.45"))
            ax.set_title("Static-BAR1 P2P: writes go straight into the peer's BAR1", loc="left")
            ax.text(0.2, 0.5, "same switch: turns inside the PLX · cross switch: up through the CPU root complex", fontsize=8, color=INK2)
    return save(fig, 1, "topology")


# ---------------------------------------------------------------- 2. address map
def chart_address_map():
    TB0 = 0x30000000000
    G = 1 << 30
    rows = [
        ("BIOS layout\n(stock)", [("A", 0x3ffe8000000, 224 << 20, STAGED, "switch A: 224 MiB\nat the top"),
                                   ("B", 0x31000000000, (64 << 30) + (32 << 20), P2P, "switch B: 1 card, 64 GiB")]),
        ("kernel re-layout\n(rescan, 64 GiB BAR1)", [("A", 0x31000000000, (128 << 30) + (64 << 20), STAGED, "A: 128 GiB + 64 MiB\n2nd BAR1 'no space'"),
                                                       ("B", 0x34000000000, (64 << 30) + (32 << 20), P2P, "B: 64 GiB")]),
        ("hand layout\n(setpci + kexec)", [("A", 0x30ffe000000, (128 << 30) + (64 << 20), P2P, "A: 2 × 64 GiB"),
                                            ("B", 0x34ffe000000, (128 << 30) + (64 << 20), P2P, "B: 2 × 64 GiB")]),
    ]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    for i, (label, wins) in enumerate(rows):
        y = 2 - i
        ax.add_patch(Rectangle((0, y - 0.22), 1024, 0.44, fc="#f0efec", ec="none"))
        for name, start, size, col, txt in wins:
            w = max(size / G, 5)
            x = min((start - TB0) / G, 1024 - w)
            ax.add_patch(Rectangle((x, y - 0.22), w, 0.44, fc=col, ec=SURFACE, lw=1.5))
            right = x > 900
            below = name == "B"
            ax.text(x + w if right else x, y - 0.27 if below else y + 0.27, txt.replace("\n", " · "),
                    va="top" if below else "bottom", ha="right" if right else "left", fontsize=7.5, color=INK2)
    ax.set_yticks([2, 1, 0], [r[0] for r in rows])
    ax.set_xlim(0, 1024)
    ax.set_ylim(-0.65, 2.75)
    ax.set_xlabel("offset in CPU2's 64-bit window (GiB, 0x300_0000_0000 + x)")
    ax.set_title("Where the switch windows sit in the 1 TiB aperture", loc="left")
    ax.grid(axis="y", visible=False)
    # zoom: one switch window, kernel order vs hand order
    ax2.set_xlim(0, 200)
    ax2.set_ylim(-0.7, 1.7)
    for y, parts, lab in (
        (1, [(0, 64, P2P, "BAR1 A1"), (64, 0.03, STAGED, "BAR3"), (128, 64, "#d0cfca", "BAR1 A2 needs 128→192")], "kernel: BAR3 after BAR1"),
        (0, [(-0.03, 0.03, STAGED, ""), (0, 64, P2P, "BAR1 A1"), (63.97, 0.03, STAGED, ""), (64, 64, P2P, "BAR1 A2")], "hand: BAR3 below each BAR1"),
    ):
        for x, w, c, t in parts:
            ax2.add_patch(Rectangle((x, y - 0.25), max(w, 1.2), 0.5, fc=c, ec=SURFACE, lw=1.5))
            if t:
                ax2.text(x + max(w, 1.2) / 2, y, t, ha="center", va="center", fontsize=7.5, color="white" if c == P2P else INK)
        ax2.text(0, y + 0.38, lab, fontsize=8, color=INK2)
    ax2.axvline(128 + 0.0625, color=STAGED, lw=1, ls=":")
    ax2.text(129, 1.55, "kernel window ends\n32 MiB short", fontsize=7, color=STAGED, va="top")
    ax2.set_yticks([])
    ax2.set_xlabel("GiB from the switch window base")
    ax2.set_title("Two 64 GiB BAR1s in one switch window", loc="left")
    ax2.grid(axis="y", visible=False)
    return save(fig, 2, "address-map")


# ---------------------------------------------------------------- 3+4. bandwidth & latency
def chart_bandwidth_latency():
    st, d2h = latency("p2p-latency-stock.json")
    pp, _ = latency("p2p-latency-4card-peer.json")
    sizes = sorted(st)
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, data, lab, col in ((a, st, "host-staged (stock driver)", STAGED), (a, pp, "direct P2P (static BAR1)", P2P), (a, d2h, "single-card D2H (reference)", THIRD)):
        ax.plot(sizes, [b_ / data[b_] / 1e3 for b_ in sizes], marker="o", color=col, label=lab)
    a.set_xscale("log", base=2)
    a.set_xticks(sizes, [human(s) for s in sizes], rotation=30)
    a.set_ylabel("GB/s (median)")
    a.set_title("Copy bandwidth vs message size", loc="left")
    a.legend(loc="upper left")
    a.annotate(f"P2P {sizes[-1] / pp[sizes[-1]] / 1e3:.2f} vs staged {sizes[-1] / st[sizes[-1]] / 1e3:.2f} GB/s\n(large copies: staged wins slightly)",
               xy=(sizes[-1], sizes[-1] / pp[sizes[-1]] / 1e3), xytext=(sizes[2], 2.0), fontsize=7.5, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED))
    for data, lab, col in ((st, "host-staged", STAGED), (pp, "direct P2P", P2P), (d2h, "D2H ref.", THIRD)):
        b.plot(sizes, [data[s] for s in sizes], marker="o", color=col, label=lab)
    b.set_xscale("log", base=2)
    b.set_yscale("log")
    b.set_xticks(sizes, [human(s) for s in sizes], rotation=30)
    b.set_ylabel("µs per copy (median, log)")
    b.set_title("Copy latency (log–log)", loc="left")
    b.legend(loc="upper left")
    ratio = [st[s] / pp[s] for s in sizes]
    bars = c.bar(range(len(sizes)), ratio, color=[P2P if r > 1 else STAGED for r in ratio], width=0.6)
    c.axhline(1, color=BASE, lw=1)
    for r_, rect in zip(ratio, bars):
        c.text(rect.get_x() + rect.get_width() / 2, r_ + 0.03, f"{r_:.2f}×", ha="center", fontsize=8, color=INK)
    c.set_xticks(range(len(sizes)), [human(s) for s in sizes], rotation=30)
    c.set_ylabel("staged time ÷ P2P time")
    c.set_title("P2P speed-up by size (>1 = P2P faster)", loc="left")
    return save(fig, 3, "bandwidth-latency")


# ---------------------------------------------------------------- 4. 12-pair matrix + switch split
def chart_pairs():
    pp, _ = latency("p2p-latency-4card-peer.json")
    d = load("p2p-latency-4card-peer.json")
    sw = {0: "A", 1: "A", 2: "B", 3: "B"}
    m = [[None] * 4 for _ in range(4)]
    for r in d["rows"]:
        if r["kind"] == "peer" and r["bytes"] == 1 << 28:
            m[r["src"]][r["dst"]] = r["GBps"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.4]})
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#1c5cab"])
    import numpy as np
    arr = np.array([[np.nan if v is None else v for v in row] for row in m])
    a.imshow(arr, cmap=cmap, vmin=5.0, vmax=6.0)
    for i in range(4):
        for j in range(4):
            a.text(j, i, "—" if i == j else f"{arr[i][j]:.2f}\nGB/s ✓", ha="center", va="center", fontsize=8, color="white" if i != j else MUTED)
    a.set_xticks(range(4), [f"GPU{j}\n(sw {sw[j]})" for j in range(4)])
    a.set_yticks(range(4), [f"GPU{i} (sw {sw[i]})" for i in range(4)])
    a.set_xlabel("destination")
    a.set_ylabel("source")
    a.grid(False)
    a.set_title("All 12 ordered pairs, 256 MiB copies\n(✓ = every byte verified on the host)", loc="left")
    sizes = [4096, 65536, 1 << 20, 1 << 24]
    groups = {"same switch (0↔1, 2↔3)": {(0, 1), (1, 0), (2, 3), (3, 2)},
              "cross switch (0/1 ↔ 2/3)": {(0, 2), (2, 0), (0, 3), (3, 0), (1, 2), (2, 1), (1, 3), (3, 1)}}
    w = 0.38
    for k, (lab, prs) in enumerate(groups.items()):
        us, _ = latency("p2p-latency-4card-peer.json", prs)
        vals = [us[s] for s in sizes]
        bars = b.bar([x + (k - 0.5) * w for x in range(len(sizes))], vals, w, color=(P2P, THIRD)[k], label=lab)
        for v, r in zip(vals, bars):
            b.text(r.get_x() + r.get_width() / 2, v * 1.04, f"{v:.0f}", ha="center", fontsize=7.5, color=INK)
    b.set_yscale("log")
    b.set_xticks(range(len(sizes)), [human(s) for s in sizes])
    b.set_ylabel("µs per copy (median, log)")
    b.set_title("Same switch vs across switches: no measurable difference", loc="left")
    b.legend(loc="upper left")
    return save(fig, 4, "pairs-switch")


# ---------------------------------------------------------------- 5. NCCL
def chart_nccl():
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.4))
    for name, lab, col, ls in (("nccl-0,1.log", "2 GPU same switch · SHM (no P2P)", STAGED, "-"),
                               ("nccl-p2p-0,1.log", "2 GPU same switch · P2P", P2P, "-"),
                               ("nccl-0,1,2.log", "3 GPU · SHM (no P2P)", STAGED, "--"),
                               ("nccl-p2p-0,1,2.log", "3 GPU · P2P", P2P, "--")):
        d = nccl(name)
        s = sorted(d)
        a.plot(s, [d[x]["busbw_GBps"] for x in s], marker="o", color=col, ls=ls, label=lab)
        b.plot(s, [d[x]["median_us"] for x in s], marker="o", color=col, ls=ls, label=lab)
    for ax in (a, b):
        ax.set_xscale("log", base=2)
        s = sorted(nccl("nccl-0,1.log"))
        ax.set_xticks(s, [human(x) for x in s], rotation=30)
    b.set_yscale("log")
    a.set_ylabel("bus bandwidth, GB/s")
    b.set_ylabel("µs per all-reduce (median, log)")
    a.set_title("NCCL all-reduce (bf16) bandwidth", loc="left")
    b.set_title("NCCL all-reduce latency", loc="left")
    a.legend(loc="upper left")
    return save(fig, 5, "nccl")


# ---------------------------------------------------------------- 6. Qwen TP2
def chart_qwen():
    runs = [("cross switch", "qwen-tp2-cross", "qwen-tp2-cross-p2p"),
            ("same switch", "qwen-tp2-same", "qwen-tp2-same-p2p"),
            ("cross + MTP k=3", "qwen-tp2-cross-mtp3", "qwen-tp2-cross-mtp3-p2p")]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4.4))
    w = 0.38
    for k, (state, col) in enumerate((("no P2P (NCCL over host)", STAGED), ("P2P (vLLM custom all-reduce)", P2P))):
        dec, pre = [], []
        for _, off, on in runs:
            s, _ = qwen((off, on)[k])
            dec.append(s["decode900"][0])
            pre.append(s["prefill_long"][1])
        for ax, vals in ((a, dec), (b, pre)):
            bars = ax.bar([x + (k - 0.5) * w for x in range(3)], vals, w, color=col, label=state)
            for v, r in zip(vals, bars):
                ax.text(r.get_x() + r.get_width() / 2, v * 1.01, f"{v:.0f}", ha="center", fontsize=7.5, color=INK)
    for ax in (a, b):
        ax.set_xticks(range(3), [r[0] for r in runs])
    a.set_ylabel("decode tok/s, 1 user, 900-token prompt")
    b.set_ylabel("prefill tok/s, 6.6k prompt")
    a.set_title("Qwen3.8-27B W4A16, TP2: decode", loc="left")
    b.set_title("TP2: prefill", loc="left")
    a.legend(loc="upper left")
    for off, on, lab, col in (("qwen-tp2-cross", "qwen-tp2-cross-p2p", "cross", P2P), ("qwen-tp2-same", "qwen-tp2-same-p2p", "same", THIRD)):
        _, c0 = qwen(off)
        _, c1 = qwen(on)
        cs = sorted(c0)
        c.plot(cs, [100 * (c1[x] / c0[x] - 1) for x in cs], marker="o", color=col, label=f"{lab} switch")
    c.axhline(0, color=BASE, lw=1)
    c.set_xticks([1, 4, 8, 16])
    c.set_xlabel("concurrent requests")
    c.set_ylabel("aggregate tok/s change with P2P, %")
    c.set_title("TP2 under load: P2P gain vs concurrency", loc="left")
    c.legend(loc="lower left")
    return save(fig, 6, "qwen-tp2")


# ---------------------------------------------------------------- 7. GLM TP4 tok/s
GLM_RUNS = [("tp4-p2p-off", "P2P off", STAGED), ("tp4-p2p-off-repembed", "P2P off + replicated embed", THIRD),
            ("tp4-p2p-on", "P2P on", P2P), ("tp4-p2p-on-repembed", "P2P on + replicated embed", "#86b6ef"),
            ("tp4-p2p-on-1stage", "P2P on + 1stage all-reduce", "#184f95")]


def glm_runs():
    return [r for r in GLM_RUNS if os.path.exists(os.path.join(REC, "glm", r[0], "receipts", "decode-structured-c1.json"))]


def chart_glm():
    runs = glm_runs()
    work = ("structured", "coding", "prose")
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4.6))
    w = 0.8 / len(runs)
    for k, (run, lab, col) in enumerate(runs):
        c1 = [glm(run, f"{x}-c1")["tok_s_median"] for x in work]
        c8 = [glm(run, f"{x}-c8")["aggregate_tok_s_median"] for x in work]
        for ax, vals in ((a, c1), (b, c8)):
            bars = ax.bar([x + (k - (len(runs) - 1) / 2) * w for x in range(3)], vals, w, color=col, label=lab, edgecolor=SURFACE, lw=1)
            for v, r in zip(vals, bars):
                ax.text(r.get_x() + r.get_width() / 2, v + 4, f"{v:.0f}", ha="center", fontsize=6.8, color=INK, rotation=90)
    for ax, t in ((a, "1 user, streaming tok/s"), (b, "8 users, aggregate tok/s")):
        ax.set_xticks(range(3), work)
        ax.set_ylabel(t)
    a.set_title("GLM-5.3-Flash TP4 (DFlash2, 180 W): P2P hurts decode here", loc="left")
    b.set_title("…and hurts more under load", loc="left")
    a.legend(loc="upper right", fontsize=7.5)
    return save(fig, 7, "glm-tp4")


# ---------------------------------------------------------------- 8. GLM per-step
def chart_glm_step():
    runs = glm_runs()
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 4.4))
    fig.subplots_adjust(wspace=0.75)
    labs = [r[1] for r in runs]
    for ax, phase, title in ((a, "structured-c1", "1 user"), (b, "structured-c8", "8 users")):
        vals = [glm(r[0], phase).get("decode_ms_per_draft_step_median") for r in runs]
        acc = [glm(r[0], phase).get("accepted_per_step_median") for r in runs]
        bars = ax.barh(range(len(runs)), vals, color=[r[2] for r in runs], height=0.6)
        for v, ac, r in zip(vals, acc, bars):
            ax.text(v + 0.3, r.get_y() + r.get_height() / 2, f"{v:.1f} ms · {ac:.2f} tok/step" if ac else f"{v:.1f} ms", va="center", fontsize=8, color=INK)
        ax.set_yticks(range(len(runs)), labs)
        ax.set_xlim(0, max(vals) * 1.55)
        ax.invert_yaxis()
        ax.set_xlabel("ms per draft-verify step (serving-cycle ratio)")
        ax.set_title(f"Structured prompt, {title}: cost per step", loc="left")
        ax.grid(axis="y", visible=False)
    return save(fig, 8, "glm-step")


# ---------------------------------------------------------------- 9. where P2P helps
def chart_summary():
    st, _ = latency("p2p-latency-stock.json")
    pp, _ = latency("p2p-latency-4card-peer.json")
    q0, qc0 = qwen("qwen-tp2-cross")
    q1, qc1 = qwen("qwen-tp2-cross-p2p")
    m0, _ = qwen("qwen-tp2-cross-mtp3")
    m1, _ = qwen("qwen-tp2-cross-mtp3-p2p")
    items = [
        ("copy 64 KiB (latency)", st[65536] / pp[65536] - 1),
        ("copy 1 MiB (latency)", st[1 << 20] / pp[1 << 20] - 1),
        ("copy 256 MiB (bandwidth)", st[1 << 28] / pp[1 << 28] - 1),
        ("Qwen TP2 decode, 1 user", q1["decode900"][0] / q0["decode900"][0] - 1),
        ("Qwen TP2 + MTP3 decode", m1["decode900"][0] / m0["decode900"][0] - 1),
        ("Qwen TP2 prefill 6.6k", q1["prefill_long"][1] / q0["prefill_long"][1] - 1),
        ("Qwen TP2 aggregate, 16 users (cross switch)", qc1[16] / qc0[16] - 1),
        ("Qwen TP2 aggregate, 16 users (same switch)", qwen("qwen-tp2-same-p2p")[1][16] / qwen("qwen-tp2-same")[1][16] - 1),
    ]
    if os.path.exists(os.path.join(REC, "glm", "tp4-p2p-on")):
        for ph, lab in (("structured-c1", "GLM TP4 structured, 1 user"), ("coding-c1", "GLM TP4 code, 1 user"),
                        ("structured-c8", "GLM TP4 structured, 8 users")):
            k = "aggregate_tok_s_median" if ph.endswith("c8") else "tok_s_median"
            items.append((lab, glm("tp4-p2p-on", ph)[k] / glm("tp4-p2p-off", ph)[k] - 1))
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(items) + 1.2))
    vals = [100 * v for _, v in items]
    ax.barh(range(len(items)), vals, color=[P2P if v >= 0 else STAGED for v in vals], height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + (1 if v >= 0 else -1), i, f"{v:+.0f}%", va="center", ha="left" if v >= 0 else "right", fontsize=8, color=INK)
    ax.axvline(0, color=BASE, lw=1)
    ax.set_yticks(range(len(items)), [l for l, _ in items])
    ax.invert_yaxis()
    ax.set_xlabel("change with P2P on (%; blue = P2P better, orange = worse)")
    ax.set_title("Where static-BAR1 P2P pays on a PLX + Broadwell host", loc="left")
    ax.grid(axis="y", visible=False)
    lo, hi = min(vals), max(vals)
    ax.set_xlim(min(lo * 1.25, -10), max(hi * 1.25, 10))
    return save(fig, 9, "summary")


ALL = [chart_topology, chart_address_map, chart_bandwidth_latency, chart_pairs, chart_nccl,
       chart_qwen, chart_glm, chart_glm_step, chart_summary]

if __name__ == "__main__":
    for f in ALL:
        print(f())
