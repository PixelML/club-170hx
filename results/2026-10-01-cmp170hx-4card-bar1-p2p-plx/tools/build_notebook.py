#!/usr/bin/env python3
"""Build notebooks/2026-10-01-cmp170hx-4card-bar1-p2p-plx-cuda.ipynb from the receipts in this
directory. Run from the repository root, then execute the notebook:

  python3 results/2026-10-01-cmp170hx-4card-bar1-p2p-plx/tools/build_notebook.py
  jupyter nbconvert --to notebook --execute --inplace notebooks/2026-10-01-cmp170hx-4card-bar1-p2p-plx-cuda.ipynb
"""
import nbformat as nbf

EXP = "2026-10-01-cmp170hx-4card-bar1-p2p-plx"
PREV = "2026-10-01-cmp170hx-3card-p2p-plx"
NB = f"notebooks/{EXP}-cuda.ipynb"
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
CH = f"../assets/charts/{EXP}"

cells = [
    md(f"""# Static-BAR1 P2P on 4× CMP 170HX behind PLX switches — works on all 12 pairs; helps TP2, hurts TP4 decode

| Metric | Value |
|---|---|
| Direct P2P | **12/12 ordered pairs**, every byte verified, 5.79 GB/s each (bare metal and inside an LXC container) |
| Copy latency, P2P vs host-staged | **16 vs 22 µs** at 4 KiB · **25 vs 41 µs** at 64 KiB · **194 vs 343 µs** at 1 MiB |
| Large copies | 5.79 GB/s P2P vs 6.27 GB/s staged (P2P −8 %) |
| Qwen3.8-27B W4A16 TP2, 1 user | **80.0 vs 71.8 tok/s** (+11 %); with MTP k=3 **110.8 vs 97.6** (+14 %) |
| GLM-5.3-Flash TP4, 1 user structured | P2P on **293** vs off **396 tok/s** (−26 %) — leave P2P off for TP4 here |
| What it took | serialized BAR1 resize + a hand-programmed PCIe layout (BAR3 below each 64 GiB BAR1) adopted via kexec |

Earlier attempt on the same host, without P2P: [`{PREV}`](../results/{PREV}/README.md).
Evidence: [`results/{EXP}/`](../results/{EXP}/README.md)

![topology]({CH}-1-topology.png)"""),
    code(f"""# --- Status cell -------------------------------------------------------
# Replays the committed receipts under results/<experiment>/ and re-renders the charts.
# The measurements need the patched driver + layout described in section 3.
import os, sys, json, statistics, importlib.util
EXPERIMENT = "{EXP}"
RESULTS_DIR = os.path.join("..", "results", EXPERIMENT)
print("receipts:", RESULTS_DIR)"""),
    code("""from IPython.display import display, Markdown, Image


def load(*p):
    with open(os.path.join(RESULTS_DIR, "receipts", *p)) as f:
        return json.load(f)


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    display(Markdown("\\n".join(out)))


spec = importlib.util.spec_from_file_location("charts", os.path.join(RESULTS_DIR, "tools", "charts.py"))
charts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(charts)
env = json.load(open(os.path.join(RESULTS_DIR, "environment.json")))
table(["Item", "Value"], [(k, v) for k, v in env.items()])"""),
    md("""## 1. TL;DR

- **P2P works on all four cards.** `cuCtxEnablePeerAccess` succeeds on every pair and a keyed
  256 MiB copy lands byte-exact on all 12 ordered pairs, same-switch and across switches, on
  bare metal and from inside an LXC container sharing the host driver.
- **It is a latency win, not a bandwidth win.** Small and medium copies are 1.4–1.8× faster;
  above 16 MiB the host-staged path is slightly faster (6.27 vs 5.79 GB/s).
- **Workload result splits.** Qwen3.8-27B TP2 gains 11–14 % single-user decode (vLLM switches to
  its custom all-reduce). GLM-5.3-Flash TP4 *loses* 25–44 %: with P2P off vLLM uses a faster
  host-staged all-reduce on this PLX + Broadwell path, and matches the published P2P-off numbers.
- **Getting there** needed two fixes beyond the published P2P patches: a serialized BAR1 resize
  (parallel resizes behind one switch corrupted the kernel) and a hand-programmed PCIe layout,
  because the BIOS leaves no room for two 64 GiB BAR1s behind one switch."""),
    md(f"""## 2. Results (measured)

### 2.1 Why two cards behind one switch had no BAR1 — and the layout that fixes it

![address map]({CH}-2-address-map.png)

Each card needs a 64 GiB BAR1 for static-BAR1 P2P. The BIOS puts switch A's window at the very
top of CPU2's 1 TiB 64-bit aperture, so it cannot grow. A kernel remove/rescan sizes the window
for two 64 GiB BARs but places each card's 32 MiB BAR3 right after its BAR1, leaving the second
64 GiB-aligned slot 32 MiB short. Placing BAR3 *below* each BAR1 fits both."""),
    code("""for name in ("layout-before.txt", "layout-after.txt"):
    print("=" * 20, name)
    print(open(os.path.join(RESULTS_DIR, "receipts", name)).read())"""),
    md(f"""### 2.2 Copy bandwidth and latency: P2P vs host-staged

![bandwidth and latency]({CH}-3-bandwidth-latency.png)"""),
    code("""st, d2h = charts.latency("p2p-latency-stock.json")
pp, _ = charts.latency("p2p-latency-4card-peer.json")
nopeer, _ = charts.latency("p2p-latency-bar1.json")
table(["Size", "staged (stock) µs", "P2P driver, peer access off µs", "P2P µs", "D2H ref µs", "P2P speed-up"],
      [(charts.human(s), f"{st[s]:.1f}", f"{nopeer[s]:.1f}", f"{pp[s]:.1f}", f"{d2h[s]:.1f}", f"{st[s] / pp[s]:.2f}×") for s in sorted(st)])"""),
    md("""The middle column is the P2P-capable driver with peer access **not** enabled: it matches the stock
driver, so `cuMemcpyPeer` still stages through host memory until `cuCtxEnablePeerAccess` is called."""),
    md(f"""### 2.3 All 12 pairs, and same switch vs across switches

![pairs]({CH}-4-pairs-switch.png)"""),
    code("""print(open(os.path.join(RESULTS_DIR, "receipts", "p2p-verify-4card-host.txt")).read())"""),
    md(f"""### 2.4 NCCL all-reduce

![nccl]({CH}-5-nccl.png)

With P2P on, NCCL moves 2-GPU traffic over `P2P/CUMEM` instead of shared memory: same-switch
bus bandwidth rises from 3.7 to 4.8 GB/s. Three GPUs (one pair cross-switch) gain less and fall
behind SHM at large sizes."""),
    code("""rows = []
for name, lab in (("nccl-0,1.log", "2 GPU SHM"), ("nccl-p2p-0,1.log", "2 GPU P2P"), ("nccl-0,1,2.log", "3 GPU SHM"), ("nccl-p2p-0,1,2.log", "3 GPU P2P")):
    d = charts.nccl(name)
    for b in (8192, 1 << 20, 1 << 28):
        rows.append((lab, charts.human(b), f"{d[b]['median_us']:.0f}", f"{d[b]['busbw_GBps']:.2f}"))
table(["Run", "Size", "µs", "busbw GB/s"], rows)"""),
    md(f"""### 2.5 Qwen3.8-27B W4A16, TP2 (vLLM sm80, 180 W)

![qwen tp2]({CH}-6-qwen-tp2.png)

With P2P vLLM uses its custom all-reduce (`['CUSTOM', 'PYNCCL']`) instead of PyNCCL only.
Single-user decode gains 11 % (cross-switch) and 8 % (same switch); prefill moves ±2 %. Under
load the same-switch pair loses ground (−18 % at 16 users): that pair includes the hottest card,
so part of it may be thermal, not P2P."""),
    code("""rows = []
for lab, off, on in (("cross", "qwen-tp2-cross", "qwen-tp2-cross-p2p"), ("same", "qwen-tp2-same", "qwen-tp2-same-p2p"), ("cross + MTP3", "qwen-tp2-cross-mtp3", "qwen-tp2-cross-mtp3-p2p")):
    s0, c0 = charts.qwen(off)
    s1, c1 = charts.qwen(on)
    rows.append((lab, f"{s0['decode900'][0]:.1f} → {s1['decode900'][0]:.1f}", f"{s0['prefill_long'][1]:.0f} → {s1['prefill_long'][1]:.0f}",
                 f"{c0[16]:.0f} → {c1[16]:.0f}"))
table(["TP2 pair", "decode tok/s (no P2P → P2P)", "prefill tok/s", "aggregate @16"], rows)"""),
    md(f"""### 2.6 GLM-5.3-Flash TP4 (Morrowmake recipe 1.6.0, DFlash2, 180 W): P2P hurts here

![glm tp4]({CH}-7-glm-tp4.png)

![glm step]({CH}-8-glm-step.png)

Same drafter acceptance (6.69 tokens per step at one user) in every run, so the difference is
pure step time: **19.4 ms with P2P off vs 26.1 ms with P2P on**. TP4 decode issues about a hundred
small collectives per step; on this host the host-staged all-reduce vLLM selects without P2P is
faster than the BAR1 path through the PLX switches and the Broadwell root complex. With P2P off
the box matches the recipe's published P2P-off numbers (396 / 306 / 193 tok/s single user,
799 / 720 / 516 at eight users, vs 394 / 377 / 181 and 798 / 683 / 545). The recipe's P2P gain
(437 tok/s) was measured on cards on EPYC root ports without PLX switches. Replicated embedding
adds ~1 %; the `1stage` all-reduce is much worse. Full GLM write-up: separate notebook."""),
    code("""rows = []
for run, lab, _ in charts.glm_runs():
    r = [lab]
    for ph in ("structured-c1", "coding-c1", "prose-c1"):
        r.append(f"{charts.glm(run, ph)['tok_s_median']:.1f}")
    for ph in ("structured-c8", "coding-c8", "prose-c8"):
        r.append(f"{charts.glm(run, ph)['aggregate_tok_s_median']:.0f}")
    r.append(f"{charts.glm(run, 'structured-c1')['decode_ms_per_draft_step_median']:.1f}")
    rows.append(r)
table(["Config", "1u struct", "1u code", "1u prose", "8u struct", "8u code", "8u prose", "ms/step 1u"], rows)"""),
    md(f"""### 2.7 Where P2P pays

![summary]({CH}-9-summary.png)"""),
    code("""for f in charts.ALL:
    f()  # re-render every chart from the receipts
print("charts re-rendered:", len(charts.ALL))"""),
    md(f"""## 3. Reproduce

**Hardware.** Two or more memory-unlocked CMP 170HX at x16, forced airflow, live temperature
monitoring. The layout step is board-specific: it hard-codes this host's bus numbers.

1. Driver: upstream cmpunlocker + [`patches/rebar-serialize.patch`](../results/{EXP}/patches/rebar-serialize.patch)
   + the four BAR1-P2P patches from admunch888/cmpunlocker (`driver/patches/p2p-*.patch`), declared in
   the build's patch order and `common/constants.yaml`; install with `--no-iommu --no-gen2-service`.
2. Layout, with no driver bound: edit `CARDS`/`UPSTREAM` in [`tools/cmp-bar-layout.py`](../results/{EXP}/tools/cmp-bar-layout.py)
   for your board, `--dry-run`, then `--unlock`, then `kexec -l /boot/vmlinuz-$(uname -r) --initrd=/boot/initrd.img-$(uname -r) --reuse-cmdline && systemctl kexec`.
3. Load the driver with `NVreg_RegistryDwords="RMForceStaticBar1=1;RMPcieP2PType=1;..."` and check
   `nvidia-smi -q -d MEMORY` shows 65536 MiB BAR1 on every card.
4. Verify with a keyed copy test (not `canAccessPeer` alone): `p2p-verify.py` from admunch888/cmpunlocker,
   then [`tools/p2p_latency.py --peer`](../results/{EXP}/tools/p2p_latency.py).
5. Per-boot automation: [`tools/boot/`](../results/{EXP}/tools/boot/) (modprobe guard in the initramfs +
   a oneshot unit that programs the layout and kexecs once, falling back to non-P2P on any mismatch).

For TP4 GLM-5.3-Flash leave `VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE=0` on hosts like this one."""),
    md("""## 4. Appendix

<details>
<summary>Limits and open items</summary>

- Only the CPU2 switch pair was tested; the CPU1 switch board's link was down during these runs.
- The same-switch Qwen concurrency loss overlaps with the hottest card (GPU0 reached 82–85 °C in
  some runs); not separated from P2P effects.
- GLM numbers use the recipe's default DFlash2 drafter (CC BY-NC-ND 4.0) as a benchmark only; an
  Apache-2.0 drop-in drafter failed to load on this engine (KV page-size mismatch) and is open.
- `p2p-verify` outputs are console captures; the latency receipts are JSON from `tools/p2p_latency.py`.
</details>"""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, NB)
print("wrote", NB)
