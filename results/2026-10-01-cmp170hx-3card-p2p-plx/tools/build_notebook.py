#!/usr/bin/env python3
"""Build notebooks/2026-10-01-cmp170hx-3card-p2p-plx-cuda.ipynb from the receipts
in this directory. Run from the repository root, then execute the notebook:

  python3 results/2026-10-01-cmp170hx-3card-p2p-plx/tools/build_notebook.py
  jupyter nbconvert --to notebook --execute --inplace notebooks/2026-10-01-cmp170hx-3card-p2p-plx-cuda.ipynb
"""
import nbformat as nbf

EXP = "2026-10-01-cmp170hx-3card-p2p-plx"
NB = f"notebooks/{EXP}-cuda.ipynb"
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md(f"""# GPU-to-GPU copies on 3× CMP 170HX behind PLX switches (Gen2 x16) — no P2P, host-staged

| Metric | Value |
|---|---|
| Direct P2P | **not available** — `cuDeviceCanAccessPeer` = 0 on all 6 pairs (driver without BAR1-P2P patches) |
| Peer copy, 256 MiB | **6.2–6.3 GB/s** (driver's host-staged path; single-card D2H link = 6.68 GB/s) |
| Peer copy, 4 KiB | **22 µs** (D2H alone: 9.5 µs) |
| NCCL all-reduce, 256 MiB, 2 GPUs | **3.69 GB/s** same switch · **3.85 GB/s** across switches (bus bandwidth, SHM transport) |

![copy and all-reduce bandwidth](../assets/charts/{EXP}.png)

```bash
python3 results/{EXP}/tools/p2p_probe.py > p2p-copy.json
```

Evidence: [`results/{EXP}/`](../results/{EXP}/README.md)"""),
    code("""# --- Status cell -------------------------------------------------------
# LIVE = False replays the committed receipts under results/<experiment>/.
# LIVE = True re-runs tools/p2p_probe.py and tools/p2p_latency.py on this
# machine's GPUs (needs libcuda; no endpoint, no network).
import os

EXPERIMENT = "2026-10-01-cmp170hx-3card-p2p-plx"
RESULTS_DIR = os.path.join("..", "results", EXPERIMENT)
LIVE = False
print("LIVE =", LIVE, "| receipts:", RESULTS_DIR)"""),
    code("""import json, statistics, subprocess, sys
from IPython.display import display, Markdown


def load(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return json.load(f)


def load_jsonl(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return [json.loads(l) for l in f if l.strip()]


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    display(Markdown("\\n".join(out)))


if LIVE:
    copy = json.loads(subprocess.run([sys.executable, os.path.join(RESULTS_DIR, "tools/p2p_probe.py")], capture_output=True, text=True, check=True).stdout)
    lat = json.loads(subprocess.run([sys.executable, os.path.join(RESULTS_DIR, "tools/p2p_latency.py")], capture_output=True, text=True, check=True).stdout)
else:
    copy = load("p2p-copy.json")
    lat = load("p2p-latency.json")
nccl = load_jsonl("nccl-allreduce.jsonl")
env = load("environment.json")
print(len(copy["pairs"]), "ordered pairs,", len(lat["rows"]), "latency rows,", len(nccl), "all-reduce rows")"""),
    md("""## 1. TL;DR

**Measured.** On this host, with the upstream memory unlock and no P2P driver patches, the
three cards cannot map each other's memory: every `cuDeviceCanAccessPeer` is 0 and
`cuCtxEnablePeerAccess` returns `CUDA_ERROR_PEER_ACCESS_UNSUPPORTED` (217). Copies between
cards still work and are correct (alias-proof check below): the driver stages them through
host memory. At Gen2 x16 that staged path runs at **~94 % of the single-card link rate** for
large transfers, so a true P2P path could not add much bandwidth here — its value would be
latency on small messages (staged: 22 µs at 4 KiB, 340 µs at 1 MiB) and freeing host memory
bandwidth. Same-switch and cross-switch pairs measure the same, because nothing takes the
switch-local path.

NCCL picks its **SHM** transport for every pair. Two-GPU all-reduce reaches 3.7–3.8 GB/s bus
bandwidth at 256 MiB and costs 120–145 µs at 8 KiB — the number a tensor-parallel decode step
pays per all-reduce."""),
    code("""pins = [
    ("Cards", env["cards"]),
    ("Unlock / driver", env["unlock"]),
    ("P2P driver patches", env["p2p_patches"]),
    ("PCIe", env["pcie_link"]),
    ("Topology", env["topology"]),
    ("ACS", env["acs"]),
    ("Power limit (W)", ", ".join(map(str, env["power_limit_w"]))),
    ("Host", env["host"]),
    ("Copy tools", env["cuda_driver_api"]),
    ("NCCL runtime", env["nccl_runtime"]),
]
table(["Pin", "Value"], pins)"""),
    md("""## 2. Visible results

### Peer access and alias-proof copy (256 MiB, 10 timed repetitions)

The destination buffer is pre-filled with byte `0xFE`, a value that never occurs in the
source pattern. A pair passes only if, after `cuMemcpyPeer`, every sampled destination byte
equals the source — a mapping that silently aliased local memory would read back `0xFE`.
"Staged (serial)" is an explicit D2H-then-H2D loop through one pinned buffer, for reference."""),
    code("""rows = []
for p in copy["pairs"]:
    rows.append([f"GPU{p['src']} → GPU{p['dst']}", p["can_access_peer"], p["enable_peer_access_rc"],
                 "pass" if p["alias_proof_pass"] else "FAIL", p["peer_copy_GBps"], p["host_staged_GBps"]])
table(["Pair", "canAccessPeer", "enablePeerAccess rc", "Alias-proof copy", "cuMemcpyPeer (GB/s)", "Staged, serial (GB/s)"], rows)"""),
    md("### Copy time vs message size (median, µs; GB/s in brackets)"),
    code("""from collections import defaultdict
peer = defaultdict(list)
d2h = {}
for r in lat["rows"]:
    if r["kind"] == "peer":
        peer[r["bytes"]].append(r)
    else:
        d2h[r["bytes"]] = r


def human(n):
    return f"{n >> 20} MiB" if n >= 1 << 20 else f"{n >> 10} KiB"


rows = []
for size in sorted(peer):
    ps = peer[size]
    med = statistics.median(x["median_us"] for x in ps)
    same = statistics.median(x["median_us"] for x in ps if {x["src"], x["dst"]} == {0, 1})
    cross = statistics.median(x["median_us"] for x in ps if 2 in (x["src"], x["dst"]))
    rows.append([human(size), f"{d2h[size]['median_us']} ({d2h[size]['GBps']})", f"{round(same, 1)}", f"{round(cross, 1)}",
                 f"{round(size / med / 1e3, 2)}"])
table(["Message", "D2H, one card: µs (GB/s)", "Peer, same switch: µs", "Peer, cross switch: µs", "Peer, all pairs: GB/s"], rows)"""),
    md("### NCCL all-reduce (bf16 sum, median of 20, bus bandwidth = algbw × 2(n−1)/n)"),
    code("""groups = defaultdict(dict)
for r in nccl:
    groups[r["gpus"]][r["bytes"]] = r
sizes = sorted({r["bytes"] for r in nccl})
headers = ["Message"] + [f"{g}: µs / busbw GB/s" for g in groups]
rows = [[human(s)] + [f"{groups[g][s]['median_us']} / {groups[g][s]['busbw_GBps']}" for g in groups] for s in sizes]
table(headers, rows)
print("NCCL transport:", sorted({r["nccl_transport"] for r in nccl}))"""),
    code("""import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import Image

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
xs = sorted(peer)
a1.plot([x / 1024 for x in xs], [d2h[x]["GBps"] for x in xs], marker="o", label="D2H, one card (link reference)")
a1.plot([x / 1024 for x in xs], [statistics.median(p["GBps"] for p in peer[x]) for x in xs], marker="s", label="cuMemcpyPeer, host-staged")
a1.set_xscale("log", base=2); a1.set_xlabel("Message size (KiB, log2)"); a1.set_ylabel("Bandwidth (GB/s)")
a1.set_title("GPU-to-GPU copy, no P2P"); a1.grid(alpha=.3); a1.legend(fontsize=8)
for g in groups:
    a2.plot([s / 1024 for s in sizes], [groups[g][s]["busbw_GBps"] for s in sizes], marker="o", label=g)
a2.set_xscale("log", base=2); a2.set_xlabel("Message size (KiB, log2)"); a2.set_ylabel("Bus bandwidth (GB/s)")
a2.set_title("NCCL all-reduce (SHM transport)"); a2.grid(alpha=.3); a2.legend(fontsize=8)
fig.suptitle("3x CMP 170HX, Gen2 x16 behind PLX switches, measured 2026-10-01", fontsize=10)
fig.tight_layout()
chart = os.path.join("..", "assets", "charts", EXPERIMENT)
fig.savefig(chart + ".png", dpi=130); fig.savefig(chart + ".svg")
plt.close(fig)
display(Image(chart + ".png"))"""),
    md(f"""## 3. Reproduce

**Hardware.** Two or more memory-unlocked CMP 170HX, forced airflow, live temperature
monitoring (stop at 80 °C core). The copy tools need only the NVIDIA driver (`libcuda`); the
all-reduce needs a CUDA-enabled PyTorch.

```bash
# peer access + alias-proof copy + bandwidth, every ordered pair
python3 results/{EXP}/tools/p2p_probe.py > p2p-copy.json
# copy time vs message size (4 KiB .. 256 MiB)
python3 results/{EXP}/tools/p2p_latency.py > p2p-latency.json
# NCCL all-reduce, e.g. two cards
docker run --rm --gpus '"device=0,1"' --ipc=host -v $PWD/results/{EXP}/tools:/t \\
  --entrypoint torchrun ghcr.io/pixelml/club-170hx:vllm-glm53-sm80-pp-20260905 \\
  --nproc-per-node 2 /t/nccl_allreduce.py
```

Check the peer-access state and topology first:

```bash
nvidia-smi topo -m
nvidia-smi topo -p2p r     # GNS = not supported
```

### If several unlocked cards share a PCIe switch

The memory unlock resizes BAR1 to 64 GB on every card at driver load, from one probe thread
per GPU running at the same time. Cards behind a shared PLX switch then release and
re-assign the same bridge windows concurrently, which corrupted the kernel's resource tree on
this host (**measured:** crash in `__release_resource` under `pci_resize_resource`, slab
corruption, `Bad swap file entry`, random "BAR 1: no space" and cards failing to boot, within
15–90 s of every boot). [`patches/rebar-serialize.patch`](../results/{EXP}/patches/rebar-serialize.patch)
serializes the resize with one mutex; with it applied the host booted cleanly 9 times in a
row with all cards at Gen2 x16 and 64 GB. Hosts with one card per root port (no shared
switch) do not hit this."""),
    md("""## 4. Appendix

<details>
<summary>What was not tested, limitations, and next steps</summary>

- **Untested: true BAR1 P2P.** A community fork of the unlock (bayley/cmpunlocker) reports
  working BAR1 P2P on 4028GR-class hosts at 1.68 GB/s per direction on Gen2 **x4** links,
  with four driver patches, `RMForceStaticBar1=1;RMPcieP2PType=1`, and ACS redirect disabled.
  Those patches were not applied here, so this notebook is the no-P2P baseline for that A/B.
  Note that the staged path measured here (6.2 GB/s at x16) is already ~3.7× the fork's x4 P2P
  figure; the comparison that matters at x16 is small-message latency and all-reduce time.
- ACS request/completion redirect is enabled on the switch downstream ports (firmware
  default). It is irrelevant without P2P; with P2P it would force same-switch traffic through
  the root complex (the fork reports 1.20 vs 1.68 GB/s).
- One 256 MiB probe per pair, 10 timed repetitions; latency medians over 30 (≤16 MiB) or 8
  (larger) copies; all-reduce median of 20 after 5 warm-ups. No error bars beyond that.
- Power limit was the card default (250 W); copies are link-bound, not power-bound.
- Host topology details that are not needed to reproduce (bus addresses, serials, host
  names) are omitted by design.

</details>"""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
nbf.write(nb, NB)
print("wrote", NB)
