# CMP 170HX card facts, PCIe fallback, and the W4A16 quant question

Research note. Read-only research; no configuration was changed to produce it.
Local measurements below came from the four-card test node on 2026-09-02.

## Decision box

**Use W4A16 via Intel AutoRound: yes, for the 2-card and single-node-tight
configurations. Keep FP8 for the 4-card default.**

- An INT4 AutoRound checkpoint of the base model already exists upstream:
  [`Intel/DeepSeek-V4-Flash-W4A16-AutoRound`](https://huggingface.co/Intel/DeepSeek-V4-Flash-W4A16-AutoRound),
  built with the RTN fast path (`--iters 0`), group size 128, vLLM support
  via [vllm-project/vllm#45645](https://github.com/vllm-project/vllm/pull/45645).
  This is our own DeepSeek-V4-Flash-Vision-Exp fork, so this artifact is a
  reference for method and expected size, not a drop-in weight file.
- Why W4A16: on this card the decode bottleneck is HBM2e bandwidth, not
  compute. vLLM's Marlin kernel dequantizes both FP8 and INT4 weights to
  BF16 before the GEMM on SM80, so the FP8 path already pays a dequant tax;
  INT4 halves the bytes moved from HBM per weight read, which is the actual
  lever on a bandwidth-bound part (Part 3).
- Where to run the quantization: not on a CMP 170HX (64 GiB, no P2P, slow
  PCIe makes CPU offload traffic painful) and not on the 128 GB DGX Spark
  GB10 unless `low_gpu_mem_usage` CPU-offload is used — a 168 GB FP8 source
  plus AutoRound's own working set will not fit in 128 GB unified memory
  without offload. Best fit: a multi-GPU box with ≥168 GB aggregate VRAM
  (matches the Intel recipe's 5x80GB pattern in their DeepSeek-V3 doc), or
  our own 4x64 GiB CMP rig with `device_map` spread across all 4 cards and
  `low_gpu_mem_usage=True`.
- ETA: Intel's own reference point is a 72B model in 37 minutes on one GPU
  in "light" iterative mode; for RTN (`--iters 0`) a 170 GB-class MoE is
  disk-I/O- and dequant-bound, not iteration-bound — order 1-3 hours end to
  end including calibration data load and shard packing. The 200-iteration
  default mode would cost meaningfully more (iterative SignSGD tuning per
  block) and is not required for the RTN-based DeepSeek-V4-Flash artifact
  that already ships.
- Runner-up: W8A16 (FP8 weight-only via Marlin, what we run today) for
  quality-sensitive routes, or a mixed scheme — INT4 routed experts +
  BF16/FP8 attention and shared layers, the same split
  `canada-quant/DeepSeek-V4-Flash-W4A16-FP8` uses — if MoE expert accuracy
  at 4-bit/group-128 turns out to regress specific benchmarks.

Measured current PCIe link state of our 4 cards (this VM, idle, 2026-09-02):

| GPU | Gen (cur/max) | Width (cur/max) | LnkCap (physical) |
|---|---|---|---|
| 0 | 1 / 1 | x8 / x16 (**downgraded**) | 2.5GT/s, x16 |
| 1 | 1 / 1 | x16 / x16 | 2.5GT/s, x16 |
| 2 | 1 / 1 | x16 / x16 | 2.5GT/s, x16 |
| 3 | 1 / 1 | x16 / x16 | 2.5GT/s, x16 |

All four report `Speed 2.5GT/s (ok)` — Gen1, not "downgraded" from a higher
capability; `LnkCap` itself caps at 2.5GT/s, so the guest's own PCIe
capability register, not just `LnkSta`, is Gen1. GPU 0 alone shows a real
width downgrade (x8 of x16 capable). See Part 2 for what this means and how
to check whether it is cosmetic.

---

## Part 1 — The card

### Silicon

The CMP 170HX ships on the **GA100-105F** die — the same Ampere GA100
silicon as the A100, cut down. Public teardown and benchmark data
(TechPowerUp, an arXiv survey of mining-GPU silicon, and independent
reviews) converge on:

- **70 SMs**, **4,480 CUDA cores**, **280 Tensor Cores** (not the
  108 SM / 6,912-core / 432-Tensor-Core config of the full A100 PCIe 80GB)
- **8 GB HBM2e** stock, 4096-bit bus, ~1,493-1,500 GB/s theoretical
  bandwidth (bus width matches a 4-stack HBM2e configuration)
- **250 W TDP**, 8-pin power only, **no display output**, **no NVLink**
- **PCIe interface stock: Gen1 x4** (some listings, and the CMP 170HX 10GB
  variant, are documented as Gen4-electrical-signaled but negotiated at
  x4 lane width; treat "PCIe 4.0 x4" spec-sheet listings as the connector's
  maximum capability, not what the mining SKU is strapped to run at)
- **No FP8/FP4 Tensor Core support** — FP8 arrived with Hopper; GA100 (A100
  and the CMP 170HX) tops out at FP16/BF16/TF32 Tensor Core math

Sources: [TechPowerUp CMP 170HX database entry](https://www.techpowerup.com/289310/nvidia-cmp-170hx-mining-card-tested-based-on-ga100-gpu-sku), [arXiv 2505.03782 — mining-GPU survey](https://arxiv.org/pdf/2505.03782), [topcpu.net CMP 170HX vs A100 PCIe 80GB comparison](https://www.topcpu.net/en/gpu-c/CMP-170HX-vs-A100-PCIe-80-GB), [niconiconi "All GB/s without FLOPS" teardown/review](https://niconiconi.neocities.org/tech-notes/nvidia-cmp-170hx-review/).

### Tensor throughput vs A100 — it is gated, not just smaller

The niconiconi review and the `170th-Street` community benchmark repo both
document that NVIDIA did not just disable SMs — it throttled the surviving
Tensor Cores at the instruction-issue level: a single MMA instruction has a
fixed 256-cycle latency that cannot be hidden by instruction-level
parallelism, and only 4 warps per SM may issue Tensor Core instructions
concurrently (an unrestricted A100 allows far more). Non-tensor FP32 FMA
throughput is separately throttled to ~0.39 TFLOPS. Measured FP16 compute
on one review landed around 42 TFLOPS — "RTX 2060 territory" — versus the
A100's Tensor Core FP16 ceiling of 312 TFLOPS dense (624 TFLOPS sparse).
Community reports after the compute-throughput unlock patch claim ~173
TFLOPS BF16, still well under A100 but a large jump from the stock gate.

Sources: [niconiconi review](https://niconiconi.neocities.org/tech-notes/nvidia-cmp-170hx-review/), [amoghmunikote/170th-Street fp16-performance.md](https://github.com/amoghmunikote/170th-street/blob/master/benchmarks-and-performance/fp16-performance.md), [DevQuasar "The almost A100"](https://devquasar.com/hardware/the-almost-a100-nvidia-cmp-170hx/), [NVIDIA A100 TF32 blog (TFLOPS reference table)](https://blogs.nvidia.com/blog/2020/03/24/tensorfloat-32-precision-format).

### Why it is cheap

The CMP 170HX was purpose-built for Ethereum mining (164 MH/s hash rate at
launch) with memory capacity deliberately capped near the Ethash DAG file
size (then under 5 GB), which is why 8 GB was "enough" for the intended
job. Ethereum's move to proof-of-stake stranded the entire CMP line; cards
that once sold for roughly $5,000 dropped to $250-500 on second-hand
markets, sold heavily out of China/Vietnam mining-farm liquidations. Prices
spiked again (reportedly $100 to over $1,000 overnight) after the 2026
unlock tools made the cards usable for AI inference.

Sources: [VideoCardz CMP 170HX spotting](https://videocardz.com/newz/nvidia-cmp-170hx-cryptomining-card-spotted-passive-design-offering-164-mh-s-hash-rate), [Tom's Hardware — unlock coverage](https://www.tomshardware.com/pc-components/gpus/nvidia-crypto-mining-gpus-hacked-to-restore-locked-away-vram-in-order-to-feed-ai-boom-software-mod-unlocks-64gb-of-vram-on-usd250-cmp-170hx), [WCCFTech price coverage](https://wccftech.com/nvidia-cmp-170hx-8-10-gb-prices-explode-over-1000-usd-as-tool-unlocks-hidden-64-80gb-vram/).

### The 64 GB memory unlock — firmware, not soldering

GA100 is a monolithic package: the full complement of HBM2e stacks is
physically present on every die, including the ones sold as 8 GB or
10 GB cards. NVIDIA disables capacity by OTP fuse and firmware/strap
configuration, not by omitting silicon. The unlock (`cmpunlocker`, by
Amogh Munikote) exploits a signature-loading bug in the Falcon security
microcontroller's BootROM to bypass those firmware locks and reprogram the
memory geometry straps, exposing 64 GB on an 8 GB card (40 GB on a 10 GB
card). It also restores gated SM compute throughput. This is a **firmware/
driver-patch mod**, not a hardware memory upgrade — no chips are added or
replaced. Persistence requires patched, DKMS-installed kernel modules (a
plain userspace patch does not survive reboot or driver reload).

Reliability caveat directly from the tooling's own docs: "software cannot
promise that the disabled HBM on your particular binned card is reliable.
Do not load a model merely because `nvidia-smi` says 65536 MiB." Field
reports (see below) put failure rates near 50% in one buyer group, with
failures appearing only once workloads exceed the original 8 GB mining
window — which is why club-170hx's HARDWARE.md and QC.md gate every card
on a large-memory stress test before accepting it, not just `nvidia-smi`
enumeration.

Sources: [amoghmunikote/cmpunlocker](https://github.com/amoghmunikote/cmpunlocker), [Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/nvidia-crypto-mining-gpus-hacked-to-restore-locked-away-vram-in-order-to-feed-ai-boom-software-mod-unlocks-64gb-of-vram-on-usd250-cmp-170hx), [Korben summary](https://korben.info/en/nvidia-cmp-170hx-vram-unlocked-firmware.html), local: `/home/ubuntu/WIP/repos/club-170hx/docs/HARDWARE.md`, `docs/QC.md`.

### The PCIe Gen/width mod — capacitor mod for width, vBIOS/strap for speed (unconfirmed)

The clearest documented mechanism is a **hardware capacitor mod for lane
width**, not a resistor strap: 12 of the CMP 170HX's 16 PCIe data lanes are
missing their 0402 AC-coupling capacitors on the PCB, which forces the link
to negotiate at x4 regardless of the slot. Soldering the missing
capacitors restores x16 width — first documented working by Amogh Munikote.
This only affects **width**, not **speed**: the result of the capacitor mod
alone is Gen1 x16 (~4 GB/s), not Gen2. A separate community fork
(`bendy2/cmpunlocker`) adds a software "Gen2" patch on top of the driver
patch set, targeting the PCIe generation register, but a GA100 vBIOS
strap-table analysis ([JRex286 gist](https://gist.github.com/JRex286/84cd3921788d2ffbc1e9bf8b6f2c9396))
found the CMP 170HX's vBIOS bus-init tables are inside the
signature/MAC-verified region, meaning a naive vBIOS byte edit for speed
will fail signature checks; and one fork's own docs state Gen3/Gen4 are
blocked by a real OTP silicon fuse (`FUSE_PCIE_GEN23_DIS`) that software
cannot clear. A `thaurock-x/CMP-170HX-64GB-Unlocked-VBIOS` repo claims a
one-shot 64 GB + x16 + higher-speed vBIOS, but its claims conflict with the
strap-table research and should be treated with the same skepticism the
gist author expresses — verify independently and keep an original vBIOS
backup before flashing.

**Our own cards**: club-170hx's INSTALLATION.md records that "the
experimental Gen2 path also caused guest/QEMU hangs and is excluded from
this baseline" — i.e. the Gen2 patch was tried and rejected as unstable in
our own environment, independent of the Proxmox link-speed question in
Part 2. Our sellers separately mod'd the cards to Gen2 x16 electrically
(per KNOWLEDGE.md, "x16 mod matters... seller modded Gen2 x16"), but what
the guest currently negotiates is Gen1 (measured above) — consistent with
either the guest-side Gen2 patch being disabled, or the Proxmox/QEMU
root-port cap described in Part 2.

Sources: [170th Street — PCIe Capacitor Mod](https://170th-street.gitbook.io/hx/modifications/pcie-capacitor-mod), [JRex286 GA100 vBIOS strap-table gist](https://gist.github.com/JRex286/84cd3921788d2ffbc1e9bf8b6f2c9396), [thaurock-x/CMP-170HX-64GB-Unlocked-VBIOS](https://github.com/thaurock-x/CMP-170HX-64GB-Unlocked-VBIOS), local: `/home/ubuntu/WIP/cmp170hx-knowledge/KNOWLEDGE.md`, `/home/ubuntu/WIP/repos/club-170hx/docs/INSTALLATION.md`.

### Measured HBM bandwidth on the 64 GB variant

The prompt's ~1,215 GB/s figure could not be traced to a specific,
citable Reddit post in this pass (the LocalLLaMA thread search returned
adjacent coverage — the unlock announcement, buyer-guide sites, and the
official `llama.cpp` CUDA-performance discussion thread — but not the
exact post with that number). It is directionally consistent with public
material: theoretical peak is ~1,493-1,500 GB/s, and one review reports the
610.43.03 driver raising achievable bandwidth from ~1.6 to ~1.8 TB/s by
tightening HBM timing (tRCD 18ns → 14ns) — note that figure is *above*
the commonly cited 1.5 TB/s spec figure and should be read as a
timing-tuned measurement, not the stock number. A measured ~1,215 GB/s
(about 80% of the ~1,500 GB/s theoretical peak) is a plausible real-world
STREAM-style result and in line with typical HBM2e efficiency, but treat it
as **unverified** pending the original post. Our own local benchmark notes
(`KNOWLEDGE.md` §3) log the same "independent field data" figure
third-hand ("Reddit r/LocalLLaMA, Aug 2026 ... ~1215 GB/s") without a
direct link — same caveat applies there.

Sources: [niconiconi review (driver bandwidth note)](https://niconiconi.neocities.org/tech-notes/nvidia-cmp-170hx-review/), [ggml-org/llama.cpp discussion #15013 (CMP 170HX entry)](https://github.com/ggml-org/llama.cpp/discussions/15013), local: `/home/ubuntu/WIP/cmp170hx-knowledge/KNOWLEDGE.md` §3.

### Community sources

- [amoghmunikote/cmpunlocker](https://github.com/amoghmunikote/cmpunlocker) — the primary unlock tool (~300 stars/156 forks at indexing), with forks adding multi-card support (`bendy2/cmpunlocker`) and an earlier Falcon-exploit approach with a persistence daemon (`d3dx9/cmpunlocker`).
- [170th-Street](https://170th-street.gitbook.io/hx/) — Munikote's community knowledge base for the card (mods, benchmarks).
- `Ithrial/ninfer-cmp170hx` and `allover326` repos named in the task brief were not found in this search pass under those exact names/orgs on GitHub or in web results — could not verify or cite; flag for a follow-up search with corrected spelling/org if these are known to exist internally.
- club-3090 / club-170hx — this repo's own sibling project pattern (community hardware playbooks for consumer/mining-surplus GPUs used for AI).
- [r/LocalLLaMA CMP 170HX discussion](https://github.com/ggml-org/llama.cpp/discussions/15013) (llama.cpp side of the same community conversation, directly citable) — treat any specific Reddit "1215 GB/s" post as unverified per above.

---

## Part 2 — PCIe: what the link speed actually costs

### (a) Weight load time for a 168 GB checkpoint

Using the per-direction figures given: Gen1 x16 ≈ 4 GB/s, Gen2 x16 ≈ 8 GB/s,
Gen4 x16 ≈ 32 GB/s. Splitting 168 GB evenly across 4 cards (42 GB/card) and
assuming host→device transfer is the bottleneck (not disk read):

| Link | Per-card transfer time (42 GB) | 4-card aggregate if serialized on one bus |
|---|---|---|
| Gen1 x16 (~4 GB/s) | ~10.5 s | ~42 s |
| Gen2 x16 (~8 GB/s) | ~5.3 s | ~21 s |
| Gen4 x16 (~32 GB/s) | ~1.3 s | ~5.3 s |

In practice this rarely dominates: a single NVMe SSD reads at roughly
3.5-7 GB/s, so at Gen1 x16 the **PCIe link and the disk are close to
matched** — total load time in the tens of seconds either way, plus
deserialization/dequant CPU time on top. At Gen2 or Gen4 the disk becomes
the bottleneck, not the PCIe link — so the mod mostly matters when the
checkpoint is already resident in page cache/RAM (repeated reloads,
container restarts) or when the disk itself is fast NVMe RAID. This matches
club-170hx's own note that "model loading at Gen2 x4 is ~2 min/card pain;
x16 removes it" — i.e. lane **width**, not raw generation, was the acute
problem at x4.

### (b) Pipeline-parallel inter-stage activation traffic (hidden=4096)

Per-token activation vector at a pipeline-stage boundary is
`hidden_dim × dtype_bytes` = 4096 × 2 bytes (BF16) = **8 KiB**.

- **Decode** (1 token/step): 8 KiB per stage crossing. At Gen1 x16 (4 GB/s)
  that is ~2 microseconds of pure transfer time — negligible next to
  kernel-launch and synchronization overhead, let alone compute time.
  Generation-of-PCIe-link makes essentially no difference at decode time.
- **Prefill** (2,941 tokens): 2,941 × 8 KiB ≈ **23 MB** per stage
  crossing.
  - Gen1 x16: 23 MB / 4 GB/s ≈ **5.7 ms**
  - Gen2 x16: 23 MB / 8 GB/s ≈ **2.9 ms**
  - Gen4 x16: 23 MB / 32 GB/s ≈ **0.7 ms**

  A few milliseconds per stage boundary, a handful of boundaries for a
  3-4-card pipeline, is small next to the matmul time for 2,941 tokens
  through a 4096-hidden transformer block (tens of milliseconds per stage
  on this class of card). PP communication cost is real but not the
  limiter at either Gen1 or Gen2 for this model shape.

### (c) Tensor-parallel all-reduce cost per layer (hidden=4096), no P2P

TP needs a full all-reduce of the activation tensor after each sharded
matmul — typically twice per transformer layer (post-attention projection,
post-FFN down-projection). At **decode**, the payload per all-reduce is the
same tiny 8 KiB seen in (b) — the problem with TP here is not bandwidth,
it is **per-op latency multiplied by op count**, especially when GPUs
cannot DMA directly to each other (no NVLink, and PCIe peer-to-peer is
frequently disabled or unavailable under virtualization/passthrough,
forcing traffic to stage through host RAM). Each such round trip costs on
the order of tens of microseconds of fixed latency (kernel launch, PCIe
round trip through the host bridge, NCCL synchronization) regardless of the
8 KiB payload size.

For this model: 43 layers × 2 all-reduces/layer = **86 collective ops per
decode token**. Even at a conservative ~50-75 microseconds fixed latency
per op (no-P2P, staged through host memory), that is roughly **4-6.5 ms of
pure communication overhead added to every decode token** — before any
compute. At a target of ~150 tok/s per card (≈6.7 ms/token budget), that
overhead alone can consume the entire per-token time budget. This is the
concrete reason **pipeline parallelism wins on this fabric**: PP crosses
the inter-card boundary once per pipeline stage (3 crossings for a 4-way
split, independent of the 43-layer depth), not once or twice per layer.
club-170hx's CLUSTER.md reaches the same conclusion from measurement
("Tensor parallel: use only after measurement... All-reduce traffic can
erase compute gains on slow PCIe"); the numbers above are the reasoning
that measurement is consistent with, not a substitute for re-measuring on
our own fabric.

### Proxmox PCIe link-speed fallback: is it cosmetic or real?

**Both mechanisms exist in the wild, and they produce the same symptom, so
they must be told apart on the host, not the guest:**

1. **QEMU generic PCIe root-port default (cosmetic capability report).**
   QEMU's generic `pcie-root-port` model has historically defaulted to
   reporting 2.5 GT/s (Gen1) x1 in its own PCIe capability registers,
   independent of what the passed-through physical device can actually do.
   Alex Williamson's 2018 QEMU patch series added `x-speed`/`x-width`
   properties to override this per the machine type; from `pc-q35-4.0`
   onward these can be raised. **Where this applies, the guest-reported
   Gen/width is purely a reported capability of the emulated bridge — real
   DMA still moves at the physical host link's negotiated speed.** The
   Proxmox wiki states this directly for the PCI-vs-PCIe passthrough flag:
   it "does not mean that PCIe capable devices that are passed through as
   PCI devices will only run at PCI speeds."
2. **A real, physical downstream-link downgrade.** Separately and
   commonly, the *host's own* `lspci -vv` on the physical slot can show
   `LnkSta` below `LnkCap` — this is a real electrical/training problem
   (ASPM power-saving downclock when idle, a BIOS PCIe link-speed setting
   left on a conservative value, a marginal riser/re-timer, or in rare
   cases a genuine board/slot limitation). If the *host* is downgraded,
   the guest cannot see a faster link than the host has, regardless of any
   QEMU property.

**Our own measurement is case (2)-shaped, not case (1)-shaped**: `LnkCap`
itself (the capability register, which VFIO passes through from the real
device on properly configured passthrough) reads 2.5 GT/s on all four
cards — not just `LnkSta`. That is consistent with the cards' **stock GA100
mining-SKU PCIe capability actually being Gen1**, with the seller's
"Gen2 x16" mod either not applied to the capability register the guest
sees, or (per our own INSTALLATION.md) deliberately not enabled on this
baseline because the "experimental Gen2 path... caused guest/QEMU hangs."
GPU 0's `x8` width (vs. `x16` capable) is a separate, real training issue
worth checking under load and after reseat.

**Exact commands to verify, host then guest:**

```bash
# On the Proxmox HOST, physical link as seen by the platform (ground truth):
lspci -nn | grep -i nvidia                     # find each BDF
lspci -vv -s <bdf> | grep -iE 'LnkCap|LnkSta|LnkCtl2'
# LnkSta below LnkCap on the HOST = a real electrical/training problem.
# LnkSta at or near LnkCap on the host but low in the guest = cosmetic (case 1).

# Re-check the host link UNDER LOAD, not idle -- ASPM can downclock at idle:
# (run a GPU workload in the guest, then immediately re-run the lspci above on the host)

# Host dmesg for the physical negotiation at boot, before vfio binds:
dmesg | grep -iE 'pcie|limited by|LnkSta'

# On the GUEST, current runtime state (what we ran above):
nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv
nvidia-smi -q | grep -A4 -iE 'link|pcie'
lspci -vv -s <bdf> | grep -iE 'LnkCap|LnkSta|LnkCtl'

# Confirm VM machine type and PCIe passthrough flag (Proxmox VM config):
qm config <vmid> | grep -iE 'machine|hostpci'
# want: machine: q35 (or q35 with a version >= 4.0), and hostpciN: ...,pcie=1
```

**Least risky fix to try, in order:**

1. **Re-check under load, not idle** (free, no config change). If host
   `LnkSta` rises under load, the idle reading was ASPM power-saving, not a
   fault — the finding above was captured at idle.
2. **Confirm `machine: q35` with a recent version and `pcie=1` on each
   `hostpciN` line** (`qm config <vmid>`), rather than i440fx or a bare PCI
   attach — this is a config check, not a live change, and is a
   prerequisite for any of the deeper fixes.
3. **If the machine type predates `pc-q35-4.0`**, upgrading the VM's
   machine-type version (Proxmox VM Options → Machine) lets QEMU's default
   root-port speed/width rise above 2.5 GT/s x1 without needing the
   experimental `x-speed`/`x-width` args at all — try this before hand
   editing `args:`.
4. **Only if (1)-(3) don't move the guest number and the *host* itself
   shows the downgrade**: check host BIOS PCIe link-speed setting (set to
   Gen2/Gen3 explicitly, not "Auto"), and disable ASPM in host BIOS (not
   just `pcie_aspm=off`, which per the kernel maintainers' own clarification
   leaves firmware-enabled ASPM untouched — the kernel flag and the BIOS
   setting are not equivalent).
5. **Last resort, most manual**: the QEMU `args: -global
   pcie-root-port.x-speed=8 -global pcie-root-port.x-width=16` override —
   only after confirming the *host* physical link genuinely supports Gen2,
   since this setting only changes what the guest is told, not what the
   silicon can do.

Given our own docs record that the experimental Gen2 driver-side patch
caused guest hangs on this exact baseline, the safer next step is (1)-(3)
above (idle-vs-load re-check, q35/`pcie=1` confirmation, machine-type
bump) before touching the Gen2 patch or hand-editing QEMU args again.

Sources: [QEMU PCIe root-port speed/width patch series (Alex Williamson, 2018)](https://patchew.org/QEMU/154222737752.9288.484557356059052047.stgit@gimli.home/154222868930.9288.2011388158102939509.stgit@gimli.home/), [qemu/docs/pcie.txt](https://github.com/qemu/qemu/blob/master/docs/pcie.txt), [Proxmox PCI(e) Passthrough wiki](https://pve.proxmox.com/wiki/PCI(e)_Passthrough), [Proxmox forum — PCIe GPU passthrough / PCIe link speed (solved)](https://forum.proxmox.com/threads/pcie-gpu-passthrough-pcie-link-speed.51681/), [kernel.org clarification that `pcie_aspm=off` does not undo firmware-enabled ASPM](https://lkml.iu.edu/hypermail/linux/kernel/2404.3/06648.html), local: `/home/ubuntu/WIP/repos/club-170hx/docs/INSTALLATION.md`, `docs/CLUSTER.md`.

---

## Part 3 — Is W4A16 the right quant target on SM80?

### Format comparison

| Format | SM80 kernel path in vLLM | Size (this model) | Decode speed driver | Notes |
|---|---|---|---|---|
| FP8 block-quant (current) | Weight-only FP8 via **FP8 Marlin** ([vLLM PR #5975](https://github.com/vllm-project/vllm/pull/5975)) — Ampere has no native FP8 Tensor Core, so weights are dequantized FP8→BF16 in the kernel | 168 GB | Bytes moved from HBM = 1 byte/weight, plus dequant compute | What we run today; MoE support for FP8 Marlin has been a gap on vLLM (tracked in [vLLM issue #17579](https://github.com/vllm-project/vllm/issues/17579)) |
| INT4 W4A16 (Marlin, AWQ/GPTQ/compressed-tensors) | **Marlin W4A16**, dense and (with caveats) MoE | ~95 GB (est., per task brief) | Bytes moved = 0.5 byte/weight — half of FP8 | Group size 128 typical; a reported MoE TP-scale-sharding bug blocks W4A16 MoE under TP>2 per a community project's bug report — verify against current vLLM main before relying on TP>2 |
| INT8 W8A16 | Marlin/compressed-tensors INT8 weight-only | ~170 GB | Bytes moved = 1 byte/weight, ~same as FP8 | No real size or bandwidth win over FP8 on this card; only useful if INT8 has an accuracy or kernel-maturity edge FP8-Marlin lacks |
| AWQ / GPTQ (4-bit) | Both route to Marlin on SM80 when symmetric | ~95 GB | Same as generic W4A16 | AutoAWQ export is 4-bit only, asymmetric; AutoGPTQ supports 2/3/4/8-bit but its asymmetric kernel has documented accuracy issues at low bit-width |

### Why 4-bit weights can be *faster* than FP8 on this card

vLLM's Marlin kernel family dequantizes packed low-bit weights (FP8 or
INT4) to BF16 **in the GEMM kernel itself** before the tensor-core matmul —
Ampere has no native FP8 or INT4 tensor-core path, so both formats pay a
dequant step. The part of the pipeline that differs is **bytes read from
HBM per weight**: FP8 reads 1 byte/weight, INT4 reads 0.5 byte/weight (plus
small per-group scale/zero-point overhead at group size 128). On a card
whose decode throughput is bandwidth-bound — which every note in this repo
and the wider community confirms the CMP 170HX is, given its gated Tensor
Cores and abundant HBM bandwidth — halving the bytes-per-weight read from
HBM can directly translate into close to a 2x decode throughput
improvement, since the dequant-and-matmul compute itself is not the
limiter. This is the same reasoning documented in vLLM's own FP8 Marlin PR:
"performance gains are higher on GPUs with less memory bandwidth" relative
to compute — the CMP 170HX inverts that ratio even further toward
bandwidth-bound than a 3090 or A10.

Source: [vLLM PR #5975 — FP8 Marlin for Ampere](https://github.com/vllm-project/vllm/pull/5975).

### Quality risk at 4-bit for MoE experts, group size 128

Group size 128 is the standard, well-tested granularity for both AWQ and
GPTQ/Marlin INT4 and is what the existing `Intel/DeepSeek-V4-Flash-W4A16-AutoRound`
checkpoint uses. Risk is concentrated in **routed experts that see little
calibration traffic** — with 256 routed experts and top-k routing, many
experts are exercised rarely in a modest calibration set, so their
per-channel scale/zero-point statistics are noisier than a dense model's.
The Intel recipe mitigates this specifically by **excluding** certain
layers from 4-bit (`--ignore_layers compressor,indexer.weights_proj` and
keeping `wo_a` at 16-bit) rather than blanket-quantizing everything to
INT4 — the same pattern `canada-quant/DeepSeek-V4-Flash-W4A16-FP8` follows
by keeping attention at FP8 and only putting routed experts at INT4. This
is direct precedent for a **mixed scheme** as the fallback if all-INT4
regresses quality: INT4 experts (bulk of the 168 GB) + higher-precision
attention/shared layers (small fraction of total parameters, so the size
cost of keeping them at BF16/FP8 is modest).

### What fits

| Configuration | Total VRAM | FP8 (168 GB) | INT4 (~95 GB) | INT8 (~170 GB) |
|---|---|---|---|---|
| 4x64 GiB | 256 GiB | Fits, ~66% used (current) | Fits with large headroom for KV cache | Fits, ~66% used |
| 2x64 GiB | 128 GiB | Does not fit (168 > 128, no headroom for KV cache even if it did) | **Fits** (~95 GB of 128 GiB, ~33 GB headroom for KV cache/activations) | Does not fit |
| 1x64 GiB | 64 GiB | Impossible | Impossible (95 > 64) | Impossible |

This confirms the task brief's framing directly: **W4A16 is the format
that turns a 2-card deployment from impossible into workable**, and is the
only one of the three candidates that leaves meaningful headroom for KV
cache on 2 cards. 1x64 GiB is not reachable by quantization alone at this
model size regardless of format.

### Existing PixelML evidence

Our own measured Qwen3.8-27B W4A16 AutoRound (dbirks) run reproduced
136-147 tok/s decode on a single CMP 170HX (`KNOWLEDGE.md` §3: 147.7 tok/s
decode, 76 ms TTFT, 2156 tok/s prefill, 57.8 GB VRAM, 255 W, using
DFlash2 speculative decoding). That is the direct internal precedent for
"W4A16 + Marlin decodes fast on this exact card," on a smaller dense-ish
model; DeepSeek-V4-Flash-Vision-Exp's 256-expert MoE structure is the
open variable this precedent does not cover — MoE-specific Marlin kernel
maturity and the TP-scale-sharding caveat above are the things to verify
before assuming the same multiplier holds.

**Bottom line for Part 3**: W4A16 is the right target for any 2-card
deployment and is worth trying even on 4 cards for the decode-speed
argument alone, given the current setup is bandwidth-bound and Marlin
dequantizes both formats to BF16 anyway. Runner-up: keep W8A16 (current
FP8 path) as the quality-safe 4-card default, and hold the mixed
INT4-experts/FP8-attention scheme in reserve if pure INT4 regresses
specific evals.

---

## Part 4 — Intel AutoRound

### The method

AutoRound is a **SignSGD-based learned-rounding** post-training
quantization method (originally from Intel Neural Compressor, now a
standalone library). Per block, it initializes learnable rounding-value
and min/max-clipping parameters, then runs **sign gradient descent** —
not plain SGD or Adam — over a calibration set for a default of **200
steps**, because block-wise reconstruction produces noisy gradients that
SignSGD is more robust to than magnitude-sensitive optimizers. After
tuning, parameters are clamped and the block is packed to the target
bit-width and exported.

This differs from the two other common PTQ methods:
- **GPTQ**: one-shot, uses second-order (Hessian) curvature information to
  choose per-weight rounding that minimizes layer output error, no
  gradient-descent training loop.
- **AWQ**: one-shot, activation-aware — protects the small subset of
  weight channels that correspond to large-magnitude activations via a
  per-channel scaling factor, not a learned per-weight rounding decision.

At 4-bit, all three are broadly comparable in accuracy; AutoRound's
advantage grows at 3-bit and 2-bit, where its iterative optimization
outperforms one-shot methods more clearly (Intel's own headline example:
an INT2-mixed DeepSeek-R1 quant retaining 97.9% of baseline accuracy).

Sources: [intel/auto-round GitHub](https://github.com/intel/auto-round), [intel/auto-round step_by_step.md](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md), [HF Transformers AutoRound docs](https://huggingface.co/docs/transformers/en/quantization/auto_round).

### Output formats and SM80 compatibility

| Export format | Best-suited runtime | vLLM/SGLang on SM80 |
|---|---|---|
| `auto_round` | CPU, Intel GPU, CUDA, HPU; supports 2/3/4/8-bit and mixed precision | Supported via native AutoRound integration in vLLM/SGLang (Intel-SGLang collaboration announced) |
| `auto_gptq` | Symmetric CUDA quant, 2/3/4/8-bit | Routes to Marlin on SM80 when symmetric — the common path for W4A16 on this card |
| `auto_awq` | Asymmetric CUDA 4-bit only | Also Marlin-compatible on SM80; AWQ/GPTQ both require `--sym` for Marlin |
| `gguf` | CPU / llama.cpp ecosystem, experimental in AutoRound | Not a vLLM/SGLang path — relevant only if the deployment target is llama.cpp, which our own benchmark comparison already shows losing to vLLM+speculation by ~3x on this card (`KNOWLEDGE.md` §3) |
| `llm_compressor` / compressed-tensors | vLLM-native compressed-tensors format | Directly supported in vLLM, including on SM80 |

### Memory needs for a 168 GB source model

- **`low_gpu_mem_usage`**: offloads intermediate calibration *activations*
  (not weights) to CPU, at a documented cost of ~20% more tuning time
  (project README frames it as up to ~30% slower for ~20 GB VRAM saved).
  This does not by itself make a 168 GB model quantizable on one 64 GB
  card — it trims activation memory, not the resident weight footprint.
- **`device_map`**: accepts `auto`, a specific device, or (for large models)
  a per-layer dict. AutoRound's own docs demonstrate quantizing
  DeepSeek-V3-BF16 (1.4T parameters) using **five 80GB GPUs** with an
  explicit `device_map` — i.e. the documented pattern for a DeepSeek-scale
  MoE is "spread the full model across enough aggregate VRAM to hold it,"
  not "quantize from a single small GPU with heavy offload." A filed
  GitHub issue notes `device_map=auto` can fail to balance memory evenly
  on multi-GPU setups for even a much smaller (8B) model — for a
  168 GB-class model, an explicit per-layer `device_map` is the safer
  starting point over `auto`.
- **128 GB DGX Spark GB10, unified memory**: 168 GB of FP8 source weights
  alone exceed 128 GB before any calibration-activation or optimizer-state
  memory is added. This is not workable without CPU offload of weights
  (beyond what `low_gpu_mem_usage` covers) or a lower-memory quant path —
  the RTN fast path with model-file-streaming (see below) is the relevant
  option here, not the default 200-iteration mode.
- **64 GB single CMP 170HX with CPU offload**: workable only via heavy
  CPU-RAM staging and slow, since the whole 168 GB source cannot be
  resident — expect this to be disk/RAM-bandwidth-bound and much slower
  than the 4-card option, and it exercises exactly the low-P2P-bandwidth
  PCIe path this document already flags as weak on our rig.
- **Our 4x64 GiB rig (256 GiB aggregate)**: the best fit we have. 168 GB of
  FP8 weights fit with room to spare across 4 cards using an explicit
  `device_map`, without needing CPU offload for weights at all.

Sources: [intel/auto-round step_by_step.md (DeepSeek-V3 5x80GB example, `low_gpu_mem_usage` definition)](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md), [intel/auto-round README](https://github.com/intel/auto-round/blob/main/README.md), [intel/auto-round issue #1636 — device_map=auto memory imbalance](https://github.com/intel/auto-round/issues/1636).

### `--iters 0` RTN vs `--iters 200` default, and DeepSeek-V4 support

AutoRound supports a **pure round-to-nearest (RTN) fast path**
(`--iters 0`, typically paired with `--disable_opt_rtn`), which the docs
describe as bit-exact-compatible with the standard flow's output format,
just skipping the SignSGD tuning loop entirely. RTN mode also unlocks a
**low-disk-memory flow**: for models without local files already
downloaded, it streams and quantizes one shard at a time, deleting each
source shard after processing — directly relevant for a 168 GB checkpoint
where holding both the FP8 source and INT4 output on disk simultaneously
may not be desirable. RTN also supports `--layer_config` for per-layer
bit overrides and `--ignore_layers`, with automatic detection of layers to
skip (MoE gates, MTP layers) based on the model config.

**Confirmed on HuggingFace**: [`Intel/DeepSeek-V4-Flash-W4A16-AutoRound`](https://huggingface.co/Intel/DeepSeek-V4-Flash-W4A16-AutoRound)
exists, made by the **Intel org**, and its model card states it was
generated **"with RTN mode"** — i.e. the fast `--iters 0` path, not the
200-iteration default — using
`--ignore_layers compressor,indexer.weights_proj --layer_config "{'wo_a':{bits:16}}"`.
vLLM support is linked via [vllm-project/vllm#45645](https://github.com/vllm-project/vllm/pull/45645).
A sibling [`Intel/DeepSeek-V4-Pro-W4A16-AutoRound`](https://huggingface.co/Intel/DeepSeek-V4-Pro-W4A16-AutoRound)
also exists but explicitly states vLLM/SGLang are **not** currently
supported for that larger Pro variant — confirming AutoRound's own
`deepseek_v4` architecture support is real and upstream-tracked, but not
uniformly wired into serving engines across every DeepSeek-V4 size yet.
The model's tags confirm `deepseek_v4` architecture recognition
(`DeepseekV4ForCausalLM`) and `auto-round`/`4-bit` tags in the
`transformers`/`safetensors` ecosystem. This is strong direct precedent
that **AutoRound already handles this architecture family**, though our
target is the Vision-Exp fork with a vision tower and 3 DSpark draft
layers layered on top — those additions are not covered by the vanilla
DeepSeek-V4-Flash artifact and would need their own `--ignore_layers`/
`--layer_config` treatment (vision tower and draft layers are exactly the
kind of small-but-precision-sensitive components the Intel recipe already
carves out for the base text model).

A second, independently produced quant —
[`canada-quant/DeepSeek-V4-Flash-W4A16-FP8`](https://huggingface.co/canada-quant/DeepSeek-V4-Flash-W4A16-FP8)
(recipe published at [canada-quant/dsv4-flash-w4a16-fp8](https://github.com/canada-quant/dsv4-flash-w4a16-fp8)) —
uses INT4 for routed experts and FP8 block-128 for attention, produced on
8x H200, and reports loading on Hopper and consumer Blackwell; it notes
the Intel artifact "explicitly excluded vLLM and SGLang at the time,"
though the Intel card now links a merged/open vLLM PR, so that gap may
already be closing.

### Wall-time estimate for a 170 GB-class MoE

Intel's own stated reference point is **37 minutes for a 72B model on a
single GPU** in the iterative "light" mode. For RTN (`--iters 0`) on a
170 GB-class MoE, there is no per-block gradient-descent loop to dominate
wall time — the job becomes **I/O- and packing-bound**: reading ~168 GB of
source shards, computing per-group scales, and writing ~95 GB of packed
INT4 output. At realistic disk/NVMe throughput (a few GB/s) this is a
low-single-digit-hours job (rough estimate: 1-3 hours), not a multi-day
one; the default 200-iteration mode would add real wall time on top of
that I/O floor because each block requires forward passes against
calibration data, and should be reserved for cases where RTN's accuracy
is shown to be insufficient (RTN is generally weaker than tuned modes at
very low bit-widths, but 4-bit/group-128 RTN is a much easier target than
2-bit, which is consistent with Intel shipping the DeepSeek-V4-Flash
quant as RTN rather than the 200-iteration default).

### llm-compressor (vLLM project) comparison

llm-compressor is vLLM's own quantization tool for producing
`compressed-tensors` checkpoints (AWQ, GPTQ, W8A8/W8A16 schemes) with
first-class vLLM loading support. Its practical advantage over AutoRound
for our purposes is **zero export/format friction** — output lands
directly in vLLM's native `compressed-tensors` format without needing an
extra format-conversion step. Its disadvantage for this specific task is
that `deepseek_v4`-architecture support (MoE routing, MTP/draft layers,
vision-tower handling) is not confirmed as upstream-supported in the way
Intel has already demonstrated for AutoRound with a working, published
checkpoint — this document did not find an equivalent public
llm-compressor DeepSeek-V4 MoE quant to compare against directly. Given
that a working AutoRound RTN artifact for the base model already exists
and is vLLM-loadable, AutoRound is the lower-risk near-term path; revisit
llm-compressor if/when a `deepseek_v4` compressed-tensors recipe is
published upstream.

---

## Local sources consulted

- `/home/ubuntu/WIP/cmp170hx-knowledge/KNOWLEDGE.md`
- `/home/ubuntu/WIP/repos/club-170hx/docs/HARDWARE.md`
- `/home/ubuntu/WIP/repos/club-170hx/docs/INSTALLATION.md`
- `/home/ubuntu/WIP/repos/club-170hx/docs/CLUSTER.md`
- `/home/ubuntu/WIP/cmpunlocker/README.md`, `/home/ubuntu/WIP/cmpunlocker/docs/INSTALLATION.md`
- Local measurement commands run on the four-card test node (2026-09-02): `nvidia-smi -q`, `nvidia-smi --query-gpu=...`, `lspci -vv -s <bdf>` for each of the four enumerated GPUs
