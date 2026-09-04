# Topology and parallelism on CMP 170HX: what the link really is, and why PP beats TP here

This page is the card/link-level companion to [LESSONS.md §c](LESSONS.md#c-topology-pipeline-parallel-vs-tensor-parallel)
(per-checkpoint topology findings) and [CLUSTER.md](CLUSTER.md) (multi-node
design). Read this page first if you are choosing tensor parallel (TP) vs
pipeline parallel (PP) on this card; read the other two for a specific
checkpoint's numbers or for scaling past one node.

Every claim below is labeled **measured**, **inferred**, **community-reported**,
or **untested**, per [AGENTS.md](../AGENTS.md). "Measured" means measured on
this club's four-card CMP 170HX test node unless a different source is named.

## 1. What the link really is

**The CMP 170HX's PCIe link is Gen1 x16 by design (measured).** Host-side
`lspci -vv` on the hypervisor shows every tested card advertising
`LnkCap: Speed 2.5GT/s, Width x16` — that is PCIe Gen1, and it is the card's
own capability register, not a hypervisor or passthrough artifact. An x16
riser cannot raise it: the ceiling is set in the card's vBIOS, not the
cabling. See [HARDWARE.md — PCIe link status](HARDWARE.md#pcie-link-status)
for the original measurement.

**A card that trains at x8 on a bifurcated slot is at that slot's ceiling,
not a fault (measured).** On the tested board (dual-socket, seven
physically populated slots), some root ports bifurcate x16 into x8/x8
between two neighboring slots. A card in one of those slots training at x8
instead of the board's advertised x16 is normal bifurcated-slot behavior,
not riser damage or a passthrough bug — confirmed by swapping the riser (a
new x16 riser did not change the width) and by swapping the card into the
position (the x8 ceiling stayed with the slot, not the card). `dmidecode`
slot tables on this board class are unreliable (list fewer slots than
exist, mark used slots "Unknown"); do not use them to identify a physical
slot.

**No P2P under VFIO passthrough; NCCL/collective traffic goes through host
memory (measured).** The tested fabric has no NVLink and no working P2P
over PCIe in this passthrough configuration. Every inter-card tensor
transfer — an all-reduce operand, a pipeline hand-off — round-trips through
host RAM rather than card-to-card DMA.

**Bandwidth by link width, PCIe Gen1, one direction (inferred from the
PCIe Gen1 spec: ~250 MB/s per lane per direction).** Not independently
re-measured on this box; treat as the theoretical ceiling, not an achieved
number.

| Link | Per-direction bandwidth (Gen1, theoretical) |
|---|---:|
| x1 | ~250 MB/s |
| x8 | ~2 GB/s |
| x16 | ~4 GB/s |

A card in an x8 bifurcated slot with no P2P is moving inter-card tensors
through host memory over a link that theoretically tops out around 2 GB/s
per direction — call this the operating envelope for any collective or
hand-off strategy on this box, not a number to plan multi-card throughput
past.

## 2. Why pipeline parallel beats tensor parallel on this card

**Tensor parallel does one all-reduce per layer per token (measured, this
lane; consistent with the independent per-checkpoint finding in
[LESSONS.md §c](LESSONS.md#c-topology-pipeline-parallel-vs-tensor-parallel)).**
Every layer boundary is a synchronization point: all four ranks block until
the slowest one's reduce lands. On a Gen1 fabric with no P2P, that is
latency-bound, not bandwidth-bound — the round trip through host memory
costs the same whether the tensor is large or small. Measured on
GLM-5.3-Flash AWQ W4A16, TP4, MTP k=3 (`ghcr.io/pixelml/club-170hx:vllm-glm53-sm80-20260903`,
fork `PixelML/sm80vllm@f6fbf3b854`): **60.4 tok/s c=1 decode (median of 5,
peak 77.9)**, **~38 tok/s aggregate decode at c=8** (three rounds: 36.3 /
38.8 / 38.1 tok/s, 8/8 requests OK each round; text-only, mm-encoder
profiling skipped — see §3).

**Pipeline parallel does one hidden-state hand-off per stage boundary
instead (measured on this card's PP4 boot; see §3).** With N pipeline
stages there are N-1 hand-offs per token per forward pass, each moving one
hidden-state tensor once, versus TP's per-layer all-reduce. The per-step
cost on a PP fabric therefore looks like a small fixed overhead per hop
plus a cost that scales with how much is actually verified per step — which
is exactly the lever speculative decoding pulls.

**Community-reported cost model, 8-stage PP, from
[promisezackr/glm53-flash-170hx-pp8](https://github.com/promisezackr/glm53-flash-170hx-pp8)
(commit `90ec72e`, Apache-2.0, derivative of vLLM):** measured with
per-stage CUDA events on their 8-card PCIe Gen2 x4 box, step time
`≈ 32 ms fixed + 2.1 ms × verified tokens per step`. The fixed part is
attributed to the serial 8-stage pipeline (per-stage attention/expert
compute plus hop and host-side prep/sampling latency); the variable part
to active-expert weight bandwidth, which scales with tokens × top-k. This
is their number on their hardware (Gen2 x4, 8 cards, NVFP4 + DFlash2), not
reproduced on this club's Gen1 x8/x16 four-card box — carried here as the
shape of the cost model, not as our own measurement. Under that model,
accepted draft length per step is the throughput lever: a wider accepted
block amortizes the fixed 32 ms over more tokens.

**Community-reported result at that cost model's target operating point:**
GLM-5.3-Flash NVFP4, PP8, DFlash2 k=7, 8x CMP 170HX at PCIe Gen2 x4 (a
faster link than this club's Gen1 x8 boxes): single-stream decode 123
tok/s (code), 37 tok/s (prose) — the source explicitly reports acceptance
and hence throughput varies by workload, so both numbers are named with
their content type, not averaged. Steady-state aggregate at 16 concurrent
streams: 276 tok/s. Source:
[promisezackr/glm53-flash-170hx-pp8](https://github.com/promisezackr/glm53-flash-170hx-pp8),
commit `90ec72e9525e90be701e742c70a20c4154418307`, README benchmark tables.
**Attribution:** Apache-2.0, derivative of vLLM; their SM8x port itself
credits [344303947/vllm170hx](https://github.com/344303947/vllm170hx) for the
base SM8x patch set and [bayley/vllm-170hx-glm5](https://github.com/bayley)
for PP-on-170HX validation. We have not run their patch set; the numbers
above are cited for comparison, not reused as our own measurement.

**Our PP4 today is not faster than TP4 — state this plainly.** GLM-5.3-Flash
AWQ W4A16, PP4 (`VLLM_PP_LAYER_PARTITION=14,12,12,7`), `--enforce-eager`,
`--gpu-memory-utilization 0.92`, on the same four-card box and checkpoint
as the TP4 numbers above (measured 2026-09-04):

| Metric | PP4 | TP4 (numbers of record) |
|---|---:|---:|
| c=1 decode, median tok/s | 3.35 (MTP k=5) | 60.4 (MTP k=3) |
| c=1 decode, spec off | 6.1 (median of 3) | not separately measured |
| c=8 aggregate tok/s (3 rounds) | 17.1 / 18.1 / 18.7 | 36.3 / 38.8 / 38.1 |
| Uncached prefill, ~2,900-token prompt, tok/s | 1,053 | 818 |

PP4 wins on prefill (a single large batch spreads across stages, and
prefill has no per-step decode bubble to pay) but loses badly on decode.
Two separable causes, isolated with a same-config MTP-on/MTP-off A/B on
this box (measured 2026-09-04): (a) MTP under PP on this fork's current
state produces clean text but only ~1.8x the spec-off throughput (6.1 vs
3.35 tok/s) — most of the collapse is not MTP; (b) the base PP4 pipeline
itself, with `--enforce-eager` (no CUDA graphs to hide the bubble), is
structurally slow on this partition regardless of speculative decoding.
**Community-reported fix for the MTP-under-PP part:** upstream vLLM's PP
path skips loading the MTP draft model's `embed_tokens` / `shared_head.head`
weights from the checkpoint under pipeline parallelism, leaving them
randomly initialized and driving acceptance toward zero — documented as
patch category 3 ("MTP under PP") in
[promisezackr/glm53-flash-170hx-pp8](https://github.com/promisezackr/glm53-flash-170hx-pp8#what-the-patches-do).
Porting that fix to this club's `glm53-sm80` fork is in progress and
unmeasured here; **PP4 on the current unpatched fork is not a viable
alternative to TP4 for decode throughput**, full stop, until that port
lands and is re-measured.

**PP is also expected to transfer to A100 boxes without NVLink (inferred,
not tested on A100).** The same argument — one hand-off per stage boundary
beats one all-reduce per layer when there is no fast interconnect — does
not depend on anything specific to the CMP 170HX; it applies to any
no-NVLink multi-GPU box. Untested on this club's own hardware pool since
we have no A100s; carried here as a design hypothesis for anyone planning
a similar box.

## 3. PP boot facts

All measured on the four-card CMP 170HX box, GLM-5.3-Flash AWQ W4A16,
2026-09-04, image `ghcr.io/pixelml/club-170hx:vllm-glm53-sm80-20260903`.

- **`VLLM_PP_LAYER_PARTITION` is honoured exactly.** Setting
  `VLLM_PP_LAYER_PARTITION=14,12,12,7` on a 45-layer model produced layer
  ranges 0–14 / 14–26 / 26–38 / 38–45 across the four pipeline stages — a
  direct check of the env var against the resulting per-stage layer count.
- **`--gpu-memory-utilization 0.80` leaves negative KV headroom on this
  partition; `0.92` serves.** At 0.80, each card spends its whole memory
  budget (0.80 × 64 GiB) on weights (44.5–45.5 GiB/card for this
  partition) plus activation/CUDA-graph overhead, leaving no room for KV
  cache — the engine fails at KV-cache allocation with a negative
  "Available KV cache memory" value, independent of `--max-model-len`
  (tried at both 524,288 and 262,144, both failed the same way). Raising
  `--gpu-memory-utilization` to 0.92 fixed it: KV cache allocated
  (1,248,304 tokens), server reached `Application startup complete`.
- **`--enforce-eager` PP step time is roughly 160 ms.** Consistent with the
  c=1 decode collapse above — enforce-eager disables CUDA-graph capture,
  which on TP4 hides much of the per-step latency; PP4 pays it directly.
- **Prefill favors PP4 over TP4 at this partition.** Uncached prefill on a
  ~2,900-token prompt: **1,053 tok/s on PP4 vs 818 tok/s on TP4**, same
  checkpoint, same box, same day.

## 4. Speculative decoding on a communication-bound box

The compute-bound advice published elsewhere — Unsloth's B200 sweep for
GLM-5.3-Flash MTP in llama.cpp found `n=2 > n=3 > n=5` and recommends
stopping around k=2 (community-reported, not reproduced by this club) —
may invert on this box (**inferred**; a draft-depth sweep is in progress,
not yet concluded). The reasoning: on a compute-bound box, every extra
verified token in a deeper draft costs real FLOPs that are not free. On
this Gen1/no-P2P box, the per-step cost model from §2
(`fixed + small·tokens`, community-reported shape) means the marginal cost
of verifying a longer accepted run is small relative to the fixed per-step
overhead — so a deeper draft that raises mean accepted length should pay
for itself more easily here than on a compute-bound box. This is a
hypothesis awaiting our own sweep, not a result; do not treat "deeper is
better here" as measured until the sweep publishes numbers.

**Acceptance is workload-dependent, not a single number.** The community
PP8 source above reports roughly a 3x spread in single-stream decode
between structured/code workloads (123 tok/s) and prose (37 tok/s) at the
same k=7 DFlash2 configuration (community-reported). Every published
throughput cell in this lane's future sweep tables names its workload
(code / prose / json / math / counting, or equivalent) rather than
reporting one aggregate number — a single "tok/s" figure without a named
workload is not comparable across runs on this box.

## 5. Quant formats on SM80

**NVFP4 checkpoints run via Marlin W4A16 dequant on this card — the same
execution class as AWQ W4A16, not native FP4 tensor-core math (measured /
inferred).** SM80 has no native FP4 tensor cores (see
[LESSONS.md §a](LESSONS.md#a-kernel-and-format-compatibility-on-sm80) for
the full format-compatibility table); an NVFP4-stored checkpoint on this
box dequantizes through the same Marlin W4A16 path used for AWQ, and
should be expected to perform similarly to AWQ W4A16 rather than to a
native-FP4 box. This is measured for the dequant-path mechanism (Marlin
selects on compute capability, confirmed elsewhere in this repo) and
inferred for the specific NVFP4-vs-AWQ performance parity claim — we have
not run a controlled NVFP4-vs-AWQ A/B on the same checkpoint on this box
to confirm the numbers land the same.

**Keep AWQ for A100 transfer.** Where a recipe needs to run unchanged on
both this club's CMP 170HX pool and an A100 box, prefer AWQ W4A16 over
NVFP4: A100 is also SM80, so the same Marlin dequant argument applies
there, and AWQ is the better-tested path on this club's own hardware today
(see the [GLM-5.3-Flash run table](models/glm-5.3-flash.md#run-on-cmp-170hx)).

## Attribution

- [promisezackr/glm53-flash-170hx-pp8](https://github.com/promisezackr/glm53-flash-170hx-pp8),
  commit `90ec72e9525e90be701e742c70a20c4154418307` — PP8 cost model
  (§2), community PP8 + DFlash2 benchmark numbers (§2, §4). License:
  Apache-2.0, derivative of vLLM. Config and code from that repository are
  cited for comparison only in this document; nothing from it has been
  vendored into this club's fork as of this page's publication. That
  repository's own credits chain: [344303947/vllm170hx](https://github.com/344303947/vllm170hx)
  (SM8x patch base) and [bayley/vllm-170hx-glm5](https://github.com/bayley)
  (PP-on-170HX validation) — carried here for completeness, not verified
  by this club.
- Unsloth's B200 MTP depth-sweep guidance for GLM-5.3-Flash (§4):
  community-reported, cited for the compute-bound comparison point only,
  not reproduced on this club's hardware.
- The GLM-5.3-Flash SM80 vLLM fork this club's own TP4/PP4 measurements
  run on is `PixelML/sm80vllm` branch `glm53-sm80`; its own upstream lineage
  is documented in [UPSTREAM-SM80-NOTES.md](UPSTREAM-SM80-NOTES.md).

## See also

- [LESSONS.md §c](LESSONS.md#c-topology-pipeline-parallel-vs-tensor-parallel) —
  per-checkpoint PP vs TP findings (DeepSeek-V4-Flash-0731, layer-partition
  skew, `gpu-memory-utilization` thresholds for that checkpoint).
- [HARDWARE.md — PCIe link status](HARDWARE.md#pcie-link-status) — the
  original link-width measurement this page's §1 builds on.
- [CLUSTER.md](CLUSTER.md) — topology guidance when scaling past one node.
- [models/glm-5.3-flash.md](models/glm-5.3-flash.md) — the full GLM-5.3-Flash
  run table and reproduce steps.
