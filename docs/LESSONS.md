# What we learned running inference on CMP 170HX

This page collects lessons from every PixelML model attempt on NVIDIA CMP
170HX (GA100 silicon, SM80 / compute capability 8.0, 64 GiB HBM2e per card,
no FP4 or FP8 tensor cores, no NVLink). Numbers come from the model
repositories linked in [MODEL-STATUS.md](MODEL-STATUS.md) and from this
club's own [Benchmarks](BENCHMARKS.md), [Cooling and power](COOLING-AND-POWER.md),
and [QC](QC.md) pages. Read each number next to its source before repeating it;
runtime, checkpoint, and power policy differ between rows below.

## a. Kernel and format compatibility on SM80

SM80 has no FP4 or FP8 tensor-core math. Kernels written for SM89 (Ada FP8)
or SM90+ (Hopper, sparse-MLA, FlashInfer fast paths) fail to compile or fail
at runtime on SM80. The practical rule: FP8 and FP4 **storage formats**
dequantize fine on SM80; FP8/FP4 **tensor-core compute** does not exist here.

| Format / kernel | Works on SM80? | Evidence | Workaround |
|---|---|---|---|
| FP8 weight-only via Marlin | Yes | Qwen3.8-27B W4A16-AutoRound Marlin repack auto-selects for compute capability 8.0 | None needed |
| FP8 activations / tensor-core FP8 math | No | DeepSeek-V4 docs state FP8 math needs SM89+; GLM `*-W4AFP8` quants rejected on SM80 | Use FP8 as a storage/KV format only, not as compute |
| FP8 KV cache | Yes (software conversion) | DeepSeek-V4 asserts `fp8_ds_mla` layout requires FP8 KV; runs correctly on SM80 | None needed; costs decode/prefill throughput (see section d) |
| NVFP4 / MXFP4 native tensor-core execution | No | GLM-5.3-Flash NVFP4 targets SM12x FlashInfer backends only; Blackwell-only NVFP4/W4A4 unavailable on SM80 (Ninfer scope note) | None; pick a different quant format |
| MXFP4-stored expert weights, dequantized | Yes | DeepSeek-V4-Flash-0731's native checkpoint (MXFP4 experts + FP8 e4m3 attention) serves at 83.3-98.1 tok/s on 3-4 CMP 170HX cards | Dequant path, not native FP4 GEMM |
| AWQ / GPTQ INT4 via Marlin | Yes | Qwen3.8-27B W4A16-AutoRound + Marlin repack is the fastest working config measured (147.7 tok/s @ 255 W) | None needed |
| W8A16 | Works, but slower | Qwen3.8-27B official W8A16 checkpoint: 31.6 tok/s vs 47.5 tok/s for W4A16 on the same card and protocol | Prefer W4A16; the card is bandwidth-bound, so denser weights are strictly slower |
| BF16 | Yes | Baseline dtype across every repo | None needed |
| DeepGEMM (TF32 prenorm GEMM) | No | `RuntimeError: DeepGEMM backend is unavailable`; upstream gates the fallback on package presence, not on architecture | Triton fallback, selected by an architecture-aware predicate (`is_deep_gemm_supported()`) |
| CuteDSL kernels | No | CuteDSL `fused_indexer_q` RoPE + FP8 quant kernel: NVVM backend fails to compile for target `sm_80` | Pure-torch fallback reproducing the same numerics (GPT-J interleaved RoPE, power-of-two per-token scales) |
| Triton `fp8e4nv` stores | No | Triton compile error: `type fp8e4nv not supported in this architecture` | Torch BF16 fallback, gated on compute capability < 9 |
| tilelang FP8 x FP4 GEMM | No | Device assert on SM80; the kernel requires SM89 FP8 MMA | FP4 dequant to BF16, then a BF16 GEMM; vision tower runs BF16 SDPA instead |
| flash_attention (FLASH_ATTN backend) | Yes | Qwen `CTX=fast` profile uses FLASH_ATTN with BF16 KV at full context | None needed |
| flashinfer fast top-k (JIT) | Conditional | JIT needs `nvcc` and `curand.h` in the image; missing headers silently fall back to `torch.topk`, costing about 2x on the DFlash2 selector | Symlink `curand.h` from the pip CUDA package into the CUDA include path; clear the flashinfer JIT cache |
| `fast_hadamard_transform` CUDA extension | No | Needed for the Vision-Exp indexer path on SM80; the compiled extension is not usable there | Pure-torch Sylvester Hadamard transform (`H @ H.T = n * I` checked for exactness) |
| DSpark speculative decoding | Yes | +1.93x aggregate decode on DeepSeek-V4-Flash-0731 (PP4); keeps winning through 64 concurrent requests under pipeline parallel | Must match `num_speculative_tokens` to the checkpoint's exact `dspark_block_size` (see section d) |
| MTP speculative decoding | Yes, needs patches | Stock vLLM blocks MTP under pipeline parallelism in three places (patched by `allover326/vllm-dsa-mtp-sm80`); the newer `GPUModelRunnerV2` path needed a second, separate patch for the same restriction | Apply both patch sets; verify with an import + undefined-name (pyflakes) check, not just `py_compile` |
| lm-head-draft (full 248k-row draft head) | Works, but hurts | -10.4% decode vs a truncated 40k-row draft head (119.2 vs 133.1 tok/s, Ninfer trick A/B) | Do not enable full-head drafting on this stack |
| DFlash2 speculative decoding | Yes | 2.30x speedup on RTX 3090, 1.21-1.61x card-for-card decode/prefill advantage of 170HX over 3090 at the same recipe | None needed |

## b. Runtime build lessons

- **The precompiled vLLM wheel lacks `vllm._C` for these SM80-specific ops.**
  `VLLM_USE_PRECOMPILED=1` silently ships without the custom op, and the
  engine fails at CUDA-graph capture, not at import. A full source build is
  required whenever a patch touches `csrc/`.
- **Full source build:** `TORCH_CUDA_ARCH_LIST=8.0`, compiled with a real CUDA
  toolkit image (`nvidia/cuda:13.0.2-cudnn-devel` or similar; pip CUDA wheels
  alone give an `nvcc`/FlashInfer header mismatch). Measured compile time was
  about 62 minutes for the editable-wheel CUDA stage (49-way parallel
  compilation, `ccache` active), plus roughly 13 minutes to export the image;
  total wall time around 78 minutes for a from-scratch build. A rebuild
  triggered by a late `COPY` layer re-runs the full 62-minute compile even
  when earlier layers are cached.
- **Image sizes:** the build-stage image measured about 40.6 GB; the final
  exported serving image measured 12.8 GB.
- **Editable installs pay for themselves.** vLLM installed with `pip install -e .`
  means `/vllm/vllm/...` is live code. Patched Python files can be bind-mounted
  read-only over the checkout and take effect with no rebuild — the
  DeepSeek-V4-Flash-0731 recipe ships five patched files this way. Only
  changes under `csrc/` force a real rebuild.
- **Verify a patched build with three checks, in order of what they catch:**
  `py_compile` (syntax only, passes on code that cannot run), an import smoke
  test (catches import-time breakage), and an undefined-name scan (pyflakes;
  catches a patch that calls a helper no patch defines). A real shipped bug
  passed the first two checks and only the third caught it.

## c. Topology: pipeline parallel vs tensor parallel

The tested CMP 170HX fabric has **no NVLink and no P2P over PCIe Gen2 x4**
(about 1.0 GB/s measured bus bandwidth between cards). This one fact decides
the topology choice.

| Strategy | Behavior on this fabric |
|---|---|
| Tensor parallel | Performs 2 all-reduces per layer (86 collectives per forward pass on a 43-layer model). Communication-bound at every sequence length: prefill measures flat at ~800 tok/s from 1.5k to 77k tokens on TP4 |
| Pipeline parallel | Moves the payload once per stage boundary (3 hand-offs for 4 stages) and overlaps transfer with compute: about 28x less data on the wire than TP |

Measured effect: **PP beats TP by 6.6x on prefill** (5,321 vs 801 tok/s at
77k context) and roughly 2x on aggregate decode, on the same DeepSeek-V4-Flash-0731
checkpoint and cards. Prefill time-to-first-token at 100k context: PP4 14.6 s
vs TP4 87.3 s.

**Layer partition skew.** The last pipeline rank always carries `lm_head` and
the speculative drafter, so an even split overloads it.

- **3 cards:** `VLLM_PP_LAYER_PARTITION=15,15,13` is required, not optional.
  The default even split (`15,14,14`) fails during the Marlin FP4 expert
  repack because the last rank also carries `lm_head` and the DSpark drafter.
- **4 cards:** `11,11,11,10` is the only stable partition found for
  DeepSeek-V4-Flash-Vision-Exp. Two alternative partitions (`12,12,12,7` and
  `12,12,11,8`) failed before serving any traffic (exit 137, and a
  first-request device-side assert). Rebalancing the DeepSeek-V4-Flash-0731
  4-card partition away from an even split grew the KV pool about 85%
  (798,660 to 1,476,563 tokens at `max-model-len 163840`) by removing an
  8.7 GiB rank imbalance.

**`gpu-memory-utilization` thresholds.** Higher is not always safer:

- 3-card DeepSeek-V4-Flash-0731 needs **0.95**; 0.85 and 0.93 both fail KV
  allocation on the lm_head-heavy last rank (weights 51.8 GiB + activations +
  CUDA graphs leave about 9 GiB; 0.95 leaves 7.7 GiB of KV, which fits).
- 4-card DeepSeek-V4-Flash-0731 uses **0.85**. Raising it further takes
  headroom from activations and CUDA-graph capture; at 0.90 with the DSpark
  draft resident, capture OOMs.

**Why alternative partitions fail:** the failure modes are consistent —
either the process exits before serving (exit 137, killed by memory pressure)
or the engine boots but the first real request hits a device-side assert in
the Marlin expert repack or the draft path. Neither failure mode is a
hardware fault; both are layer-count-to-memory mismatches on the heaviest
rank.

## d. Speculative decoding

Speculative method and draft depth (`k`, or `num_speculative_tokens`) must
match the checkpoint, not a rule of thumb.

| Checkpoint | Method | k | Mean acceptance length | Aggregate decode |
|---|---|---:|---|---:|
| DeepSeek-V4-Flash-0731 (PP4) | DSpark | 5 | 3.03 (per-position 0.730/0.569/0.372/0.226/0.131) | 98.1 tok/s |
| DeepSeek-V4-Flash-0731 (PP4) | DSpark | 7 | 1.43-2.51 | 60.3 tok/s (worse — acceptance never extends past ~3 tokens, so extra drafts are pure waste) |
| DeepSeek-V4-Flash-0731 (PP3, 180 W local) | DSpark | 5 | 5.07-5.32 (81-86%) | 73.4-116.6 tok/s by content type, 83.3 aggregate |
| DeepSeek-V4-Flash-Vision-Exp (PP4) | DSpark | 6 | not reported | 59.78 tok/s warm single-stream |
| Qwen3.8-27B (1 card, 255 W) | DFlash2 | 7 | 2.56-2.80 | 147.7 tok/s |
| Qwen3.8-27B (1 card, 180 W local) | DFlash2 | 7 | 2.85-3.32 (better than the 255 W run) | 135.3-140.3 tok/s |

**vLLM enforces `num_speculative_tokens >= dspark_block_size`** for
DeepSeek-V4-Flash-0731 (5 for that checkpoint). Below it, output is garbled,
not merely lower-acceptance. Above it measures worse, not better.

**lm-head-draft hurts.** The full 248k-row draft head lost 10.4% decode
against a truncated 40k-row draft head (119.2 vs 133.1 tok/s, same server,
same protocol). The wider acceptance of the full head does not pay for its
6x wider draft GEMM.

**FP8 KV cost, measured directly (Qwen3.8-27B, DFlash2 k=7, same server):**
131k-context FP8 KV vs BF16 KV: **-3.4% decode** (135.8 vs 140.5 tok/s) and
**-19.5% prefill** (1,741 vs 2,161 tok/s). FP8 KV buys long context on one
card; it is not free.

**DSpark output is not bit-reproducible at temperature 0**, on either the
patched or the stock upstream path — this is a property of DSpark, not of any
local patch. Plain (non-speculative) decode on the same server is
self-deterministic. If bit-reproducible output matters, drop
`speculative-config` entirely.

**Concurrency behavior differs by topology.** On pipeline parallel, DSpark
keeps winning through 64 concurrent requests (712.8 vs 472.0 tok/s plain at
c=64). On tensor parallel, DSpark goes *negative* above about 8 concurrent
requests. Pipeline parallel leaves bubbles the drafter can fill; tensor
parallel does not.

## e. Memory and storage

**Host-RAM exhaustion from eager safetensors loading.** The default eager
load strategy reads each whole shard into host RAM with `f.read()`. Four
pipeline ranks loading a 156 GiB checkpoint concurrently, each reading
roughly 3.2 GB shards, exhausted a 94 GiB host and the kernel OOM-killed
workers silently at 7 of 48 shards (no traceback). Fix: drop
`--safetensors-load-strategy eager` and use the default streaming
`safe_open`, verified correct on both bf16 and float8_e8m0fnu tensor shards.

**NFS is slow for cold model loads.** Measured aggregate physical-read
throughput over NFS during a stalled load: **31.01 MiB/s**. At that rate a
179 GB checkpoint takes about 92 minutes of raw I/O. A local NVMe-class tier
(about 500 MiB/s conservative, up to about 820 MB/s NVMe-class) cuts that to
roughly 4-6 minutes — a 16-25x improvement on raw I/O, with total
listener-ready time expected to fall from 75-90 minutes to 8-15 minutes once
deserialization and kernel init are included.

**Staging pattern: manifest + `READY.json`, fail-closed.** For a one-time
NVMe staging copy of an already-verified checkpoint:

1. Freeze a canonical manifest on the source (source revision, transform code
   or image digest, exact per-file bytes and SHA-256, shard/rank count,
   parsed shard headers).
2. Copy resumably into a unique `<target>/.staging/<asset-id>.partial`
   directory. Never overwrite an existing ready tree.
3. Verify destination bytes, hashes, and every shard header against the
   manifest.
4. `fsync`, then atomically rename to the final path, then write
   `READY.json` only after every gate passes.
5. Bind the final tree read-only into the serving container.
6. Startup preflight requires the target to be a distinct real mount (by
   filesystem UUID), read-write, at or above its free-space floor, with a
   valid manifest and a valid `READY.json`. Any failure aborts startup —
   there is no automatic fallback to slower shared storage or to root.

**Root-disk floors.** Two floors were enforced as stop conditions during
staging and builds: a **10% free** floor on the root disk during a build
(observed at 12.8% free, treated as passing but tight), and a **20% free**
floor on serving storage before a container is allowed to start.

**Docker/build cache placement.** Move container build and runtime caches off
the root disk and onto the larger model-storage volume. A root disk observed
at 88% full is a stop condition for any further build activity until caches
are relocated.

## f. Power and thermal

| Policy | Measured effect |
|---|---|
| 180 W (per-card benchmark cap) | Qwen3.8-27B decode 135.3-140.3 tok/s across 3 local cards; SM clock 1140-1275 MHz under load |
| 255 W (rental reference cap) | Same recipe: 147.7 tok/s; SM clock 1455 MHz |
| 180 W vs 255 W, same recipe | Local best card reached 95% of the 255 W decode rate (140.3 vs 147.7 tok/s) at 71% of the power. Prefill: 91% of the 255 W rate. TTFT is worse locally (about 190 ms vs 76 ms), attributable to the PCIe Gen2 x4 host link, not the power cap |
| 125 W (quiet/idle policy) | Administration, downloads, idle serving |

**A caution on an earlier reading.** A single-card run once reported 47.0
tok/s at 180 W and was initially read as "the power cap costs 3x." That
number was contaminated by host contention (an overlapping model build) and a
pinned KV-cache allocation left over from a 24 GB-card default; a clean
rerun at the same 180 W cap reached 135.3-140.3 tok/s. **The 180 W vs 255 W
gap is about 5%, not 3x.** Always re-verify a surprising power number before
trusting it.

**Passive cooling is not optional.** Community field reports: passive cards
reach 85 C within 1-2 minutes without directed airflow. Club measurements:
stop thresholds are **80 C core / 85 C memory**. A four-card idle group
(180 W cap per card, forced blower airflow, 0% utilization) measured about
**141 W total group power**, 37-38 C core, 41-51 C memory — idle heat from
four cards is real heat; an open frame with no ducted airflow is not
sufficient cooling by itself.

**Power-brake (`PWRBRK#`, PCIe sideband pin B30)** is a separate, board-level
hazard: some workstation boards assert it on x16 slots, permanently capping a
card near 88 W of a 250 W budget and dropping clocks to about 1140 MHz
(measured fp16 throughput 39.3 vs 155.7 TFLOPS healthy). Check
`nvidia-smi -q | grep -A1 "HW Power Brake Slowdown"` before blaming any other
part of the stack for low throughput; it reports `Active` only when the
platform itself is asserting the signal.

## g. Failure modes and recovery

| Signal | Meaning | Recovery |
|---|---|---|
| Xid 79, "fallen off the bus" | Card left the PCIe bus; PCI config reads as an invalid header | Function-level reset, secondary-bus reset, runtime power changes, and remove/rescan all failed in the measured case. A true cold power cycle (standby rails discharged) was required |
| Xid 154, "Node Reboot Required" | Follow-on to a bus-drop event | Full VM/host reboot |
| Xid 43 (software-classified) | Draft-path embedding assert at high concurrency (measured at c=16 on a DSpark ladder) | Not a hardware or ECC fault; restart the server process |
| NVRM VA-space corruption after an OOM kill storm | Kernel log shows NVRM assertion failures on every GPU (`pool_alloc.c`, `vaspace_api.c`); `cuInit` returns `CUDA_ERROR_NO_DEVICE` host-wide | Reloading `nvidia_uvm` alone does not clear it. Full sequence: `rmmod nvidia_uvm nvidia` then `modprobe nvidia nvidia_uvm` restores all devices without a VM reboot |
| `--gpus all` assigns zero devices after a crash | Stale cgroup state left behind by an OOM crash; reproduced in a minimal test container | Use an explicit device list (for example `--gpus '"device=0,1,2,3"'`) instead of `all` |
| Ranks stuck in D state during NFS-backed weight load | `wchan folio_wait_bit_common` — uninterruptible page wait while `mmap`-ing shards over NFS; `rchar` stays near-static because `mmap` page faults do not increment it | This is loading, not hanging. Confirm with a bounded read-throughput sample (physical reads increasing) before treating it as stuck |
| NCCL store timeout (600 s) | One rank stalls on an NFS page-in for a large shard while its peers reach the rendezvous store first; the skew exceeds the store timeout and `torchrun` SIGTERMs the stalled rank | Root cause is storage-read skew across ranks, not a network or NCCL defect; stage weights locally (section e) to remove the skew |
| Container seccomp blocks `pidfd_getfd` | A CPU-offload worker process needs `pidfd_getfd`; common container seccomp profiles deny it | Works on bare metal; inside a container, either allow the syscall or avoid the code path that needs CPU offload (fits when the table it offloads is small enough for VRAM) |

## h. QC

- **Field failure rate is high above the mining window.** Independent
  reports put failures around 50% among buyers, and faults appear **only**
  above the 8 GiB region a mining rig would ever touch — the cards pass
  `nvidia-smi` and die only under a large model's full memory footprint.
  This is the reason for a strict pre-purchase gate, not an optional
  courtesy to the seller.
- **`memtest_vulkan` gate.** v0.5.0, run sequentially per card (multi-GPU
  Vulkan device selection is unreliable in that version), about 10 minutes
  per card. Exit code 64 means a setup error, not a card failure; an error
  string in the per-GPU log means FAIL; a clean exit or an expected timeout
  is a PASS candidate.
- **`vram-gate` gate.** Allocates about 61 GiB, writes a PRNG pattern,
  verifies, idles 20 s, re-verifies, runs a 60 s warm burn, then a full
  second verification pass. One JSON verdict per card
  (`{"gpu":N,"pci_id":"...","vram":"65536 MiB","lnk":"2,16","write_read":"1","verdict":"PASS"}`).
  Any single mismatched byte is a reject, regardless of seller explanation.

Full acceptance ladder and isolation procedure: [QC](QC.md).

## i. Benchmark methodology

- **Count tokens from the final `usage.completion_tokens`, never from SSE
  event counts.** Under speculative decoding a single server-sent event can
  carry several accepted tokens (roughly the acceptance length). An early
  harness counted events and reported 43.3 "tok/s" for a run that was
  actually 110-130 real tok/s once the usage field was read correctly — more
  than a 2x undercount from a counting bug, not a hardware difference.
- **Aggregate vs single-stream are different metrics; do not compare them
  across rows.** Aggregate throughput is total completion tokens across every
  concurrent request in a load level, divided by wall time for that level.
  A widely circulated reference number (212.68 tok/s) is an 8-stream
  aggregate; the correct comparison point for a single-stream measurement on
  this hardware is roughly 120-148 tok/s.
- **Use a fixed prefill fixture for cross-run comparison.** The Vision-Exp
  benchmark protocol standardized on an uncached 2,941-input-token prompt so
  prefill numbers are comparable across runs and topologies.
- **Greedy, 400-token completion ladder.** Both DeepSeek-V4-Flash-0731 and
  DeepSeek-V4-Flash-Vision-Exp report decode by content type (technical,
  prose, code) at temperature 0, 400 generated tokens each, because
  speculative acceptance is strongly content-dependent — a single prompt is
  not a measurement.
- **Warm up once, then take at least 3 repetitions.** The Qwen protocol uses
  1 warmup request plus 3 measured samples. A confirmed case: an *unchanged*
  server produced aggregate numbers ranging from 101.9 to 130.6 tok/s across
  four back-to-back runs — a single run at n=1 is not a measurement.
- **Expect about 5% node-to-node spread** even on the identical recipe and
  protocol. The same Qwen3.8-27B configuration measured 140.5 tok/s on one
  physical card and 147.7 tok/s on another; treat differences under about 5%
  as card-to-card variance, not a real regression.
- **Discard the first request after a cold boot.** It carries Triton JIT
  compilation time and reads roughly 4x low.
- **Disable prefix caching for prefill benchmarks.** Without
  `--no-enable-prefix-caching`, a repeated benchmark prompt can hit the
  prefix cache and report a multi-hundred-token prompt "prefilling" in a
  fraction of a second — a measurement artifact, not a real number.

## See also

- [MODEL-STATUS.md](MODEL-STATUS.md) — every model attempted on this card, one table.
- [BENCHMARKS.md](BENCHMARKS.md) — the normalized, sanitized measurement ledger.
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — symptom-to-cause diagnosis at the hardware/driver layer.
- [COOLING-AND-POWER.md](COOLING-AND-POWER.md) — power policy and thermal detail.
- [QC.md](QC.md) — the full acceptance ladder and isolation matrix.
