# CMP 170HX public research digest (2026-09-02)

Web-only literature and community-source review. Not verified by this project unless marked "our measurement." Every claim carries a source tag:
**[paper]** arXiv/peer-reviewed, **[vendor]** NVIDIA/OEM doc, **[community]** forum/repo/blog with measured numbers, **[rumor]** claimed, not verified, **[unverified]** single source, no corroboration.

## Starting paper: arXiv 2505.03782

**[paper]** "Exploration of Cryptocurrency Mining-Specific GPUs in AI Applications: A Case Study of CMP 170HX" (Xing Kangwei, 2025). https://arxiv.org/abs/2505.03782

Central technique: patch CUDA source to disable Fused Multiply-Add (FMA) contraction, recovering throttled FP32 compute — reports >15x FP32 uplift and >3x LLM-inference improvement at some precisions using OpenCL benchmark, mixbench, and a llama-based benchmark harness. This is a compute-throttle workaround, distinct from and earlier than the 2026 VRAM/PCIe unlock work described below. The paper's listed silicon specs (GA100-105F-A1, 70 SM, 4480 CUDA cores, 8 GB HBM2e, 1493 GB/s theoretical bandwidth) partly conflict with other community sources reporting 8960 CUDA cores for the same 70-SM count; treat the CUDA-core figure as unresolved between sources.

## 1. Silicon and specs

**[community]** GA100-105F-A1 die, 70 SMs, Compute Capability 8.0 (same architecture family as A100). Stock VRAM 8 GB (10de:20c2) or 10 GB (10de:2082) variant, HBM2e, 4096-bit bus.

Measured figures (niconiconi teardown, https://niconiconi.neocities.org/tech-notes/nvidia-cmp-170hx-review/):
- Bandwidth (clpeak): ~1,150–1,355 GB/s depending on vector width.
- FP32 FMA (stock, gated): ~395 GFLOPS. FP32 without FMA (patched): ~6,285 GFLOPS.
- FP16: ~42 TFLOPS.
- INT32: ~12,500 GIOPS; hashcat MD5 ~44,000 MH/s (~68% of a real A100).
- Tensor-core mode (`gpu_burn -tc`) measured identical to plain non-FMA FP32 — suggestive that Tensor Cores are not delivering expected uplift on this card, but this is inference from a single benchmark, not a confirmed gating mechanism. **[community, unverified interpretation]**
- Power: 250 W TDP, 8-pin connector; idle ~30 W; workload-dependent draw 60–180 W depending on kernel type.
- Cooling: no onboard fan; community water-block solution (Bykski `N-TESLA-A100-X-V2`) holds ~45 °C at 180 W with a 360 mm radiator.

One market source cites ~700–800 GB/s "real world" bandwidth for unlocked cards, notably lower than niconiconi's clpeak measurements — likely reflecting partially-defective unlocked memory banks rather than a hard ceiling. **[community, unverified vs. niconiconi]**

## 2. The 64 GB VRAM unlock

**[community]** Mechanism: reported exploit of a signature-loading bug in the GPU's Falcon BootROM security microcontroller, entirely software-based, no physical modification. Primary tool: `cmpunlocker` (GPL), usage `sudo ./install.sh --profile=8gb|10gb` + cold reboot. Requires Linux x86-64, root, NVIDIA's open-kernel driver 610.43.0x, and Secure Boot disabled.

- 8 GB cards reportedly unlock to 64 GB; 10 GB cards to 40 GB. A claimed 80 GB configuration for 10 GB cards is contested — one detailed community wiki reports it was "built, tested, and rejected as unstable," while at least one press outlet reported it as achieved. **[community, conflicting]**
- Reliability is inconsistent: NVIDIA fuses off HBM2e regions that failed factory test, so unlocking exposes whatever is physically present, good or defective — no way to predict outcome per card. QC relies on tools like `memtest_vulkan` to test and disable bad banks.
- A specific "~50% failure rate above 8GB" statistic was not found in any source reviewed.
- Caution: at least one separate GitHub repo distributes an "unlocked VBIOS" dump rather than the GPL software-unlock route; secondary sources warn this carries real risk (bricking, unverified provenance) versus the documented cmpunlocker method.
- Linux-only in every source found; no working Windows path reported anywhere.

## 3. PCIe

**[community, measured]** Stock: PCIe Gen1 x4 electrical (`lspci`: `LnkCap: Speed 2.5GT/s, Width x16`; `LnkSta: Speed 2.5GT/s, Width x4 (downgraded)`; measured throughput ~0.85 GB/s). `LnkCap2` on stock cards lists only 2.5GT/s (Gen1) as a supported speed.

- **x16 width mod (hardware):** 12 of 16 PCIe lanes are missing AC-coupling capacitors on the PCB. Soldering ~24 capacitors (0402 package, exact value reported inconsistently across sources) restores x16 width. At least one named community member reported success in April 2026. Measured result: stock Gen1 x4 ~1.58 Gbit/s vs. modded Gen1 x16 ~6.37 GB/s. Guide: 170th Street gitbook, https://170th-street.gitbook.io/hx/modifications/pcie-capacitor-mod.
- **Gen2 software claim:** A community patch reportedly raises link speed to Gen2 purely via register override, no soldering. This claim is **contested** — it conflicts with our own cards, which continue to report LnkCap 2.5GT/s on the host, and with independent teardown data showing the stock LnkCap2 field lists only 2.5GT/s as supported at all. Treat as **rumor/unreliable pending independent reproduction.**
- **Gen3/Gen4:** Universally reported as blocked by a factory-burned OTP fuse, with no known software bypass. No credible report of a working Gen3/4 unlock was found in English or via Chinese-language search terms.

## 4. NVLink

**[community]** Classification: **not achieved, likely requires board-level rework.**

The GA100 die supports NVLink, but 170HX boards are reported (via direct teardown) to have NVLink gold-finger pads present but all supporting components unpopulated on the PCB — i.e., the physical bridge hardware was deliberately omitted, not just fused off in firmware. No source found describes a successful NVLink enable on a 170HX or any comparable GA100 mining-derivative card. Community documentation lists it as an open, unsolved problem rather than a completed unlock.

## 5. Software ecosystem (SM80 inference)

**[community]** The 170HX reports Compute Capability 8.0 (architecturally an A100 for kernel-selection purposes).

- Community vLLM forks run modern MoE models (e.g., DeepSeek-V4-Flash) on SM80 hardware by patching around upstream's Hopper-only assumptions. One such project reports ~98 tok/s decode and ~5,300 tok/s prefill on 4x 64GB-unlocked SM80 cards using pipeline-parallel execution and speculative decoding — self-reported by the project author, not independently reproduced elsewhere as of this research.
- Upstream vLLM discussion confirms SM80/A100-class support for these newer MoE architectures is feasible via Triton/PyTorch fallback paths (no new CUDA needed), with community forks existing because upstream has generally deferred official SM80 support to forks.
- Quantization compatibility on SM80 (general, vendor-documented, not CMP-specific):
  - **Works:** Marlin-kernel INT4 (AWQ/GPTQ weight-only), weight-only FP8 via a Marlin fallback kernel, INT8 W8A8 via CUTLASS.
  - **Does not work:** Native NVFP4/MXFP4 tensor-core-native paths and native FP8 activation/tensor-core execution — these require Hopper/Blackwell-class tensor cores and are not available on SM80.
- A single forum report describes running a 27B W8A8 model via vLLM on one unlocked 170HX, with decode performance close to a pair of AMD R9700 cards but prefill roughly 50% slower.

## 6. Market

**[community]** Cards traded around $100–$300 before the 2026 unlock-tool release; multiple independent outlets report a subsequent spike to roughly $1,000–$2,000+ (with some outlier asks as high as $4,000) once the VRAM unlock became public. Used real A100 40GB modules trade around $3,500 for comparison, so the economic case for the modded 170HX narrowed sharply once its own price rose. On performance, community reports place it roughly comparable to an RTX 3090 on small models and 1.5–2x slower than an RTX 5090 on 70B+ class models — but it remains one of the only budget paths to fit 64GB+ of weights on a single card. A cloud-rental listing (Vast.ai) was reported at roughly $0.24–0.27/GPU-hour. All figures are volatile and reported by secondary sources; treat as directional, not precise.

## 7. Community hubs

1. Community technical wiki covering silicon, firmware, the unlock procedure, and open problems (NVLink, ECC, PCIe Gen3/4) — the most structured single reference found.
2. `cmpunlocker` — the primary GPL unlock tool.
3. "170th Street" community site/gitbook — hardware teardown, PCIe capacitor mod guide, watercooling guide, AI/ML notes.
4. niconiconi's independent technical review/teardown — the most detailed independently measured source found (real `lspci`/`nvidia-smi` output, benchmark numbers, a schematics-based repair log).
5. NVIDIA Developer Forums and Level1Techs forum threads discussing multi-card stacking and single-card vLLM deployments.
6. GitHub projects adapting vLLM for SM80 MoE inference on this hardware.
7. This project — **club-170hx** — community-tested recipes, diagnostics, and reproducible benchmarks for CMP 170HX, with a companion project for DGX Spark comparisons.
8. Chinese-language primary sources (Bilibili/Zhihu/Taobao) were searched for but not independently found in this pass; Chinese-market activity is known only through secondary English/Taiwanese reporting.

## Verified vs. rumor

| Claim | Status |
|---|---|
| CMP 170HX = GA100-105F, SM80, same die family as A100 | Verified (community, corroborated) |
| Stock VRAM cap is firmware/OTP, not physical absence | Verified (community, multiple independent unlock reports) |
| 8GB card unlocks to 64GB | Verified (community, multiple sources agree) |
| 10GB card unlocks to 40GB | Verified (community) |
| 10GB card unlocks to 80GB | Contested / rumor (sources disagree) |
| "~50% failure rate above 8GB" | Unverified — no source found |
| Stock PCIe is Gen1 x4 | Verified (measured) |
| x16 width unlock via capacitor soldering | Verified (community, named successful builder) |
| PCIe Gen2 software unlock | Rumor / contested — conflicts with our own cards' behavior |
| PCIe Gen3/Gen4 unlock | Not achieved — OTP-fused, no bypass found |
| NVLink enable | Not achieved — board likely lacks populated bridge hardware |
| Tensor Cores software-gated | Plausible, unconfirmed — single-benchmark inference |
| SM80 vLLM MoE inference (~98 tok/s decode on 4 cards) | Community-measured, self-reported, not third-party-reproduced |
| INT4/FP8 weight-only quantization works on SM80 | Verified (vendor docs, general architecture support) |
| Native NVFP4/MXFP4/FP8-activation works on SM80 | False — Hopper/Blackwell-only |
| Price spike to $1,000–2,000+ after unlock release | Verified (multiple independent outlets) |

## What we could test on our four modded cards

Our cards currently show LnkCap 2.5GT/s on the host — i.e., the claimed PCIe Gen2 software mod has not visibly changed host-reported link capability. Ranked by expected value:

1. **Confirm or refute the PCIe Gen2 software-patch claim** with before/after `lspci -vv` and `nvidia-smi -q` logs. Cheap, no hardware risk, resolves a live community controversy directly on our hardware.
2. **Reproduce the Tensor-Core-gating test** (`gpu_burn -tc` vs. non-FMA FP32) to independently check the single-source claim that Tensor Core mode delivers no uplift over plain FP32 on this card.
3. **Measure real achievable HBM2e bandwidth** on our unlocked cards and compare against both the higher (~1,150–1,355 GB/s) and lower (~700–800 GB/s) figures reported elsewhere, to see where our specific units fall.
4. **Attempt the x16 PCIe capacitor mod on one card** and benchmark model-load time before/after — the highest-effort, highest-payoff item, testing the single most consequential unverified performance claim in the literature.
5. **Independently reproduce the reported 4-card SM80 MoE decode/prefill numbers**, since our rig already matches the hardware profile those community figures were reported on, and no third party has reproduced them yet.

---
*Compiled 2026-09-02 from public web sources. Not for use as a purchasing or safety guarantee — unlock and modification procedures described here carry real risk of hardware damage and are undertaken at the operator's own risk.*
