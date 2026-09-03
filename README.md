# Club CMP 170HX

Community-tested recipes, diagnostics, and reproducible benchmarks for running AI workloads on NVIDIA CMP 170HX cards.

This project is building a practical, low-cost SM80 compute pool for:

- large-language-model inference;
- image generation;
- video generation;
- CUDA validation and memory-heavy research workloads;
- single-node and distributed experiments.

The CMP 170HX shares useful traits with A100-class hardware, including SM80 compute capability and high-capacity HBM. It is **not an A100 replacement**: it has an unsupported software path, no display output, no NVLink, limited PCIe behavior in common passthrough setups, and unusual cooling and power requirements.

## Start here

| Goal | Guide |
|---|---|
| Understand the card and trade-offs | [Hardware](docs/HARDWARE.md) |
| Install it in a Proxmox VM | [Installation](docs/INSTALLATION.md) |
| Validate a new or used card | [QC and acceptance testing](docs/QC.md) |
| Control heat and noise | [Cooling and power](docs/COOLING-AND-POWER.md) |
| Plan a multi-card node | [Cluster design](docs/CLUSTER.md) |
| Diagnose failures | [Troubleshooting](docs/TROUBLESHOOTING.md) |
| Compare measured results | [Benchmarks](docs/BENCHMARKS.md) |
| Avoid past operator mistakes | [Operator lessons](docs/OPERATOR-LESSONS.md) |
| What the world knows about this card | [Unlock and mod research](docs/RESEARCH-CMP170HX-UNLOCKS.md) |
| Choose an AI workload | [Workload matrix](docs/WORKLOADS.md) |
| Read the consolidated lessons | [What we learned](docs/LESSONS.md) |
| See every model tried on this card | [Model status](docs/MODEL-STATUS.md) |
| Run a specific model: quick start, settings, troubleshooting | [Model guides](docs/models/README.md) |
| Read an executed experiment notebook | [Notebooks](notebooks/README.md) |

## Notebooks

Each row links one executed Jupyter notebook, top to bottom, with committed
outputs. The schema and the LIVE-replay convention are in
[notebooks/README.md](notebooks/README.md).

| Date | Experiment | Headline | Notebook | Video |
|---|---|---|---|---|
| 2026-09-02 | Four-card tensor-core correctness gate, post-incident recovery verification | 4/4 cards PASS, 74.6–78.5 TFLOP/s, 0 Xid; recovery check after a fleet-wide OOM led to a UVM fatal error and a VM reboot | [notebooks/2026-09-02-cmp170hx-health-gate.ipynb](notebooks/2026-09-02-cmp170hx-health-gate.ipynb) | — |
| 2026-09-02 | DeepSeek-V4-Flash-Vision-Exp, 4x CMP 170HX, PP4 + DSpark k=6 | 220.2 tok/s aggregate decode at c=8 (median of 3); c=16 fails with a device-side assert | [notebooks/2026-09-02-deepseek-v4-flash-vision-exp-4card-pp4-vllm.ipynb](notebooks/2026-09-02-deepseek-v4-flash-vision-exp-4card-pp4-vllm.ipynb) | [mp4](assets/video/dsv4-vision-4card-motion/dsv4-vision-4card-motion-1080x1920.mp4) |
| 2026-09-02 | DeepSeek-V4-Flash-Vision-Exp, vision on 4x CMP 170HX, PP4 + DSpark k=6 | Vision gates PASS, 10/10 image keyword match; text-only decode 119 tok/s median of 5 reps (peak 162) @ c=1, server crashed at c=4 | [notebooks/2026-09-02-deepseek-v4-flash-vision-exp-4card-vision-pp4-vllm.ipynb](notebooks/2026-09-02-deepseek-v4-flash-vision-exp-4card-vision-pp4-vllm.ipynb) | [mp4](assets/video/dsv4-vision-4card-vision-motion/dsv4-vision-4card-vision-motion-1080x1920.mp4) |
| 2026-09-02 | DeepSeek-V4-Flash-Vision-Exp highlight reel: text ladder, vision gates, 5 SM80 fixes, cross-platform tok/Wh | 10/10 image keyword match; C1 97-119, C8 220 tok/s aggregate; CMP 170HX 2.4x cheaper per token than 2x DGX Spark at c=8 | [notebooks/2026-09-02-deepseek-v4-flash-vision-exp-4card-vision-pp4-vllm.ipynb](notebooks/2026-09-02-deepseek-v4-flash-vision-exp-4card-vision-pp4-vllm.ipynb) | [mp4](assets/video/club-170hx-highlights-2026-09-02/club-170hx-highlights-2026-09-02-1080x1920.mp4) |
| 2026-09-02 | DeepSeek-V4-Flash-Vision-Exp, chart-led 20s cut: tok/s vs concurrency, tok/Wh, 4x CMP 170HX vs 2x DGX Spark | 119 tok/s median at c=1 (peak 162); c=4+ crashes on the vision build; CMP 2.4x cheaper per token, Spark 2x more efficient | [notebooks/2026-09-02-deepseek-v4-flash-vision-exp-4card-pp4-vllm.ipynb](notebooks/2026-09-02-deepseek-v4-flash-vision-exp-4card-pp4-vllm.ipynb) | [mp4](assets/video/dsv4-vision-4card-chart-20s/dsv4-vision-4card-chart-20s-1080x1920.mp4) |
| 2026-08-30 | DeepSeek-V4-Flash-0731, 3x CMP 170HX, PP3 vLLM | 83.3 tok/s aggregate decode, DSpark k=5, 180 W/card | [notebooks/2026-08-30-deepseek-v4-flash-0731-3card-pp3-vllm.ipynb](notebooks/2026-08-30-deepseek-v4-flash-0731-3card-pp3-vllm.ipynb) | — |
| 2026-08-30 | Qwen3.8-27B W4A16 AutoRound + DFlash2, 1x CMP 170HX vLLM | Best local 140.3 tok/s decode at 180 W (95% of a 255 W rented card's decode at 71% of the power cap) | [notebooks/2026-08-30-qwen3.8-27b-w4a16-dflash2-1card-vllm.ipynb](notebooks/2026-08-30-qwen3.8-27b-w4a16-dflash2-1card-vllm.ipynb) | [mp4](assets/video/qwen38-27b-motion/qwen38-27b-motion-1080x1920.mp4) |
| 2026-08-31 | GLM-5.3-Flash compatibility, CMP 170HX (negative result) | NVFP4 incompatible on SM80; llama.cpp UD-IQ4_XS fallback runs at 17.73 tok/s (c=1) | [notebooks/2026-08-31-glm-5.3-flash-compatibility-cmp170hx.ipynb](notebooks/2026-08-31-glm-5.3-flash-compatibility-cmp170hx.ipynb) | — |
| 2026-09-02 | GLM-5.3-Flash, 4x CMP 170HX, EXL3 4.05bpw, exllamav3 + TabbyAPI | 26.9-44.8 tok/s aggregate decode (c=1-c=8) at the vBIOS default 250 W, found accidental — see 2026-09-03; 20/20 golden corpus; single-request context capped at ~2,048 tokens under the standing Q8 KV cache | [notebooks/2026-09-03-glm-5.3-flash-exl3-4gpu-tabbyapi.ipynb](notebooks/2026-09-03-glm-5.3-flash-exl3-4gpu-tabbyapi.ipynb) | — |
| 2026-09-03 | GLM-5.3-Flash, 4x CMP 170HX, power cap correction | Re-measured at the verified 180 W club-standard cap: 25.2-44.6 tok/s aggregate decode (c=1-c=8), no consistent difference from the 250 W run outside noise; 180 W is now the canonical cap | [docs/models/glm-5.3-flash.md](docs/models/glm-5.3-flash.md) | — |

## Four-card rig update

![Four passively cooled CMP 170HX cards before installation](assets/four-cmp-170hx-cards.png)

I traded out three RTX 3090s and rebuilt this node around four CMP 170HXs. The reason was simple: the four cards expose 256 GiB of aggregate HBM in one box. The lab still has RTX 3090 cards and DGX Spark systems, giving us useful comparison points for consumer CUDA, low-cost SM80/HBM, and GB10. DGX Spark notes live in [club-dgx-spark](https://github.com/PixelML/club-dgx-spark).

**Measured on 2026-08-30:** all four cards enumerated in one Ubuntu guest with driver 610.43.03 and reported 65,536 MiB each. Under the current build (four cards in an open frame with one 80 mm blower on a printed duct), an idle snapshot the same day showed 37–38 °C cores, 41–51 °C memory temperatures, and about 141 W for the group at zero utilization. Cooling and power details are in [Cooling and power](docs/COOLING-AND-POWER.md).

### Four-card results: DeepSeek-V4-Flash-Vision-Exp

The four cards now serve `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` (FP8, 48 shards, revision `86f746b3`). Vision now runs on the same SM80 vLLM fork that serves text, on the same PP4 + DSpark k=6 recipe. A separate reference TP4 runtime, kept as history, gave the first-ever real-image PASS on this checkpoint on Ampere hardware.

1. **Text and vision, same recipe, PP4 + DSpark k=6.** Five boot fixes (see [Lessons](docs/LESSONS.md)) got this SM80 fork's vision path to a running server. Functional gates pass, including 10/10 image keyword match. The text-only ladder ran through c=4, where the server crashed; once the server came back up later in the session, the text+image ladder ran clean at c=1 and c=2. c=8, c=16 text-only, and text+image at c=4 and above, are not measured.
2. **Vision path, correctness, reference TP4 runtime (history).** The reference TP4 runtime with SM80 fallback patches completed the first real-image inference of this checkpoint on Ampere hardware. It decodes at about 0.9 tok/s and is a correctness result, not a performance result.

A 5-rep reproducibility check on this same c=1 recipe found a wide run-to-run
spread (48.5-161.7 tok/s), tracked to DSpark draft-acceptance swings
(0.20-0.83 accept ratio), not to the power cap. Detail:
[docs/BENCHMARKS.md](docs/BENCHMARKS.md#reproducibility-and-power-cap-2026-09-02).

| Measurement | Value | Status |
|---|---:|---|
| Functional gates | PASS (/v1/models, deterministic greedy, image keyword match 10/10) | Measured 2026-09-02 |
| Golden corpus, text (20 rows) | 15/20 keyword match, 10/20 exact-match vs. DGX Spark reference | Measured 2026-09-02, known limitation |
| Decode, c=1 (text-only) | 119 tok/s median of 5 reps (peak 162) aggregate | Measured 2026-09-02; DSpark acceptance variance drives run-to-run spread |
| Decode, c=2 (text-only) | 116.6 tok/s aggregate (median of 3) | Measured 2026-09-02 |
| Decode, c=4 (text-only) | server crashed on rep 3 of 3 (EngineCore died) | Measured 2026-09-02 |
| Decode, c=8 / c=16 (text-only), text+image (c=4 to c=16) | not measured | Not measured |
| Decode, c=1 (text+image) | 45.3 tok/s aggregate (median of 3) | Measured 2026-09-02 |
| Decode, c=2 (text+image) | 78.2 tok/s aggregate (median of 3) | Measured 2026-09-02 |
| Uncached prefill, 2,941 input tokens | 2,352 tok/s warm (918 tok/s first cold prefill) | Measured 2026-09-02 |
| Warm TTFT | 0.386 s | Measured 2026-09-02 |
| Real-image completion (reference TP4 runtime, history) | PASS, 0.9 tok/s decode | Correctness evidence only |

Full protocol, the crash detail, and the five boot fixes are in
[notebooks/2026-09-02-deepseek-v4-flash-vision-exp-4card-vision-pp4-vllm.ipynb](notebooks/2026-09-02-deepseek-v4-flash-vision-exp-4card-vision-pp4-vllm.ipynb)
and [Benchmarks](docs/BENCHMARKS.md#deepseek-v4-flash-vision-exp-four-cards). The SM80 vLLM fork and patch set build on the work of [allover326](https://github.com/allover326/vllm-dsa-mtp-sm80).

**Cross-platform context.** The same checkpoint at the same revision also runs on a two-node DGX Spark kit (GB10, vLLM, TP=2): c=1 36.9 tok/s, c=6 112.7 tok/s aggregate, uncached prefill 1,789 tok/s, vision PASS (merged evidence; a normalized 2,941/400-token rerun at c=1 48.7 is in an open PR there). Results: [DeepSeek-V4-Flash-Vision-Exp-DGX-Spark](https://github.com/PixelML/DeepSeek-V4-Flash-Vision-Exp-DGX-Spark). The two platforms use different runtimes, parallelism, and memory budgets. Read the numbers as context, not as a head-to-head comparison.

## Verified baseline

The current hardware and repeatable software baseline is:

- 4 × CMP 170HX installed, each reporting 64 GiB VRAM;
- published load tests on one-, three-, and four-card topologies;
- Ubuntu 22.04 guest on a Proxmox Q35 VM with SeaBIOS;
- Linux 6.8 and NVIDIA 610.43.03 open kernel modules;
- pinned `cmpunlocker` v0.1 patch set;
- 125 W quiet/idle policy and 180 W benchmark policy;
- forced airflow across every passive heatsink.

Exact versions matter. Treat this as a known-good reference, not a claim that every card, board, BIOS, kernel, or driver combination will work.

## Published workload results

Canonical scoreboard of PixelML measurements on CMP 170HX. The status column
separates publication-safe results from provisional or blocked measurements.
Normalized metrics, methodology, and evidence tiers:
[docs/BENCHMARKS.md](docs/BENCHMARKS.md). Raw manifests and receipts stay in
the model repositories; small redacted snapshots may be retained here when the
detailed ledger is not public.

| Workload | Quant / runtime | Topology | Decode | Aggregate | Quality / success | Status | Evidence |
|---|---|---|---|---|---|---|---|
| GLM-5.3-Flash | EXL3 4.05bpw · exllamav3 + TabbyAPI | 4 cards · manual `gpu_split` · 180 W cap (canonical) | 25.2 tok/s @ c=1 | 44.6 tok/s @ c=8 | 20/20 golden corpus | Publication-safe | [Result card](results/2026-09-03-glm-5.3-flash-exl3-4gpu-tabbyapi/README.md) · [Guide](docs/models/glm-5.3-flash.md) · [Benchmarks](docs/BENCHMARKS.md#glm-53-flash-exl3-405bpw-four-cards-exllamav3--tabbyapi-publication-safe) |
| GLM-5.3-Flash | UD-IQ4_XS · llama.cpp | 4 cards · layer split · 16k ctx · c ≤ 4 | 17.73 tok/s median @ c=1 | ~17.5–17.7 tok/s @ c=2/4 | 21/26 local tasks · 41/41 soak | Publication-safe; superseded by the EXL3 row above | [Result card](results/2026-08-30-glm-5.3-flash-ud-iq4xs-llamacpp-cmp170hx.md) · [Evidence pin](https://github.com/PixelML/GLM-5.3-Flash-CMP-170HX/blob/7fc71e00925f7b7902764aab7d08b6d923aaaea4/results/phase63/run-manifest.json) |
| Qwen3.8-27B | W4A16 AutoRound (dbirks) + DFlash2 k=7 · vLLM | 1 card, 3 cards tested @ 180 W | 136.38 tok/s mean @ 256 tokens (122.00 @ 900 tokens) | single stream | TTFT 190.8 ms; prefill 1,946 tok/s | Measured 2026-08-30 | [Repo](https://github.com/PixelML/Qwen3.8-27B-CMP-170HX) · [Benchmarks](docs/BENCHMARKS.md#qwen38-27b-nvfp4-one-card) |
| Qwen3.8-27B | Runtime A/B: vLLM + DFlash2 control vs. Ninfer sm_80 fork (MTP) | 1 card @ 180 W | 38.16 tok/s Ninfer spec-on / 29.95 spec-off vs. 138.6 tok/s vLLM control | single stream | Verdict: stay on vLLM, 3.6x slower on Ninfer despite a higher measured clock | Measured 2026-09-02, negative for Ninfer | [Results](results/2026-09-02-qwen3.8-27b-ninfer-ab/README.md) · [Benchmarks](docs/BENCHMARKS.md#qwen38-27b-runtime-ab-vllm-vs-ninfer-sm_80-fork-measured-2026-09-02) |
| DeepSeek-V4-Flash-0731 | FP8 (native FP4 experts) · SM80 vLLM fork · PP3 · DSpark k=5 | 3 cards @ 180 W | 83.3 tok/s aggregate (technical 73.4 / prose 72.4 / code 116.6) | single stream | prefill 2,965 tok/s @ 5,399 tokens; acceptance 5.07–5.32 | Measured 2026-08-30 | [Repo](https://github.com/PixelML/DeepSeek-V4-Flash-0731-CMP-170HX) · [Benchmarks](docs/BENCHMARKS.md#deepseek-v4-flash-0731-three-cards) |
| DeepSeek-V4-Flash-Vision-Exp | FP8 · SM80 vLLM fork | 4 cards · PP4 · DSpark k=6 | 97.4 tok/s (median of 3; 57.6–123.5) @ c=1 | 165.5 tok/s (median of 3; 140.3–203.2) @ c=4 · failed (device-side assert, reproduced twice) @ c=16 · 2,352 tok/s warm (362 tok/s first cold prefill) prefill | Text passed; image not served on this path | Text-only recipe; vision now measured on the same fork (see next row) | [Repo](https://github.com/PixelML/DeepSeek-V4-Flash-Vision-Exp-CMP-170HX) · [Benchmarks](docs/BENCHMARKS.md#deepseek-v4-flash-vision-exp-four-cards) |
| DeepSeek-V4-Flash-Vision-Exp | FP8 · SM80 vLLM fork, vision-enabled (Path 3) | 4 cards · PP4 · DSpark k=6 | 119 tok/s median of 5 reps (peak 162) @ c=1, text-only · 45.3 tok/s @ c=1, text+image | 116.6 tok/s @ c=2, text-only (server crashed @ c=4, EngineCore died) · 78.2 tok/s @ c=2, text+image (c=4+ not attempted) | Vision gates PASS, 10/10 image keyword match; text exact-match 10/20 vs. DGX Spark reference | Measured 2026-09-02, partial (server crash cut the text-only ladder short; c=1 spread driven by DSpark acceptance variance) | [Repo](https://github.com/PixelML/DeepSeek-V4-Flash-Vision-Exp-CMP-170HX) · [Benchmarks](docs/BENCHMARKS.md#deepseek-v4-flash-vision-exp-four-cards) |
| DeepSeek-V4-Flash-Vision-Exp | FP8 → BF16 fallback · reference TP4 runtime + SM80 patches | 4 cards · TP4 · batch 1 | 0.9 tok/s | — | Real-image completion PASS; prefill OOM above ~1,024 tokens | Correctness evidence only, history; superseded as the vision benchmark by the row above | [Repo](https://github.com/PixelML/DeepSeek-V4-Flash-Vision-Exp-CMP-170HX) · [Benchmarks](docs/BENCHMARKS.md#vision-correctness-milestone) |

`—` = not presented without sanitized stable evidence. Pending and provisional
rows are not decision-grade; they remain visible so measured learning is not
lost, but their status and blockers must stay explicit.

The two Vision-Exp rows describe one checkpoint on two runtimes. The first
row is the in-progress normalized text benchmark; it supersedes an earlier
ladder whose runtime source revision was unavailable from the running image
and whose protocol differed from the single-stream baseline. The second row
is the vision-correctness milestone; its decode rate is not a performance
claim.

## Hugging Face

Curated, verified artifacts from this club: [PixelML/club-170hx: verified on CMP 170HX (SM80)](https://huggingface.co/collections/PixelML/club-170hx-verified-on-cmp-170hx-sm80-6a97bf4edc20b52c5cf454e3).

## Repository map

```text
docs/       Hardware, setup, QC, operations, troubleshooting, and results
scripts/    Read-only inventory, model-fit, and card-validation helpers
workloads/  LLM, image, and video workload recipes and status
results/    Submission format for reproducible community results
```

## Safety first

CMP 170HX cards use passive server heatsinks. Do not run sustained workloads without directed, monitored airflow. Our default stop thresholds are 80 °C core or 85 °C memory. A cold power cycle may be required after a card falls off the PCIe bus.

The unlock path is community-maintained and unsupported by NVIDIA. Back up the machine, pin known-good artifacts, verify module provenance, and expect recovery work.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md). Benchmark claims need raw, redacted evidence and full environment metadata. Security and privacy rules are in [AGENTS.md](AGENTS.md) and [SECURITY.md](SECURITY.md).

Inspired by the community-first documentation style of [club-3090](https://github.com/noonghunna/club-3090).

## License

Apache-2.0. See [LICENSE](LICENSE).
