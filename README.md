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
| Choose an AI workload | [Workload matrix](docs/WORKLOADS.md) |

## Four-card rig update

![Four passively cooled CMP 170HX cards before installation](assets/four-cmp-170hx-cards.png)

I traded out three RTX 3090s and rebuilt this node around four CMP 170HXs. The reason was simple: the four cards expose 256 GiB of aggregate HBM in one box. The lab still has RTX 3090 cards and DGX Spark systems, giving us useful comparison points for consumer CUDA, low-cost SM80/HBM, and GB10. DGX Spark notes live in [club-dgx-spark](https://github.com/PixelML/club-dgx-spark).

**Measured on 2026-08-30:** all four cards enumerated in one Ubuntu guest with driver 610.43.03 and reported 65,536 MiB each. With a 180 W cap, the idle snapshot showed 31–32 °C core temperatures, 35–45 °C memory temperatures, 32–36 W per card, zero memory use, and zero GPU utilization. This is a bring-up result, not a four-card performance benchmark.

## Verified baseline

The current hardware and repeatable software baseline is:

- 4 × CMP 170HX installed, each reporting 64 GiB VRAM;
- published load tests on one- and three-card topologies;
- Ubuntu 22.04 guest on a Proxmox Q35 VM with SeaBIOS;
- Linux 6.8 and NVIDIA 610.43.03 open kernel modules;
- pinned `cmpunlocker` v0.1 patch set;
- 125 W quiet/idle policy and 180 W benchmark policy;
- forced airflow across every passive heatsink.

Exact versions matter. Treat this as a known-good reference, not a claim that every card, board, BIOS, kernel, or driver combination will work.

## Published workload results

### PixelML measurements

| Workload | Topology and power | Decode | Prefill and latency | Evidence |
|---|---|---:|---:|---|
| Qwen3.8-27B W4A16 + DFlash2 | 1 card at a time, 3 cards tested, 180 W cap | **136.4 tok/s mean** (133.6–140.3) for 256-token generations | 1,926–1,957 tok/s at 6.6K context; 181–201 ms TTFT | [Pinned results](https://github.com/PixelML/Qwen3.8-27B-CMP-170HX/blob/41d2c414fe0f293d77087ef18cda5896664754d6/RESULTS.md) |
| DeepSeek-V4-Flash-0731 native weights + DSpark | 3 cards, pipeline parallel, 180 W/card | **83.3 tok/s aggregate** across three 400-token prompts | **2,965 tok/s** at 5.4K context | [Pinned results](https://github.com/PixelML/DeepSeek-V4-Flash-0731-CMP-170HX/blob/5c5b5a4b45e8def82ec027737df616c55f997963/RESULTS.md) |

GLM-5.3-Flash remains a compatibility result rather than a speed result: no completed CMP 170HX serving run has been published. See the [negative-result notes](docs/BENCHMARKS.md#negative-results-matter).

### Community reference

**Community-reported, not independently reproduced by PixelML:** [allover326's four-card DeepSeek-V4-Flash run](https://github.com/allover326/deepseek-v4-cmp170hx/tree/3dd2d8817e7deae00d998edde0d227e7254ea71e) used pipeline parallelism and DSpark at 180 W/card.

| Decode | Prefill and latency | Long context | Evidence |
|---:|---:|---:|---|
| **98.1 tok/s** single-stream aggregate; **712.8 tok/s** at 64 concurrent requests | **5,207 tok/s** at 77K context; 14.6 s TTFT at 100K in the PP/TP sweep | 1,047,736-token one-shot prefill; 1,002,852-token accumulated conversation with row chunk 64 | [Pinned results](https://github.com/allover326/deepseek-v4-cmp170hx/blob/3dd2d8817e7deae00d998edde0d227e7254ea71e/RESULTS.md) |

At roughly 1.04M tokens, that community run measured 1,904 tok/s prefill, about 550 seconds to first token, and 35.6 tok/s decode. Retrieval accuracy also fell from about 100% at 150K to 30% at 900K, so the maximum window is a capacity result, not a claim that every token remains equally useful.

Do not compare these rows as a leaderboard. The models, prompts, context lengths, concurrency, and runtime patches differ. Decode rates above use generated-token counts from the final usage result or a fixed output count; they do not count streaming events as tokens.

### Model, quant, and runtime strategy

| Lane | Evidence | Current guidance |
|---|---|---|
| DeepSeek-V4 native checkpoint (MXFP4 experts + FP8 attention) with DSpark | **Measured by PixelML:** 83.3 tok/s on 3 cards | Best published large-model lane on this node. Start with pipeline parallelism, `15,15,13` layer placement, DSpark `k=5`, and FP8 KV. |
| Qwen3.8-27B W4A16 AutoRound + W4A16 DFlash2 | **Measured by PixelML:** 136.4 tok/s mean across 3 independently tested cards | Best published single-card speed lane. This is W4A16, not NVFP4; correct usage-token counting and the prepared fast variant both matter. |
| GLM-5.2 symmetric Int4/Int8 mix with an unquantized MTP head | **Community-reported:** 28.47 tok/s baseline on 8 CMP 170HX cards; MTP serving initialization verified | Useful SM80 reference for future DSA models, but not a four-card recipe. The [pinned vLLM composition](https://github.com/allover326/vllm-dsa-mtp-sm80/blob/56bba6097b06b3c0d981de3a6cef63ed394d2626/README.md) combines a Triton sparse-MLA backend with MTP under pipeline parallelism. |
| GLM-5.3-Flash NVFP4, EXL3, and AWQ attempts | **[PixelML stable summary](https://github.com/PixelML/GLM-5.3-Flash-CMP-170HX/blob/a2f22cc9641c3a95c841c6b06d58c6dcabb0f92e/README.md):** no completed serving result | Keep these as fit/runtime investigations. Do not publish throughput until a checkpoint both fits and reaches an SM80-capable serving path. |
| W4AFP8 or other FP8-activation quants | **Community-reported:** the tested GLM-5.2 paths require SM89 or newer | Reject at the fit gate unless the runtime documents an SM80-safe implementation. This does not rule out FP8 KV: the DeepSeek-V4 run above used FP8 KV successfully on SM80. |

Three rules keep repeating across our attempts and the community work:

1. Start with pipeline parallelism, not tensor parallelism, when cards communicate over slow PCIe without P2P.
2. Rebalance the final pipeline rank when it also holds the LM head or speculative draft; default equal splits can fail even when total VRAM is sufficient.
3. Check the quant details, not only the bit count. Symmetric versus asymmetric MoE weights, activation format, KV format, and whether the draft head is quantized can change compatibility.
4. Match the build to the patch. Our DeepSeek-V4 run required compiled SM80 operators; a precompiled image that applied only Python overlays failed during graph capture.
5. Verify composed patches with import and undefined-name checks, not only syntax compilation. The community composition includes this gate after a real missing-helper failure.

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
