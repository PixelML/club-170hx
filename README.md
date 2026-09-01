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

**Measured on 2026-08-30:** all four cards enumerated in one Ubuntu guest with driver 610.43.03 and reported 65,536 MiB each. This is a bring-up inventory, not a four-card performance benchmark. Under the current build (four cards in an open frame with one 80 mm blower on a printed duct), a later idle snapshot the same day showed 37–38 °C cores, 41–51 °C memory temperatures, and about 141 W for the group at zero utilization. Cooling and power details are in [Cooling and power](docs/COOLING-AND-POWER.md).

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

Canonical scoreboard of PixelML measurements on CMP 170HX. The status column
separates publication-safe results from provisional or blocked measurements.
Normalized metrics, methodology, and evidence tiers:
[docs/BENCHMARKS.md](docs/BENCHMARKS.md). Raw manifests and receipts stay in
the model repositories; small redacted snapshots may be retained here when the
detailed ledger is not public.

| Workload | Quant / runtime | Topology | Decode | Aggregate | Quality / success | Status | Evidence |
|---|---|---|---|---|---|---|---|
| GLM-5.3-Flash | UD-IQ4_XS · llama.cpp | 4 cards · layer split · 16k ctx · c ≤ 4 | 17.73 tok/s median @ c=1 | ~17.5–17.7 tok/s @ c=2/4 | 21/26 local tasks · 41/41 soak | Publication-safe | [Result card](results/2026-08-30-glm-5.3-flash-ud-iq4xs-llamacpp-cmp170hx.md) · [Evidence pin](https://github.com/PixelML/GLM-5.3-Flash-CMP-170HX/blob/7fc71e00925f7b7902764aab7d08b6d923aaaea4/results/phase63/run-manifest.json) |
| GLM-5.3-Flash | NVFP4 | 3 cards | — | — | — | Not compatible: SM121-format weights on SM80; AWQ INT4 ≈ 66 GiB/card does not fit 3 × 64 GiB | [Negative result](docs/BENCHMARKS.md#negative-results-matter) |
| Qwen3.8-27B | NVFP4 · vLLM | 1 card × 3 runs @ 180 W | — | — | — | Pending evidence repair: [repo PR 1](https://github.com/PixelML/Qwen3.8-27B-CMP-170HX/pull/1) | [Repo](https://github.com/PixelML/Qwen3.8-27B-CMP-170HX) |
| DeepSeek-V4-Flash-0731 | FP8 · vLLM pipeline | 3 cards | — | — | — | Pending evidence repair: [repo PR 1](https://github.com/PixelML/DeepSeek-V4-Flash-0731-CMP-170HX/pull/1) | [Repo](https://github.com/PixelML/DeepSeek-V4-Flash-0731-CMP-170HX) |
| DeepSeek-V4-Flash-Vision-Exp | FP8 · SM80 vLLM fork | 4 cards · PP4 | **59.78 tok/s warm** · 56.6 sustained | **169.65 tok/s @ c=4** · 325.5 tok/s prefill | Text passed; image rejected (HTTP 400) | Provisional measured text baseline; vision and immutable runtime pin still blocked | [Result summary](results/2026-08-31-deepseek-v4-flash-vision-exp-cmp170hx.md) |

`—` = not presented without sanitized stable evidence. Pending and provisional
rows are not decision-grade; they remain visible so measured learning is not
lost, but their status and blockers must stay explicit.

The Vision-Exp row is intentionally visible but not labeled publication-safe:
the key text numbers are measured, while the SM80 runtime does not yet wire the
vision tower and its source revision was unavailable from the measured image.

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
