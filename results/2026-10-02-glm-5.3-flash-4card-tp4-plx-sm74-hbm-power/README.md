# GLM-5.3-Flash TP4 on four cards behind PLX switches — 74 SMs, equalised HBM clock, power-cap sweep

Status: measured
Date: 2026-10-02

Follow-up to [2026-10-01 Morrowmake 1.6.0 TP4](../2026-10-01-glm-5.3-flash-morrowmake-1.6.0-4card-tp4-plx/README.md) on the same host, same recipe, same served configuration (P2P off + replicated embedding). Three hardware levers were changed one at a time and the decode matrix re-run after each: the SM count (70 → 74), the HBM clock of two cards (1,458 → 1,728 MHz), and the per-card power cap (165 → 100 W). A copy-drafts A/B on the PixelML/sm80vllm fork, run the evening before on the 70-SM driver, is included for completeness.

**Verdict (measured):**

- **HBM clock is the lever that moved decode.** Two of the four cards shipped with an older VBIOS that runs HBM at NDIV 54 (1,458 MHz); the other two run NDIV 64 (1,728 MHz). TP4 waits for the slowest card. Raising the two slow cards to NDIV 64 (after a hot 170tune gate) cut the decode step from 19.5 to 18.4 ms: single-user decode +5.8 to +6.3%, eight-user +3.1 to +3.8%, at identical draft acceptance. 418.0 tok/s structured single-user is the best this host has measured.
- **+4 SMs (70 → 74) did not move decode measurably.** Compared with the 70-SM run at 180 W the 74-SM run at 150 W is within −5% to +5% per prompt; the power cap differs between the two, so this is not a clean A/B.
- **Power: 140 W is free, 110 W is the efficiency peak.** 140 W keeps 98.7–99.7% of 150 W throughput for 8% less GPU power. Below 130 W the core clock falls fast: 120 W costs 7–12%, 110 W costs 13–22% but gives the most eight-user tokens per watt, 100 W loses on both. 165 W adds 0.1–1.1% and pushes HBM to 82–83 °C. The host now runs 140 W by default.
- **SM VF offset +200 at 1,410 MHz (140 W): +0.6% and −6 W, within noise.** Only the two 300 W VBIOS cards take the offset; the two 250 W VBIOS cards expose a VF offset range of [0..0] and NVML silently refuses it (measured).
- **Copy drafts (sm80vllm `VLLM_GLM5_COPY_DRAFTS`)**: edit-style replies that repeat the prompt decode 37% faster (rename task 256 → 351 tok/s, byte-identical output); the comment-edit task is 28% faster but its reply text differs; the general decode matrix is 0.3–2.2% slower.

## Hardware

- Cards: 4 × CMP 170HX, 65,536 MiB each, two per PLX PEX 8747 switch, both switches on one socket of a dual-socket Broadwell Xeon (E5-2686 v4), every link Gen2 x16
- Two VBIOS revisions on the same board part number: `92.00.67.00.01` (stock HBM NDIV 54 = 1,458 MHz) on two cards, `92.00.6D.00.0A` (stock NDIV 64 = 1,728 MHz) on the other two. Memory timings read identical on all four
- Power cap per card: 150 W for the SM and HBM comparison, then 165/150/140/130/120/110/100 W for the sweep (set live with `nvidia-smi -pl`, no server restart)
- Cooling: forced air, temperature-following fan controller (CPU-zone fans first, front fan last)
- Every link Gen2 x16: the `OPT_DISABLE_GEN3_SPEED` fuse reads 1 on all four cards, so Gen3 is not reachable in software

## Software

- Host: Proxmox VE 9.2.2, kernel 7.0.2-6-pve; the server runs in a privileged LXC container sharing the host driver
- NVIDIA driver 615.71.09 open kernel modules, [amoghmunikote/cmpunlocker](https://github.com/amoghmunikote/cmpunlocker) @ `88e39ce` plus:
  - upstream `6c442ee` "Add more SMs" (opens the reconfiguration PLM and releases one reserved TPC per enabled GPC: 70 → 74 SMs, read back as 74 by `torch.cuda.get_device_properties().multi_processor_count`)
  - [PR #60](https://github.com/amoghmunikote/cmpunlocker/pull/60) `hbm-control-plm.patch` (opens the FBPA_MEM and FBPA PLL privilege masks so host writes to the HBM clock take effect; changes nothing by itself; closed upstream, maintainer declined tuning features)
  - `rebar-serialize.patch` (same fix as upstream open [PR #59](https://github.com/amoghmunikote/cmpunlocker/pull/59)) and the four BAR1 P2P patches, as in the 2026-10-01 result. P2P is loaded but the served configuration keeps `VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE=0`
- HBM clock: [cachenetics/170tune](https://github.com/cachenetics/170tune) @ `5eb4775`. `170tune -i N hbm-gate --ndiv 64 --sweeps 12` with `GATE_TEMP=75 GATE_SOAK_MAX=600` on the two NDIV-54 cards: both passed 12/12 full-VRAM sweeps plus the compute check, peak HBM 75 °C; then `170tune persist save --ndiv 64` + `persist enable` re-applies it after every boot (the card still boots at stock first)
- Recipe: `Morrowmake/glm53-flash-cmp170hx-recipe` @ `a242b4f` (v1.6.0), `start.sh` `--gpus all` → `--device nvidia.com/gpu=all` (LXC/CDI)
- Engine image: `ghcr.io/morrowmake/vllm-cmp170hx@sha256:cc26c8abb63953a37c6c1861cadc9188e99c1cceab403600b5822740f3a82ae0` (vLLM fork `v0.30.1rc1.dev301+g3a2bf16da`)
- Copy-drafts A/B only: `pixelml/sm80vllm:tf-learnings-c93c274c8`, built locally from [PixelML/sm80vllm](https://github.com/PixelML/sm80vllm) branch `glm53-sm80-tf-learnings` (`127c6f076` base build + `c93c274c8` Python overlay). **Not published to a registry.**
- Target: [`canada-quant/GLM-5.3-Flash-W4A16-MTP`](https://huggingface.co/canada-quant/GLM-5.3-Flash-W4A16-MTP) @ `5723f4d02a`; drafter [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) @ `bf582e4eac` (CC BY-NC-ND 4.0, benchmark use only)

## Method

- Decode matrix (`run_decode.sh`): MiaAI-Lab [`tests/bench_decode.py`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/943912cdcda25f4b7e02f4626656e873c6f14847/tests/bench_decode.py) @ `943912cd` (AGPL-3.0, not vendored; `BASE` pointed at the server), structured / code / prose prompts, temperature 0, thinking off, 400 tokens, 32-token warmup, median of 5, at 1 and 8 users. Identical to the 2026-10-01 result.
- Telemetry: `nvidia-smi` at 1 s during each matrix (`telemetry.csv`). "Busy power" = mean board power over samples with utilization > 50%, summed over the four cards. Tokens per watt = eight-user structured aggregate ÷ busy power (GPU boards only, host not included).
- Power sweep (`sweep.sh`): the cap is changed live between matrices on a warm server, highest first; runs are back to back, so temperatures carry over between runs and are not a clean thermal comparison.
- Copy drafts (`bench_edit.py`): the model returns Python's `textwrap.py` with a rename or a one-line comment per function; tokens/s over the whole reply, median of 3; a sha256 of each reply shows whether output changed.
- `nvidia-smi` keeps reporting 1,458 MHz memory clock for the two re-clocked cards (cached at driver load); `170tune -i N status` reads the PLL and shows NDIV 64.

## Results

| Run | Cap | 1 user tok/s struct / code / prose | 8 users tok/s struct / code / prose | ms/step (1 user struct) | Busy GPU power | 8-user tok/s per W |
|---|---:|---:|---:|---:|---:|---:|
| 74 SM, mixed HBM | 150 W | 393.2 / 291.9 / 202.3 | 791.9 / 702.2 / 533.6 | 19.51 | 582 W | 1.36 |
| 74 SM, HBM 1,728 on all | 165 W | 418.6 / 311.0 / 214.7 | 822.4 / 735.7 / 558.9 | 18.33 | 650 W | 1.27 |
| 74 SM, HBM 1,728 on all | 150 W | **418.0 / 309.9 / 214.1** | **816.5 / 729.0 / 552.7** | 18.36 | 582 W | 1.40 |
| 74 SM, HBM 1,728 on all | **140 W** | 412.7 / 305.9 / 211.6 | 814.2 / 725.0 / 549.0 | 18.59 | 537 W | 1.52 |
| 74 SM, HBM 1,728 on all | 130 W | 394.9 / 290.8 / 201.9 | 785.9 / 696.6 / 527.9 | 19.43 | 517 W | 1.52 |
| 74 SM, HBM 1,728 on all | 120 W | 374.4 / 274.0 / 191.5 | 756.5 / 665.6 / 495.6 | 20.50 | 476 W | 1.59 |
| 74 SM, HBM 1,728 on all | 110 W | 340.1 / 241.1 / 172.0 | 706.8 / 602.1 / 447.2 | 22.56 | 436 W | **1.62** |
| 74 SM, HBM 1,728 on all | 100 W | 291.9 / 205.1 / 146.9 | 627.4 / 528.1 / 391.2 | 26.29 | 398 W | 1.58 |
| *70 SM, mixed HBM (2026-10-01)* | *180 W* | *398.4 / 307.3 / 192.9* | *812.0 / 721.9 / 518.6* | *19.3* | — | — |

Draft acceptance is 6.692 tokens per step (structured, 1 user) in every run, so every throughput difference is step time. Peak core / HBM temperature per card is in the notebook; the 165 W run reached 82 °C core and 83 °C HBM on one card, above this repository's 80 °C core stop rule (flagged, kept).

Copy-drafts A/B (sm80vllm image, 70 SM, mixed HBM, 150 W cap, one boot each):

| Copy drafts | 1 user struct / code / prose | 8 users struct / code / prose | Edit: rename (tok/s, reply sha) | Edit: comment per function (tok/s, reply sha) |
|---|---:|---:|---|---|
| off | 396.4 / 304.1 / 191.1 | 807.9 / 713.3 / 508.5 | 256.1 · `aca9129465db6368` | 254.3 · `34aaea9ff8c9387a` |
| on | 394.3 / 297.5 / 190.5 | 793.1 / 702.8 / 502.6 | 351.3 · `aca9129465db6368` | 326.6 · `2747918d44027b7b` |

7.1% of request-steps took a copied draft (server log). The rename reply is byte-identical with copy drafts on; the comment-edit reply is not. This engine is not run-to-run reproducible at temperature 0 (the same prompt twice can differ through the prefix-cache/batch path), so a changed sha is not proof that copy drafts changed the output, and an exactness gate needs pinned cache state.

## SM VF offset (+200 MHz at a 1,410 MHz ceiling), 140 W

`170tune -i N gate 200 1410 12`, hot, one card at a time: all four reported GATED (12/12 sweeps + compute; peak HBM 75 / 76 / 70 / 76 °C; GPU 2 at `GATE_TEMP=70`, its serving range; the others at 75).

**Measured caveat:** the two cards with VBIOS `92.00.67.00.01` (250 W, stock HBM NDIV 54) report an NVML GPC VF offset range of [0..+0] MHz; `nvmlDeviceSetGpcClkVfOffset` returns *Unknown Error* and reads back +0. Their gate ran at offset 0 (stock voltage, 1,410 MHz ceiling) and still reported a pass. Only the two `92.00.6D.00.0A` cards (300 W, range ±1000 MHz) take the undervolt; only they have it persisted (`OFFSET=200 CLK=1410`). The other two keep NDIV 64 only.

| Run (140 W) | 1 user struct / code / prose | 8 users struct / code / prose | ms/step | Busy GPU power |
|---|---:|---:|---:|---:|
| no offset | 412.7 / 305.9 / 211.6 | 814.2 / 725.0 / 549.0 | 18.59 | 537 W |
| +200 @ 1,410 on the two 300 W VBIOS cards | 415.0 / 308.3 / 213.1 | 816.3 / 728.1 / 553.4 | 18.49 | 531 W |

The undervolted cards reach the 1,410 MHz ceiling at 128–130 W; the other two stay power-limited at ~1,290 MHz (136–137 W), and TP4 runs at the slowest card's pace: +0.6%, −6 W, within noise. Peak HBM 66–67 °C.

Coherence probe: `coherence.json` reads `coherent: false` because the `cmp` probe ("is 9.9 greater than 9.11") ran out of tokens inside the model's reasoning preamble, before the answer. A direct request with `max_tokens` 800 answered "9.9" twice (67 tokens each, `finish_reason=stop`), measured. A probe-budget artifact, not corruption.

## Files

- `receipts/<run>/` — `decode-*.json` (per-run timings, usage, per-position acceptance), `decode-*.stdout`, `env.txt` (recipe settings, model path masked), `telemetry.csv`, `run_decode.out`
- `receipts/sm74-hbm64-off200-140w/` — the offset run, plus `coherence.json` / `coherence.stdout`
- `receipts/copy-drafts-{off,on}/` — the same plus `edit.json`, `edit.out`, `spec-metrics.txt`, `server-key-lines.txt`
- `run_decode.sh`, `sweep.sh`, `bench_edit.py` — harness; `tools/build_notebook.py` — builds the notebook from these receipts
- Notebook: [`notebooks/2026-10-02-glm-5.3-flash-4card-tp4-sm74-hbm-power-vllm.ipynb`](../../notebooks/2026-10-02-glm-5.3-flash-4card-tp4-sm74-hbm-power-vllm.ipynb)
