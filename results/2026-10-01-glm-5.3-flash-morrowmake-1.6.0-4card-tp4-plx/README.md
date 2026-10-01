# GLM-5.3-Flash — Morrowmake 1.6.0, TP4 on four cards behind two PLX switches

Status: measured
Date: 2026-10-01

Replication of [Morrowmake/glm53-flash-cmp170hx-recipe](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe) release [v1.6.0](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe/tree/v1.6.0) (MIT), `LAYOUT=tp4`, DFlash2 drafter, on a different platform from theirs: four cards at Gen2 x16 behind two PLX PEX 8747 switches (two cards per switch) on a dual-socket Broadwell Xeon. Morrowmake measured on four cards on EPYC root ports.

**Verdict (measured):** with peer-to-peer off (the recipe default) this rig matches Morrowmake's TP4 structured decode (396.0 vs 394.0 tok/s for one user, 798.6 vs 797.9 for eight) and lands within ±7% on prose and eight-user code; single-user code is 19% lower. Turning BAR1 peer-to-peer on makes TP4 decode 20–26% slower for one user and 38–48% slower for eight here, the opposite of Morrowmake's result, so the served configuration is peer-to-peer off.

## Hardware

- Cards: 4 × CMP 170HX, 65,536 MiB each, all on the second CPU socket, two per PLX PEX 8747 switch, every link Gen2 x16
- CPU: 2 × Xeon E5-2686 v4 (Broadwell, 2.3 GHz base); cards attach to one socket
- Power limit: 180 W per card (the recipe's measurement cap)
- Cooling: forced air with a temperature-following fan controller; peak core temperatures per run are in the telemetry receipts
- Peer-to-peer: BAR1 peer access enabled by a patched driver (see Software); the recipe switch `VLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE` selects whether vLLM uses it

## Software

- Host: Proxmox VE 9.2, kernel 7.0.2-6-pve; the server runs in a privileged LXC container that shares the host driver (GPUs passed with CDI, `--device nvidia.com/gpu=all`)
- NVIDIA driver: 615.71.09 open kernel modules, unlocked with [amoghmunikote/cmpunlocker](https://github.com/amoghmunikote/cmpunlocker) @ `88e39ce`, plus:
  - `rebar-serialize.patch` (serializes the per-GPU BAR1 resize so two cards behind one switch do not race; see the 2026-10-01 interconnect result)
  - the four BAR1 peer-to-peer patches from [admunch888/cmpunlocker](https://github.com/admunch888/cmpunlocker) (ported from bayley's work): `p2p-caps-override`, `p2p-bar1`, `p2p-skip-mailbox-preinit`, `p2p-readcap-override`
  - a pre-driver PCIe layout: the BIOS leaves each switch's 64-bit window too small for two 64 GB BAR1s, so before the driver loads the BARs and bridge windows are programmed by hand (each 32 MB BAR3 directly below its 64 GB-aligned BAR1) and the kernel is restarted with kexec to adopt the layout. Verified: all 12 ordered card pairs pass a byte-exact peer copy at 5.79 GB/s.
- Recipe: `Morrowmake/glm53-flash-cmp170hx-recipe` @ `a242b4fc53724bc5216f416fbbfaf7bf318b9f2c` (tag v1.6.0). One local change: `--gpus all` → `--device nvidia.com/gpu=all` in `start.sh` (needed inside LXC; `--gpus` fails there with an NVML error)
- Engine image: `ghcr.io/morrowmake/vllm-cmp170hx@sha256:cc26c8abb63953a37c6c1861cadc9188e99c1cceab403600b5822740f3a82ae0` (vLLM fork `v0.30.1rc1.dev301+g3a2bf16da`)
- Target: [`canada-quant/GLM-5.3-Flash-W4A16-MTP`](https://huggingface.co/canada-quant/GLM-5.3-Flash-W4A16-MTP) @ `5723f4d02af36366c23ace8668866ca7775855c1`, W4A16
- Drafter: [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) @ `bf582e4eacc1810f76656d1811693ff6c6737d2a` (CC BY-NC-ND 4.0; used here for benchmarking only), adaptive depth up to 7

## Command

```bash
git clone https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe && cd glm53-flash-cmp170hx-recipe
git checkout v1.6.0
printf 'LAYOUT=tp4\nMODELS_DIR=/path/to/models\nVLLM_ALLOW_PCIE_P2P_CUSTOM_ALLREDUCE=0\n' > .env
./install.sh && ./download.sh
./start.sh && ./start.sh smoke
```

Each variant below is one server boot with exactly one setting changed; the effective settings are in each variant's `env.txt` (and `container-env.txt` where captured).

## Method

- **Decode** (`run_decode.sh`): MiaAI-Lab's [`tests/bench_decode.py`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/943912cdcda25f4b7e02f4626656e873c6f14847/tests/bench_decode.py) @ `943912cd` (AGPL-3.0, not vendored; only `BASE` and `MODEL` changed), the prompts Morrowmake credits: structured (count 1→200), code (`clamp_range`), prose (hash map). Temperature 0, `chat_template_kwargs.enable_thinking=false`, 400 max tokens, one 32-token warmup, median of 5. Stream tok/s = `(completion_tokens − 1) / (end − first token)`, tokens from the final `usage`. 8-user aggregate = `sum(completion_tokens) / wall`.
  - With thinking off, this model still writes a short reasoning preamble into `content` (documented in the recipe's how-to-use); the receipts' `text_head` shows it. Morrowmake's figures use the same prompts and the same behaviour.
  - Differences from Morrowmake's protocol: no per-request nonce (affects TTFT through the prefix cache, not decode rate).
- **Cold prefill and long-context decode** (`bench_long.py`, from the 2026-09-27 result): unique nonce first, real text, `max_tokens=1` for prefill, 400 tokens for the decode curve. Run on the final configuration only, gated: every request waits until all cards are under 60 °C, and the run stops if any card reaches 84 °C.
- **Per-step time**: `decode_ms_per_draft_step_median` from the decode receipts (a serving-cycle ratio, not a kernel timing).

## Results

Decode, 1 user (stream tok/s) and 8 users (aggregate tok/s), median of 5:

| TP4 variant (one change per boot) | 1 user structured / code / prose | 8 users structured / code / prose | ms per step, 1 user structured | KV pool @ 262,144 ctx |
|---|---:|---:|---:|---:|
| **P2P off** (recipe default) | **396.0 / 306.2 / 192.5** | **798.6 / 720.0 / 515.6** | 19.4 | 1,072,150 |
| P2P off + replicated embedding | 398.4 / 307.3 / 192.9 | 812.0 / 721.9 / 518.6 | 19.3 | 1,005,199 |
| P2P on (`2stage`, recipe P2P default) | 293.4 / 228.7 / 154.3 | 450.6 / 377.9 / 319.6 | 26.1 | 1,072,150 |
| P2P on, CPU + memory pinned to the cards' socket | 294.2 / 228.9 / 154.5 | 452.4 / 381.5 / 320.3 | 26.1 | 1,073,093 |
| P2P on + replicated embedding | 296.0 / 230.4 / 155.3 | 453.2 / 381.0 / 321.4 | 25.9 | 1,006,142 |
| P2P on, `1stage` all-reduce | 214.4 / 167.9 / 109.7 | 241.6 / 205.7 / 180.8 | 35.8 | 1,073,093 |
| *Morrowmake v1.6.0, P2P off (EPYC root ports)* | *394.0 / 377.4 / 180.5* | *797.9 / 683.2 / 544.7* | — | *1,072,150* |
| *Morrowmake v1.6.0, P2P on (EPYC root ports)* | *437.4 / 404.2 / 198.6* | *840.7 / 769.4 / 589.5* | — | *1,073,093* |

DFlash2 accepted tokens per step at 1 user are identical in every variant (structured 6.69, code 5.20, prose 2.55–2.75 of up to 7): the variants differ only in step time.

Cold prefill and long-context decode, final configuration (P2P off, `receipts/tp4-p2p-off-final/`):

| Measure | This rig | Morrowmake v1.6.0 TP4 P2P off |
|---|---:|---:|
| Cold prefill, median of 5 (19.4k–35.5k-token prompts; first request = warmup, excluded) | **2,701 tok/s** | 2,669 tok/s (23.9k–37.9k prompts) |
| Decode after a prompt of 860 / 6,809 / 26,823 / 51,851 / 103,567 / 170,756 tokens | 195.8 / 171.3 / 181.8 / 215.6 / 212.1 / 163.2 tok/s | — |

The long-context task is "explain the text above", so acceptance (and speed) differs from the three fixed prompts.

## Findings

- **Measured:** P2P off against Morrowmake's TP4 P2P-off table: structured +0.5% (1 user) and +0.1% (8 users); prose +6.6% and −5.3%; code −18.9% and +5.4%. The single-user code gap (306 vs 377) is not explained here.
- **Measured:** with BAR1 peer-to-peer on, every decode step is ~35% longer (26.1 vs 19.4 ms), single-user decode drops 20–26% and 8-user aggregate 38–48%. The `1stage` kernel is worse still. Pinning to the cards' CPU socket changes nothing (within 0.3%).
- **Inferred:** TP4 decode issues dozens of small all-reduces per step. With P2P on, vLLM's custom all-reduce moves them card-to-card through BAR1 mappings across the PLX switches and, for cross-switch pairs, across the root complex of an older Xeon. With P2P off the engine selects `['HOSTSHM', 'PYNCCL']` (measured, `allreduce-backend.txt`): a host shared-memory all-reduce for messages up to 512 KiB, which is faster for these small messages on this platform. The card-to-card copy and all-reduce measurements behind this are in the [4-card BAR1 P2P notebook](../../notebooks/2026-10-01-cmp170hx-4card-bar1-p2p-plx-cuda.ipynb). Morrowmake's +11% P2P gain was measured with direct EPYC root ports and does not transfer to this topology.
- **Measured:** the replicated input embedding (`VLLM_GLM5_REPLICATED_EMBED=1`) is the fastest variant (+0.6% one user, +1.7% eight users with P2P off) and costs 6% of the KV pool (1,005,199 tokens, still 3.83 full-length requests). The headline numbers above use the recipe default (off); the faster variant is in the table.

## Excluded or flagged receipts

- `tp4-p2p-on/excluded/`: cold prefill and an interrupted long-context run from the first P2P-on boot. One card reached 85 °C during these (the decode matrix before them stayed below 80 °C), and the long-context run was stopped by the temperature guard. Not used.
- `tp4-p2p-on-numa1`: no telemetry was recorded for this run; temperatures are unknown.
- `tp4-p2p-off-final/longctx-decode.json`, row at 170,756 tokens: one card ran at 80–82 °C for 24 one-second samples during this request, above this repository's 80 °C stop rule (the run's own guard was set at 84 °C). The row is kept and flagged; the cool-down gate ran before each phase (prefill, long context), not before each request.
- Which all-reduce backend vLLM selected is recorded only for the final configuration (`receipts/tp4-p2p-off-final/allreduce-backend.txt`); the variant receipts keep the candidate list only.

## Not tested

- The Apache-2.0 drafter [`canada-quant/GLM-5.3-Flash-DFlash2-G`](https://huggingface.co/canada-quant/GLM-5.3-Flash-DFlash2-G) does not load on this engine: it has 8 full-attention layers (the incoai drafter has 5 sliding-window layers) and engine start-up stops with `Layer language_model.model.layers.3.self_attn.indexer.k_cache: page size is not divisible by the maximum page size and cannot be padded`. Its own launcher targets a different engine build (FP8 KV, block size 2304).
- PP4 on this rig with recipe 1.6.0; quality suites (HumanEval/GSM8K).
