# DeepSeek-V4.1-Flash (EXL3 3.0 bpw): TP4 + EP4 on four cards, every routed expert in VRAM, DSpark

Status: measured
Date: 2026-10-03

DeepSeek-V4.1-Flash served from the [Mia-AiLab EXL3 3.0 bpw checkpoint](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-3.0bpw) on four CMP 170HX cards, built on [0xSero/dsv41-flash-offload](https://github.com/0xSero/dsv41-flash-offload) (a single-GPU host-offload recipe) with four source changes that make it run with tensor parallelism 4 and expert parallelism 4. All 384 routed experts stay in VRAM (96 whole experts per card).

**Verdict (measured):** correct on five greedy known-answer prompts with and without the drafter. Without a drafter one user decodes at 25.4 tok/s (structured and code) and four users at 78.3 tok/s aggregate. DSpark k=5, from the checkpoint's own MTP layers, raises one user to 99.5 tok/s on structured text and 48.9 on code (four users: 214.1 / 128.2), and the edit reply stays byte-identical. Prose (10–12 tok/s) and prefill (110–360 tok/s) are bound by the Engram n-gram tables, which this recipe reads from NVMe on every step.

Notebook: [`notebooks/2026-10-03-deepseek-v4.1-flash-exl3-4card-tp4ep4-dspark-vllm.ipynb`](../../notebooks/2026-10-03-deepseek-v4.1-flash-exl3-4card-tp4ep4-dspark-vllm.ipynb)

## Hardware

- Cards: 4 × CMP 170HX, 65,536 MiB each (64 GiB unlock), PCIe Gen2; three cards at x16 and one at x8 during these runs
- Power limit: 140 W per card (`nvidia-smi -pl 140`)
- Cooling: forced air; peak core temperature 70 °C in every run (telemetry receipts)
- Peer-to-peer: not used (`--disable-custom-all-reduce`; collectives over NCCL)

## Software

- Host: Proxmox VE 9.2, kernel 7.0.2-6-pve; the server runs in a privileged LXC container sharing the host driver (GPUs passed with CDI)
- NVIDIA driver: 615.71.09 open kernel modules with [PixelML/cmpunlocker](https://github.com/PixelML/cmpunlocker) tag `cmp170hx-plx-p2p-2026-10-02` (commit `6ca4da72f078553d37089ee741cd128aa7804504`) for the 64 GiB unlock
- Served image: `ghcr.io/pixelml/club-170hx@sha256:ffba0e15b5295204a3860fae7ba93fd70fe94d824f79baa1340ab71fa1f451f8`
- Base image: `ghcr.io/pixelml/club-170hx@sha256:75683c4813efaabcc1a96ee610a96a74336677c4a8776b79b01abd83ff3c0c7b`, built from `0xSero/dsv41-flash-offload` @ `2e9da5ecdab37c232c2da887e25c33a99da738d3` with `docker build --build-arg TORCH_CUDA_ARCH_LIST=8.0 -f docker/Dockerfile .` Its Dockerfile pins `lazymio/vllm-backport@sha256:349690323ab9aba712111529ed1ca60730199205d8202f67895ffde85b451be3` (vLLM 0.13 backport with DeepSeek-V4.1), Tokha233 a100-turbo `fae324ae62ac5cef31b7d38f5d369618e1cae1fa`, exllamav3 `5be886578ec80324c2c715269387be2058724b6e` and vllm-exl3 `d3cfd394920360d69f820d2dc96f8292a9e10283`
- Our changes: [`build/`](build/) holds the Dockerfile that produced the served image from the base, the four patched files and the launcher; [`build/patches/`](build/patches/) has them as diffs:
  - `exl3.patch` (vllm-exl3, AGPL-3.0): `ReplicatedLinear` weights are not sliced by TP; the MoE split comes from the layer's own MoE parallel config, so expert parallelism keeps whole experts; a vocab-parallel lm_head loads its exact vocab range from vLLM's shard indices
  - `v4model.patch`: `DSV41_REPLICATE_SHARED=1` replicates the shared expert (2304 / 4 = 576 rows is not 128-aligned for EXL3) and scales its output by 1/4 before the MoE all-reduce
  - `v41model.patch`: the EXL3 lm_head is built with vocab padding 128 × tp so every rank's slice starts on a 128-row Hadamard block
  - `engram_disk.patch` (MIT): the disk Engram tier runs at TP > 1, each rank reading all heads from the shared mmap
- Checkpoint: [`Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-3.0bpw`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-3.0bpw) @ `c5534b90602b4090980a1c3ded1eb3b4d99a38d5` (MIT; routed experts EXL3 3 bpw, head 6 bpw, MTP 4 bpw)
- Engram tables: [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) @ `2cba9e42aa026125f3ed06c6d98c1db82f7ca027`, shards 47 and 48 (MIT)
- Drafter: DSpark over the checkpoint's `mtp.0`–`mtp.2` layers, `num_speculative_tokens` 5 (served) or 10

## Command

```bash
hf download Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-3.0bpw --revision c5534b90602b4090980a1c3ded1eb3b4d99a38d5 \
  --local-dir /path/to/models/Mia-DeepSeek-V4.1-Flash-EXL3-3.0bpw
hf download deepseek-ai/DeepSeek-V4.1-Flash --revision 2cba9e42aa026125f3ed06c6d98c1db82f7ca027 \
  --include "model-00047-of-00048.safetensors" "model-00048-of-00048.safetensors" "config.json" \
  "model.safetensors.index.json" "inference/engram.py" --local-dir /path/to/models/DeepSeek-V4.1-Flash-engram
docker run --rm --gpus '"device=0"' -v /path/to/models:/models ghcr.io/pixelml/club-170hx@sha256:75683c4813efaabcc1a96ee610a96a74336677c4a8776b79b01abd83ff3c0c7b prepare
docker run -d --name dsv41 --gpus all --ulimit memlock=-1 --shm-size 16g -p 8040:8040 \
  -v /path/to/models:/models:ro ghcr.io/pixelml/club-170hx@sha256:ffba0e15b5295204a3860fae7ba93fd70fe94d824f79baa1340ab71fa1f451f8 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}'
```

Every serve argument is in [`build/launch.sh`](build/launch.sh). Drop `--speculative-config` for the no-drafter row.

## Results (measured)

T=0, 400 tokens, median of 3, through the OpenAI API, generated tokens from the final usage object. One server boot per variant.

| variant | 1 user structured / code / prose (tok/s) | 4 users aggregate structured / code / prose (tok/s) | drafts accepted per step, 1 user (structured / code / prose) | edit reply tok/s |
|---|---|---|---|---:|
| no drafter | 25.4 / 25.4 / 12.1 | 78.3 / 73.0 / 40.7 | – | 25.1 |
| DSpark k=5 | 99.5 / 48.9 / 10.0 | 214.1 / 128.2 / 49.0 | 4.97 / 3.71 / 2.11 | 49.5 |
| DSpark k=10 | 55.5 / 53.3 / 24.7* | 45.0 / 40.3 / 23.3 | 3.93 / 3.71 / 1.69 | 49.0 |

\* The k=10 run started after about 101 of the 189 GiB of Engram tables were faulted into the page cache; the other two rows read them from NVMe. Prose depends on that far more than structured text or code.

- Known-answer checks: 5 / 5 on the DSpark k=5 server ([`receipts/correctness.json`](receipts/correctness.json)); the no-drafter server also answered 5 / 5 earlier in the session.
- Edit reply: the same 4,720-token reply (sha256 prefix `b755ea88e04201c2`) with no drafter, k=5 and k=10.
- KV pool at 65,536 context, fp8: 570,145 tokens without a drafter, 263,972 with DSpark k=5. Weights: 52.2 GiB per card, 54.4 GiB with the DSpark layers.
- Prefill (`max_tokens=1`, unique-nonce real text): 110–173 tok/s with Engram cold, 193–360 tok/s with ~101 GiB of it in the page cache. GPUs averaged about 10% utilization during the cold prefill.
- Long context, no drafter: decode 13.4–15.4 tok/s at 0.9k–45k prompt tokens.

## What failed on the way (measured)

- TP4 without expert parallelism: EXL3 refuses the 576-row shared-expert shard (fixed by the replicated shared expert plus EP).
- TP2 × PP2: stage 1's KV allocation references compressor state caches of stage-0 layers 2, 8 and 14 (`StopIteration`).
- lm_head split 32,320 rows per rank: the server emitted one token forever; the bad token ids sat at local row 32,256 of each rank (fixed by the 128-aligned split).
- DSpark k=7 is rejected: `num_speculative_tokens` must be a multiple of `n_predict` = 5.

## Receipts

- `receipts/<variant>/decode-*.json|stdout`: decode matrix, one file per prompt type and concurrency
- `receipts/<variant>/edit.json|out`: edit reply bench including the reply text
- `receipts/<variant>/spec-metrics.txt`: vLLM speculative-decoding counters (DSpark rows)
- `receipts/no-drafter/prefill.json`, `receipts/dspark-k10/prefill.json`, `receipts/no-drafter/longdecode.json`
- `receipts/<variant>/telemetry.csv`: 2 s `nvidia-smi` samples
- Bench scripts: [`bench_decode.py`](bench_decode.py), [`bench_edit.py`](bench_edit.py), [`bench_long.py`](bench_long.py) (as run; edit bench trimmed to one rename run at this speed)
