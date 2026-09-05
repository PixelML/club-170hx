# GLM-5.3-Flash on 8x CMP 170HX (Vast.ai rental) — RUNBOOK

Companion to `notebooks/2026-09-03-glm-5.3-flash-4card-tp4-vllm.ipynb`
(the reproducibility source of truth) and issue tracking this lane. This
runbook covers the parts specific to a Vast.ai rental: finding an offer,
onstart, the full measurement queue, and destroy.

## 0. Preconditions

- `vastai` CLI installed and authenticated (`vastai show user` works).
- Balance covers the planned instance-hours at the offer's `$/hr` with
  headroom for the hard cap below.
- Image `ghcr.io/pixelml/club-170hx:vllm-glm53-sm80-20260903` pulls
  anonymously (`docker manifest inspect <image>` from any machine).
- Checkpoint `wtdcode/GLM-5.3-Flash-AWQ-W4A16` is public on HF (verified via
  `curl https://huggingface.co/api/models/wtdcode/GLM-5.3-Flash-AWQ-W4A16`).

## 1. Find an offer

```bash
scripts/rental/find_offer.sh 8
```

**`verified=any` is required** — vastai hides unverified hosts by default,
and every CMP 170HX 8x host seen so far is unverified. Without this flag
the 8x listings silently disappear (looks like "no inventory" when it
isn't). Pick the offer with the best `pcie_bw_gbs` (Gen2 x16 > Gen1 x4)
and enough `disk_gb` (need >= 450 for weights + image + receipts).

## 2. Rent

**Decision (confirmed 2026-09-04): no nested docker.** The image has no
`docker`/`dockerd` binary in it (`docker run --rm --entrypoint /bin/bash
<image> -c 'which docker dockerd'` returns nothing), so the instance is
rented with the club-170hx image itself as the instance's container, and
`vllm serve` is exec'd directly inside it (no `docker run` inside the
rental). `launch_tp4.sh` / `launch_pp4.sh` / `launch_tp8.sh` and
`run_queue.sh` already assume this (nohup + PID file, plain log files
under `/workspace/logs`, not `docker logs`).

**Exact rental command** (offer `49884606` as of 2026-09-04: 8x CMP
170HX, PCIe Gen2 x16 / 6.4 GB/s, 65.5 GB/card, 512 GB RAM, 3238 GB disk,
$3.293/hr, driver/CUDA 13.3, verified=false — re-run
`scripts/rental/find_offer.sh 8` first, since offer IDs are single-use
and availability changes; use the top `pcie_bw_gbs` row):

```bash
vastai create instance 49884606 \
  --image ghcr.io/pixelml/club-170hx:vllm-glm53-sm80-20260903 \
  --disk 450 \
  --ssh --direct \
  --onstart scripts/rental/onstart.sh
```

`--onstart <file>` (not `--onstart-cmd "$(cat ...)"`) avoids CLI arg
length limits — `onstart.sh` is ~90 lines. `--ssh --direct` gives a
direct SSH endpoint for running `run_queue.sh` interactively and for
`collect.sh`'s scp-back step (`vastai show instances-v1` / `vastai ssh-url
<id>` after creation resolves the host/port).

At $3.293/hr the 4.5 h / $17 session cap (2026-09-04 balance-adjusted)
caps the rental at roughly 4.3 h of wall time before the $ cap binds —
budget accordingly and destroy the moment the queue ends or the cap
nears, per the hard rules below.

## 3. Checkpoint download gate

The onstart script downloads the 190.8 GB checkpoint and aborts if the
byte count is short. Before committing to the full queue, watch the
early transfer rate and project the ETA:

```bash
# after ~2 min of hf download running, from the instance:
du -sb /workspace/models/glm53 2>/dev/null
```

If the projected ETA for the full 190.8 GB exceeds **60 minutes**, abort
and destroy the instance (bad network draw) rather than burn the clock.

## 4. Run the queue

```bash
NUM_GPUS=8 bash scripts/rental/run_queue.sh
```

Steps (see `run_queue.sh` for detail), each gated by `wait_ready` with a
15 min boot timeout and writing JSON receipts under
`/workspace/receipts/<step>/`:

0. Preflight (link gen/width, topo, P2P, disk, driver/CUDA).
1. TP4 recipe of record, cards 0-3 — **mm-profiling flag OFF first**; the
   step auto-retries WITH `--limit-mm-per-prompt` only if the flag-off
   boot fails, and records which path worked in `mm-flag-result.txt`.
   Then: sanity, c=1 x5, prefill 2,900 tok, warm TTFT x3, c=8 x3, context
   sweep 4k/16k/64k, 20-prompt lossless check, boot time.
2. TP4 variants: P2P/custom-all-reduce path (only if P2P available), k=2,
   k=5 — c=1 and c=8 cells only.
3. PP4 + MTP x5, partition 14/12/12/7, enforce-eager — sanity + c=1 x3 +
   acceptance-length grep from the log. Decides whether the PP4 breakage
   seen on the 4x home box is the box or the build.
4. TP8 across all 8 cards (skipped if `NUM_GPUS<8`), then 2x TP4
   replicas on separate ports for aggregate c=16 (also skipped if
   `NUM_GPUS<8`).
5. Collect: tar receipts + trimmed logs to `/workspace/*.tar.gz`.

Resumable: each step's receipts dir gets a `DONE` marker; re-running
`run_queue.sh` skips completed steps. Delete a step's `DONE` file to
force a re-run of just that step.

## 5. Hard rules (abort/destroy triggers)

- Any single step hangs > 30 min.
- Spend approaches the session's hard cap.
- Xid, ECC, or driver errors appear in `nvidia-smi` or dmesg.
- Push whatever receipts exist first, then destroy.

## 6. Collect and destroy

```bash
scp -P <port> root@<host>:/workspace/glm53-8x170hx-receipts-*.tar.gz .
# commit receipts to the evidence repo branch (never main), per club-170hx AGENTS.md
vastai destroy instance <INSTANCE_ID>
vastai show instances   # confirm empty / instance gone
```

## Expected times (from the 4x home-box recipe; rental will differ)

| Phase | Expected |
|---|---|
| Image pull (if not baked into the rented image) | ~5-10 min (40.6 GB, cached layers likely) |
| Checkpoint download | measure with 1-shard sample; abort if ETA > 60 min for 190.8 GB |
| TP4 boot | ~8-10 min (weights load twice: main + MTP draft) |
| Each bench phase (gate/prefill/ttft/decode_c1) | 1-5 min |
| c8_stability (3 rounds) | ~5-10 min |
| Full queue (steps 0-5, 8 cards) | ~2-3 hours including 3 boots (TP4, PP4, TP8) + 2 replicas |

## 7. Training + PP4 extraction rental (2026-09-05)

New purpose, same branch/scripts family: parallel drafter training (block_size x lr sweep)
plus, if the box has >=4 spare 64GB cards, a PP4 slice-C hidden-state extraction. Recipe
of record comes from the drafter lane's "vast bundle" on `seanphan/pixelml#108`; scripts
here are a direct-exec translation (no nested docker — same finding as the inference
image: no docker binary inside `ghcr.io/pixelml/club-170hx:vllm-glm53-sm80-pp-20260905`).

Caps: total spend <= $17 (balance ~$18.66), <= 6h instance time, destroy on completion.

### Offer priority (via `find_offer.sh`, always `verified=any`)

1. 10x64GB Gen1x4 (49715704-class, ~$1.61/hr): 4 cards extraction + 6 cards training.
2. Else 8x64GB Gen2x16 (49884606, $3.29/hr): 4 extraction + 4 training, ~5h max.
3. Else any 4x64GB offer: training only (skip slice C).

Disk >= 450GB required in all cases. Math at commit `e03679f1` (checkpoints exported-set
only, not the frozen shared weights — 9.4GB/run instead of 19GB/run): 178GB AWQ + ~41GB
slice C at 1M tokens + 18GB training data (sliceB/target-shared/ref-drafter) + ~56GB
checkpoints (6 runs) + ~29GB image = **~322GB of 500**. (At the superseded commit this
was ~380GB — workable but far tighter; re-pin to `e03679f1` before renting.)

### Sequence

1. Rent with `scripts/rental/onstart_train.sh` as onstart (torch-import sanity only; no
   checkpoint pull here — that's extraction-only and gated separately).
2. From the orchestrating machine (not the rental): run `scripts/rental/transfer_specdec.sh`
   with `SPECDEC_SOURCE`, `SPECDEC_DEST_HOST`, `SPECDEC_DEST_PORT` set. It does a 1GB rate
   test each hop, then relays tools tarball (sha256 `ba8e8b00...`, commit `e03679f1`), sliceB
   (14GB), target-shared.safetensors (1.9GB), ref-drafter (2.2GB) via a local staging dir
   (source has no route to the rental). Ends with the mandatory gate:
   `tap hc_post-materialized+stream-mean tokens 455367 shards 9 files 9` then `OK`.
   **Stop if this string differs — do not train on it.**
3. On the rental: `scripts/rental/run_training.sh [NUM_GPUS_TRAIN]` — runs the reference
   band check first (`ref_eval2.py`, must land `IN BAND (~36%)` against alpha 0.3614),
   then launches all six runs FROM SCRATCH in parallel (one ~2.5h wave; no `--init-from`
   chaining off bs8 — against a 6h cap two sequential ~2.4h waves risks the cap). If
   extraction finishes early and hours remain, re-run bs13/bs17 with `--init-from` then.
4. If >=4 spare cards for extraction: `scripts/rental/run_extraction.sh eta-check` first
   (178GB AWQ checkpoint, 1-shard sample). Abort slice C (not training) if ETA > 60 min.
   Else `run_extraction.sh download` then `run_extraction.sh extract 1000000` (PP4,
   `--no-batch`, default 1M tokens). **Do not trust a flat ~2h20m estimate** — Gen1x4
   throughput for PP4's per-step cross-stage hidden-state hop is unmeasured on this
   topology. Read the extractor's own `[shard 0] ... tok/s` line (~7 min in) and re-plan:
   120 tok/s -> 2h20m, 80 -> 3h30m, 60 -> 4h40m for 1M tokens. If shard 0 comes in under
   ~70 tok/s, cut the token target (not the run) — `--resume` checkpoints every shard, so
   stopping early yields a short clean slice rather than a truncated unusable one. Reserve
   >=30 min at the end for rsync-back + destroy regardless of how far extraction got.
5. rsync checkpoints + `acceptance.json` + slice-C shards back to
   `/library/models/specdec-data/vast-2026-09-05/` on the source host.
6. `vastai destroy instance <ID>`; confirm via `vastai show instances-v1`.
7. Post spend + per-run alpha vs reference (0.3614) + slice-C size to both `#108` and
   `#107` — never to public repos (AGENTS.md).

### Notes

- Training data footprint (~18GB) is small; the AWQ checkpoint (178GB) is extraction-only
  and downloaded directly on the rental via `hf download`, not relayed from the source.
- `run_training.sh` and `run_extraction.sh` translate the bundle's `docker run --gpus
  device=N ...` per-card commands into `CUDA_VISIBLE_DEVICES=N` + `nohup` + PID file,
  consistent with `launch_tp4.sh`/`launch_pp4.sh`/`launch_tp8.sh` above.
