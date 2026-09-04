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

Rent using the image directly as the instance's container (avoids nested
docker — the recipe's `docker run` commands in the queue run other
containers *inside* this one only if the offer's Vast template supports
docker-in-docker; the club-170hx recipe instead launches `vllm serve`
processes as additional bind-mounted containers via the onstart script,
so verify `docker info` works inside the rented container before running
the queue):

```bash
vastai create instance <OFFER_ID> \
  --image ghcr.io/pixelml/club-170hx:vllm-glm53-sm80-20260903 \
  --disk 450 \
  --onstart-cmd "$(cat scripts/rental/onstart.sh)"
```

If the offer's template does not support docker-in-docker, rent a plain
docker-capable Vast template instead and run the club image manually
inside it (`docker pull ... && docker run ...` per `launch_tp4.sh` etc.)
rather than fighting nested docker.

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
