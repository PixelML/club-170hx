# Rental scripts — 8x CMP 170HX on Vast.ai

Operator scripts for renting a multi-card CMP 170HX box by the hour and running
this club's GLM-5.3-Flash work on it. They exist because the club's own node has
four cards: anything that needs eight, or that needs a healthy PCIe link, has to
be rented.

**Status: prepared, not yet executed.** Nothing here has produced a published
measurement. No rental in this bundle has been run to completion, so every
timing in the runbook is a projection from the four-card box, not a measured
rental figure. Treat the whole directory as a plan with executable parts.

Two purposes, added in that order, sharing the same launcher and gate helpers:

| Purpose | Entry point | State |
|---|---|---|
| 8-card inference measurement (TP4 / PP4 / TP8 sweep) | `run_queue.sh` | prepared, never run |
| Drafter training + PP4 hidden-state extraction | `run_training.sh`, `run_extraction.sh` | prepared, never run |

The second purpose was added on 2026-09-05, after the four-card PP4 lane became
the [recipe of record](../../docs/models/glm-5.3-flash.md). The inference queue
is therefore aimed at what four cards cannot answer — TP8, an eight-stage
pipeline, and the same recipe on a healthy link — rather than at re-measuring
the recipe of record.

## Files

| File | What it does |
|---|---|
| `RUNBOOK.md` | The operator procedure for both purposes: find an offer, rent, gate the checkpoint download, run, collect, destroy. Read it before anything else. |
| `find_offer.sh` | Lists CMP 170HX offers, `verified=any`, sorted by PCIe bandwidth. |
| `onstart.sh` | Instance bootstrap for the inference queue: checkpoint download with a byte-count gate. |
| `onstart_train.sh` | Instance bootstrap for the training run: torch-import sanity only, no checkpoint pull. |
| `launch_tp4.sh`, `launch_pp4.sh`, `launch_tp8.sh` | Direct-exec `vllm serve` launchers, one topology each. No nested docker: the rented instance runs the club image itself. |
| `run_queue.sh` | The full measurement queue, resumable via per-step `DONE` markers. |
| `run_training.sh`, `run_extraction.sh` | Drafter training sweep and PP4 slice extraction. |
| `transfer_specdec.sh` | Relays training data from an internal source host to the rental, with a rate test per hop. |
| `collect.sh` | Tars receipts and trimmed logs for transfer back. |
| `bench/` | The measurement harness the queue calls: decode, prefill/TTFT, concurrency stability, context sweep, lossless check. |

## Before you run any of this

- The scripts assume the club's public image and a public checkpoint. They carry
  no credentials, no host names and no internal paths. Anything internal is read
  from an environment variable that fails closed when unset (`SPECDEC_SOURCE`,
  `SPECDEC_DEST_HOST`, `SPECDEC_DEST_PORT`, `SRC_DATA_DIR`).
- Spend caps and destroy triggers are in `RUNBOOK.md` section 5. They are the
  point of the runbook, not paperwork around it.
- Receipts from a rental go to a branch in the evidence repository, never
  straight to `main`, and are sanitized the same way every other result set in
  this club is. Rental-specific identifiers — instance ids, SSH endpoints, offer
  ids — do not belong in a published receipt.
