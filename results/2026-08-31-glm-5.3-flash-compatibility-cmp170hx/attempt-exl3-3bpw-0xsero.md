# Attempt — EXL3 3.0 bpw (0xSero) — not attempted

Status: not attempted (artifact not servable on any identified runtime)
Date: 2026-08-30

## Checkpoint

- Repository: [0xSero/GLM-5.3-Flash-EXL3-3.0bpw](https://huggingface.co/0xSero/GLM-5.3-Flash-EXL3-3.0bpw)
  @ commit `8b099bf276507a17faea920deff3f62d5597fb52`
- 130-shard `layers/` layout (community-reported from the checkpoint card at
  the pinned commit; not independently re-measured here)
- Requires a custom loader (`requires_custom_loader: true` in the checkpoint
  metadata at the pinned commit)

## Quality gate (repo-reported, external)

The checkpoint card reports KL divergence 0.153, perplexity delta 0.093, and
top-1 agreement 0.873 against the FP8 reference on the card's own protocol
(community-reported; not reproduced here). That self-reported panel fails the
quality bar this project set for a candidate.

## Runtime status

- `runtime_status: pending_full_server` in the checkpoint metadata at the
  pinned commit — no complete serving path exists on llama.cpp or ExLlamaV3.
- Tokenizer files are absent from the checkpoint (observed in the repo tree
  at the pinned commit), so even a partial load cannot produce text.

## Outcome

Not attempted. No identified runtime can serve this artifact, and the
checkpoint's own reported quality gate fails. Recorded so the negative result
stays visible next to the fitting candidates.
