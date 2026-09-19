# CED overlay on apollo-2 — session handoff (2026-09-17)

## Where the implementation stands

The v1 fills implementation is **designed to the line level and drafted** but **not yet
validated against the live stack**. Committed artifacts:

- `docs/llkvapprox-flash-next/DESIGN-CED-overlay.md` — injection design (per-layer fills,
  delayed-HC-combine carry, metadata constraints, bench protocol)
- `docs/llkvapprox-flash-next/apollo2-vllm-src.tgz` — extracted stack source (model.py,
  QSA attention/indexer, hyperconnection ops, mamba layer)
- `docs/llkvapprox-flash-next/CED-overlay-patch-notes.md` — the three-patch spec
- `docs/llkvapprox-flash-next/ced_fills_draft.py` — fill-injection draft (imperfect:
  the FA rope call doesn't match the vLLM convention yet — the vLLM layer applies rope
  inside `_project_qkv_gate` (fused qk_rmsnorm_rope) while the draft uses a standalone
  rotate; the GDN chunk kernel import path is also container-specific)
- `receipts/` — baseline curve (2.2k-70k), MoE-skip ablation (negative), shape-noise
  floor (0.0), all eager-engine receipts

## The next session's work (in order)

1. **Fix the FA fill rope**: use the layer's own `_project_qkv_gate` on the
   projector-predicted rows (it applies k_norm + rope + gate in one fused/eager
   dispatch — the exact convention). The k prediction is post-k_norm pre-RoPE;
   the fill path should be: rope via the layer's rotary on the predicted k,
   then do_kv_cache_update. The indexer ik fill uses `qsa_compress_groups_with_ratio`
   (which norms + ropes + stores + pools internally given raw keys + positions).
2. **Debug the profile-pass crash** (CED env active during engine init crashes with
   NoneType.shape — the CED branch must be inactive during the profile/dummy run;
   guard: `if not env VLLM_PROFILE or forward_context has real metadata`).
3. **Canary**: short prompt, greedy, compare vs the pre-patch baseline generation.
4. **A/B bench**: the grid (512/1k/2k/6.6k) with CED on vs off, plus 32k/64k.
5. **Uplift notebook** + PR.

## Hardware note

apollo-2's server is currently healthy on STOCK serving (the CED bind-mount and env
were rolled back after the v0 ablation). The staged CED files are at `~/ced/model.py`
on apollo-2 and in the club-170hx PR branch. The `.env` EXTRA_DOCKER_ARGS needs the
bind-mount + env re-added for the CED runs (the exact line is in the git log:
commit 4be047f and later on `vllm-baseline-apollo2`).

## Measurements that stand (receipted, on the committed receipts)

| Result | Value |
|---|---|
| vLLM baseline prefill | 1,854 → 7,538 tok/s (2.2k → 70k), production kernels |
| Eager-engine CED speedup | 2.19–2.27× (T ≥ 1k) |
| Trained projector cosines | 0.95–0.99 (holdout = train) |
| Trained student greedy match | 75.9% (suffix-1, eager stack) |
| MoE-skip ablation | negative: quality-fatal + no timing gain on fused kernels |
| Shape-noise floor | 0.0 (model fully deterministic) |
| Kernel non-associativity | ~10% state error for 255+1 chunk split |
