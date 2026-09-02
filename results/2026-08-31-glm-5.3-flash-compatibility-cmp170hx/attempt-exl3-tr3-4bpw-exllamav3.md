# Attempt — EXL3/TR3 4 bpw on ExLlamaV3

Status: compatibility-only, failed
Date: 2026-08-30

## Checkpoint

- [Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw) @ `25a44fdbf16862a46b7cc9921142c6c81350af2f`
  (byte-identical mirror of brandonmusic snapshot `5ab363a8`)
- Brandonmusic source snapshot pinned at
  [brandonmusic/GLM-5.3-Flash-tr3-4bpw](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw)
  `285507247eb5f031f35c52af5014ea33e1c3e69f` (Hugging Face API, checked
  2026-08-30); the 120-shard weights byte total below was re-verified at this
  revision.
- Size: 175,642,157,752 bytes = 175.64 GB = 163.58 GiB across 120 shards
  (measured: Hugging Face API blob-size sum at the pinned revision, checked
  2026-08-30). Earlier "~171.8 GiB / ~176 GB installed" in this file and the
  root README was a unit-conversion error against that same byte total; the
  static-fit paragraph below now uses the corrected total.
- EXL3/TR3 4 bpw, uniform K4 group codebooks, routed experts only
- License: ShapleyMCG License 1.0 (source-available) — check redistribution terms before mirroring

## Runtime

- ExLlamaV3 (pin `c5d9c657`, 0.0.43) via the SM121 fork's Docker overlay
- SM80 support: **untested on SM80; distribution is SM121-only** — the overlay
  builds `sm_121a` cubins on arm64, and the image docs explicitly disallow
  x86_64/QEMU use. The quant-method capability gate in the overlay returns 80
  (Ampere), which is why this path is the most plausible future candidate, but a
  capability gate is not a working SM80 stack (inferred, primary-source: overlay
  source in the DGX repository, file `exl3/overlay/exl3.py` around line 480).

## Static fit calculation

- 175.64 GB installed / 4 cards = ~43.91 GB/card = ~40.90 GiB/card if
  perfectly weight-balanced at TP=4 (measured HF blob sum at the pinned rev,
  arithmetic exact; expert-layout skew remains unverified).
- On paper this leaves ~23 GiB/card for CUDA context + KV — the most favorable
  candidate on memory. BUT: routed experts dominate the byte count,
  and TP=4 weight balance is not guaranteed for expert layouts; treat the fit as
  unverified until a real attempt loads the model.

## Execution status and outcome

Not executed. Two independent blockers, either one fatal on its own:

1. No SM80 build of the ExLlamaV3 extension exists (would require rebuilding
   `exllamav3_ext` for `sm_80` — untested, undocumented, plausibly blocked
   by the NoPE-MLA attention architecture).
2. The attention path is sparse-MLA targeting SM12x backends; SM80 has no
   equivalent. An SM80 port needs a non-sparse or differently-sparse attention
   implementation that no runtime provides today.

## Blocker

SM121-only binary distribution plus a missing SM80 attention backend. The
checkpoint itself is the closest thing to viable; the software around it is not.

## Evidence

- Quant capability gate: DGX repository, `exl3/overlay/exl3.py` — audited via
  the repository's `exl3/overlay/exl3.py` file at the commit referenced in its
  README at the time of review; the short git sha recorded earlier here could
  not be resolved upstream, so no abbreviated pin is reproduced as provenance.
- Size and shard layout: EXL3 recipe documentation in the DGX repository
- Upstream ExLlamaV3 targets Ampere+ (community-reported)

## Re-run instructions

Blocked until both items above exist. Once an SM80 build is available: stage the
checkpoint in shared model storage, TP=4 across the four-card node, verify
per-card weight placement before accepting the fit estimate, 180 W policy,
forced airflow, stop at 80 C core / 85 C memory, abort on Xid.
