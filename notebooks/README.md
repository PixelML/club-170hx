# Notebooks

This directory holds one executed Jupyter notebook per experiment. Each
notebook runs top to bottom and every cell output is committed. The
collection grows over time; older notebooks are not rewritten, only
superseded by a newer file that links back to them.

## File naming

```
YYYY-MM-DD-<model>-<topology>-<runtime>.ipynb
```

- `YYYY-MM-DD` is the measurement date, not the commit date.
- `<model>` is a short, lowercase, hyphenated model name.
- `<topology>` names card count and parallelism, for example `4card-pp4`.
- `<runtime>` names the serving engine, for example `vllm` or `llamacpp`.

Example: `2026-09-02-deepseek-v4-flash-vision-exp-4card-pp4-vllm.ipynb`.

## Required sections

Every notebook uses the same six sections, in this order. `notebooks/TEMPLATE.ipynb`
has the section headers and helper cells already in place.

1. **Identity and pins** — model name and revision, shard count, image digest,
   runtime source commit and patches, torch/vLLM (or other engine) versions,
   topology, and hardware. State each pin as measured; do not guess a version.
2. **Preflight and safety** — the safety limits and stop conditions from
   `AGENTS.md` (thermal ceilings, Xid/ECC checks, storage checks), and
   confirmation that the run followed them.
3. **Load gate** — the readiness and deterministic-output check that must
   pass before any measurement counts.
4. **Functional gates** — protocol-level checks: sampling settings, token
   accounting method, fixture sizes, and any pass/fail gate other than the
   load gate.
5. **Measurements** — the tables and charts for the experiment's metrics.
6. **Publication and limitations** — what the result does and does not show,
   known failures, and links to the evidence source and any related
   comparison repository.

## The LIVE flag

Each notebook has a status cell near the top with a `LIVE` flag.

- `LIVE = False` (default) replays the committed receipts under
  `results/<experiment>/`. This is what gets executed and committed. It needs
  no GPU and no network access.
- `LIVE = True` runs the same harness code against a running
  OpenAI-compatible endpoint. The endpoint URL comes from an environment
  variable (for example `DSV4_URL`), never a literal address in the
  notebook. Use this mode only for local reproduction; do not commit a
  notebook that was executed with `LIVE = True`, because the receipts it
  reads may not be sanitized.

## Publication rules

These follow `AGENTS.md` and apply to every notebook and every file under
`results/<experiment>/`:

- Commit every cell output. A notebook with cleared outputs is not published.
- No private IPs, hostnames, container names, PIDs, PCI bus ids, or storage
  paths. Use the same `<...>` masking convention as the evidence bundle.
- Label every claim as measured, inferred, community-reported, or untested.
- State units and denominators on every axis and every table column.
- Preserve negative and failed results; do not delete a failed configuration
  from a table.

## Docker image

For DeepSeek-V4-Flash-Vision-Exp on SM80 (CMP 170HX), a prebuilt vLLM
image replaces a from-source build that takes about 60 minutes:

```bash
docker pull ghcr.io/pixelml/club-170hx:vllm-deepseek-v4-sm80-20260902
```

This image is text-path only; it does not run the vision encoder path.
For the fork lineage, patch list, launch command, security scan, and a
`vast-onstart.sh` script for one-command bring-up on a rented box, see
`docs/DOCKER-IMAGE.md` in the
[DeepSeek-V4-Flash-Vision-Exp-CMP-170HX](https://github.com/PixelML/DeepSeek-V4-Flash-Vision-Exp-CMP-170HX)
evidence repository.

## Adding a notebook

1. Copy `notebooks/TEMPLATE.ipynb` to the new file name.
2. Copy the sanitized receipts for the experiment into
   `results/<experiment>/` (README, RESULTS.md or equivalent, raw JSON,
   harness script, protocol script, telemetry, and a log tail).
3. Fill in each section using the receipts. Do not invent a number that is
   not in a receipt.
4. Set `LIVE = False` and execute top to bottom:
   `jupyter nbconvert --to notebook --execute --inplace notebooks/<file>.ipynb`.
5. Export any chart to `assets/charts/` as PNG and SVG.
6. Add a row to the Notebooks table in the top-level `README.md` and link the
   notebook from the relevant section of `docs/BENCHMARKS.md`.
7. Run the pre-publication gate in `AGENTS.md` before committing.
