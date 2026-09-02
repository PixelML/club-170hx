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

## The first screen

Mobile readers do not scroll past one screen, so the very first cell is a
short markdown hero cell, before section 1 and before the `LIVE` status
cell. It holds, in this order and nothing else:

1. A one-line title naming the model and the hardware.
2. A four-row metrics table with units: decode at c=1, best aggregate,
   prefill, TTFT. Use "untested" for a row with no receipt instead of
   guessing a number.
3. The chart image at its `assets/charts/*.png` relative path, or, for a
   vision notebook with a passing run, the proof image with the model's
   one-line answer underneath it. A notebook with no passing run yet
   states that plainly instead of showing a fake image.
4. The pull command, one code line (`hf download ...` or `docker pull ...`).
5. One link line: the release, the evidence repository, or the
   Hugging Face collection. When the model has a guide page under
   `docs/models/`, that link line points there instead — the guide is
   the durable how-to (settings, troubleshooting, current numbers); this
   notebook is the executed proof that one specific run happened. A model
   with no guide page yet keeps the release/evidence/collection link.

No pins, no protocol text, no prose paragraph belongs in this cell. If the
notebook has an embedded video, that cell comes second, right after the
hero cell. Everything else — pins, protocol, the full tables, Reproduce,
and the appendix — follows below as section 1 onward.
`notebooks/TEMPLATE.ipynb` and `notebooks/TEMPLATE-vision.ipynb` carry this
hero cell already filled with placeholders.

## Required sections

Every notebook reads top to bottom in four sections, in this order.
`notebooks/TEMPLATE.ipynb` and `notebooks/TEMPLATE-vision.ipynb` have the
section headers and helper cells already in place. A reader who stops after
section 1 gets the verdict; a reader who stops after section 3 can
reproduce the run; nobody has to read the appendix unless they want the
failure history.

1. **TL;DR.** Nothing above this but the notebook title and the `LIVE`
   status cell. One key-metrics table (pass/fail, headline throughput
   numbers, cold boot time, power) and one pins table (model revision,
   runtime commit, image tag, driver, topology, quantization). Keep this
   section short enough to screenshot.
2. **Visible results.** Every chart, every results table, and, for a
   vision notebook, the fixture images shown inline next to the model's
   answer. This is where a reader checks the claim in section 1 against
   the evidence.
3. **Reproduce.** Hardware requirements (card, VRAM per card, card count,
   PCIe, power, cooling), the `docker pull` and `docker run` commands, the
   snapshot download, the launch command, the bench command, and the
   expected boot time. A reader with the right hardware can run this
   section and get the same numbers.
4. **Appendix.** Collapsed under a `<details>` heading so it stays out of
   the main scroll. Every failed attempt with its exact error class and
   its fix or its dead end, how the approach evolved, safety/preflight
   notes, cost and limitations, and links to the evidence source. Long
   narrative belongs here, not in sections 1 to 3.

Identity/pins, preflight, load gate, functional gates, measurements, and
publication/limitations are still the content categories from the older
six-section convention; they now nest inside these four sections instead
of each getting a top-level heading. Preflight and load-gate detail move
to the appendix; a one-line preflight confirmation ("stayed within limits")
is enough in section 1 or 3.

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

The package is public. Pushed digest:
`sha256:90a1419e8ceaad3542153ef4e2a1d94a69b9af03cce7b0a1b267dd1dad55b9d7`.
For exact reproducibility, pull by digest instead of tag:

```bash
docker pull ghcr.io/pixelml/club-170hx@sha256:90a1419e8ceaad3542153ef4e2a1d94a69b9af03cce7b0a1b267dd1dad55b9d7
```

This image is text-path only; it does not run the vision encoder path.
For the fork lineage, patch list, launch command, security scan, and a
`vast-onstart.sh` script for one-command bring-up on a rented box, see
`docs/DOCKER-IMAGE.md` in the
[DeepSeek-V4-Flash-Vision-Exp-CMP-170HX](https://github.com/PixelML/DeepSeek-V4-Flash-Vision-Exp-CMP-170HX)
evidence repository.

## Adding a notebook

1. Copy `notebooks/TEMPLATE.ipynb` to the new file name, or
   `notebooks/TEMPLATE-vision.ipynb` for an experiment that exercises image
   input.
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
