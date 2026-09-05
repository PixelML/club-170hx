#!/usr/bin/env python3
"""Build the Qwen3.8 CMP recipe notebook from reviewable source strings."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "recipes/qwen3.8-27b-dflash2/reproduce.ipynb"
_CELL_INDEX = 0


def source_lines(text: str) -> list[str]:
    text = text.strip("\n") + "\n"
    return text.splitlines(keepends=True)


def markdown(text: str) -> dict[str, object]:
    global _CELL_INDEX
    cell_id = f"cell-{_CELL_INDEX:02d}"
    _CELL_INDEX += 1
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source_lines(text)}


def code(text: str) -> dict[str, object]:
    global _CELL_INDEX
    cell_id = f"cell-{_CELL_INDEX:02d}"
    _CELL_INDEX += 1
    return {
        "cell_type": "code",
        "id": cell_id,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines(text),
    }


CELLS = [
    markdown(
        """
# Qwen3.8-27B W4A16 + DFlash2 on CMP 170HX

## TL;DR

**Measured:** 136.38 output tok/s mean for 256-token single-stream decode and 1,946 input tok/s at a 6,603-token prompt, repeated on three CMP 170HX cards at 180 W. This notebook preserves the clean result snapshot and is the runnable path from a fresh Ubuntu host to an editable API request.

![Measured performance](assets/performance.png)

Evidence class: **MEASURED**. The sanitized, usage-accounted receipts are pinned by SHA-256 in [recipe.json](recipe.json). **Limitation:** the legacy performance receipts did not retain completion text; a live run validates and displays response text plus the final API usage object before benchmarking.
"""
    ),
    markdown(
        """
## Requirements

- Ubuntu 22.04, one visible CMP 170HX with 64 GiB, a working NVIDIA driver, and directed airflow.
- Python 3.12, `git`, `curl`, `jq`, build tools, and enough model-storage space for roughly 40 GiB plus caches.
- Network access to GitHub, PyPI, and Hugging Face. The pinned public artifacts do not require a committed token.
- Readable kernel telemetry through `journalctl --dmesg`; the live path refuses to start if Xid monitoring is unavailable.
- Safety stop: 80 °C core, 85 °C memory when supported, NVIDIA Xid/device loss, missing telemetry, unsafe storage, or a conflicting workload.
"""
    ),
    markdown(
        """
## Configure

Edit only this cell. Keep `RUN_LIVE=False` when reading the committed result. Set `PIXELML_RUN_LIVE=1` and `PIXELML_MODEL_ROOT` on the target host before running all cells. `PIXELML_API_KEY` is optional; when absent, the notebook creates an in-memory key and never writes it to disk.
"""
    ),
    code(
        """
from pathlib import Path
import csv, json, os, secrets, signal, subprocess, sys, tempfile, time

RUN_LIVE = os.environ.get("PIXELML_RUN_LIVE", "0") == "1"
GPU_INDEX = int(os.environ.get("PIXELML_GPU_INDEX", "0"))
PORT = int(os.environ.get("PIXELML_PORT", "18020"))
MODEL_ROOT_VALUE = os.environ.get("PIXELML_MODEL_ROOT", "").strip()
MODEL_ROOT = Path(MODEL_ROOT_VALUE).expanduser().resolve() if MODEL_ROOT_VALUE else None
RUNTIME_ROOT = Path(os.environ.get("PIXELML_RUNTIME_ROOT", "~/.local/share/pixelml/qwen-serving")).expanduser().resolve()
PROMPT = "Explain why speculative decoding can emit fewer SSE events than completion tokens."

RECIPE_DIR = Path.cwd().resolve()
if not (RECIPE_DIR / "recipe.json").exists():
    RECIPE_DIR = (RECIPE_DIR / "recipes" / "qwen3.8-27b-dflash2").resolve()
REPO_ROOT = RECIPE_DIR.parents[1]
RUN_DIR = Path(tempfile.gettempdir()) / "pixelml-qwen-recipe-run"
PINS = json.loads((RECIPE_DIR / "recipe.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(RECIPE_DIR))
from live_benchmark import inspected_completion, run_suite
from live_support import FailClosedTelemetryGuard, preflight_snapshot, verify_runtime_versions
from result_pipeline import write_summary_csv

server = None
guard = None
print(json.dumps({
    "run_live": RUN_LIVE,
    "gpu_index": GPU_INDEX,
    "port": PORT,
    "model_revision": PINS["model_revision"],
    "runtime_pin": PINS["runtime_pin"],
    "runtime_dependencies": PINS["runtime_dependencies"],
}, indent=2, sort_keys=True))
"""
    ),
    markdown(
        """
## Preflight

This read-only gate prints only public-safe fields. It refuses live work when telemetry or kernel error visibility is unavailable, the expected card is absent, the selected card is busy, storage is unconfigured, or a temperature threshold is reached.
"""
    ),
    code(
        """
def run(command, **kwargs):
    return subprocess.run(command, check=True, text=True, **kwargs)

if not RUN_LIVE:
    print("RECORDED MODE — validated measured outputs below; no hardware command was run.")
else:
    if MODEL_ROOT is None:
        raise RuntimeError("set PIXELML_MODEL_ROOT to node-local model storage")
    if not MODEL_ROOT.parent.is_dir():
        raise RuntimeError("configured model-storage parent does not exist")
    snapshot = preflight_snapshot(GPU_INDEX)
    if float(snapshot["memory_used_mib"]) >= 1024 or float(snapshot["utilization_gpu_pct"]) >= 5:
        raise RuntimeError("selected GPU is not free; do not interrupt its owner")
    free_gib = __import__("shutil").disk_usage(MODEL_ROOT.parent).free / 2**30
    if free_gib < 40:
        raise RuntimeError(f"need at least 40 GiB free, found {free_gib:.1f}")
    print(json.dumps({"status": "PREFLIGHT PASS", "free_model_storage_gib": round(free_gib, 1), "telemetry": snapshot}, indent=2, sort_keys=True))
"""
    ),
    markdown(
        """
## Install the pinned runtime

This idempotent cell clones the exact public runtime commit, installs every direct serving dependency from `requirements.lock`, applies the pinned patch set, and verifies resolved versions. It does not start a GPU workload.
"""
    ),
    code(
        """
if RUN_LIVE:
    install = r'''set -euo pipefail
sudo apt-get update -qq
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev build-essential patch git curl jq ca-certificates
install -d -m 0755 "$RUNTIME_ROOT"
if [ ! -d "$RUNTIME_ROOT/.git" ]; then
  git clone https://github.com/syv-ai/qwen38-27b-rtx3090 "$RUNTIME_ROOT"
fi
git -C "$RUNTIME_ROOT" fetch origin 69ba4d0688c6ae76cb9d3c4a5c3b36445e1b040c
git -C "$RUNTIME_ROOT" checkout --detach 69ba4d0688c6ae76cb9d3c4a5c3b36445e1b040c
cd "$RUNTIME_ROOT"
if [ ! -x venv/bin/python ]; then python3.12 -m venv venv; fi
venv/bin/pip install --disable-pip-version-check -r "$RECIPE_DIR/requirements.lock"
sudo install -d -m 0755 /app
sudo ln -sfn "$RUNTIME_ROOT" /app/qwen-serving
sudo ln -sfn "$RUNTIME_ROOT/venv" /app/venv
sudo ln -sfn "$RUNTIME_ROOT/prepare" /app/prepare
NVIDIA_NVCC=$(venv/bin/python -c 'import nvidia.cuda_nvcc, os; print(os.path.join(os.path.dirname(nvidia.cuda_nvcc.__file__), "bin"))')
sudo install -d -m 0755 /usr/local/cuda/bin /usr/local/cuda/include
sudo ln -sfn "$NVIDIA_NVCC/nvcc" /usr/local/cuda/bin/nvcc
CURAND_H=$(find venv -name curand.h -print -quit)
[ -n "$CURAND_H" ] || { echo 'curand.h missing from pinned environment' >&2; exit 1; }
sudo ln -sfn "$(realpath "$CURAND_H")" /usr/local/cuda/include/curand.h
rm -rf "$HOME/.cache/flashinfer"
SP=$(venv/bin/python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')
if ! grep -q dflash2-backport "$SP/vllm/engine/arg_utils.py" 2>/dev/null; then
  for patch_file in patches/*.patch; do patch -p1 -N -d "$SP" < "$patch_file"; done
fi
grep -q dflash2-backport "$SP/vllm/engine/arg_utils.py"
'''
    environment = dict(os.environ, RUNTIME_ROOT=str(RUNTIME_ROOT), RECIPE_DIR=str(RECIPE_DIR))
    run(["bash", "-lc", install], env=environment)
    resolved = run([
        str(RUNTIME_ROOT / "venv/bin/python"), "-c",
        "import json,sys;sys.path.insert(0,sys.argv[1]);from live_support import verify_runtime_versions;print(json.dumps(verify_runtime_versions(),sort_keys=True))",
        str(RECIPE_DIR),
    ], capture_output=True).stdout.strip()
    print(f"PINNED RUNTIME PASS {resolved}")
else:
    print("Install skipped in recorded mode; requirements.lock remains the immutable live input.")
"""
    ),
    markdown(
        """
## Prepare the pinned model artifacts

The helper uses immutable Hugging Face revisions, performs the same W4A16 preparation, assembles the fast overlay, and fetches the W4A16 DFlash2 draft. Downloads are resumable and go only to the configured node-local model root.
"""
    ),
    code(
        """
if RUN_LIVE:
    run([
        str(RUNTIME_ROOT / "venv/bin/python"),
        str(RECIPE_DIR / "prepare_pinned_models.py"),
        "--runtime-root", str(RUNTIME_ROOT),
        "--model-root", str(MODEL_ROOT),
    ])
    print("Pinned model artifacts ready.")
else:
    print("Model preparation skipped in recorded mode.")
"""
    ),
    markdown(
        """
## Start the OpenAI-compatible service

The service uses one card, DFlash2 `k=7`, the fast W4A16 target, BF16 KV, and the measured 65,536-token profile. A fail-closed guard begins before service launch, continuously checks GPU/device/kernel telemetry, and terminates only this notebook-owned service on a safety violation.
"""
    ),
    code(
        """
def stop_owned_service():
    if server is not None and server.poll() is None:
        os.killpg(server.pid, signal.SIGTERM)

if RUN_LIVE:
    expected = PINS["runtime_dependencies"]
    resolved = verify_runtime_versions(expected, version_reader=lambda name: subprocess.check_output([
        str(RUNTIME_ROOT / "venv/bin/python"), "-c",
        "import importlib.metadata,sys;print(importlib.metadata.version(sys.argv[1]))", name,
    ], text=True).strip())
    if resolved != expected:
        raise RuntimeError("runtime version verification failed")
    api_key = os.environ.get("PIXELML_API_KEY") or secrets.token_hex(32)
    os.environ["PIXELML_API_KEY"] = api_key
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    guard = FailClosedTelemetryGuard(
        GPU_INDEX,
        RUN_DIR / "telemetry.jsonl",
        on_violation=stop_owned_service,
    )
    guard.start()
    environment = dict(os.environ)
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(GPU_INDEX),
        "VLLM_API_KEY": api_key,
        "SPEC": "dflash2", "CTX": "fast", "MAX_SEQS": "1",
        "DFLASH_TOKENS": "7", "PORT": str(PORT), "GPU_UTIL": "0.90", "KV_MEM": "",
        "MODEL": str(MODEL_ROOT / "Qwen3.8-27B-W4A16-AutoRound-fast"),
        "DRAFT": str(MODEL_ROOT / "Qwen3.8-27B-DFlash2-W4A16"),
        "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1",
        "FLASHINFER_DISABLE_VERSION_CHECK": "1", "VLLM_V2_CUDAGRAPH_MEM_MIB": "1400",
    })
    try:
        log_handle = (RUN_DIR / "server-local.log").open("w")
        server = subprocess.Popen(
            ["bash", "single-user/start_qwen.sh"],
            cwd=RUNTIME_ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            guard.assert_safe()
            check = subprocess.run(["curl", "-fsS", "--max-time", "3", f"http://127.0.0.1:{PORT}/health"], capture_output=True)
            if check.returncode == 0:
                print("SERVICE READY WITH FAIL-CLOSED TELEMETRY")
                break
            if server.poll() is not None:
                raise RuntimeError("service exited before health gate; inspect the local redacted log")
            time.sleep(1)
        else:
            raise TimeoutError("service did not become ready within 30 minutes")
    except Exception:
        stop_owned_service()
        guard.stop()
        raise
else:
    print("Service start skipped in recorded mode.")
"""
    ),
    markdown(
        """
## Inspect a functional response

Before benchmarking, the live path requires non-empty model text and a complete final usage object. The clean response is displayed in the notebook; credentials and raw server logs remain local and are never printed.
"""
    ),
    code(
        """
if RUN_LIVE:
    guard.assert_safe()
    inspected = inspected_completion(f"http://127.0.0.1:{PORT}", os.environ["PIXELML_API_KEY"], PROMPT)
    guard.assert_safe()
    print(inspected["response"])
    print("usage:")
    print(json.dumps(inspected["usage"], indent=2, sort_keys=True))
else:
    print("RECORDED LIMITATION — legacy performance receipts contain final usage counts but not completion text; run live to capture and inspect both.")
"""
    ),
    markdown(
        """
## Benchmark and recorded output

The live path runs the local usage-token-counted harness. Every measured request must end with `usage.completion_tokens`. Live JSONL is validated into `results/summary.csv`, and the chart reads that CSV directly. SSE event counts are never treated as tokens.
"""
    ),
    code(
        """
summary_path = RECIPE_DIR / "results/summary.csv"
if RUN_LIVE:
    live_results = RUN_DIR / "live-results.jsonl"
    run_suite(f"http://127.0.0.1:{PORT}", os.environ["PIXELML_API_KEY"], live_results, guard.assert_safe)
    guard.assert_safe()
    with live_results.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "gpu_peaks", **guard.peaks}, sort_keys=True) + "\\n")
    write_summary_csv({"live-run": live_results}, summary_path)
    print("Live usage records validated into results/summary.csv.")
else:
    write_summary_csv({
        "card-0": RECIPE_DIR / "results/recorded-receipts.jsonl",
        "card-1": RECIPE_DIR / "results/recorded-receipts-card-1.jsonl",
        "card-2": RECIPE_DIR / "results/recorded-receipts-card-2.jsonl",
    }, summary_path)
    print("Recorded sanitized receipts validated into results/summary.csv.")

rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
columns = ["card", "decode_256_tok_s", "decode_900_tok_s", "prefill_prompt_tokens", "prefill_tok_s", "ttft_ms"]
print(" | ".join(columns))
print(" | ".join(["---"] + ["---:"] * (len(columns) - 1)))
for row in rows:
    print(" | ".join(row[column] for column in columns))
"""
    ),
    code(
        """
run([
    sys.executable,
    str(REPO_ROOT / "scripts/render_recipe_chart.py"),
    "--spec", str(RECIPE_DIR / "chart-spec.json"),
    "--results", str(RECIPE_DIR / "results/summary.csv"),
    "--output", str(RECIPE_DIR / "assets/performance.png"),
])
print("Chart regenerated from results/summary.csv: assets/performance.png")
"""
    ),
    markdown(
        """
## Try your own prompt

Edit `PROMPT` in the configuration cell or below. With the service ready, the final literal `curl` prints the model response and the authoritative final usage object.
"""
    ),
    code(
        """
PROMPT = "Write a compact Python function that validates a topological ordering. Return code only."
os.environ["PIXELML_PROMPT"] = PROMPT
os.environ["PIXELML_PORT"] = str(PORT)
print(PROMPT)
"""
    ),
    code(
        r'''
%%bash
set -euo pipefail
if [ "${PIXELML_RUN_LIVE:-0}" != "1" ]; then
  echo "Set PIXELML_RUN_LIVE=1, rerun from Configure, then run this cell."
  exit 0
fi
test -n "${PIXELML_API_KEY:-}" || { echo "PIXELML_API_KEY is unavailable in this kernel" >&2; exit 1; }
BODY=$(jq -n --arg prompt "$PIXELML_PROMPT" '{model:"qwen3.8-27b",prompt:$prompt,max_tokens:256,temperature:0.0,stream:false}')
RESPONSE=$(curl --fail-with-body -sS "http://127.0.0.1:$PIXELML_PORT/v1/completions" \
  -H "Authorization: Bearer $PIXELML_API_KEY" \
  -H 'Content-Type: application/json' \
  -d "$BODY")
printf '%s\n' "$RESPONSE" | jq -er '.choices[0].text'
printf '\nusage:\n'
printf '%s\n' "$RESPONSE" | jq -e '.usage | {prompt_tokens, completion_tokens, total_tokens}'
'''
    ),
]


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
