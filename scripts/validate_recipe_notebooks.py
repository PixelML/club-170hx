#!/usr/bin/env python3
"""Fail closed when a published recipe notebook is incomplete or unsafe."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


REQUIRED_SECTIONS = (
    "TL;DR",
    "Requirements",
    "Configure",
    "Preflight",
    "Install",
    "Start",
    "Benchmark",
    "Try your own prompt",
)
PROHIBITED = (
    re.compile(
        r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|"
        r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}|"
        r"100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])(?:\.\d{1,3}){2})\b"
    ),
)
PROHIBITED_TEXT = (
    "PIXELML_API_SECRET_FILE",
    "pixelml-qwen38-api-key",
)
REQUIRED_RUNTIME_VERSIONS = {
    "vllm": "0.27.1",
    "torch": "2.13.0",
    "transformers": "5.15.0",
    "tokenizers": "0.22.2",
    "huggingface_hub": "1.27.0",
    "flashinfer-python": "0.6.16.post3",
    "flashinfer-cubin": "0.6.13",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_recipe(manifest_path: Path) -> list[str]:
    errors: list[str] = []
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in (
        "title",
        "status",
        "hardware",
        "model",
        "model_revision",
        "runtime_pin",
        "runtime_dependencies",
        "requirements_sha256",
        "notebook",
        "chart",
        "results",
        "evidence",
    ):
        if not manifest.get(key):
            errors.append(f"{manifest_path}: missing manifest field {key}")

    notebook_path = root / manifest.get("notebook", "")
    chart_path = root / manifest.get("chart", "")
    result_paths = [root / item for item in manifest.get("results", [])]
    for path in (notebook_path, chart_path, *result_paths):
        if not path.is_file():
            errors.append(f"{manifest_path}: missing {path.relative_to(root)}")
    if not notebook_path.is_file():
        return errors

    requirements_path = root / "requirements.lock"
    if not requirements_path.is_file():
        errors.append(f"{manifest_path}: requirements.lock is missing")
    elif sha256(requirements_path) != manifest.get("requirements_sha256"):
        errors.append(f"{manifest_path}: requirements.lock hash does not match manifest")
    if manifest.get("runtime_dependencies") != REQUIRED_RUNTIME_VERSIONS:
        errors.append(f"{manifest_path}: runtime dependency pins are incomplete or changed")

    evidence = manifest.get("evidence", [])
    if not isinstance(evidence, list):
        errors.append(f"{manifest_path}: evidence must be a local file-level receipt list")
    else:
        for item in evidence:
            if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
                errors.append(f"{manifest_path}: malformed evidence entry")
                continue
            if "://" in str(item["path"]):
                errors.append(f"{manifest_path}: evidence must not link a repository-wide tree")
                continue
            evidence_path = root / str(item["path"])
            if not evidence_path.is_file():
                errors.append(f"{manifest_path}: missing evidence file {item['path']}")
            elif sha256(evidence_path) != item["sha256"]:
                errors.append(f"{manifest_path}: evidence hash mismatch for {item['path']}")

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        errors.append(f"{notebook_path}: notebook must use nbformat 4")
    cells = notebook.get("cells", [])
    markdown = "\n".join("".join(cell.get("source", [])) for cell in cells if cell.get("cell_type") == "markdown")
    code = "\n".join("".join(cell.get("source", [])) for cell in cells if cell.get("cell_type") == "code")
    for section in REQUIRED_SECTIONS:
        if section.lower() not in markdown.lower():
            errors.append(f"{notebook_path}: missing section {section}")
    if "curl " not in code or "/v1" not in code:
        errors.append(f"{notebook_path}: final editable curl request is missing")
    if "usage" not in code.lower() or "completion_tokens" not in code:
        errors.append(f"{notebook_path}: usage-token accounting guard is missing")
    if "FailClosedTelemetryGuard" not in code or "assert_safe" not in code:
        errors.append(f"{notebook_path}: fail-closed telemetry guard is missing")
    if "write_summary_csv" not in code or "--results" not in code:
        errors.append(f"{notebook_path}: live receipts are not wired through summary and chart")
    if not any(cell.get("outputs") for cell in cells if cell.get("cell_type") == "code"):
        errors.append(f"{notebook_path}: no clean recorded outputs are preserved")
    for index, cell in enumerate(cells):
        if cell.get("cell_type") == "code" and not isinstance(cell.get("execution_count"), int):
            errors.append(f"{notebook_path}: code cell {index} was not freshly executed")
    if not cells or cells[-1].get("cell_type") != "code" or "curl " not in "".join(cells[-1].get("source", [])):
        errors.append(f"{notebook_path}: notebook must end with the editable curl cell")

    chart_spec_path = root / "chart-spec.json"
    summary_path = root / "results/summary.csv"
    if chart_spec_path.is_file() and summary_path.is_file():
        chart_spec = json.loads(chart_spec_path.read_text(encoding="utf-8"))
        if '"values"' in json.dumps(chart_spec):
            errors.append(f"{chart_spec_path}: chart measurements must come from summary.csv")
        with summary_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or [])
            rows = list(reader)
        referenced = {
            chart_spec.get("x_column"),
            *(series.get("column") for panel in chart_spec.get("panels", []) for series in panel.get("series", [])),
        }
        if None in referenced or not referenced <= columns:
            errors.append(f"{chart_spec_path}: chart columns do not match summary.csv")
        if not rows or rows[-1].get("card") != "mean":
            errors.append(f"{summary_path}: summary must end with a mean row")

    text_paths = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".md", ".py", ".txt", ".csv", ".lock", ".ipynb"}
    ]
    public_text = "\n".join(path.read_text(encoding="utf-8") for path in text_paths)
    for pattern in PROHIBITED:
        if pattern.search(public_text):
            errors.append(f"{manifest_path}: prohibited public identifier matched {pattern.pattern}")
    lowered = public_text.lower()
    for prohibited in PROHIBITED_TEXT:
        if prohibited.lower() in lowered:
            errors.append(f"{manifest_path}: prohibited public text matched {prohibited}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=Path.cwd(), type=Path)
    args = parser.parse_args()
    manifests = sorted(args.root.glob("recipes/*/recipe.json"))
    if not manifests:
        raise SystemExit("no published recipe manifests found")
    errors = [error for manifest in manifests for error in validate_recipe(manifest)]
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"validated {len(manifests)} reproducible recipe notebook(s)")


if __name__ == "__main__":
    main()
