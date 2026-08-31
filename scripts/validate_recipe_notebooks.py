#!/usr/bin/env python3
"""Fail closed when a published recipe notebook is incomplete or unsafe."""

from __future__ import annotations

import argparse
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


def validate_recipe(manifest_path: Path) -> list[str]:
    errors: list[str] = []
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("title", "status", "hardware", "model", "model_revision", "runtime_pin", "notebook", "chart", "results"):
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
    if not any(cell.get("outputs") for cell in cells if cell.get("cell_type") == "code"):
        errors.append(f"{notebook_path}: no clean recorded outputs are preserved")

    public_text = manifest_path.read_text(encoding="utf-8") + "\n" + notebook_path.read_text(encoding="utf-8")
    for pattern in PROHIBITED:
        if pattern.search(public_text):
            errors.append(f"{manifest_path}: prohibited public identifier matched {pattern.pattern}")
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
