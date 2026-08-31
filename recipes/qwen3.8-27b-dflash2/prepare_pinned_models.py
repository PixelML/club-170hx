#!/usr/bin/env python3
"""Prepare the exact public Qwen + DFlash2 artifacts used by this recipe."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download


BASE_REPO = "dbirks/Qwen3.8-27B-W4A16-AutoRound"
BASE_REVISION = "1f05c441c4e64ae0549de44fa9ea5a6d43610314"
FAST_REPO = "syvai/qwen3.8-27b-3090-fast-variant"
FAST_REVISION = "124c14e7e8c7d2f5402933b9af368e772a9fcf0c"
DRAFT_REPO = "syvai/Qwen3.8-27B-DFlash2-W4A16"
DRAFT_REVISION = "4d30ec736ffc6b8688dc2ae2b502d9b48bdec279"


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    args = parser.parse_args()
    runtime_root = args.runtime_root.resolve()
    model_root = args.model_root.resolve()
    base = model_root / "Qwen3.8-27B-W4A16-AutoRound"
    fast = model_root / "Qwen3.8-27B-W4A16-AutoRound-fast"
    draft = model_root / "Qwen3.8-27B-DFlash2-W4A16"
    model_root.mkdir(parents=True, exist_ok=True)

    snapshot_download(BASE_REPO, revision=BASE_REVISION, local_dir=base)
    environment = dict(os.environ, BASE_MODEL_DIR=str(base), FAST_VARIANT="0", DFLASH2="0")
    subprocess.run(["bash", "docker/prepare.sh"], cwd=runtime_root, env=environment, check=True)

    fast.mkdir(parents=True, exist_ok=True)
    for source in sorted(base.glob("model-0000*-of-00007.safetensors")):
        if source.name != "model-00007-of-00007.safetensors":
            link_or_copy(source, fast / source.name)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "processor_config.json",
        "quantization_config.json",
    ):
        source = base / name
        if source.exists():
            link_or_copy(source, fast / name)

    overlay = Path(snapshot_download(FAST_REPO, revision=FAST_REVISION))
    for name in (
        "model-00007-of-00007.safetensors",
        "model_extra_tensors.safetensors",
        "mtp_draft_vocab_ids.pt",
        "config.json",
        "model.safetensors.index.json",
    ):
        shutil.copy2(overlay / name, fast / name)

    snapshot_download(
        DRAFT_REPO,
        revision=DRAFT_REVISION,
        local_dir=draft,
        allow_patterns=["*.json", "*.safetensors", "README.md"],
    )
    print("pinned model artifacts ready")


if __name__ == "__main__":
    main()
