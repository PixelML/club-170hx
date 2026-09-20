#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Mask infrastructure identifiers in the committed receipts.

The JSON receipts are written through the probe's own mask(); engine logs are
captured as text and go through this pass before publication. Rules follow
`notebooks/README.md`: no storage paths, no PIDs, no PCI bus ids, no hostnames.
Idempotent, so it is safe to re-run after regenerating a receipt.

  python sanitize_receipts.py [--check]
"""

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECEIPTS = HERE.parent / "receipts"

RULES = [
    # Storage paths: model directories, workspaces, caches.
    (re.compile(r"/library/models/[^\s'\"]+"), "<model-dir>"),
    (re.compile(r"/home/[a-z0-9_-]+/WIP/[^\s'\"]*"), "<workspace>"),
    (re.compile(r"/home/[a-z0-9_-]+/venv[^\s'\"]*"), "<venv>"),
    (re.compile(r"/home/[a-z0-9_-]+/\.cache/[^\s'\"]*"), "<cache>"),
    (re.compile(r"/home/[a-z0-9_-]+"), "<home>"),
    (re.compile(r"/tmp/[A-Za-z0-9._-]+"), "<tmp>"),
    # Identifiers.
    (re.compile(r"\bpid=\d+"), "pid=<pid>"),
    (re.compile(r"\bpid \d+\b"), "pid <pid>"),
    (re.compile(r"00000000:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]"), "<pci>"),
    (re.compile(r"\bGPU-[0-9a-f-]{36}\b"), "<gpu-uuid>"),
    (re.compile(r"\b127\.0\.0\.1:\d+"), "<endpoint>"),
    (re.compile(r"\b0\.0\.0\.0:\d+"), "<endpoint>"),
    # Hostnames of the participating nodes.
    (re.compile(r"\b(agent-sandbox|chimera|apollo-?\d?|sean-apollo-\d)\b"), "<host>"),
]

TEXT_SUFFIXES = {".log", ".txt", ".jsonl"}


def sanitize_text(text):
    for pattern, replacement in RULES:
        text = pattern.sub(replacement, text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report files that would change instead of rewriting")
    args = ap.parse_args()

    changed = []
    for path in sorted(RECEIPTS.iterdir()):
        if path.suffix not in TEXT_SUFFIXES or not path.is_file():
            continue
        original = path.read_text(errors="replace")
        cleaned = sanitize_text(original)
        if cleaned != original:
            changed.append(path.name)
            if not args.check:
                path.write_text(cleaned)
    if args.check:
        print("files that would change:", ", ".join(changed) or "none")
        return 1 if changed else 0
    print("sanitized:", ", ".join(changed) or "nothing to change")
    return 0


if __name__ == "__main__":
    sys.exit(main())
