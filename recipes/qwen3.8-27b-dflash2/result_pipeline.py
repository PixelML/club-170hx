#!/usr/bin/env python3
"""Validate benchmark JSONL and write the chart's single structured source."""

from __future__ import annotations

import csv
import json
import os
import statistics
import tempfile
from pathlib import Path


FIELDS = (
    "card",
    "decode_256_tok_s",
    "decode_900_tok_s",
    "prefill_prompt_tokens",
    "prefill_tok_s",
    "ttft_ms",
    "peak_core_c",
    "peak_memory_c",
)
CASES = ("decode256", "decode900", "prefill_long")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path.name}:{line_number}: invalid JSON") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path.name}:{line_number}: record must be an object")
            records.append(record)
    return records


def summarize_records(records: list[dict[str, object]], card: str) -> dict[str, object]:
    summary_records = [record for record in records if record.get("type") == "summary"]
    summaries = {str(record.get("case")): record for record in summary_records}
    if len(summary_records) != len(CASES) or set(summaries) != set(CASES):
        raise ValueError(f"{card}: expected exactly {', '.join(CASES)} summaries")
    for case, summary in summaries.items():
        if int(summary.get("samples", 0)) < 3:
            raise ValueError(f"{card}/{case}: at least three measured samples required")
        usage = [
            record.get("usage")
            for record in records
            if record.get("type") == "usage"
            and record.get("case") == case
            and record.get("phase") == "measured"
        ]
        if len(usage) < int(summary["samples"]):
            raise ValueError(f"{card}/{case}: authoritative measured usage records missing")
        for item in usage:
            if not isinstance(item, dict) or not isinstance(item.get("completion_tokens"), int):
                raise ValueError(f"{card}/{case}: usage.completion_tokens missing")

    peaks = [record for record in records if record.get("type") == "gpu_peaks"]
    if len(peaks) != 1:
        raise ValueError(f"{card}: one sanitized gpu_peaks record required")
    peak = peaks[0]
    row: dict[str, object] = {
        "card": card,
        "decode_256_tok_s": float(summaries["decode256"]["decode_tok_s"]),
        "decode_900_tok_s": float(summaries["decode900"]["decode_tok_s"]),
        "prefill_prompt_tokens": int(summaries["prefill_long"]["prompt_tokens"]),
        "prefill_tok_s": float(summaries["prefill_long"]["prefill_tok_s"]),
        "ttft_ms": float(summaries["decode256"]["mean_ttft_ms"]),
        "peak_core_c": float(peak["peak_core_c"]),
        "peak_memory_c": float(peak["peak_memory_c"]),
    }
    for field in FIELDS[1:]:
        if float(row[field]) <= 0:
            raise ValueError(f"{card}: {field} must be positive")
    return row


def write_summary_csv(sources: dict[str, Path], output_path: Path) -> list[dict[str, object]]:
    """Validate receipt files and atomically write rows plus a mean row."""
    if not sources:
        raise ValueError("at least one benchmark receipt is required")
    rows = [summarize_records(read_jsonl(path), card) for card, path in sources.items()]
    mean: dict[str, object] = {"card": "mean"}
    for field in FIELDS[1:]:
        values = [float(row[field]) for row in rows]
        value = statistics.mean(values)
        mean[field] = int(value) if field == "prefill_prompt_tokens" and value.is_integer() else round(value, 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=output_path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows([*rows, mean])
        temporary = Path(handle.name)
    os.replace(temporary, output_path)
    return [*rows, mean]
