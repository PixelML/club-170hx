#!/usr/bin/env python3
"""Fail-closed telemetry and runtime-pin checks for the live CMP recipe."""

from __future__ import annotations

import importlib.metadata
import json
import re
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


CORE_LIMIT_C = 80.0
MEMORY_LIMIT_C = 85.0
XID_PATTERN = re.compile(r"(?:NVRM:\s*Xid|Xid\s*\(|fallen off the bus)", re.IGNORECASE)
UNSUPPORTED = {"N/A", "[Not Supported]", "Not Supported", ""}
RUNTIME_VERSIONS = {
    "vllm": "0.27.1",
    "torch": "2.13.0",
    "transformers": "5.15.0",
    "tokenizers": "0.22.2",
    "huggingface_hub": "1.27.0",
    "flashinfer-python": "0.6.16.post3",
    "flashinfer-cubin": "0.6.13",
}


class SafetyViolation(RuntimeError):
    """A public-safe, terminal live-run safety failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"live safety gate failed: {code}")


def verify_runtime_versions(
    expected: Mapping[str, str] = RUNTIME_VERSIONS,
    version_reader: Callable[[str], str] = importlib.metadata.version,
) -> dict[str, str]:
    """Require exact versions for every serving-critical Python package."""
    resolved: dict[str, str] = {}
    for distribution, required in expected.items():
        try:
            installed = version_reader(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise RuntimeError(f"required runtime package missing: {distribution}") from error
        if installed != required:
            raise RuntimeError(
                f"runtime version mismatch for {distribution}: expected {required}, got {installed}"
            )
        resolved[distribution] = installed
    return resolved


class FailClosedTelemetryGuard:
    """Continuously stop an owned live run on missing or unsafe telemetry.

    The guard records only generic measurements. Command output and errors are
    never copied into the receipt because kernel and device tools may include
    machine identifiers.
    """

    QUERY_FIELDS = (
        "name,memory.total,memory.used,utilization.gpu,"
        "temperature.gpu,temperature.memory"
    )

    def __init__(
        self,
        gpu_index: int,
        receipt_path: Path | None,
        *,
        command_runner: Callable[..., Any] = subprocess.run,
        on_violation: Callable[[], None] | None = None,
        sample_interval_s: float = 0.5,
        started_epoch_s: float | None = None,
    ) -> None:
        self.gpu_index = gpu_index
        self.receipt_path = receipt_path
        self.command_runner = command_runner
        self.on_violation = on_violation
        self.sample_interval_s = sample_interval_s
        self.started_epoch_s = started_epoch_s or time.time()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._failure: SafetyViolation | None = None
        self._handle = None
        self.peaks = {"peak_core_c": 0.0, "peak_memory_c": 0.0}

    def _run(self, command: Sequence[str]) -> str:
        try:
            completed = self.command_runner(
                list(command),
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception as error:
            raise SafetyViolation("telemetry_unavailable") from error
        return str(completed.stdout)

    @staticmethod
    def _number(value: str, *, optional: bool = False) -> float | None:
        if value in UNSUPPORTED:
            if optional:
                return None
            raise SafetyViolation("telemetry_unavailable")
        try:
            return float(value)
        except ValueError as error:
            raise SafetyViolation("telemetry_unavailable") from error

    def sample_once(self) -> dict[str, object]:
        """Read one complete GPU and kernel-error sample or raise."""
        raw = self._run(
            [
                "nvidia-smi",
                "-i",
                str(self.gpu_index),
                f"--query-gpu={self.QUERY_FIELDS}",
                "--format=csv,noheader,nounits",
            ]
        )
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if len(lines) != 1:
            raise SafetyViolation("device_missing")
        fields = [value.strip() for value in lines[0].split(",")]
        if len(fields) != 6:
            raise SafetyViolation("telemetry_unavailable")
        name, total_raw, used_raw, util_raw, core_raw, memory_raw = fields
        if "CMP 170HX" not in name.upper():
            raise SafetyViolation("unexpected_device")
        total_mib = self._number(total_raw)
        used_mib = self._number(used_raw)
        utilization_pct = self._number(util_raw)
        core_c = self._number(core_raw)
        memory_c = self._number(memory_raw, optional=True)
        assert total_mib is not None and used_mib is not None
        assert utilization_pct is not None and core_c is not None
        if total_mib < 64000:
            raise SafetyViolation("unexpected_device")
        if core_c >= CORE_LIMIT_C:
            raise SafetyViolation("core_temperature_limit")
        if memory_c is not None and memory_c >= MEMORY_LIMIT_C:
            raise SafetyViolation("memory_temperature_limit")

        kernel = self._run(
            [
                "journalctl",
                "--dmesg",
                f"--since=@{int(self.started_epoch_s)}",
                "--no-pager",
                "--output=cat",
            ]
        )
        if XID_PATTERN.search(kernel):
            raise SafetyViolation("nvidia_xid_or_device_loss")

        sample: dict[str, object] = {
            "type": "telemetry",
            "card": "selected",
            "memory_total_mib": total_mib,
            "memory_used_mib": used_mib,
            "utilization_gpu_pct": utilization_pct,
            "core_temp_c": core_c,
            "memory_temp_c": memory_c,
        }
        self.peaks["peak_core_c"] = max(self.peaks["peak_core_c"], core_c)
        if memory_c is not None:
            self.peaks["peak_memory_c"] = max(self.peaks["peak_memory_c"], memory_c)
        return sample

    def _write(self, record: dict[str, object]) -> None:
        if self._handle is not None:
            self._handle.write(json.dumps(record, sort_keys=True) + "\n")
            self._handle.flush()

    def _fail(self, failure: SafetyViolation) -> None:
        callback = None
        with self._lock:
            if self._failure is None:
                self._failure = failure
                self._write({"type": "safety_stop", "reason": failure.code})
                callback = self.on_violation
        self._stop.set()
        if callback is not None:
            try:
                callback()
            except Exception:
                pass

    def _loop(self) -> None:
        while not self._stop.wait(self.sample_interval_s):
            try:
                self._write(self.sample_once())
            except SafetyViolation as failure:
                self._fail(failure)
                return

    def start(self) -> None:
        """Synchronously validate once, then begin continuous monitoring."""
        if self._thread is not None:
            raise RuntimeError("telemetry guard already started")
        if self.receipt_path is not None:
            self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.receipt_path.open("w", encoding="utf-8")
        try:
            self._write(self.sample_once())
        except SafetyViolation as failure:
            self._fail(failure)
            self.stop()
            raise
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def assert_safe(self) -> None:
        """Raise the first terminal safety failure without exposing raw data."""
        with self._lock:
            failure = self._failure
        if failure is not None:
            raise failure

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=6)
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def preflight_snapshot(gpu_index: int) -> dict[str, object]:
    """Return one sanitized safety sample without starting a workload."""
    return FailClosedTelemetryGuard(gpu_index, None).sample_once()
