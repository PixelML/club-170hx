from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
RECIPE = REPO_ROOT / "recipes/qwen3.8-27b-dflash2"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


live_support = load_module("qwen_live_support", RECIPE / "live_support.py")
live_benchmark = load_module("qwen_live_benchmark", RECIPE / "live_benchmark.py")
result_pipeline = load_module("qwen_result_pipeline", RECIPE / "result_pipeline.py")


class FakeRunner:
    def __init__(self, gpu_outputs: list[str], kernel_output: str = "clean") -> None:
        self.gpu_outputs = iter(gpu_outputs)
        self.kernel_output = kernel_output

    def __call__(self, command, **_kwargs):
        if command[0] == "nvidia-smi":
            return SimpleNamespace(stdout=next(self.gpu_outputs))
        if command[0] == "journalctl":
            return SimpleNamespace(stdout=self.kernel_output)
        raise AssertionError(f"unexpected command: {command[0]}")


SAFE_GPU = "NVIDIA CMP 170HX, 65536, 0, 0, 45, 55\n"


class SafetyGuardTests(unittest.TestCase):
    def guard(self, gpu: str = SAFE_GPU, kernel: str = "clean"):
        return live_support.FailClosedTelemetryGuard(
            0,
            None,
            command_runner=FakeRunner([gpu], kernel),
        )

    def test_safe_sample_is_sanitized(self) -> None:
        sample = self.guard().sample_once()
        self.assertEqual(sample["card"], "selected")
        self.assertNotIn("gpu_index", sample)
        self.assertEqual(sample["core_temp_c"], 45.0)

    def test_core_and_memory_limits_fail_closed(self) -> None:
        with self.assertRaisesRegex(live_support.SafetyViolation, "core_temperature_limit"):
            self.guard("NVIDIA CMP 170HX, 65536, 0, 0, 80, 55\n").sample_once()
        with self.assertRaisesRegex(live_support.SafetyViolation, "memory_temperature_limit"):
            self.guard("NVIDIA CMP 170HX, 65536, 0, 0, 45, 85\n").sample_once()

    def test_missing_device_and_xid_fail_closed(self) -> None:
        with self.assertRaisesRegex(live_support.SafetyViolation, "device_missing"):
            self.guard("").sample_once()
        with self.assertRaisesRegex(live_support.SafetyViolation, "nvidia_xid_or_device_loss"):
            self.guard(kernel="NVRM: Xid (PCI: redacted): 79").sample_once()

    def test_telemetry_command_failure_fails_closed(self) -> None:
        def broken(*_args, **_kwargs):
            raise subprocess.TimeoutExpired("nvidia-smi", 5)

        guard = live_support.FailClosedTelemetryGuard(0, None, command_runner=broken)
        with self.assertRaisesRegex(live_support.SafetyViolation, "telemetry_unavailable"):
            guard.sample_once()

    def test_background_violation_calls_owned_service_stop(self) -> None:
        stopped: list[bool] = []
        runner = FakeRunner(
            [SAFE_GPU, "NVIDIA CMP 170HX, 65536, 0, 100, 80, 55\n"]
        )
        guard = live_support.FailClosedTelemetryGuard(
            0,
            None,
            command_runner=runner,
            on_violation=lambda: stopped.append(True),
            sample_interval_s=0.01,
        )
        guard.start()
        deadline = time.monotonic() + 1
        while not stopped and time.monotonic() < deadline:
            time.sleep(0.01)
        guard.stop()
        self.assertEqual(stopped, [True])
        with self.assertRaisesRegex(live_support.SafetyViolation, "core_temperature_limit"):
            guard.assert_safe()


class PublicationPipelineTests(unittest.TestCase):
    def test_pinned_harness_emits_authoritative_usage(self) -> None:
        benchmark = live_benchmark.Benchmark("http://127.0.0.1:1", "test-only", lambda: None)
        benchmark.stream_completion = lambda _prompt, max_tokens: (
            0.1,
            1.1,
            {"prompt_tokens": 11, "completion_tokens": max_tokens, "total_tokens": 11 + max_tokens},
        )
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "receipt.jsonl"
            with receipt.open("w", encoding="utf-8") as handle:
                benchmark.run_case(handle, "decode256", "prompt", 256, 11)
            records = result_pipeline.read_jsonl(receipt)
        measured_usage = [
            record for record in records if record.get("type") == "usage" and record.get("phase") == "measured"
        ]
        self.assertEqual(len(measured_usage), 3)
        self.assertTrue(all(record["usage"]["completion_tokens"] == 256 for record in measured_usage))

    def test_runtime_versions_are_exact(self) -> None:
        resolved = live_support.verify_runtime_versions(
            {"one": "1.2.3"}, version_reader=lambda _name: "1.2.3"
        )
        self.assertEqual(resolved, {"one": "1.2.3"})
        with self.assertRaisesRegex(RuntimeError, "version mismatch"):
            live_support.verify_runtime_versions(
                {"one": "1.2.3"}, version_reader=lambda _name: "1.2.4"
            )

    def test_recorded_receipts_feed_summary_and_chart(self) -> None:
        sources = {
            "card-0": RECIPE / "results/recorded-receipts.jsonl",
            "card-1": RECIPE / "results/recorded-receipts-card-1.jsonl",
            "card-2": RECIPE / "results/recorded-receipts-card-2.jsonl",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "summary.csv"
            image = root / "chart.png"
            result_pipeline.write_summary_csv(sources, summary)
            with summary.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[-1]["card"], "mean")
            self.assertEqual(float(rows[-1]["decode_256_tok_s"]), 136.38)
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/render_recipe_chart.py"),
                    "--spec",
                    str(RECIPE / "chart-spec.json"),
                    "--results",
                    str(summary),
                    "--output",
                    str(image),
                ],
                check=True,
            )
            self.assertGreater(image.stat().st_size, 10_000)

    def test_chart_spec_contains_columns_not_measurements(self) -> None:
        spec = json.loads((RECIPE / "chart-spec.json").read_text(encoding="utf-8"))
        self.assertNotIn('"values"', json.dumps(spec))
        self.assertTrue(
            all("column" in series for panel in spec["panels"] for series in panel["series"])
        )


if __name__ == "__main__":
    unittest.main()
