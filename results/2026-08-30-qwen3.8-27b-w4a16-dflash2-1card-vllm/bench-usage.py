#!/usr/bin/env python3
"""Usage-token-counted Qwen3.8 benchmark used by the v9 CMP 170HX run."""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


P256 = "Write a story about a robot who learns to paint."
LONG = (
    "summarize the following text. "
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega "
) * 200


def emit(record: dict[str, object]) -> None:
    """Print one JSON record immediately."""
    print(json.dumps(record, sort_keys=True), flush=True)


class TelemetrySampler:
    """Capture raw and peak nvidia-smi telemetry for one physical GPU."""

    FIELDS = (
        "utilization.gpu,power.draw,memory.used,clocks.sm,clocks.mem,"
        "temperature.gpu,temperature.memory,pstate"
    )

    def __init__(self, gpu_index: int, output: Path) -> None:
        self.gpu_index = gpu_index
        self.output = output
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.peaks: dict[str, float] = {
            "utilization_gpu_pct": 0,
            "power_w": 0,
            "memory_used_mib": 0,
            "sm_clock_mhz": 0,
            "memory_clock_mhz": 0,
            "core_temp_c": 0,
            "memory_temp_c": 0,
        }

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5)

    def _sample(self) -> None:
        names = list(self.peaks) + ["pstate"]
        with self.output.open("w", encoding="utf-8") as handle:
            while not self.stop_event.is_set():
                try:
                    completed = subprocess.run(
                        [
                            "nvidia-smi",
                            "-i",
                            str(self.gpu_index),
                            f"--query-gpu={self.FIELDS}",
                            "--format=csv,noheader,nounits",
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    values = [value.strip() for value in completed.stdout.split(",")]
                    record: dict[str, object] = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "gpu_index": self.gpu_index,
                    }
                    for name, value in zip(names, values, strict=True):
                        if name == "pstate":
                            record[name] = value
                        else:
                            number = float(value)
                            record[name] = number
                            self.peaks[name] = max(self.peaks[name], number)
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()
                except Exception as error:  # Telemetry must not abort inference.
                    handle.write(json.dumps({"telemetry_error": str(error)}) + "\n")
                    handle.flush()
                self.stop_event.wait(0.5)


class Benchmark:
    """Run the exact v9 completion and long-prefill request shapes."""

    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def post(self, path: str, body: dict[str, object], timeout: int = 600):
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body).encode(),
            headers=self.headers,
        )
        return urllib.request.urlopen(request, timeout=timeout)

    def token_count(self, prompt: str) -> int:
        with self.post(
            "/tokenize", {"model": "qwen3.8-27b", "prompt": prompt}
        ) as response:
            return len(json.load(response)["tokens"])

    def stream_completion(self, prompt: str, max_tokens: int) -> tuple[float, float, int]:
        body = {
            "model": "qwen3.8-27b",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        started = time.perf_counter()
        ttft = None
        event_count = 0
        completion_tokens = None
        with self.post("/v1/completions", body) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    break
                if ttft is None:
                    ttft = time.perf_counter() - started
                event_count += 1
                message = json.loads(payload)
                usage = message.get("usage") or {}
                if usage.get("completion_tokens") is not None:
                    completion_tokens = int(usage["completion_tokens"])
        if ttft is None:
            raise RuntimeError("stream ended without a first token")
        if completion_tokens is None:
            raise RuntimeError(
                "server omitted usage.completion_tokens; refusing event-counted result"
            )
        total = time.perf_counter() - started
        emit(
            {
                "type": "stream_accounting",
                "completion_tokens": completion_tokens,
                "sse_events": event_count,
            }
        )
        return ttft, total, completion_tokens

    def run_case(
        self,
        name: str,
        prompt: str,
        max_tokens: int,
        prompt_tokens: int,
        warmups: int = 1,
        samples: int = 3,
    ) -> dict[str, object]:
        for _ in range(warmups):
            self.stream_completion(prompt, max_tokens)

        results = []
        for index in range(samples):
            ttft, total, completion_tokens = self.stream_completion(prompt, max_tokens)
            decode_tps = (completion_tokens - 1) / (total - ttft)
            record = {
                "type": "sample",
                "case": name,
                "sample": index + 1,
                "prompt_tokens": prompt_tokens,
                "ttft_ms": round(ttft * 1000, 1),
                "total_s": round(total, 3),
                "completion_tokens": completion_tokens,
                "decode_tok_s": round(decode_tps, 2),
                "prefill_tok_s": round(prompt_tokens / ttft, 1),
            }
            emit(record)
            results.append((ttft, total, completion_tokens))

        mean_ttft = sum(result[0] for result in results) / samples
        mean_total = sum(result[1] for result in results) / samples
        mean_tokens = sum(result[2] for result in results) / samples
        summary = {
            "type": "summary",
            "case": name,
            "samples": samples,
            "prompt_tokens": prompt_tokens,
            "mean_ttft_ms": round(mean_ttft * 1000, 1),
            "mean_total_s": round(mean_total, 3),
            "mean_completion_tokens": round(mean_tokens, 1),
            "decode_tok_s": round(
                (mean_tokens - 1) / (mean_total - mean_ttft), 2
            ),
            "prefill_tok_s": round(prompt_tokens / mean_ttft, 1),
        }
        emit(summary)
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--gpu-index", required=True, type=int)
    parser.add_argument("--telemetry-output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    emit(
        {
            "type": "metadata",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "gpu_index": args.gpu_index,
            "protocol": "v9-usage-token-counted",
        }
    )
    benchmark = Benchmark(args.base_url, args.api_key)
    prompt_tokens = {
        "story": benchmark.token_count(P256),
        "long": benchmark.token_count(LONG),
    }
    emit({"type": "prompt_tokens", **prompt_tokens})

    sampler = TelemetrySampler(args.gpu_index, args.telemetry_output)
    sampler.start()
    try:
        benchmark.run_case("decode256", P256, 256, prompt_tokens["story"])
        benchmark.run_case("decode900", P256, 900, prompt_tokens["story"])
        benchmark.run_case("prefill_long", LONG, 8, prompt_tokens["long"])
    finally:
        sampler.stop()
    emit({"type": "gpu_peaks", **sampler.peaks})


if __name__ == "__main__":
    main()
