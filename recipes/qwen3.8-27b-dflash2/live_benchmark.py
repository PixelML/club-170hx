#!/usr/bin/env python3
"""Usage-accounted OpenAI-compatible benchmark for the CMP recipe."""

from __future__ import annotations

import json
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path


P256 = "Write a story about a robot who learns to paint."
LONG = (
    "summarize the following text. "
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega "
) * 200


class Benchmark:
    """Run fixed request shapes and retain authoritative final usage objects."""

    def __init__(self, base_url: str, api_key: str, safety_check: Callable[[], None]) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.safety_check = safety_check

    def post(self, path: str, body: dict[str, object], timeout: int = 600):
        self.safety_check()
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(body).encode(),
            headers=self.headers,
        )
        return urllib.request.urlopen(request, timeout=timeout)

    def token_count(self, prompt: str) -> int:
        with self.post("/tokenize", {"model": "qwen3.8-27b", "prompt": prompt}) as response:
            tokens = json.load(response).get("tokens")
        self.safety_check()
        if not isinstance(tokens, list):
            raise RuntimeError("tokenizer response omitted tokens")
        return len(tokens)

    def stream_completion(
        self,
        prompt: str,
        max_tokens: int,
    ) -> tuple[float, float, dict[str, int]]:
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
        usage = None
        with self.post("/v1/completions", body) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    break
                if ttft is None:
                    ttft = time.perf_counter() - started
                message = json.loads(payload)
                candidate = message.get("usage") or {}
                if candidate.get("completion_tokens") is not None:
                    usage = {
                        "prompt_tokens": int(candidate["prompt_tokens"]),
                        "completion_tokens": int(candidate["completion_tokens"]),
                        "total_tokens": int(candidate["total_tokens"]),
                    }
        self.safety_check()
        if ttft is None:
            raise RuntimeError("stream ended without a first token")
        if usage is None:
            raise RuntimeError("server omitted final usage.completion_tokens")
        return ttft, time.perf_counter() - started, usage

    def run_case(
        self,
        handle,
        name: str,
        prompt: str,
        max_tokens: int,
        prompt_tokens: int,
        *,
        warmups: int = 1,
        samples: int = 3,
    ) -> None:
        for index in range(warmups):
            _, _, usage = self.stream_completion(prompt, max_tokens)
            self._emit(handle, {"type": "usage", "case": name, "phase": "warmup", "sample": index + 1, "usage": usage})

        measurements = []
        for index in range(samples):
            ttft, total, usage = self.stream_completion(prompt, max_tokens)
            completion_tokens = usage["completion_tokens"]
            decode_tps = (completion_tokens - 1) / (total - ttft)
            self._emit(handle, {"type": "usage", "case": name, "phase": "measured", "sample": index + 1, "usage": usage})
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
            self._emit(handle, record)
            measurements.append((ttft, total, completion_tokens))

        mean_ttft = sum(item[0] for item in measurements) / samples
        mean_total = sum(item[1] for item in measurements) / samples
        mean_tokens = sum(item[2] for item in measurements) / samples
        self._emit(
            handle,
            {
                "type": "summary",
                "case": name,
                "samples": samples,
                "prompt_tokens": prompt_tokens,
                "mean_ttft_ms": round(mean_ttft * 1000, 1),
                "mean_total_s": round(mean_total, 3),
                "mean_completion_tokens": round(mean_tokens, 1),
                "decode_tok_s": round((mean_tokens - 1) / (mean_total - mean_ttft), 2),
                "prefill_tok_s": round(prompt_tokens / mean_ttft, 1),
            },
        )

    @staticmethod
    def _emit(handle, record: dict[str, object]) -> None:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def run_suite(
    base_url: str,
    api_key: str,
    output_path: Path,
    safety_check: Callable[[], None],
) -> None:
    """Run all three cases and write sanitized JSONL atomically enough to audit."""
    benchmark = Benchmark(base_url, api_key, safety_check)
    prompt_tokens = {
        "story": benchmark.token_count(P256),
        "long": benchmark.token_count(LONG),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("w", encoding="utf-8") as handle:
            benchmark._emit(handle, {"type": "protocol", "name": "usage-token-counted-v1"})
            benchmark.run_case(handle, "decode256", P256, 256, prompt_tokens["story"])
            benchmark.run_case(handle, "decode900", P256, 900, prompt_tokens["story"])
            benchmark.run_case(handle, "prefill_long", LONG, 8, prompt_tokens["long"])
    except Exception:
        safety_check()
        raise
    safety_check()


def inspected_completion(base_url: str, api_key: str, prompt: str) -> dict[str, object]:
    """Return only response text and the authoritative final usage object."""
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=json.dumps(
            {
                "model": "qwen3.8-27b",
                "prompt": prompt,
                "max_tokens": 128,
                "temperature": 0.0,
                "stream": False,
            }
        ).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.load(response)
    choices = payload.get("choices") or []
    usage = payload.get("usage") or {}
    if not choices or not isinstance(choices[0].get("text"), str):
        raise RuntimeError("functional response omitted text")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if not isinstance(usage.get(key), int):
            raise RuntimeError(f"functional response omitted usage.{key}")
    return {"prompt": prompt, "response": choices[0]["text"], "usage": {key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")}}
