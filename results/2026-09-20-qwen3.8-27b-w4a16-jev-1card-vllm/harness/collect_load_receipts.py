#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Freeze the production-load receipts for the Jev-endpoint experiment.

The numbers come from the first real workload served by this endpoint: an
LLM-as-judge annotation pilot (the Jev contract, five permutation reads per
annotation, 16 client workers) that ran for one continuous stretch against the
served engine. Three inputs are aggregated, none of which is committed raw:

  <pilot-ledger>    the pilot's per-annotation JSONL ledger (one record per
                    annotation: timestamp, input tokens, billed cost). The raw
                    file stays outside this repository because it names pool
                    items; only the per-minute aggregates are committed.
  <metrics-t0> / <metrics-t1>
                    two snapshots of the engine's Prometheus /metrics endpoint,
                    taken <window> seconds apart under load. The deltas are the
                    engine-side counters for exactly that window.
  <dmon>            `nvidia-smi dmon -i <gpu> -s pucm -d 2` telemetry captured
                    during the same window.

  python collect_load_receipts.py <pilot-ledger> <metrics-t0> <metrics-t1> \
      <window-seconds> <window-end-utc> <dmon> <gpu-index>

Writes receipts/load/{load-summary.json,load-ledger-minutes.json,
metrics-delta.json,gpu-telemetry.json}. Nothing here identifies hosts, ports,
paths or pool items; annotations are counted, never shipped.
"""

import json
import re
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "receipts" / "load"


def parse_metrics(path):
    metrics = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        key, value = line.rsplit(None, 1)
        try:
            metrics[key] = metrics.get(key, 0.0) + float(value)
        except ValueError:
            pass
    return metrics


def metric_total(metrics, name):
    return sum(v for k, v in metrics.items()
               if k == name or k.startswith(name + "{"))


def metric_bucket_delta(m0, m1, name):
    def buckets(m):
        d = {}
        for k, v in m.items():
            if k.startswith(name + "{"):
                le = re.search(r'le="([^"]+)"', k)
                if le:
                    d[le.group(1)] = d.get(le.group(1), 0.0) + v
        return d
    b0, b1 = buckets(m0), buckets(m1)
    return {k: b1.get(k, 0.0) - b0.get(k, 0.0) for k in set(b0) | set(b1)}


def main():
    ledger_path, t0_path, t1_path, window_s, window_end, dmon_path, gpu_idx = (
        sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]),
        datetime.fromisoformat(sys.argv[5].replace("Z", "+00:00")),
        sys.argv[6], sys.argv[7])

    records = [json.loads(line) for line in open(ledger_path) if line.strip()]
    stamps = [datetime.fromisoformat(r["ts"].replace("Z", "+00:00"))
              for r in records]
    order = sorted(range(len(records)), key=lambda i: stamps[i])
    records = [records[i] for i in order]
    stamps = [stamps[i] for i in order]
    t_start, t_end = stamps[0], stamps[-1]
    span_s = (t_end - t_start).total_seconds()

    failed = [r for r in records if r.get("failed")]
    input_tokens = [r["input_tokens"] for r in records]
    sorted_tokens = sorted(input_tokens)

    # steady state = last 30 minutes of the frozen ledger
    t_steady = t_end - timedelta(minutes=30)
    steady = [(r, t) for r, t in zip(records, stamps) if t >= t_steady]
    steady_span = (t_end - t_steady).total_seconds()

    per_hour = {}
    for r, t in zip(records, stamps):
        hour = t.strftime("%H")
        per_hour.setdefault(hour, []).append(r["input_tokens"])

    # per-minute aggregate for the chart
    per_minute = {}
    for r, t in zip(records, stamps):
        minute = t.strftime("%H:%M")
        d = per_minute.setdefault(minute, {"reads": 0, "input_tokens": 0})
        d["reads"] += 1
        d["input_tokens"] += r["input_tokens"]
    # engine window
    m0, m1 = parse_metrics(t0_path), parse_metrics(t1_path)
    req = metric_total(m1, "vllm:e2e_request_latency_seconds_count") - \
        metric_total(m0, "vllm:e2e_request_latency_seconds_count")
    e2e_sum = metric_total(m1, "vllm:e2e_request_latency_seconds_sum") - \
        metric_total(m0, "vllm:e2e_request_latency_seconds_sum")
    prompt_tok = metric_total(m1, "vllm:prompt_tokens_total") - \
        metric_total(m0, "vllm:prompt_tokens_total")
    gen_tok = metric_total(m1, "vllm:generation_tokens_total") - \
        metric_total(m0, "vllm:generation_tokens_total")
    cached_tok = metric_total(m1, "vllm:prompt_tokens_cached_total") - \
        metric_total(m0, "vllm:prompt_tokens_cached_total")
    pq = metric_total(m1, "vllm:prefix_cache_queries_total") - \
        metric_total(m0, "vllm:prefix_cache_queries_total")
    ph = metric_total(m1, "vllm:prefix_cache_hits_total") - \
        metric_total(m0, "vllm:prefix_cache_hits_total")

    e2e_delta = metric_bucket_delta(m0, m1, "vllm:e2e_request_latency_seconds_bucket")
    e2e_bounds = {}
    for bound in ("10.0", "15.0", "20.0", "30.0", "60.0"):
        if bound in e2e_delta:
            e2e_bounds[f"within_{int(float(bound))}s"] = e2e_delta[bound]

    window_start = window_end - timedelta(seconds=window_s)
    ledger_in_window = sum(1 for t in stamps if window_start <= t <= window_end)
    tokens_in_window = sum(
        r["input_tokens"] for r, t in zip(records, stamps)
        if window_start <= t <= window_end)

    # telemetry
    rows = []
    for line in open(dmon_path):
        p = line.split()
        # -o T prepends HH:MM:SS; columns: gpu pwr gtemp mtemp sm mem enc dec
        # jpg ofa mclk pclk fb bar1 ccpm
        if len(p) >= 13 and p[0][:2].isdigit() and p[1] == str(gpu_idx):
            rows.append({
                "time_utc": p[0], "power_w": float(p[2]),
                "gpu_temp_c": float(p[3]), "mem_temp_c": float(p[4]),
                "sm_util_pct": float(p[5]), "mem_util_pct": float(p[6]),
                "mem_clock_mhz": float(p[11]), "sm_clock_mhz": float(p[12]),
            })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gpu-telemetry.json").write_text(json.dumps(rows, indent=1) + "\n")

    metrics_delta = {
        "window_s": window_s,
        "requests_finished": req,
        "e2e_latency_mean_s": e2e_sum / req if req else None,
        "e2e_latency_bounds": e2e_bounds,
        "prompt_tokens": prompt_tok,
        "prompt_tokens_per_s": prompt_tok / window_s,
        "generation_tokens": gen_tok,
        "generation_tokens_per_request": gen_tok / req if req else None,
        "cached_tokens": cached_tok,
        "prefix_cache_queries": pq,
        "prefix_cache_hits": ph,
        "prefix_cache_hit_rate_pct": ph / pq * 100 if pq else None,
        "num_requests_running_at_t1": metric_total(
            m1, "vllm:num_requests_running"),
        "num_requests_waiting_at_t1": metric_total(
            m1, "vllm:num_requests_waiting"),
    }
    (OUT / "metrics-delta.json").write_text(
        json.dumps(metrics_delta, indent=1) + "\n")

    with open(OUT / "load-ledger-minutes.json", "w") as f:
        json.dump({
            "unit": "UTC minute; aggregate of the pilot ledger, no per-item data",
            "minutes": [{"minute_utc": k, **v}
                        for k, v in sorted(per_minute.items())],
        }, f, indent=1)
        f.write("\n")

    summary = {
        "kind": "production-load-aggregate",
        "workload": ("LLM-as-judge annotation pilot speaking the Jev contract: "
                     "five permutation reads per annotation, 16 client "
                     "workers, one continuous run"),
        "source": ("aggregate of the pilot's per-annotation ledger and the "
                   "engine's /metrics deltas; the raw ledger stays outside "
                   "this repository"),
        "ledger": {
            "annotations": len(records),
            "failed": len(failed),
            "start_utc": t_start.isoformat(),
            "end_utc": t_end.isoformat(),
            "span_h": span_s / 3600,
            "input_tokens_total": sum(input_tokens),
            "input_tokens_mean": statistics.mean(input_tokens),
            "input_tokens_p95": sorted_tokens[int(0.95 * len(sorted_tokens))],
            "usd_billed": sum(r.get("billed_usd", 0.0) or 0.0 for r in records),
            "annotations_per_h_avg": len(records) / span_s * 3600,
            "steady_last30_annotations_per_min": len(steady) / steady_span * 60,
            "steady_last30_input_tokens_per_s":
                sum(r["input_tokens"] for r, _ in steady) / steady_span,
        },
        "per_hour_utc": [
            {"hour_utc": h, "annotations": len(v),
             "mean_input_tokens": statistics.mean(v)}
            for h, v in sorted(per_hour.items())],
        "engine_window": {
            "window_s": window_s,
            "ledger_annotations_in_window": ledger_in_window,
            "ledger_input_tokens_in_window": tokens_in_window,
            "requests_per_annotation": (metrics_delta["requests_finished"]
                                        / ledger_in_window),
            **metrics_delta,
        },
        "gpu_window": {
            "samples": len(rows),
            "sample_period_s": 2,
            "power_w_mean": statistics.mean(r["power_w"] for r in rows),
            "power_w_min": min(r["power_w"] for r in rows),
            "power_w_max": max(r["power_w"] for r in rows),
            "sm_util_pct_mean": statistics.mean(r["sm_util_pct"] for r in rows),
            "gpu_temp_c_range": [min(r["gpu_temp_c"] for r in rows),
                                 max(r["gpu_temp_c"] for r in rows)],
            "mem_temp_c_range": [min(r["mem_temp_c"] for r in rows),
                                 max(r["mem_temp_c"] for r in rows)],
            "sm_clock_mhz": sorted({r["sm_clock_mhz"] for r in rows}),
            "mem_clock_mhz": sorted({r["mem_clock_mhz"] for r in rows}),
        },
    }
    (OUT / "load-summary.json").write_text(
        json.dumps(summary, indent=1) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
