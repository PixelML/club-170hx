#!/usr/bin/env python3
"""Run the club recipe's usage-accounted suite against a local server with a
temperature/Xid guard, then write telemetry next to the receipts.

usage: run_qwen_suite.py <recipe_dir> <out.jsonl> <gpu_indices comma list>
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

recipe_dir, out_path, gpus = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, recipe_dir)
from live_benchmark import run_suite  # noqa: E402

CORE_STOP_C = 80
telemetry = []
violation = []


def sample():
    q = subprocess.run(
        ["nvidia-smi", "-i", gpus, "--query-gpu=index,temperature.gpu,temperature.memory,power.draw,power.limit,clocks.sm,pcie.link.gen.current,pcie.link.width.current",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout
    rows = []
    for line in q.strip().splitlines():
        idx, t, tm, p, pl, sm, gen, width = [x.strip() for x in line.split(",")]
        rows.append({"gpu": int(idx), "core_c": float(t), "mem_c": None if tm in ("N/A", "[N/A]") else float(tm),
                     "power_w": float(p), "limit_w": float(pl), "sm_mhz": int(sm), "gen": int(gen), "width": int(width)})
    return rows


def watcher(stop):
    base_xid = subprocess.run(["bash", "-c", "dmesg | grep -c Xid"], capture_output=True, text=True).stdout.strip()
    while not stop.is_set():
        rows = sample()
        telemetry.append({"t": round(time.time(), 1), "gpus": rows})
        if any(r["core_c"] >= CORE_STOP_C for r in rows):
            violation.append(f"core temperature >= {CORE_STOP_C} C")
        xid = subprocess.run(["bash", "-c", "dmesg | grep -c Xid"], capture_output=True, text=True).stdout.strip()
        if xid != base_xid:
            violation.append("new Xid in kernel log")
        stop.wait(2)


def safety_check():
    if violation:
        raise RuntimeError("SAFETY STOP: " + violation[0])


stop = threading.Event()
th = threading.Thread(target=watcher, args=(stop,), daemon=True)
th.start()
try:
    run_suite("http://127.0.0.1:18020", os.environ["VLLM_API_KEY"], out_path, safety_check)
finally:
    stop.set()
    th.join()
    with open(out_path.with_suffix(".telemetry.jsonl"), "w") as f:
        for rec in telemetry:
            f.write(json.dumps(rec) + "\n")
peak = {}
for rec in telemetry:
    for r in rec["gpus"]:
        p = peak.setdefault(r["gpu"], {"core_c": 0, "power_w": 0})
        p["core_c"] = max(p["core_c"], r["core_c"])
        p["power_w"] = max(p["power_w"], r["power_w"])
print(json.dumps({"peaks": peak, "violations": violation}))
for line in out_path.read_text().splitlines():
    rec = json.loads(line)
    if rec.get("type") == "summary":
        print(rec["case"], "decode", rec["decode_tok_s"], "prefill", rec["prefill_tok_s"], "ttft_ms", rec["mean_ttft_ms"])
