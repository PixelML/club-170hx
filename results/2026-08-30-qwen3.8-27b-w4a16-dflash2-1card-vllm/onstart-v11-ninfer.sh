#!/usr/bin/env bash
# qwen38-vast-onstart-v11-ninfer.sh — vast.ai onstart: native NInfer build + serve + bench
#
# Image: nvidia/cuda:13.1.1-devel-ubuntu24.04
#   The requested tag nvidia/cuda:13.1.1-devel-ubuntu25.10 DOES NOT EXIST
#   (verified via hub.docker.com/v2/repositories/nvidia/cuda/tags on 2026-08-28:
#    no ubuntu25.10 CUDA image exists at all; 13.1.1-devel ships only
#    ubuntu24.04 / ubuntu22.04). Closest 13.1.x devel tag chosen:
#    13.1.1-devel-ubuntu24.04. (The fork Dockerfile itself pins
#    13.1.2-devel-ubuntu24.04, which also exists.)
#
# NOTE: the fork consumes CMAKE_CUDA_ARCHITECTURES, NOT NINFER_CUDA_ARCHITECTURES.
#   CMakeLists.txt validates CMAKE_CUDA_ARCHITECTURES against ^(80|86|89)$; the
#   Dockerfile maps its NINFER_CUDA_ARCHITECTURES build-arg via
#   -DCMAKE_CUDA_ARCHITECTURES=${NINFER_CUDA_ARCHITECTURES}. Mirrored here.
#
# Extra apt packages beyond the requested list (required by CMakeLists.txt with
# NINFER_BUILD_APPS=ON, mirroring the Dockerfile): pkg-config,
# libavcodec-dev libavformat-dev libavutil-dev libswscale-dev (FFMPEG >= 60/58/7),
# libcurl4-openssl-dev (libcurl >= 7.85 via pkg-config).
set -ux
date -u
nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader || true

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq || true
apt-get install -y --no-install-recommends \
  build-essential cmake ninja-build git curl python3 ca-certificates \
  pkg-config libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
  libcurl4-openssl-dev

mkdir -p /app /models
[ -d /app/ninfer/.git ] || git clone --depth 1 https://github.com/Ithrial/ninfer-cmp170hx /app/ninfer

# Native build (no docker) — mirrors the Dockerfile build stage:
cmake -S /app/ninfer -B /app/ninfer/build -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DNINFER_BUILD_APPS=ON -DBUILD_TESTING=OFF -DNINFER_BUILD_BENCHMARKS=OFF
cmake --build /app/ninfer/build --parallel --target ninfer ninfer-serve
/app/ninfer/build/apps/ninfer-serve --help >/dev/null 2>&1 && echo "BUILD-OK"

# Model artifact — exact URL from scripts/download-qwen38.sh (~18.2 GB), resumable:
if [ ! -f /models/qwen3_8_27b.ninfer ]; then
  curl -L -C - --fail --output /models/qwen3_8_27b.ninfer \
    'https://huggingface.co/neroued/Qwen3.8-27B-NInfer/resolve/main/qwen3_8_27b.ninfer'
fi
ls -l /models/qwen3_8_27b.ninfer

cat > /app/bench-ninfer.py <<'PYEOF'
#!/usr/bin/env python3
"""bench-ninfer.py — v11 measurement harness for NInfer (engine=ninfer).

Protocol-identical to the v9/v10 vLLM harness (warmup 1 + 3 samples,
decode256/decode900/prefill_long, temperature 0.0, ignore_eos, streaming,
usage.completion_tokens counted, nvidia-smi peak sampler) with two engine
mandated deltas:
  - POSTs /v1/chat/completions: the fork has NO /v1/completions route
    (verified in src/serve/http_server.cpp register_routes()).
  - Token counts come from usage in the final SSE chunk; /tokenize is
    never called (no such endpoint in the fork).

Writes nothing to disk; stdout only (one JSON object per line).
"""

import json
import subprocess
import threading
import time
import urllib.request

BASE_URL = "http://127.0.0.1:18020"
MODEL = "qwen3.8-27b"
KEY = "<bench-api-key>"
HDR = {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}

peak = {"util": 0, "power": 0, "mem": 0, "clk": 0, "temp": 0}
stop = False


def sample():
    global stop
    while not stop:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,memory.used,clocks.sm,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5).stdout.strip()
            u, p, m, c, t = map(float, out.split(","))
            peak["util"] = max(peak["util"], u)
            peak["power"] = max(peak["power"], p)
            peak["mem"] = max(peak["mem"], m)
            peak["clk"] = max(peak["clk"], c)
            peak["temp"] = max(peak["temp"], t)
        except Exception:
            pass
        time.sleep(1)


def req_stream(prompt, maxtok):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": maxtok, "temperature": 0.0, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(BASE_URL + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=HDR)
    t0 = time.perf_counter()
    ttft = None
    n = 0
    ctok = None
    ptok = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for line in resp:
            if line.startswith(b"data: "):
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    break
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n += 1
                try:
                    obj = json.loads(payload)
                    u = obj.get("usage") or {}
                    if u.get("completion_tokens") is not None:
                        ctok = u["completion_tokens"]
                    if u.get("prompt_tokens") is not None:
                        ptok = u["prompt_tokens"]
                except Exception:
                    pass
    if ctok is None:
        ctok = n
    return ttft, time.perf_counter() - t0, ctok, ptok


def bench(name, prompt, maxtok, warm, samples):
    for i in range(warm):
        req_stream(prompt, maxtok)
    res = []
    for i in range(samples):
        ttft, total, n, ptok = req_stream(prompt, maxtok)
        row = {"engine": "ninfer", "run": name, "i": i,
               "ttft_ms": round(ttft * 1000, 1), "total_s": round(total, 3),
               "out_tok": n, "decode_tok_s": round((n - 1) / (total - ttft), 2)}
        if ptok is not None:
            row["prompt_tokens"] = ptok
        print(json.dumps(row), flush=True)
        res.append((ttft, total, n, ptok))
    ttft = sum(r[0] for r in res) / len(res) * 1000
    tot = sum(r[1] for r in res) / len(res)
    nn = sum(r[2] for r in res) / len(res)
    ptoks = [r[3] for r in res if r[3] is not None]
    summary = {"engine": "ninfer", "summary": name,
               "mean_ttft_ms": round(ttft, 1), "mean_total_s": round(tot, 3),
               "mean_out_tok": round(nn, 1),
               "decode_tok_s": round((nn - 1) / (tot - ttft / 1000), 2)}
    if ptoks:
        summary["prompt_tokens"] = ptoks[0]
        summary["prefill_tok_s"] = round(ptoks[0] / (ttft / 1000), 1)
    print(json.dumps(summary), flush=True)


def main():
    print(json.dumps({"engine": "ninfer", "bench": "v11", "base_url": BASE_URL,
                      "model": MODEL}), flush=True)
    threading.Thread(target=sample, daemon=True).start()
    P256 = "Write a story about a robot who learns to paint."
    LONG = ("summarize the following text. " +
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega ") * 200
    bench("decode256", P256, 256, 1, 3)
    bench("decode900", P256, 900, 1, 3)
    bench("prefill_long", LONG, 8, 1, 3)
    time.sleep(2)
    stop = True
    time.sleep(1)
    print(json.dumps({"engine": "ninfer", "gpu_peak": peak}), flush=True)


if __name__ == "__main__":
    main()
PYEOF

# Serve on loopback:18020, single GPU, CUDA graphs ON (no --no-cuda-graph), MTP3 + lm-head draft.
# Flag set mirrors scripts/run-qwen38-c1.sh (port 18020 per the vLLM CTX=fast baseline).
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
nohup /app/ninfer/build/apps/ninfer-serve /models/qwen3_8_27b.ninfer \
  --host 127.0.0.1 --port 18020 \
  --model-id qwen3.8-27b \
  --max-context 65536 --kv-capacity 65536 \
  --max-concurrency 1 --max-pending-requests 16 \
  --prefill-chunk 1024 --kv-dtype int8 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  >/tmp/ninfer-serve.log 2>&1 &
echo "server pid $!" | tee /tmp/ninfer-serve.pid

HEALTH=0
for i in $(seq 1 150); do
  if curl -sf -m 3 http://127.0.0.1:18020/health >/dev/null; then HEALTH=1; echo "HEALTH-OK i=$i"; break; fi
  sleep 10
done
[ "$HEALTH" = "1" ] || { echo "HEALTH-NEVER-OK"; tail -n 60 /tmp/ninfer-serve.log; exit 0; }

sleep 20
python3 /app/bench-ninfer.py
echo "BENCH-DONE"
grep -i -E "accept|spec|throughput" /tmp/ninfer-serve.log | tail -n 40 || true
tail -n 30 /tmp/ninfer-serve.log || true
