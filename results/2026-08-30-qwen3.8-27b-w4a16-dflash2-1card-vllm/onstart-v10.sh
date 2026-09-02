#!/usr/bin/env bash
# v10: test the two Ninfer tricks from the Reddit 5090 post, adapted to the
# syv vLLM stack, on ONE rented 170HX 64GB instance via a 4-boot matrix:
#   BOOT A  dflash2 k=7 CTX=fast  -- baseline (expect ~147.7 tok/s decode)
#   BOOT B  mtp k=4 CTX=fast      -- control for C (truncated 40k draft head)
#   BOOT C  mtp k=4 CTX=fast MTP_DRAFT_VOCAB=0 -- "lm-head-draft": draft with
#           the full shared 248k lm_head instead of the truncated head
#   BOOT D  dflash2 k=7 CTX=long  -- fp8 KV, 131072 ctx: the syv-native
#           equivalent of Ninfer's "KV offload to host" capability test
#           (on 64GB the pool fits in VRAM; --host-kv itself is pointless here)
# Setup phase is byte-identical to v9 (proven). Bench protocol identical to
# v9: streaming, count usage.completion_tokens (not SSE events), temp 0.
set -ux
date -u
nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader || true

mkdir -p /root/.ssh && chown root:root /root/.ssh && chmod 700 /root/.ssh
touch /root/.ssh/authorized_keys && chown root:root /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys

apt-get update -qq || true
DEBIAN_FRONTEND=noninteractive apt-get install -y python3.12 python3.12-venv python3.12-dev build-essential patch git curl ca-certificates || true

mkdir -p /app && cd /app
[ -d qwen-serving/.git ] || git clone --depth 1 https://github.com/syv-ai/qwen38-27b-rtx3090 qwen-serving
ln -sfn /app/qwen-serving/venv /app/venv
ln -sfn /app/qwen-serving/prepare /app/prepare
cd qwen-serving

if [ ! -x venv/bin/vllm ]; then
  python3.12 -m venv venv
  venv/bin/pip install --upgrade pip wheel
  venv/bin/pip install -r docker/requirements.txt
  venv/bin/pip install flashinfer-python flashinfer-cubin==0.6.13 || true
fi

mkdir -p /usr/local/cuda/bin
NVIDIA_NVCC=$(venv/bin/python -c 'import nvidia.cuda_nvcc, os; print(os.path.join(os.path.dirname(nvidia.cuda_nvcc.__file__), "bin"))' 2>/dev/null || true)
if [ -n "$NVIDIA_NVCC" ] && [ -x "$NVIDIA_NVCC/nvcc" ]; then
  ln -sf "$NVIDIA_NVCC/nvcc" /usr/local/cuda/bin/nvcc
fi
/usr/local/cuda/bin/nvcc --version | tail -1 || true

# curand.h for the flashinfer fast-topk JIT (v8 lesson)
CURAND_H=$(find /app/qwen-serving/venv -name curand.h 2>/dev/null | head -1 || true)
mkdir -p /usr/local/cuda/include
if [ -n "$CURAND_H" ]; then
  ln -sf "$CURAND_H" /usr/local/cuda/include/curand.h
  echo "CURAND-FROM-VENV: $CURAND_H"
else
  DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-curand-13-0 || true
  find /usr/local/cuda* /opt/cuda* -name curand.h 2>/dev/null | head -1 | while read -r h; do ln -sf "$h" /usr/local/cuda/include/curand.h; done
  echo "CURAND-FROM-APT-OR-MISSING"
fi
ls -l /usr/local/cuda/include/ 2>/dev/null || true
rm -rf /root/.cache/flashinfer

SP=$(venv/bin/python -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' | tail -n1)
grep -q dflash2-backport "$SP/vllm/engine/arg_utils.py" 2>/dev/null || for p in patches/*.patch; do echo "== $p"; patch -p1 -d "$SP" < "$p" || true; done

export HF_HOME=/cache/huggingface
export PATH=/app/venv/bin:$PATH
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 FLASHINFER_DISABLE_VERSION_CHECK=1
export VLLM_API_KEY=<bench-api-key>

BASE=/app/qwen-serving/models/Qwen3.8-27B-W4A16-AutoRound
BASE_MODEL_DIR=$BASE bash docker/prepare.sh || echo "PREPARE-FAILED-AGAIN"
if [ -f "$BASE-fast/model.safetensors.index.json" ]; then
  echo "FAST-VARIANT-PRESENT"
else
  echo "FAST-VARIANT-MISSING fetching explicitly"
  venv/bin/python prepare/fetch_fast_variant.py "$BASE" "$BASE-fast" || echo "FAST-VARIANT-FETCH-FAILED"
fi
venv/bin/python prepare/fetch_dflash2.py || echo "DFLASH2-FETCH-NONZERO"

# ---- bench harness (protocol identical to v9; takes boot tag from argv) ----
cat > /app/bench.py <<'PYEOF'
import json, subprocess, sys, threading, time, urllib.request
BOOT = sys.argv[1] if len(sys.argv) > 1 else "unknown"
BASE_URL = "http://127.0.0.1:18020"
KEY = "<bench-api-key>"
HDR = {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}
peak = {"util": 0, "power": 0, "mem": 0, "clk": 0, "temp": 0}
stop = False
def sample():
    global stop
    while not stop:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,memory.used,clocks.sm,temperature.gpu",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip()
            u, p, m, c, t = map(float, out.split(","))
            peak["util"] = max(peak["util"], u); peak["power"] = max(peak["power"], p)
            peak["mem"] = max(peak["mem"], m); peak["clk"] = max(peak["clk"], c)
            peak["temp"] = max(peak["temp"], t)
        except Exception:
            pass
        time.sleep(1)
threading.Thread(target=sample, daemon=True).start()

def post(path, body, timeout=600):
    req = urllib.request.Request(BASE_URL + path, data=json.dumps(body).encode(), headers=HDR)
    return urllib.request.urlopen(req, timeout=timeout)

def ntokens(prompt):
    with post("/tokenize", {"model": "qwen3.8-27b", "prompt": prompt}) as r:
        return len(json.load(r)["tokens"])

def req_stream(prompt, maxtok):
    body = {"model": "qwen3.8-27b", "prompt": prompt, "max_tokens": maxtok,
            "temperature": 0.0, "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True}}
    t0 = time.perf_counter(); ttft = None; n = 0
    ctok = None
    with post("/v1/completions", body) as resp:
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
                except Exception:
                    pass
    if ctok is None:
        ctok = n
    return ttft, time.perf_counter() - t0, ctok

def bench(name, prompt, maxtok, warm, samples, ptok):
    for i in range(warm):
        req_stream(prompt, maxtok)
    res = []
    for i in range(samples):
        ttft, total, n = req_stream(prompt, maxtok)
        print(json.dumps({"boot": BOOT, "run": name, "i": i, "ttft_ms": round(ttft * 1000, 1), "total_s": round(total, 3),
                          "out_tok": n, "decode_tok_s": round((n - 1) / (total - ttft), 2)}), flush=True)
        res.append((ttft, total, n))
    ttft = sum(r[0] for r in res) / len(res) * 1000
    tot = sum(r[1] for r in res) / len(res)
    nn = sum(r[2] for r in res) / len(res)
    print(json.dumps({"boot": BOOT, "summary": name, "prompt_tokens": ptok, "mean_ttft_ms": round(ttft, 1),
                      "mean_total_s": round(tot, 3), "mean_out_tok": round(nn, 1),
                      "decode_tok_s": round((nn - 1) / (tot - ttft / 1000), 2),
                      "prefill_tok_s": round(ptok / (ttft / 1000), 1)}), flush=True)

P256 = "Write a story about a robot who learns to paint."
LONG = ("summarize the following text. " +
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega ") * 200
t256 = ntokens(P256); tlong = ntokens(LONG)
print(json.dumps({"boot": BOOT, "prompt_tokens": {"story": t256, "long": tlong}}), flush=True)
bench("decode256", P256, 256, 1, 3, t256)
bench("decode900", P256, 900, 1, 3, t256)
bench("prefill_long", LONG, 8, 1, 3, tlong)
time.sleep(2); stop = True; time.sleep(1)
print(json.dumps({"boot": BOOT, "gpu_peak": peak}), flush=True)
PYEOF

cd /app/qwen-serving

run_boot() {
  NAME=$1; shift
  echo "=== BOOT $NAME START env: $* ==="
  pkill -f 'vllm serve' 2>/dev/null || true
  sleep 8
  pkill -9 -f 'vllm serve' 2>/dev/null || true
  for i in 1 2 3 4 5 6 7 8; do
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$USED" -lt 2000 ] && break
    echo "waiting GPU drain: ${USED}MiB"
    sleep 10
  done
  env "$@" bash single-user/start_qwen.sh >/tmp/qwen38-server-$NAME.log 2>&1 &
  echo "server pid $!" | tee /tmp/qwen38-server.pid
  HEALTH=0
  for i in $(seq 1 150); do
    if curl -sf -m 3 http://127.0.0.1:18020/health >/dev/null; then HEALTH=1; echo "HEALTH-OK-$NAME i=$i"; break; fi
    sleep 10
  done
  if [ "$HEALTH" != 1 ]; then
    echo "HEALTH-NEVER-OK-$NAME"
    tail -n 80 /tmp/qwen38-server-$NAME.log
    return 1
  fi
  sleep 20
  venv/bin/python /app/bench.py "$NAME"
  echo "BENCH-DONE-$NAME"
  grep -i -E "accept|spec" /tmp/qwen38-server-$NAME.log | tail -n 20 || true
  return 0
}

echo "=== MATRIX START $(date -u) ==="
run_boot A SPEC=dflash2 CTX=fast   MAX_SEQS=1 DFLASH_TOKENS=7 PORT=18020 GPU_UTIL=0.90 KV_MEM=
run_boot B SPEC=mtp      CTX=fast   MAX_SEQS=1 DRAFT_TOKENS=4   PORT=18020 GPU_UTIL=0.90
run_boot C SPEC=mtp      CTX=fast   MAX_SEQS=1 DRAFT_TOKENS=4   PORT=18020 GPU_UTIL=0.90 MTP_DRAFT_VOCAB=0
run_boot D SPEC=dflash2 CTX=long   MAX_SEQS=1 DFLASH_TOKENS=7 PORT=18020 GPU_UTIL=0.90 KV_MEM=
echo "=== MATRIX DONE $(date -u) ==="
nvidia-smi --query-gpu=name,memory.used,power.draw --format=csv,noheader || true
