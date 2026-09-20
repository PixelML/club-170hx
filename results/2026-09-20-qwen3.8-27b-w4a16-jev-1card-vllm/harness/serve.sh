#!/usr/bin/env bash
# Launch the Qwen3.8-27B Jev engine on one CMP 170HX (SM80).
#
#   MODEL_DIR=/models/Qwen3.8-27B-GPTQ-4bit ./serve.sh
#
# Serves the W4A16 GPTQ checkpoint with the fork runtime's vLLM, then the Jev
# interposer reads label logits from /v1/completions. Two flags matter for the
# Jev path and neither is a default:
#
#   --logprobs-mode processed_logprobs   the default (raw_logprobs) computes
#                                        logprobs *before* allowed_token_ids,
#                                        so the label distribution is missing
#   --max-logprobs 128                   one logprob per label (62 symbols max)
#
# The read path also neutralises the checkpoint's generation_config defaults
# (top_k=20, top_p=0.95) per request; see jev_server.py.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
VENV=${VENV:-$HOME/venv-qwen}
MODEL_DIR=${MODEL_DIR:?set MODEL_DIR to the W4A16 GPTQ checkpoint}
PORT=${PORT:-18030}
JEV_PORT=${JEV_PORT:-18031}
GPU=${GPU:-0}
LOG=${LOG:-$HERE/../receipts/serve.log}
JEV_LOG=${JEV_LOG:-$HERE/../receipts/jev-server.log}

echo "engine: $MODEL_DIR on GPU $GPU, port $PORT, log $LOG"
CUDA_VISIBLE_DEVICES=$GPU \
VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
"$VENV/bin/vllm" serve "$MODEL_DIR" \
  --served-model-name qwen3.8-27b-jev \
  --port "$PORT" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-logprobs 128 \
  --logprobs-mode processed_logprobs \
  --enable-prefix-caching \
  2>&1 | tee "$LOG" &

for _ in $(seq 1 120); do
  if curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null; then break; fi
  sleep 5
done
curl -sf -m 3 "http://127.0.0.1:$PORT/health" >/dev/null || { echo "engine never became healthy"; exit 1; }
echo "engine healthy"

echo "interposer: port $JEV_PORT -> http://127.0.0.1:$PORT, log $JEV_LOG"
nohup "$VENV/bin/python" "$HERE/jev_server.py" \
  --upstream "http://127.0.0.1:$PORT" \
  --tokenizer "$MODEL_DIR" \
  --model qwen3.8-27b-jev \
  --port "$JEV_PORT" >"$JEV_LOG" 2>&1 &

for _ in $(seq 1 24); do
  if curl -sf -m 3 "http://127.0.0.1:$JEV_PORT/health" >/dev/null; then
    echo "interposer healthy"; exit 0
  fi
  sleep 2
done
echo "interposer never became healthy"; exit 1
