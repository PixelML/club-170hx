#!/bin/bash
# DeepSeek-V4-Flash-Vision, three arms on the published image: TP4 SYS on (published), TP4 SYS off, PP4 SYS off.
# Set MODEL (the checkpoint folder) and BENCH (a folder with protocol.sh, bench_harness.py, correct.py, probe.py, vision_check.py
# from results/2026-10-04-deepseek-v4-flash-vision-exp-4card-tp4-autoresearch-vllm/{build,autoresearch}). protocol.sh serves on :18098.
set -u
MODEL=${MODEL:?set MODEL}; BENCH=${BENCH:?set BENCH}; CACHE=${CACHE:-$HOME/.cache/dsv4v}
IMG=ghcr.io/pixelml/club-170hx@sha256:97ecb29396abc87a9a4f99e18e561f97dc610ca1d79788819d374f5fff1205ba
watchdog() { while sleep 5; do t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | sort -n | tail -1); [ "$t" -ge 80 ] && { echo "WATCHDOG ${t}C"; docker stop dsv4v; return; }; done; }
arm() {  # arm NAME PARALLEL-ARGS ENV-FLAGS...
  local n=$1 par=$2; shift 2
  docker rm -f dsv4v >/dev/null 2>&1; sleep 10
  docker run -d --name dsv4v --gpus all --ipc=host --ulimit memlock=-1 --shm-size 16g -p 18098:8000 \
    -e HF_HUB_OFFLINE=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_ENGINE_ITERATION_TIMEOUT_S=1800 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False "$@" \
    -v $MODEL:/model:ro -v $CACHE:/root/.cache \
    $IMG vllm serve /model --served-model-name dsv4v $par --kv-cache-dtype fp8 --block-size 256 \
    --max-model-len 16384 --max-num-batched-tokens 2048 --trust-remote-code --gpu-memory-utilization 0.90 \
    --max-num-seqs 8 --disable-custom-all-reduce --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --no-enable-flashinfer-autotune --tokenizer-mode deepseek_v4 \
    --speculative-config '{"method":"dspark","num_speculative_tokens":6}' \
    --hf-overrides '{"architectures":["DeepseekV4ForConditionalGeneration"]}' --limit-mm-per-prompt '{"image":2}'
  watchdog & WD=$!
  bash $BENCH/protocol.sh $n; kill $WD 2>/dev/null
}
arm x16-tp4-sys   "--tensor-parallel-size 4"   -e NCCL_P2P_LEVEL=SYS
arm x16-tp4-nosys "--tensor-parallel-size 4"
arm x16-pp4-nosys "--pipeline-parallel-size 4" -e VLLM_PP_LAYER_PARTITION=11,11,11,10
docker rm -f dsv4v >/dev/null 2>&1
