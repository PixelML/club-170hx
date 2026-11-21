#!/bin/bash
# run-vllm.sh <gpu> <port> <tag> [extra vllm serve args]  — run on the GPU host
# vllm/vllm-openai:inkling-2-mini @sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e
# thinkingmachines/Inkling-2-Mini @ed40e7da625f594fc9c4990f5f66be24873c22d7 (NVFP4 experts + MXFP8 dense)
G=$1; P=$2; TAG=$3; shift 3
mkdir -p /work/cache
docker rm -f "inkling-vllm-$TAG" >/dev/null 2>&1
docker run -d --name "inkling-vllm-$TAG" --gpus "device=$G" --network host --ipc host \
  -v /models/inkling2-mini:/models/inkling2-mini:ro -v /work/cache:/root/.cache \
  -v /work/patches:/patches:ro ${VLLM_PATCH_MOUNTS} \
  vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e \
  /models/inkling2-mini/Inkling-2-Mini --served-model-name inkling --host 127.0.0.1 --port "$P" \
  --max-model-len 32768 --reasoning-parser inkling "$@"
