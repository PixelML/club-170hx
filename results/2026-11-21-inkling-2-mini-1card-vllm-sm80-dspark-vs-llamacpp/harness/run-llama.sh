#!/bin/bash
# run-llama.sh <gpu> <port> <tag> [extra llama-server args]  — run on the GPU host
# llama.cpp PR #32695 @6196d980 (sm_80 build), ggml-org/Inkling-2-Mini-GGUF IQ4_NL @db61c5b4
G=$1; P=$2; TAG=$3; shift 3
docker rm -f "inkling-llama-$TAG" >/dev/null 2>&1
docker run -d --name "inkling-llama-$TAG" --gpus "device=$G" --network host \
  -v /work/src/llama.cpp/build/bin:/opt/llama:ro -v /models/inkling2-mini:/models/inkling2-mini:ro \
  -e LD_LIBRARY_PATH=/opt/llama nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  /opt/llama/llama-server -m /models/inkling2-mini/gguf/Inkling-2-Mini-IQ4_NL.gguf --alias inkling \
  --host 127.0.0.1 --port "$P" -ngl 999 -fa on --jinja --no-warmup "$@"
