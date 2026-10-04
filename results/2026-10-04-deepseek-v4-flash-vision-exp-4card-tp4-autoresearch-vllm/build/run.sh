#!/bin/bash
# DeepSeek-V4-Flash-Vision-Exp on 4x CMP 170HX. LAYOUT=tp4|pp4, P2P=on|off.
LAYOUT=${LAYOUT:-tp4}; P2P=${P2P:-on}; EP=${EP:-0}; DRAFT=${DRAFT:-1}
[ "$EP" = 1 ] && EPARG="--enable-expert-parallel" || EPARG=""
[ "$DRAFT" = 1 ] && SPEC=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":6}") || SPEC=()
IMG=ghcr.io/pixelml/club-170hx@sha256:b26232f8f041c988d3285e2278c9f5001cc49f96131bb0b22a9d38b5e5e061cd
docker rm -f dsv4v >/dev/null 2>&1
if [ "$LAYOUT" = pp4 ]; then PAR="--pipeline-parallel-size 4"; LENV="-e VLLM_PP_LAYER_PARTITION=11,11,11,10"; else PAR="--tensor-parallel-size 4"; LENV=""; fi
if [ "$P2P" = on ]; then
  PENV="-e NCCL_P2P_LEVEL=SYS -e VLLM_FORCE_CUSTOM_AR_PCIE=1 -e VLLM_CUSTOM_AR_MAX_BYTES=${CAR_MAX:-131072} -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False -v <work>/car.py:/vllm/vllm/distributed/device_communicators/custom_all_reduce.py:ro"
  PARG=""
else
  PENV="-e NCCL_P2P_DISABLE=1"; PARG="--disable-custom-all-reduce"
fi
exec docker run -d --name dsv4v --device nvidia.com/gpu=all --ipc=host --ulimit memlock=-1 --shm-size 16g -p 0.0.0.0:18098:8000 \
  -e HF_HUB_OFFLINE=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_ENGINE_ITERATION_TIMEOUT_S=1800 \
  $LENV $PENV -v /path/to/DeepSeek-V4-Flash-Vision-Exp:/model:ro -v <cache>:/root/.cache \
  $IMG vllm serve /model --served-model-name dsv4v $PAR $PARG $EPARG --kv-cache-dtype fp8 \
  --block-size 256 --max-model-len 16384 --max-num-batched-tokens 2048 \
  --trust-remote-code --gpu-memory-utilization 0.90 --max-num-seqs 8 \
  --no-enable-flashinfer-autotune --tokenizer-mode deepseek_v4 \
  "${SPEC[@]}" \
  --hf-overrides "{\"architectures\":[\"DeepseekV4ForConditionalGeneration\"]}" \
  --limit-mm-per-prompt "{\"image\": 2}" "$@"
