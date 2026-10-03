#!/bin/bash
# DSV4.1 TP4 on 4x CMP 170HX: the reference entrypoint (dsv41-flash-offload 2e9da5e) env and args,
# all routed experts VRAM-resident (empty host plan), no host DMA / CPU tier / expert cache.
set -eu
ROOT=/opt/dsv41; MODELS=/models; PACK=$MODELS/DSV41-EXL3-3090-D010
export DSV41_PACK=$PACK DSV41_ENGRAM_DISK=${DSV41_ENGRAM_DISK:-1} DSV41_ENGRAM_LIB=$ROOT/build/engram_disk.so
export DSV41_REPLICATE_SHARED=1 EXL3_HOST_EXPERTS=/plan/empty-plan.json EXL3_DMA_GEMM=0 EXL3_HOST_DMA=0 DSV41_EC=0 DSV41_CPU_TIER=0
export VLLM_EXL3_MOE_KERNEL=exllamav3 VLLM_EXL3_FAT_THRESHOLD=32
export OMP_NUM_THREADS=8 VLLM_PLUGINS=vllm_exl3 VLLM_ENGINE_READY_TIMEOUT_S=3600
export FLASHINFER_DISABLE_VERSION_CHECK=1 TORCH_CUDA_ARCH_LIST=8.0 FLASHINFER_CUDA_ARCH_LIST=8.0 PYTHONUNBUFFERED=1
exec vllm serve "$PACK" --host 0.0.0.0 --port 8040 --tensor-parallel-size 4 --enable-expert-parallel --quantization exl3 \
  --served-model-name deepseek-v4.1-flash --max-logprobs -1 \
  --max-model-len ${MAX_LEN:-65536} --max-num-seqs 4 --max-num-batched-tokens ${CHUNK:-8192} \
  --kv-cache-dtype fp8 --gpu-memory-utilization ${GPU_UTIL:-0.94} \
  --no-enable-prefix-caching --language-model-only --tokenizer-mode deepseek_v41 --trust-remote-code \
  $( [ "${CUSTOM_AR:-0}" = 1 ] || echo --disable-custom-all-reduce ) --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"custom_ops\":[\"all\"]}" \
  --cudagraph-capture-sizes 1 2 4 6 8 12 16 24 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 "$@"
