#!/bin/bash
# launch-1017.sh: mounted over /launch.sh in dsv41-1017-exl3:sm80-v3p1 (bench container, host-driver mode).
# = the image's baked /launch.sh (0xSero dsv41-flash-offload 2e9da5e entrypoint env + our Oct 3-4 changes) with:
#   PACK -> /models/view (1017 overlay from prepare_1017.py), DSV41_REPLICATE_SHARED from env (default 0),
#   MAX_SEQS + CG_SIZES from env (8 users with DSpark k=5 = 48 tokens per decode step -> capture up to 48),
#   SPEC from env (DSpark on/off/k), everything else unchanged.
set -eu
ROOT=/opt/dsv41; MODELS=/models; PACK=${PACK:-$MODELS/view}
export DSV41_PACK=$PACK DSV41_ENGRAM_DISK=${DSV41_ENGRAM_DISK:-1} DSV41_ENGRAM_LIB=$ROOT/build/engram_disk.so
export DSV41_REPLICATE_SHARED=${DSV41_REPLICATE_SHARED:-0} EXL3_HOST_EXPERTS=/plan/empty-plan.json EXL3_DMA_GEMM=0 EXL3_HOST_DMA=0 DSV41_EC=0 DSV41_CPU_TIER=0
# FAT_THRESHOLD must be >= the largest capture size (48 = 8 seqs x (k5+1)): above it exl3.py:1454 does a host
# .item() sync, which CUDA-graph capture forbids (Oct 18 boot 4 crash at 32). Exact: fused kernel covers these rows.
export VLLM_EXL3_MOE_KERNEL=exllamav3 VLLM_EXL3_FAT_THRESHOLD=${FAT_THRESHOLD:-48}
export OMP_NUM_THREADS=8 VLLM_PLUGINS=vllm_exl3 VLLM_ENGINE_READY_TIMEOUT_S=3600
export FLASHINFER_DISABLE_VERSION_CHECK=1 TORCH_CUDA_ARCH_LIST=8.0 FLASHINFER_CUDA_ARCH_LIST=8.0 PYTHONUNBUFFERED=1
K=${SPEC_K:-5}
SPEC_ARGS=()
[ "$K" != 0 ] && SPEC_ARGS=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$K}")
echo "launch-1017: PACK=$PACK K=$K MAX_SEQS=${MAX_SEQS:-8} GPU_UTIL=${GPU_UTIL:-0.92} CUSTOM_AR=${CUSTOM_AR:-0} REPL=$DSV41_REPLICATE_SHARED INT4=${DSV41_ENGRAM_INT4:-?} NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-} NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-}"
exec vllm serve "$PACK" --host 0.0.0.0 --port 8040 --tensor-parallel-size 4 --enable-expert-parallel --quantization exl3 \
  --served-model-name deepseek-v4.1-flash --max-logprobs -1 \
  --max-model-len ${MAX_LEN:-65536} --max-num-seqs ${MAX_SEQS:-8} --max-num-batched-tokens ${CHUNK:-8192} \
  --kv-cache-dtype fp8 --gpu-memory-utilization ${GPU_UTIL:-0.92} \
  --no-enable-prefix-caching --language-model-only --tokenizer-mode deepseek_v41 --trust-remote-code \
  $( [ "${CUSTOM_AR:-0}" = 1 ] || echo --disable-custom-all-reduce ) --compilation-config "{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"custom_ops\":[\"all\"]}" \
  --cudagraph-capture-sizes ${CG_SIZES:-1 2 4 6 8 12 16 24 32 48} \
  --enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 "${SPEC_ARGS[@]}" "$@"
