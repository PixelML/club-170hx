#!/bin/bash
# launch-curtis-1017.sh: mounted over /launch.sh in dsv41-1017-exl3:sm80-tcmoe1 (bench container, host-driver mode). Pixel ML, Oct 18 2026.
# = the first TC-MoE launcher + serve-flag knobs, so one image can run our flags or the serve flags Curtis White
#   (@CurtisAccelerate) published for 1017 (github.com/CurtisAccelerate/DeepSeekv41-Flash-1017-EXL3-CMP170HX @ f5cb5051, README).
#   That repo has no licence file: its flags are used here as facts; no text or code is copied from it.
# TC-MoE kernels + hook: CurtisAccelerate/DeepSeekv41-Flash-EXL3-TC-MoE @cd83ade1 (= 88eb6e13 code, Apache-2.0), kernels by
#   Morrowmake (vllm-cmp170hx @18305948, Apache-2.0); TP align/rotate/coop routing: vcruz305/vllm-exl3 #27 #36 #40 (AGPL-3.0).
# Knobs (env): EP=1|0  CHUNK  MAX_LEN  MAX_SEQS  GPU_UTIL  CG_SIZES  CGMODE=FULL_DECODE_ONLY|FULL_AND_PIECEWISE  ASYNC=0|1  PREFIX=0|1
# Kept 1017 fixes: FAT_THRESHOLD >= max capture size (48), int4 Engram in RAM (DSV41_ENGRAM_DISK=0 set by run-curtis-1017.sh),
#   bias_vl skips (model.py, dspark.py mounts), docker --init (run script).
# Not ported (not public): DSV41_DSPARK_V1 + VLLM_USE_V2_MODEL_RUNNER=0 (Curtis runs DSpark on the v1 runner); we stay on the
#   v2 runner, where our DSpark works. EXL3 arena cache (load time only).
set -eu
ROOT=/opt/dsv41; MODELS=/models; PACK=${PACK:-$MODELS/view}
export DSV41_PACK=$PACK DSV41_ENGRAM_DISK=${DSV41_ENGRAM_DISK:-1} DSV41_ENGRAM_LIB=$ROOT/build/engram_disk.so
export DSV41_REPLICATE_SHARED=${DSV41_REPLICATE_SHARED:-0} EXL3_HOST_EXPERTS=/plan/empty-plan.json EXL3_DMA_GEMM=0 EXL3_HOST_DMA=0 DSV41_EC=0 DSV41_CPU_TIER=0
export VLLM_EXL3_MOE_KERNEL=exllamav3 VLLM_EXL3_FAT_THRESHOLD=${FAT_THRESHOLD:-48}
export OMP_NUM_THREADS=8 VLLM_PLUGINS=vllm_exl3 VLLM_ENGINE_READY_TIMEOUT_S=3600
export FLASHINFER_DISABLE_VERSION_CHECK=1 TORCH_CUDA_ARCH_LIST=8.0 FLASHINFER_CUDA_ARCH_LIST=8.0 PYTHONUNBUFFERED=1
K=${SPEC_K:-5}
SPEC_ARGS=()
[ "$K" != 0 ] && SPEC_ARGS=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$K}")
CGMODE=${CGMODE:-FULL_DECODE_ONLY}
EXTRA=()
[ "${EP:-1}" = 1 ] && EXTRA+=(--enable-expert-parallel)
[ "${ASYNC:-0}" = 1 ] && EXTRA+=(--async-scheduling)
if [ "${PREFIX:-0}" = 1 ]; then EXTRA+=(--enable-prefix-caching); else EXTRA+=(--no-enable-prefix-caching); fi
[ "${CUSTOM_AR:-0}" = 1 ] || EXTRA+=(--disable-custom-all-reduce)
echo "launch-curtis: EP=${EP:-1} CHUNK=${CHUNK:-8192} MAX_LEN=${MAX_LEN:-65536} CGMODE=$CGMODE CG_SIZES=${CG_SIZES:-default} ASYNC=${ASYNC:-0} PREFIX=${PREFIX:-0}" \
  "COOP=${VLLM_EXL3_COOP:-0} ALIGN=${VLLM_EXL3_MOE_TP_ALIGN:-0} ROT=${VLLM_EXL3_MOE_TP_ROTATE:-0} TCV2=${VLLM_EXL3_TC_MOE_V2:-0} TCPRE=${VLLM_EXL3_TC_MOE_PREFILL:-0}" \
  "HAR=${VLLM_HOST_ALLREDUCE:-0} K=$K MAX_SEQS=${MAX_SEQS:-8} GPU_UTIL=${GPU_UTIL:-0.92} INT4=${DSV41_ENGRAM_INT4:-?} NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-}"
exec vllm serve "$PACK" --host 0.0.0.0 --port 8040 --tensor-parallel-size 4 --quantization exl3 \
  --served-model-name deepseek-v4.1-flash --max-logprobs -1 \
  --max-model-len ${MAX_LEN:-65536} --max-num-seqs ${MAX_SEQS:-8} --max-num-batched-tokens ${CHUNK:-8192} \
  --kv-cache-dtype fp8 --gpu-memory-utilization ${GPU_UTIL:-0.92} \
  --language-model-only --tokenizer-mode deepseek_v41 --trust-remote-code "${EXTRA[@]}" \
  --compilation-config "{\"cudagraph_mode\":\"$CGMODE\",\"custom_ops\":[\"all\"]}" \
  --cudagraph-capture-sizes ${CG_SIZES:-1 2 4 6 8 12 16 24 32 48} \
  --enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41 "${SPEC_ARGS[@]}" "$@"
