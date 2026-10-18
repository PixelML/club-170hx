#!/bin/bash
# run-1017.sh ARM [extra vllm serve args]  -- (re)start the dsv41-1017 container for one bench arm.
# Runs INSIDE the bench container (host-driver mode). Any variable below can be overridden from the caller's env.
# Arms (all: TP4 + EP4, int4 Engram pinned in RAM, fp8 KV, 64k ctx, port 8040; power cap is set on the HOST):
#   p2poff        NCCL_P2P_DISABLE=1, custom AR off          (what a VM-per-card host can do; Curtis White's setting)
#   p2pauto       NCCL picks (P2P where its topology rules allow), custom AR off   (= Oct 4 "NCCL only", best c1)
#   p2psys        NCCL_P2P_LEVEL=SYS (P2P across both PLX switches), custom AR off
#   car128k       NCCL_P2P_LEVEL=SYS + vLLM custom AR over PCIe P2P up to 128 KiB (= image default "hybrid")
#   carall        custom AR for all sizes (Oct 3: -22..-35% c4; only for completeness)
#   <arm>-k0      same, DSpark off      <arm>-k10   DSpark k=10      <arm>-seqs4   MAX_SEQS=4 (Oct 4 config)
#   <arm>-repl1   DSV41_REPLICATE_SHARED=1 (whole shared expert per card, Oct 3 path)
#   <arm>-util94  GPU_UTIL=0.94 (Oct 4 value; host driver has no WPR fix -> watch for Xid 31)
#   suffixes combine: p2psys-k0-seqs4
set -euo pipefail
ARM=${1:?usage: run-1017.sh ARM [serve args]}; shift
IMG=${IMG:-dsv41-1017-exl3:sm80-v3p1}
NAME=dsv41-1017
base=${ARM%%-*}
NCCL_ENV=()
CUSTOM_AR=${CUSTOM_AR:-0}; CAR_MAX=${CAR_MAX:-0}
case "$base" in
  p2poff)  NCCL_ENV=(-e NCCL_P2P_DISABLE=1) ;;
  p2pauto) NCCL_ENV=() ;;
  p2psys)  NCCL_ENV=(-e NCCL_P2P_LEVEL=SYS) ;;
  car128k) NCCL_ENV=(-e NCCL_P2P_LEVEL=SYS); CUSTOM_AR=1; CAR_MAX=131072 ;;
  carall)  NCCL_ENV=(-e NCCL_P2P_LEVEL=SYS); CUSTOM_AR=1; CAR_MAX=0 ;;
  *) echo "unknown arm base '$base'"; exit 2 ;;
esac
SPEC_K=${SPEC_K:-5}; MAX_SEQS=${MAX_SEQS:-8}; CG_SIZES=${CG_SIZES:-1 2 4 6 8 12 16 24 32 48}
REPL=${REPL:-0}; GPU_UTIL=${GPU_UTIL:-0.92}
case "$ARM" in *-k0*)  SPEC_K=0 ;; esac
case "$ARM" in *-k10*) SPEC_K=10 ;; esac
case "$ARM" in *-seqs4*) MAX_SEQS=4; CG_SIZES="1 2 4 6 8 12 16 24" ;; esac
case "$ARM" in *-repl1*) REPL=1 ;; esac
case "$ARM" in *-util94*) GPU_UTIL=0.94 ;; esac

docker rm -f $NAME >/dev/null 2>&1 || true
sleep 10   # Oct 4 gotcha: docker rm -f then an immediate docker run races
set -x
docker run -d --init --name $NAME --device nvidia.com/gpu=all --ulimit memlock=-1 --shm-size 16g -p 0.0.0.0:8040:8040 \
  -v /models/dsv41-1017:/models:ro -v /root/dsv41-cache:/root/.cache \
  -v /root/dsv41-tp4/launch-1017.sh:/launch.sh:ro \
  -v /root/dsv41-tp4/patch/model.py:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/nvidia/model.py:ro \
  -v /root/dsv41-tp4/patch/dspark.py:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/nvidia/dspark.py:ro \
  -e CUSTOM_AR=$CUSTOM_AR -e VLLM_FORCE_CUSTOM_AR_PCIE=$CUSTOM_AR -e VLLM_CUSTOM_AR_MAX_BYTES=$CAR_MAX \
  -e DSV41_ENGRAM_INT4=${DSV41_ENGRAM_INT4:-1} -e DSV41_ENGRAM_DISK=${DSV41_ENGRAM_DISK:-0} -e DSV41_ENGRAM_SINGLE=${ENGRAM_SINGLE:-0} -e DSV41_REPLICATE_SHARED=$REPL \
  -e EXL3_DELEGATE_ATTRS=${EXL3_DELEGATE_ATTRS:-0} -e VLLM_EXL3_LOG_MOE_ROUTING=${LOG_ROUTING:-1} \
  -e SPEC_K=$SPEC_K -e MAX_SEQS=$MAX_SEQS -e "CG_SIZES=$CG_SIZES" -e GPU_UTIL=$GPU_UTIL \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,P2P,SHM \
  "${NCCL_ENV[@]}" \
  --restart no --entrypoint /launch.sh "$IMG" "$@"
