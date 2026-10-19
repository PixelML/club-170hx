#!/bin/bash
# run-curtis-1017.sh ARM [extra vllm serve args] -- (re)start the dsv41-1017 container for one arm of the Curtis-stack replication
# on OUR quant (PixelML/DeepSeek-V4.1-Flash-1017-EXL3-3bpw @3d095b72, int4 Engram). Runs INSIDE the bench container. Pixel ML, Oct 18 2026.
# Image dsv41-1017-exl3:sm80-tcmoe1 (image id sha256:97d358644223603e25c4f316208eadf82d06444a4e29d00f2778a93734f2f11d). Credits: see launch-curtis-1017.sh.
# All arms: NCCL_P2P_DISABLE=1 + custom AR off (= today's p2poff baseline and Curtis's peer-off setting), int4 Engram in RAM, DSpark k=5.
#
# Arm bases (one new thing at a time; each arm is independent, a failing arm does not block the next):
#   tcep       new image, every new switch off, EP4                   -> regression check vs p2poff (104.4/75.4/150.1)
#   epcoop     EP4 + exl3_moe_coop for <= 42 rows                     -> coop alone, no TP change
#   tptp       MoE TP4, Hadamard-aligned 640/640/512/512 + rotation, stock exl3_moe kernel
#   tpcoop     tptp + coop                                            (= Curtis "graphs + coop" stack)
#   tcfull     tpcoop + TC moe_out_v2 (43-64 rows) + moe_out_prefill (65-2048 rows), max-num-batched-tokens 2048
#   tcfullhar  tcfull + Morrowmake host-staged all-reduce (VLLM_HOST_ALLREDUCE=1, vendored /root/tcmoe/hostar)   (= Curtis shipped)
# Aliases from the plan: base=tcep  coop=tpcoop-cf  cooptc=tcfull-cf  cooptchar=tcfullhar-cf
# Suffixes (combine):
#   -cf      Curtis serve flags: cudagraph FULL_AND_PIECEWISE, capture sizes 1,2,3,4,6,8,12,18,24,30,36,42,48, --async-scheduling,
#            --enable-prefix-caching
#   -sync    with -cf: drop --async-scheduling        -nopfx  with -cf: drop prefix caching      -fdo  keep FULL_DECODE_ONLY
#   -util97  GPU_UTIL=0.97 (Curtis value; CAUTION host driver has no WPR fix -> watch dmesg for Xid 31)
#   -len262k MAX_LEN=262144 (needs the KV pool; check boot.txt / "GPU KV cache size")
#   -k0      DSpark off
set -euo pipefail
ARM=${1:?usage: run-curtis-1017.sh ARM [serve args]}; shift
case "$ARM" in
  base) ARM=tcep ;; coop) ARM=tpcoop-cf ;; cooptc) ARM=tcfull-cf ;; cooptchar) ARM=tcfullhar-cf ;;
esac
IMG=${IMG:-dsv41-1017-exl3:sm80-tcmoe1}
NAME=dsv41-1017
base=${ARM%%-*}
CUSTOM_AR=${CUSTOM_AR:-0}; CAR_MAX=${CAR_MAX:-0}
NCCL_ENV=(-e NCCL_P2P_DISABLE=1)
EP=1; CHUNK=8192; TC_ENV=(); MNT=()
ALIGN=(-e VLLM_EXL3_MOE_TP_ALIGN=128 -e VLLM_EXL3_MOE_TP_ROTATE=1)
TC=(-e VLLM_EXL3_TC_MOE_V2=1 -e VLLM_EXL3_TC_MOE_PREFILL=1 -e VLLM_EXL3_TC_PREFILL_ROWS=2048)
case "$base" in
  tcep)      ;;
  epcoop)    TC_ENV=(-e VLLM_EXL3_COOP=1) ;;
  tptp)      EP=0; TC_ENV=("${ALIGN[@]}") ;;
  tpcoop)    EP=0; TC_ENV=("${ALIGN[@]}" -e VLLM_EXL3_COOP=1) ;;
  tcfull)    EP=0; CHUNK=2048; TC_ENV=("${ALIGN[@]}" -e VLLM_EXL3_COOP=1 "${TC[@]}") ;;
  tcfullhar) EP=0; CHUNK=2048; TC_ENV=("${ALIGN[@]}" -e VLLM_EXL3_COOP=1 "${TC[@]}" -e VLLM_HOST_ALLREDUCE=1)
             MNT=(-v /root/tcmoe/hostar/cuda_communicator.py:/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/cuda_communicator.py:ro
                  -v /root/tcmoe/hostar/dsv41_host_shm_all_reduce.py:/usr/local/lib/python3.12/dist-packages/dsv41_host_shm_all_reduce.py:ro) ;;
  *) echo "unknown arm base '$base'"; exit 2 ;;
esac
SPEC_K=${SPEC_K:-5}; MAX_SEQS=${MAX_SEQS:-8}; CG_SIZES=${CG_SIZES:-1 2 4 6 8 12 16 24 32 48}
REPL=${REPL:-0}; GPU_UTIL=${GPU_UTIL:-0.92}; MAX_LEN=${MAX_LEN:-65536}
CGMODE=FULL_DECODE_ONLY; ASYNC=0; PREFIX=0
case "$ARM" in *-cf*) CGMODE=FULL_AND_PIECEWISE; CG_SIZES="1 2 3 4 6 8 12 18 24 30 36 42 48"; ASYNC=1; PREFIX=1 ;; esac
case "$ARM" in *-sync*)   ASYNC=0 ;; esac
case "$ARM" in *-nopfx*)  PREFIX=0 ;; esac
case "$ARM" in *-fdo*)    CGMODE=FULL_DECODE_ONLY ;; esac
case "$ARM" in *-util97*) GPU_UTIL=0.97 ;; esac
case "$ARM" in *-len262k*) MAX_LEN=262144 ;; esac
case "$ARM" in *-k0*)     SPEC_K=0 ;; esac

docker rm -f $NAME >/dev/null 2>&1 || true
sleep 10   # Oct 4 gotcha: docker rm -f then an immediate docker run races
set -x
docker run -d --init --name $NAME --device nvidia.com/gpu=all --ulimit memlock=-1 --shm-size 16g -p 0.0.0.0:8040:8040 \
  -v /models/dsv41-1017:/models:ro -v /root/dsv41-cache:/root/.cache \
  -v /root/dsv41-tp4/launch-curtis-1017.sh:/launch.sh:ro \
  -v /root/dsv41-tp4/patch/model.py:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/nvidia/model.py:ro \
  -v /root/dsv41-tp4/patch/dspark.py:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/nvidia/dspark.py:ro \
  "${MNT[@]}" \
  -e CUSTOM_AR=$CUSTOM_AR -e VLLM_FORCE_CUSTOM_AR_PCIE=$CUSTOM_AR -e VLLM_CUSTOM_AR_MAX_BYTES=$CAR_MAX \
  -e DSV41_ENGRAM_INT4=${DSV41_ENGRAM_INT4:-1} -e DSV41_ENGRAM_DISK=${DSV41_ENGRAM_DISK:-0} -e DSV41_ENGRAM_SINGLE=${ENGRAM_SINGLE:-0} -e DSV41_REPLICATE_SHARED=$REPL \
  -e EXL3_DELEGATE_ATTRS=${EXL3_DELEGATE_ATTRS:-0} -e VLLM_EXL3_LOG_MOE_ROUTING=${LOG_ROUTING:-1} \
  -e EP=$EP -e CHUNK=$CHUNK -e MAX_LEN=$MAX_LEN -e CGMODE=$CGMODE -e ASYNC=$ASYNC -e PREFIX=$PREFIX "${TC_ENV[@]}" \
  -e SPEC_K=$SPEC_K -e MAX_SEQS=$MAX_SEQS -e "CG_SIZES=$CG_SIZES" -e GPU_UTIL=$GPU_UTIL \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,P2P,SHM \
  "${NCCL_ENV[@]}" \
  --restart no --entrypoint /launch.sh "$IMG" "$@"
