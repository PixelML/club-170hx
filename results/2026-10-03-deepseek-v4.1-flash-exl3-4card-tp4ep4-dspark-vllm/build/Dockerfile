# DeepSeek-V4.1-Flash EXL3 3.0 bpw on 4x CMP 170HX (sm_80): TP4 + EP4, all routed experts in VRAM.
# v2 (2026-10-04): + vLLM custom all-reduce on PCIe P2P for messages below 128 KiB.
# v3 (served 2026-10-04): + optional 4-bit Engram table pinned in host RAM (DSV41_ENGRAM_DISK=0 DSV41_ENGRAM_INT4=1).
# Base: 0xSero/dsv41-flash-offload @ 2e9da5ecdab37c232c2da887e25c33a99da738d3 built with
#   docker build --build-arg TORCH_CUDA_ARCH_LIST=8.0 -f docker/Dockerfile .
ARG BASE=ghcr.io/pixelml/club-170hx@sha256:75683c4813efaabcc1a96ee610a96a74336677c4a8776b79b01abd83ff3c0c7b
FROM ${BASE}
# 1. vllm-exl3 (AGPL-3.0): ReplicatedLinear is unsharded; MoE split read from the MoE parallel
#    config (EP keeps whole experts); vocab-parallel lm_head loaded by vLLM's 128-aligned shard range.
COPY exl3.py /usr/local/lib/python3.12/dist-packages/vllm_exl3/exl3.py
# 2. DeepSeek-V4 MLP (vLLM backport, Apache-2.0): DSV41_REPLICATE_SHARED=1 replicates the shared
#    expert (2304/4 = 576 is not 128-aligned for EXL3) and scales its output by 1/tp before the MoE all-reduce.
COPY v4model.py /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/model.py
# 3. DeepSeek-V4.1 model: EXL3 lm_head built with vocab padding 128 * tp.
COPY v41model.py /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/nvidia/model.py
# 4. Engram disk tier (MIT): TP > 1; optional single-rank lookup (DSV41_ENGRAM_SINGLE=1, off: measured no gain).
COPY engram_disk.py /opt/dsv41/dsv41/engram_disk.py
# 5. vLLM custom all-reduce (Apache-2.0): VLLM_FORCE_CUSTOM_AR_PCIE=1 treats P2P-capable PCIe GPUs as fully
#    connected (vLLM's P2P self-test still runs); VLLM_CUSTOM_AR_MAX_BYTES caps it, NCCL above the cap.
COPY custom_all_reduce.py /usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/custom_all_reduce.py
# 6. DeepSeek-V4.1 Engram (vLLM backport, Apache-2.0): DSV41_ENGRAM_INT4=1 reads int4 rows (offset-binary,
#    fp16 scale per 32 values, 144 B/row) from the pinned host table with a Triton lookup kernel.
COPY engram.py /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/engram.py
# 7. Converter: fp8 Engram shards -> int4 shards with the same tensor names (run once, CPU only).
COPY engram_int4.py /opt/dsv41/engram_int4.py
# Launcher: TP4 + EP4, no host experts; CUSTOM_AR=1 drops --disable-custom-all-reduce.
COPY empty-plan.json /plan/empty-plan.json
COPY launch.sh /launch.sh
ENV CUSTOM_AR=1 VLLM_FORCE_CUSTOM_AR_PCIE=1 VLLM_CUSTOM_AR_MAX_BYTES=131072 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False DSV41_ENGRAM_SINGLE=0 DSV41_ENGRAM_INT4=0
ENTRYPOINT ["/launch.sh"]
