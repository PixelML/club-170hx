# DeepSeek-V4-Flash-Vision-Exp on 4x CMP 170HX (sm_80), TP4. Fixes found by the 2026-10-04 autoresearch loop.
# Base: the 2026-09-02 vision image (allover326 SM80 fork f8ea5bb163c161ef38b401d055cc5fd4a934091a + 8-file
# SM80/DSpark patch set + vision port), editable vLLM at /vllm.
ARG BASE=ghcr.io/pixelml/club-170hx@sha256:b26232f8f041c988d3285e2278c9f5001cc49f96131bb0b22a9d38b5e5e061cd
FROM ${BASE}
# 1. Root-cause fix: @eager_break_during_capture decides at call time. At import time the breakable-cudagraph switch
#    was still off, so DSV4 attention was captured into piecewise graphs and replayed stale KV for 16/32-token prefills.
COPY breakable_cudagraph.py /vllm/vllm/compilation/breakable_cudagraph.py
# 2. Split-K sparse-MLA decode kernels also on CUDA sm8x (attention decode kernel 56 -> 39.5 us).
COPY rocm_aiter_mla_sparse.py /vllm/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py
# 3. DSpark: skip the MTP n_predict divisibility check (allows k=5/7 to boot; k=6 is served).
COPY speculative.py /vllm/vllm/config/speculative.py
# 4. Custom all-reduce on PCIe with verified P2P (VLLM_FORCE_CUSTOM_AR_PCIE=1). Served config disables it: NCCL is faster here.
COPY custom_all_reduce.py /vllm/vllm/distributed/device_communicators/custom_all_reduce.py
