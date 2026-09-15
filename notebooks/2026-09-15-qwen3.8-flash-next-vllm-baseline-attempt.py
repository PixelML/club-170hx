# %% [markdown] cell:0
# Qwen3.8-Flash-Next on 2x CMP 170HX (SM80) — vLLM serving attempt: precise hardware blocker (negative result)

| Item | Value |
|---|---|
| Attempt | vLLM baseline serve of Qwen3.8-Flash-Next AWQ-W4A16, TP2 on 2x CMP 170HX 64GB |
| Outcome | **Fails at load: PLE offload worker requests 102.4 GB pinned host RAM; VM has 31 GB** |
| Verdict | Full-fidelity vLLM serving of Qwen3.8-Flash-Next is **arithmetically impossible on this VM** (any PLE placement: VRAM or RAM) |
| What would work | (a) host with >=128 GB RAM + 2x 64GB-class GPUs (W4A16 + PLE offload), or (b) 96-128GB Blackwell SM100+ (NVFP4 checkpoint, per the Wombat/Pennyroyal recipes) |
| Interim evidence | eager-HF engine (same box): 2.19-2.27x CED prefill speedup, trained projector suffix-1 greedy match 75.9% — different stack, labeled as such |

Companion notebook: `2026-09-15-qwen3.8-flash-next-llkvapprox-2card` (eager-engine oracle + trained-student receipts).
# %% [markdown] cell:1
# Why vLLM (and what was attempted)

vLLM provides the production kernel path this model wants: fused int4 grouped-GEMM MoE
(Marlin-class, for the 512-expert top-10 layers), flash-class attention with the lightning-indexer
sparse masks, and paged KV. The LLKVApprox claim (skip layers 24..47 during prefill via a trained
projector, keep decode exact) is a *relative* speedup over whatever kernel stack serves the model —
so the intended experiment was: (1) vLLM baseline prefill TTFT/tok-s, (2) CED variant with our
trained projector injected, same stack, matched comparison.

Serving recipe attempted (prior art: `serve-qwen38-flash.sh` on this host; image
`lazymio/vllm-backport:v0.9.0-sm80` — a v0.9.0-line backport carrying the `qwen4_exp` model
implementation and the `VLLM_PLE_CPU_OFFLOAD` shim): TP2 over GPU 0,1, expert parallel,
`--language-model-only`, max-model-len 8192, MTP off, reference conv fallback.

# %% [code] cell:2
# --- The failure, verbatim ----------------------------------------------
# From receipts/vllm-gate1-serve-failure.log (full log committed alongside):
tail_line = open("../receipts/2026-09-15-qwen3.8-flash-next-vllm-baseline-attempt/vllm-gate1-serve-failure.log").read()
key = "DefaultCPUAllocator: can't allocate memory: you tried to allocate 102400491520 bytes"
print("failure present in log:", key in tail_line)
print("requested:", 102400491520 / 2**30, "GiB  (the BF16 PLE n-gram table, exactly)")
print("host RAM installed: 31 GB")
# %% [markdown] cell:3
# ### The arithmetic (all checkpoint variants vs this VM)
#
| PLE placement | requirement | this VM | verdict |
|---|---|---|---|
| Offload to host RAM (BF16 table, as shipped in AWQ-W4A16) | 102.4 GB pinned | 31 GB total / ~22 avail | **fails (measured)** |
| Offload to host RAM (FP8 table, as in NVFP4 exports) | 47.7 GB pinned | 31 GB | **fails (arithmetic)** |
| GPU-resident, TP2 (BF16) | ~51 GB/card + experts ~30 GB/card + KV | 64 GB/card | **fails (arithmetic)** |
| GPU-resident (FP8, TP2) | ~24 GB/card + NVFP4 experts | NVFP4 kernels need SM100+; cards are SM80 | **fails (arch)** |
#
# The PLE table is architecturally mandatory: it is consumed at layer 1 of the text stack for
# every token (prefill and decode). Dropping it changes model outputs materially and would make
# any "baseline" a measurement of a different model.
# %% [markdown] cell:4
# ### What would unblock vLLM serving
#
| Host | Why it works |
|---|---|
| x86 + >=128 GB RAM + 2x 64GB-class GPUs (any arch supporting W4A16 marlin) | BF16 PLE offload fits pinned RAM |
| 1x 96GB Blackwell (SM120) + >=64GB RAM | NVFP4 checkpoint per the WombatSoftware/Pennyroyal recipes (PLE FP8 offload 47.7 GB or NVMe mode) |
| DGX Spark-class (128 GB unified, Blackwell) | NVFP4 + PLE offload; needs arm64 image rebuild |
#
# Engine-side prerequisites regardless of host: vLLM >= 0.29.0 nightly with PR #54371 (PLE
# offload), plus the `_is_deepseek_v4_eagle()` qwen4_exp eagle-gate overlay (else prefix caching
# silently disables). Both documented in the WombatSoftware recipe this attempt started from.
# %% [markdown] cell:5
# ### Interim evidence from the eager engine (different stack — labeled as such)
#
# The eager HF engine (packed int4 experts, mmap PLE) cannot serve at production speed, but it
# can compute the model exactly, and it already carries the trained projector:
#
# - CED vs full prefill, T=2048: 45.1 s vs 98.8 s (**2.19x**); T=6603: 55.4 s vs 123.6 s (**2.23x**)
# - Oracle gate: FA prior fills bit-exact (0.0); GDN grid-boundary replay exact by construction
# - Trained student, suffix-1 greedy match vs baseline: 75.9% pooled (see companion notebook)
#
# These are relative claims on one stack. The matched-stack vLLM comparison (baseline vs CED
# overlay inside vLLM) is the follow-up experiment on a host that can hold the checkpoint.
