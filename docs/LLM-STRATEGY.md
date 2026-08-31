# LLM inference strategy on CMP 170HX

Consolidated decision guide from measured results across the 170HX fleet — what to reach for first, and the traps that waste card-hours. Every claim is labeled **measured** (we ran it), **community-reported** (published, reproducible recipe exists), or **inferred** (reasoned, unverified). Last consolidated 2026-08-31.

## Three facts that drive every decision

1. **Pipeline parallel, never tensor parallel.** [community-reported, 6.6x prefill margin] On PCIe Gen2 links without P2P, TP performs 86 all-reduces per forward pass and sits flat at ~800 tok/s prefill; PP moves the payload only 3 times and reaches ~5,300 tok/s prefill. Exception: a model that fits on one card (e.g. Qwen3.8-27B at ~58 GiB) should be replicated — one serving process per card, no interconnect dependency at all. Rule: fits-one-card -> replicas; otherwise PP.
2. **Speculative decoding is the single biggest decode lever.** [measured 2.30x on our Qwen recipe; community-reported 1.93x under PP4 with wins to 64 concurrent] There is no competitive decode speed without it. Requirements: a vLLM-class runtime (NOT llama.cpp/GGUF), a symmetric quant with the MTP head unquantized, memory utilization ~0.85 with a drafter resident, and CUDA graphs ON. Never `--enforce-eager` with a drafter — it collapses to 8-10 tok/s [community-reported].
3. **Power limit sets the ceiling.** [measured, Qwen single-card gate] 125 W -> ~85-105 tok/s; 180 W -> ~115-136; 255 W -> ~148. Scaling is near-linear with achievable memory bandwidth. 180 W is a common standing cap for heat/noise reasons; know which curve your PASS threshold belongs to before judging a result.

## Best-known setting by workload

| Workload | Best-known setting | Result | Basis |
|---|---|---|---|
| Qwen3.8-27B class, one card | vLLM + DFlash2-style speculative (k=7), W4/W8 quant, 255 W | 147.7 tok/s, TTFT 76 ms | measured (single-card hosted reference) |
| Same, N cards | Replicas: one serve process per card + round-robin | ~148 tok/s/user floor at N users | measured scaling, inferred aggregate |
| GLM-5.3-Flash GGUF | llama.cpp SM80 fork | 17.73 tok/s | measured — compute-bound kernel ceiling; lane exhausted |
| GLM DSA family, 8 cards | PP8, block-size 64, V2 runner, MTP | 28.47 tok/s (GLM-5.2-744B class) | community-reported |
| DeepSeek-V4-Flash, 4 cards | PP4 + DSpark-style speculation | 98 tok/s single / 712 tok/s @ 64 concurrent | community-reported |
| DeepSeek-V4-Flash, 3 cards | PP3, k=5, FP8 KV, 180 W | 83.3 tok/s aggregate | measured (this fleet) |

## Decision order before any new LLM attempt

1. Import-probe the runtime first — no GPU, no weights: does the image register the model architecture? 15 minutes kills bad candidates for free.
2. Does the model fit one card? Replicate; skip interconnect entirely.
3. Multi-card required? PP, never TP on these links.
4. Speculative decoding possible? Always on, with the settings above. Acceptance below ~0.5 = re-tune or disable.
5. Runtime must be vLLM-class for the levers above to apply. GGUF/llama.cpp hits a compute-bound kernel ceiling on Flash-class models (~17.7 tok/s measured); do not spend more card-hours there.
6. Power: pick the curve that matches your cap and judge PASS thresholds against it.

## Hard rules (each one caused a real loss somewhere)

- Send SIGINT to the serving PID; never tree-kill a multi-GPU vLLM run — stranded CUDA contexts require a host reboot [community-reported].
- When a rank dies, chase the Xid in dmesg — the surviving ranks' gloo "connection closed" errors are noise [community-reported].
- One workload per exclusive window; no GPU co-tenancy while measuring.
- Symmetric quants only on SM80 DSA-family models; asymmetric AWQ fails MoE assertions. Keep the MTP head unquantized [community-reported].
- `--block-size 64` is required on the sparse-MLA compose stack [community-reported].
- Check the hardware power brake on any new board or slot before benchmarking: `nvidia-smi -q | grep -A1 "HW Power Brake Slowdown"`. A braked card looks exactly like "these mining cards are just slow" at ~4x loss [community-reported].

## Settled non-levers — do not re-litigate

- Tensor parallel on Gen2-no-P2P links: 6.6x worse than PP.
- FP8 KV cache: requires SM89+. Dead on GA100.
- NVLink: not supported on 170HX.
- Further llama.cpp/GGUF tuning for Flash-class models: kernel ceiling proven.

## Evidence index

- [Benchmarks](BENCHMARKS.md) — measured numbers on this fleet.
- [PixelML/Qwen3.8-27B-CMP-170HX](https://github.com/PixelML/Qwen3.8-27B-CMP-170HX) — single-card recipe and raw outputs.
- [PixelML/DeepSeek-V4-Flash-0731-CMP-170HX](https://github.com/PixelML/DeepSeek-V4-Flash-0731-CMP-170HX) — three-card PP3 + speculation recipe.
- [allover326/deepseek-v4-cmp170hx](https://github.com/allover326/deepseek-v4-cmp170hx) — four-card PP4 reference, operational lessons.
- [allover326/vllm-dsa-mtp-sm80](https://github.com/allover326/vllm-dsa-mtp-sm80) — GLM-5.2-class DSA + MTP under PP on SM80. Note: registers only the GLM-5.2 architecture class; GLM-5.3-Flash (glm5_next) needs a different runtime base.
- [PixelML/GLM-5.3-Flash-CMP-170HX](https://github.com/PixelML/GLM-5.3-Flash-CMP-170HX) — measured GGUF baseline and lane verdict.
