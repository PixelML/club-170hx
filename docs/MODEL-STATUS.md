# Model status on CMP 170HX

Every model PixelML has tried on CMP 170HX, in one table. Status definitions:

- **Measured** — a run completed and produced numbers, with receipts in the owning repository.
- **Negative** — the attempt is complete and the answer is "does not fit" or "does not run," with evidence.
- **Planned** — preparation only; no GPU time spent yet.

Do not compare headline numbers across rows. Quantization, runtime, card count,
power policy, and context length all differ. Each row links to its own
evidence; read the number next to its source.

| Model | Quant / format | Runtime | Cards | Status | Headline number | Evidence |
|---|---|---|---|---|---|---|
| Qwen3.8-27B | W4A16-AutoRound ("fast variant") | vLLM 0.27.1, DFlash2 k=7 | 1 (255 W, rented) | Measured | 147.7 tok/s decode, 2,156 tok/s prefill, 76 ms TTFT | [PixelML/Qwen3.8-27B-CMP-170HX](https://github.com/PixelML/Qwen3.8-27B-CMP-170HX) |
| Qwen3.8-27B | W4A16-AutoRound | vLLM 0.27.1, DFlash2 k=7 | 3 (180 W, local) | Measured | 133.6-140.3 tok/s decode per card, 1,926-1,957 tok/s prefill | [PixelML/Qwen3.8-27B-CMP-170HX RESULTS.md](https://github.com/PixelML/Qwen3.8-27B-CMP-170HX) |
| Qwen3.8-27B | W8A16 (official checkpoint) | vLLM 0.27.1, DFlash2 k=7 | 1 (255 W) | Measured (negative vs W4A16) | 31.6 tok/s decode, acceptance 2.9 (slower than W4A16 because the card is bandwidth-bound) | [PixelML/Qwen3.8-27B-CMP-170HX RESULTS.md](https://github.com/PixelML/Qwen3.8-27B-CMP-170HX) |
| Qwen3.8-27B | W4A16 AutoRound (dbirks) + DFlash2 k=7 | vLLM | 1 (3 cards tested, 180 W) | Measured 2026-08-30 | 136.38 tok/s mean at 256 tokens; TTFT 190.8 ms; prefill 1,946 tok/s | [Repo](https://github.com/PixelML/Qwen3.8-27B-CMP-170HX) |
| Qwen3.8-27B | Ninfer sm_80 fork (groupwise-int + w8, MTP draft-tokens 3, lm-head-draft) | Ithrial/ninfer-cmp170hx (sm_80 port) | 1 | Measured (negative on decode, positive on prefill); superseded on decode by the controlled A/B below | 39.6-41.2 tok/s decode (3.5x behind the vLLM DFlash2 stack); prefill 419,177 tok/s and 1.8 ms TTFT (chat-template prompt, different route) | [Ithrial/ninfer-cmp170hx](https://github.com/Ithrial/ninfer-cmp170hx); comparison in [PixelML/Qwen3.8-27B-CMP-170HX RESULTS.md](https://github.com/PixelML/Qwen3.8-27B-CMP-170HX) |
| Qwen3.8-27B | Ninfer sm_80 fork, controlled A/B vs. the vLLM DFlash2 recipe, int8 KV | Ithrial/ninfer-cmp170hx (sm_80 port), MTP draft-tokens 3, lm-head-draft | 1 (180 W) | Measured 2026-09-02, negative | Spec-on 38.16 tok/s decode256 (3.6x slower than the 138.6 tok/s vLLM control); spec-off 29.95 tok/s (4.6x slower); MTP gives a real ~1.28x uplift over spec-off; verdict: stay on vLLM | [results/2026-09-02-qwen3.8-27b-ninfer-ab](../results/2026-09-02-qwen3.8-27b-ninfer-ab/README.md) |
| Qwen3.6-35B-A3B | (NInfer field checkpoint) | NInfer / llama-swap | 1 | Measured (field observation, not a controlled benchmark) | 93-126 generated tok/s, 1,580-2,347 prompt tok/s | [Ithrial/ninfer-cmp170hx](https://github.com/Ithrial/ninfer-cmp170hx) |
| Qwen3.8-Flash-Next | AWQ-INT4 | vLLM (pending upstream support) | 4 (planned) | Planned; checkpoint verified, not yet run | 5 prior attempts on 6x40GB rented cards all failed before serving (vocab/TP mismatch, PLE-offload/PP conflict, Marlin sharding, MoE group-scale divisibility, container seccomp blocking `pidfd_getfd` in the CPU-offload worker); checkpoint since verified byte-for-byte on local storage (50/50 files, 38/38 shard hashes match) | [PixelML/Qwen3.8-Flash-Next-CMP-170HX](https://github.com/PixelML/Qwen3.8-Flash-Next-CMP-170HX) |
| DeepSeek-V4-Flash-0731 | Native MXFP4 experts + FP8 e4m3 attention | vLLM (SM8x fork, full source build), PP3, DSpark k=5 | 3 (180 W, local) | Measured | 83.3 tok/s aggregate decode (technical 73.4 / prose 72.4 / code 116.6), 2,965 tok/s prefill at 5,399 tokens, acceptance 5.07-5.32 | [PixelML/DeepSeek-V4-Flash-0731-CMP-170HX](https://github.com/PixelML/DeepSeek-V4-Flash-0731-CMP-170HX) |
| DeepSeek-V4-Flash-0731 | Same, community 4-card reference config | vLLM (SM8x fork), PP4, DSpark k=5 | 4 | Measured (upstream reference, not this club's own run) | 98.1 tok/s aggregate decode, 5,321 tok/s prefill (77k context), full 1,047,736-token context verified | [allover326/deepseek-v4-cmp170hx](https://github.com/allover326/deepseek-v4-cmp170hx) |
| DeepSeek-V4-Flash-Vision-Exp | FP8 e4m3, 48 shards | SM80 vLLM fork, PP4, layer partition 11,11,11,10, DSpark k=6 (text path) | 4 | Measured 2026-09-02 | Decode c=1 97.4 tok/s (median of 3; 57.6–123.5); aggregate c=4 165.5 tok/s (median of 3; 140.3–203.2); aggregate c=16 failed (device-side assert, reproduced twice); uncached prefill (2,941 tokens) 2,352 tok/s warm (362 tok/s first cold prefill); warm TTFT 0.394 s. Earlier superseded ladder: 101.21 tok/s @ c=1, 169.65 tok/s @ c=4, 325.5 tok/s prefill | [PixelML/DeepSeek-V4-Flash-Vision-Exp-CMP-170HX](https://github.com/PixelML/DeepSeek-V4-Flash-Vision-Exp-CMP-170HX) |
| DeepSeek-V4-Flash-Vision-Exp | FP8 e4m3, 48 shards | Same SM80 vLLM fork, PP4, DSpark k=6, vision path (Path 3, 5 boot fixes) | 4 | Measured 2026-09-02, partial | Vision gates PASS (10/10 image keyword match); text golden corpus 15/20 keyword, 10/20 exact-match (known limitation). Text-only decode 119 tok/s median of 5 reps (peak 162) @ c=1 — DSpark acceptance variance (0.20-0.83) drives the spread — 116.6 tok/s @ c=2, server crashed at c=4 (EngineCore died, not restarted). Text+image decode 45.3 tok/s @ c=1, 78.2 tok/s @ c=2 (aggregate, measured after the server came back up); c=4 and above not attempted. Uncached prefill 2,352 tok/s warm; warm TTFT 0.386 s | [notebooks/2026-09-02-deepseek-v4-flash-vision-exp-4card-vision-pp4-vllm.ipynb](../notebooks/2026-09-02-deepseek-v4-flash-vision-exp-4card-vision-pp4-vllm.ipynb), [docs/BENCHMARKS.md](BENCHMARKS.md#vision-on-vllm-sm80-path-3-measured) |
| DeepSeek-V4-Flash-Vision-Exp | FP8 -> BF16 fallback (reference runtime + 4 SM80 patches) | Reference TP4 runtime, batch 1 (history) | 4 | Measured (correctness, not performance) | **PASS**: first real-image completion of this checkpoint on Ampere hardware, 0.88-0.93 tok/s decode (this is a correctness result; do not read it as a performance number). Superseded as the vision-correctness milestone by the vLLM Path 3 row above, kept here as history | [PixelML/DeepSeek-V4-Flash-Vision-Exp-CMP-170HX](https://github.com/PixelML/DeepSeek-V4-Flash-Vision-Exp-CMP-170HX) |
| GLM-5.3-Flash | EXL3, 4.05 bits/weight | exllamav3 1.4.6 + TabbyAPI, manual `gpu_split` | 4 | Measured 2026-09-02 at 250 W (accidental default); re-measured 2026-09-03 at the verified 180 W club-standard cap (canonical); context-length follow-up measured 2026-09-03 | **180 W (canonical):** 25.2-44.6 tok/s aggregate decode (c=1-c=8), 20/20 golden corpus, at `cache_mode: Q8` (context capped ~2,048 tokens there). 250 W figures (26.9-44.8 tok/s) retained for comparison — no consistent throughput difference outside noise. **Resolved:** `cache_mode: FP16` validated to 262,144-token context (250,000 prompt tokens tested, no OOM/crash, needle-in-haystack PASS at 32k/250k); now the recommended default for long context; throughput ladder not re-run at 262k | [docs/models/glm-5.3-flash.md](models/glm-5.3-flash.md), [results/2026-09-03-glm-5.3-flash-exl3-4gpu-tabbyapi](../results/2026-09-03-glm-5.3-flash-exl3-4gpu-tabbyapi/README.md), full attempt history in [PixelML/GLM-5.3-Flash-CMP-170HX](https://github.com/PixelML/GLM-5.3-Flash-CMP-170HX) |
| GLM-5.3-Flash | UD-IQ4_XS GGUF | llama.cpp (unslothai DSA fork, sm_80) | 4 | Measured; superseded by the EXL3 row above | 17.73 tok/s median decode (c=1), ~17.5-17.7 tok/s at c=2/4, 21/26 evaluation tasks, 41/41 soak reps clean | [PixelML/GLM-5.3-Flash-CMP-170HX](https://github.com/PixelML/GLM-5.3-Flash-CMP-170HX) |
| GLM-5.3-Flash | NVFP4 (LibertAIDAI) | vLLM fork (SM121 image only) | — | Negative | Does not run on SM80: sparse-MLA path targets SM12x FlashInfer backends only | [PixelML/GLM-5.3-Flash-CMP-170HX](https://github.com/PixelML/GLM-5.3-Flash-CMP-170HX) |
| GLM-5.3-Flash | AWQ W4A16 (compressed-tensors, wtdcode) | vLLM sm80, PP4 partition 14,12,12,7 + native MTP k=3 | 4 (180 W) | **Measured 2026-09-05 — recipe of record** | **87.6 tok/s c=1 decode** (temp 0, median of 3, clean workload) and **67.9 tok/s** (temp 0.7 + `ignore_eos`, median of 5); decode flat at 75-79 tok/s from 2,024 to 131,042 prompt tokens; best aggregate 78.4 tok/s at c=16, prefill 1,752 tok/s at 16k, warm TTFT 9.44 s at 16k — **aggregate, prefill and TTFT measured on a degraded PCIe link (one card at Gen1 x1) and are lower bounds**; draft-depth sweep k=2/3/5/7 shows acceptance collapsing with depth (74.0% to 37.3%), k=3 optimal; boots 5/5, 995-1,263 s | [docs/models/glm-5.3-flash.md](models/glm-5.3-flash.md), [results/2026-09-05-glm-5.3-flash-4card-pp4-vllm](../results/2026-09-05-glm-5.3-flash-4card-pp4-vllm/README.md), [notebook](../notebooks/2026-09-05-glm-5.3-flash-4card-pp4-vllm.ipynb) |
| GLM-5.3-Flash | AWQ W4A16 (compressed-tensors, wtdcode) | vLLM sm80 fork (`glm53-sm80` branch, TP4 + native MTP k=3) | 4 (180 W) | Measured 2026-09-03; **superseded** by the PP4 row above | **56.4 tok/s median c=1 decode (peak 56.9)** — 2.1x the EXL3 lane — at 524,288-token context; c=8 aggregate 37.0 tok/s (below EXL3's 44.8; TP4 all-reduce over PCIe Gen1, PP4+MTP port pending) | [results/2026-09-03-glm-5.3-flash-vllm-sm80-4gpu](../results/2026-09-03-glm-5.3-flash-vllm-sm80-4gpu/README.md) |
| GLM-5.3-Flash | EXL3/TR3 4bpw | ExLlamaV3 (SM121 fork image) | — | Negative | Fits VRAM on paper (~40.9 GiB/card at TP=4) but ships SM121-only kernel binaries; no SM80 build exists | [PixelML/GLM-5.3-Flash-CMP-170HX](https://github.com/PixelML/GLM-5.3-Flash-CMP-170HX) |
| GLM-5.3-Flash | AWQ-INT4 (cyankiwi) | Upstream vLLM (if `glm5_next` were supported) | — | Negative | 198.1 GiB measured; fits on paper at TP=4 (~49.5 GiB/card) but `glm5_next` is absent from the upstream vLLM model registry; no runtime path exists yet | [PixelML/GLM-5.3-Flash-CMP-170HX](https://github.com/PixelML/GLM-5.3-Flash-CMP-170HX) |
| GLM-5.3-Flash | Official FP8 / BF16 | Any | — | Negative | Does not fit: ~76+ GiB/card (FP8) at TP=4, BF16 larger | [PixelML/GLM-5.3-Flash-CMP-170HX](https://github.com/PixelML/GLM-5.3-Flash-CMP-170HX) |
| FastH3 | BF16 baseline (Lane A) / Kijai INT8 "convrot" (Lane B) | FastVideo / ComfyUI (non-Blackwell VSA-Triton path) | 4 (planned) | Planned; blocked on license gate | Static fit only: ~34.4 GiB/card (Lane A) or ~24.28 GiB total (Lane B) vs 64 GiB/card; nothing downloaded, nothing run | [PixelML/FastH3-CMP-170HX](https://github.com/PixelML/FastH3-CMP-170HX) |

## Notes on specific rows

**Qwen3.8-Flash-Next.** Five attempts on a rented 6x40GB node all failed
before serving, each root-caused from logs: TP6 has a vocabulary size not
divisible by 6; TP2xPP3 conflicts with the required CPU-offload mode; TP4
Marlin does not support the layer sharding; TP4 Triton has MoE group scales
that fail a divisibility check; TP4 Triton plus expert parallelism reached a
healthy engine and then the offload worker died on a container seccomp
policy blocking `pidfd_getfd`. The checkpoint's large n-gram lookup table
needs either CPU offload (blocked by seccomp in containers, expected to work
bare metal) or enough VRAM to hold it directly; the second path is why this
model is a natural first test once four 64 GiB cards are locally available.

**DeepSeek-V4-Flash-Vision-Exp text path.** The row above is filled by a
normalized benchmark run using the same protocol as the DeepSeek-V4-Flash-0731
measurement: greedy 400-token completions, three repetitions, tokens counted
from `usage.completion_tokens`. Read the earlier ladder (101.21-169.65 tok/s)
as historical, not current.

**DeepSeek-V4-Flash-Vision-Exp vision path.** Vision on vLLM SM80 is now
measured, not in progress: functional gates and image correctness pass, the
text-only decode ladder ran through c=2 before a c=4 crash, and the
text+image ladder ran through c=2 once the server came back up. c=8/c=16
text-only and c=4-and-up text+image are honestly reported as not measured.
The reference TP4 runtime row above stays in the table as the historical
first-PASS milestone; it no longer needs to carry the "only vision evidence"
weight now that a vLLM-served measurement exists.

**GLM-5.3-Flash.** No quantization/runtime pairing that both fits CMP 170HX's
VRAM and has a working SM80 kernel path exists for any vLLM-served format as
of this writing; every vLLM-path row above is a documented negative result
kept for the record, not a gap in testing. Two non-vLLM lanes now work: EXL3
4.05bpw on exllamav3 1.4.6 + TabbyAPI (2026-09-02, 26.9-44.8 tok/s, the
recommended recipe) and UD-IQ4_XS GGUF on the sm_80 llama.cpp fork
(2026-08-30, 17.73 tok/s, now superseded). The EXL3 recipe's Q8 KV cache
caps single-request context at about 2,048 tokens — GLM-5.3-Flash's
DeepSeek Sparse Attention path does not support a quantized MLA cache in
this exllamav3 build, and a request over that limit fails with a 503 while
the server stays up. **Resolved 2026-09-03:** switching to `cache_mode:
FP16` lifts the cap, validated up to 262,144-token context (250,000
prompt tokens tested, no OOM/crash). FP16/262k is now the recommended
default for long context; Q8/32k remains a lower-VRAM short-context
alternative. See [docs/models/glm-5.3-flash.md](models/glm-5.3-flash.md)
for the full recipe and the fix.

**Ninfer sm_80 fork.** A genuine, buildable sm_80 port of a Blackwell-targeted
engine. A controlled A/B against the vLLM DFlash2 recipe on 2026-09-02
confirmed it is not competitive on single-stream decode: 3.6x slower with
its own MTP speculation on, 4.6x slower with speculation off. The prior
prefill/TTFT numbers used a different request route (chat-template prompt,
not the streamed decode benchmark) and were not re-measured on the same
protocol in this A/B, so they stay flagged as a separate, unreconciled
observation rather than a competing verdict. See
[results/2026-09-02-qwen3.8-27b-ninfer-ab](../results/2026-09-02-qwen3.8-27b-ninfer-ab/README.md).

## See also

- [LESSONS.md](LESSONS.md) — the kernel, runtime, topology, and failure-mode lessons behind these numbers.
- [BENCHMARKS.md](BENCHMARKS.md) — the normalized, sanitized measurement ledger this club publishes directly.
