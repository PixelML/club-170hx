# v7 series (W8A16 experiments, 2026-08-28 12:00-12:15 UTC)

Goal: apply the official vLLM recipe (Qwen/Qwen3.8-27B + DFlash2, 7 draft
tokens) and the LocalMaxxing config (lued W8A16 MTP checkpoint) to the
CMP 170HX, fixing the v6 findings.

## Fixes attempted

1. W8A16 checkpoint served directly (no W4A16 requant) - matches reference
2. KV_MEM passed as GPU_UTIL=0.90 only - but the syv launcher's CTX=fast
   default still pins kv_cache_memory_bytes=5583457484 (5.2 GiB)
3. nvcc symlinked from pip nvidia-cuda-nvcc into /usr/local/cuda/bin/ -
   flashinfer JIT ran but still logged warnings; topk compiled or fell back
   (log preserved)

## v7e results (instance <instance-11>)

| Metric | v7e (W8A16) | v6 (W4A16) |
|---|---|---|
| Output tok/s | 31.6 | 47.5 |
| TTFT | 608-621 ms | 517-521 ms |
| Acceptance length | 2.9 | 3.38 |

**Conclusion:** the CMP 170HX is memory-bandwidth-bound (~1.5 TB/s effective).
Doubling weight density (W4->W8) halves effective throughput: 47.5 -> 31.6
tok/s is a ~1.5x drop, consistent with bandwidth saturation. The reference
212 tok/s likely depends on factors not reproduced here (see RESULTS.md).

## Failed sub-attempts

- v7 (<instance-7>): onstart-cmd passed a host path; container had no such file.
  Destroyed, recreated.
- v7b (<instance-8>): DRAFT passed as lued/Qwen3.8-27B-DFlash2-W4A16 (nonexistent
  repo). Destroyed, recreated.
- v7c (<instance-9>): main model download used huggingface-cli (removed in
  hub 1.27); empty dir crashed vLLM. Fixed with 'hf download'.
- v7d (<instance-10>): same DRAFT bug as v7b. Fixed by leaving DRAFT unset
  (launcher auto-discovers local drafter).
- v7e (<instance-11>): healthy, benchmarked, captured, destroyed.
