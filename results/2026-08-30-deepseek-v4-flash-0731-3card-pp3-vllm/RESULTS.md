# DeepSeek-V4-Flash-0731 on 3x CMP 170HX - bench results (2026-08-30)

All runs: 180 W caps on all 3 cards (verified), greedy, single stream,
official deepseek-ai/DeepSeek-V4-Flash-0731, vLLM PP=3
(15,15,13), DSpark k=5, kv fp8, maxlen 16384, util 0.95.

## Decode (bench_decode3.py, 400 tok per prompt)

| prompt | tokens | elapsed s | tok/s |
|---|---|---|---|
| technical | 400 | 5.45 | 73.4 |
| open-prose | 400 | 5.52 | 72.4 |
| code | 400 | 3.43 | 116.6 |
| **AGGREGATE** | 1200 | 14.41 | **83.3** |

Reference: allover326 4-card config ~78.8 tok/s (higher total power).

## Prefill (bench_prefill.py, best of 2)

| real tokens | prefill tok/s |
|---|---|
| 784 | 1667.3 |
| 1553 | 2081.6 |
| 3091 | 2626.9 |
| 5399 | **2965.0** |

Short-prompt end-to-end decode (prefill + 400 tok): 129.2 tok/s.

## Speculative decoding (engine metrics)

- Mean acceptance length: 5.07-5.32 (k=5)
- Draft acceptance rate: 81.3-86.4%
- Per-position: 1.000 / 0.900 / 0.767 / 0.767 / 0.633 (typical)

## Server identity

- Resolved: DeepseekV4ForCausalLM + DeepSeekV4MTPModel, model_type
  deepseek_v4, 43 layers + 3 hash layers, 256 routed experts,
  expert_dtype fp4, served name dsv4s.
- Boot: 22 min weight load (NFS) + ~7 min CUDA graph capture.
