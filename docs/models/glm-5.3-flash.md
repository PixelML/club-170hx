# GLM-5.3-Flash on CMP 170HX

GLM-5.3-Flash serves on 4x CMP 170HX using the EXL3 4.05bpw quantization on
exllamav3 1.4.6, behind TabbyAPI's OpenAI-compatible server. This is the
club's first working, benchmarked lane for this model. An earlier
[compatibility review](../../notebooks/2026-08-31-glm-5.3-flash-compatibility-cmp170hx.ipynb)
found every vLLM-served checkpoint blocked on SM80 (no `glm5_next` support
upstream, sparse-MLA attention backends targeting SM90+/SM12x only) and
fell back to a slow llama.cpp GGUF path at 17.73 tok/s. This EXL3 recipe
replaces that fallback: 26.9-44.8 tok/s across the concurrency ladder, on
the same hardware.

> **Context-length warning.** With the standing `cache_mode: Q8` config,
> any single request whose context exceeds about **2,048 tokens fails with
> a 503**. GLM-5.3-Flash's DeepSeek Sparse Attention (DSA) mechanism
> activates past that point, and exllamav3's sparse-attention path does not
> yet support a quantized MLA cache. The server process itself stays up —
> only the oversized request fails. See
> [Context limit: Q8 cache and DSA](#context-limit-q8-cache-and-dsa) below
> before relying on this recipe for anything beyond short-context chat.

## Run on CMP 170HX

| Cards | Format | Runtime | gpu_split | Cache mode | Max context (single request) | Measured decode | Status |
|---|---|---|---|---|---:|---|---|
| 4 | EXL3, 4.05 bits/weight | exllamav3 1.4.6 + TabbyAPI | `[48, 48, 48, 48]` GB (manual, not tensor_parallel) | Q8 | ~2,048 tokens before DSA/Q8 fails the request | 26.9 tok/s (c=1) to 44.8 tok/s (c=8) | Measured |
| 4 | EXL3, 4.05 bits/weight | exllamav3 1.4.6 + TabbyAPI | `[48, 48, 48, 48]` GB (manual, not tensor_parallel) | FP16 | untested at concurrency; confirmed to fit VRAM | not load-tested | Untested (future work) |
| 4 | UD-IQ4_XS GGUF | llama.cpp (unslothai sm_80 fork) | layer split, `1,1,1,1` | F16 (server default) | 16,384 | 17.73 tok/s (c=1) | Measured, superseded by the row above (see [docs/MODEL-STATUS.md](../MODEL-STATUS.md)) |

## Quick start

### 1. Download the weights

```bash
pip install -U huggingface_hub
hf download turboderp/GLM-5.3-Flash-exl3 \
  --revision 2a30229e67012798ba9f0cd832bb78abf4c363d5 \
  --local-dir <weights>
```

The checkpoint is `turboderp/GLM-5.3-Flash-exl3`, branch `4.05bpw`,
revision `2a30229e67012798ba9f0cd832bb78abf4c363d5` (EXL3 quantization at
4.05 bits per weight).

### 2. Install the runtime

exllamav3 1.4.6+cu128.torch2.10.0, served through
[TabbyAPI](https://github.com/theroyallab/tabbyAPI) (an OpenAI-compatible
server). Install both per their upstream project instructions.

### 3. Configure and launch

```yaml
model_dir: <weights>
gpu_split: [48, 48, 48, 48]   # GB per card, manual split
max_seq_len: 32768
cache_size: 32768
cache_mode: Q8                 # switch to FP16 for single requests over ~2048 tokens
reasoning: true                 # required, see below
```

`tensor_parallel` raises `NotImplementedError` for
`Glm5NextForConditionalGeneration` in exllamav3 1.4.6. `gpu_split` (manual,
per-card GB) is the working topology, not TP.

### 4. First request

```bash
curl http://localhost:5000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "glm-5.3-flash",
    "messages": [{"role": "user", "content": "What is the capital of France? Answer with just the city name."}],
    "temperature": 0,
    "max_tokens": 128
  }'
```

Use the standard TabbyAPI OpenAI-compatible port; there is nothing model
family specific about the endpoint itself.

## Reasoning parsing (required)

GLM-5.3-Flash defaults to `reasoning_effort: max` with `<think>...</think>`
tags baked into its chat template. If TabbyAPI's `reasoning` config is left
`false`, chain-of-thought leaks unparsed into the visible `content` field
and burns the whole completion budget with no final answer ever produced.
Setting `reasoning: true` routes the chain-of-thought into a separate
`reasoning_content` field instead, leaving `content` for the final answer.

`max_tokens=32` (a common quick smoke-test budget) is reproducibly **not
enough**, even with reasoning correctly parsed — the model spends the whole
budget thinking and returns `content: null`. `max_tokens=128` on the same
prompt cleanly returns `content: "Paris"`. **Recommendation: use
`max_tokens >= 128` for short factual answers with reasoning enabled.**

## Context limit: Q8 cache and DSA

GLM-5.3-Flash uses DeepSeek Sparse Attention (DSA), an indexer-based sparse
attention mechanism. Per the model's own `config.json`, `index_topk: 2048`.
exllamav3 activates the sparse-attention code path once context exceeds
about 2,048 tokens, and that code path carries an explicit assertion that
it does not support a quantized (Q8) MLA cache:

```
assert qc is None, "sparse DSA over a quantized MLA cache is not supported yet; use an fp16 cache"
```

**Practical consequence:** with `cache_mode: Q8` (the VRAM-efficient
standing config used for this recipe), any single request whose context
exceeds about 2,048 tokens fails that request with a 503. The server
process itself stays up and keeps serving shorter requests normally.

Long-context use requires `cache_mode: FP16` instead. FP16 was confirmed to
still fit the same `gpu_split [48, 48, 48, 48]` with headroom (GPU0=45270,
GPU1=48078, GPU2=44526, GPU3=19316 MiB when tested), but it was **not
load-tested at concurrency**. Treat FP16 as unvalidated future work, not a
standing recommendation, until a concurrency ladder exists for it.

## Recommended settings

- **Sampling for throughput benches:** greedy decoding, exactly 400
  completion tokens, tokens counted from the final usage object.
- **Speculative decoding:** n-gram drafting was tested and gave no
  benefit — 26.9 tok/s (draft off) vs. 25.0 tok/s (draft n-gram), a 7.2%
  regression within noise. The shipped config has drafting disabled.
- **KV cache:** Q8 for the standing service (VRAM-efficient, but caps
  single-request context at about 2,048 tokens because of the DSA
  assertion above). FP16 removes the cap but is unvalidated at
  concurrency.
- **Context:** `max_seq_len` and `cache_size` are both configured at
  32,768, but the effective per-request ceiling under Q8 is about 2,048
  tokens once DSA activates.
- **Concurrency:** measured clean through c=8 with no OOM.
- **Reasoning:** `reasoning: true` is required; see above.

## Benchmarks

Measured 2026-09-02. Full receipts:
[results/2026-09-03-glm-5.3-flash-exl3-4gpu-tabbyapi/](../../results/2026-09-03-glm-5.3-flash-exl3-4gpu-tabbyapi/README.md).
Executed notebook:
[notebooks/2026-09-03-glm-5.3-flash-exl3-4gpu-tabbyapi.ipynb](../../notebooks/2026-09-03-glm-5.3-flash-exl3-4gpu-tabbyapi.ipynb).

### Concurrency ladder (greedy, exactly 400 completion tokens, 1 warmup + 3 measured reps)

| Concurrency | Aggregate tok/s (mean of 3 reps) | Mean per-request tok/s |
|---|---:|---:|
| C1 | 26.9 | 26.9 |
| C2 | 31.1 | 15.6 |
| C4 | 41.7 | 10.5 |
| C8 | 44.8 | 8.3 |

No OOM through C8. Memory and temperature watched continuously (2s
samples) through the whole ladder: no growth trend, peak 49 °C, no
Xid/ECC events.

### Prefill / TTFT (2,941-token prompt, exact token count verified against the tokenizer)

Measured with `cache_mode` temporarily set to FP16 to avoid the Q8/DSA
assertion above; reverted to Q8 for standing service afterward.

| Item | Value |
|---|---:|
| Cold (first request post-boot) | 5.57 s prompt time |
| Warm (reps 2-3, usage-reported) | mean prompt_time 0.39 s -> ~354 tok/s prefill throughput |

### Power (measured during a C4 load run, 1s samples over 60s)

| GPU | Mean W | Peak W |
|---|---:|---:|
| 0 | 65.4 | 120.0 |
| 1 | 63.0 | 101.4 |
| 2 | 56.2 | 97.4 |
| 3 | 50.1 | 78.9 |
| **Total** | **234.7** | **302.3** |

### Golden corpus quality gate

20 prompts spanning `short_factual`, `reasoning`, `code`, `json`, and
`multilingual` categories, keyword-match scoring, `max_tokens=512`. **All
20 passed.**

## Per-card memory footprint

| Card | VRAM used |
|---|---:|
| GPU0 | 48,468 MiB |
| GPU1 | 47,982 MiB |
| GPU2 | 47,982 MiB |
| GPU3 | 12,084-12,116 MiB |

Total: ~153 GiB used of a 256 GiB pool (4 x 64 GiB). GPU3 stays
under-loaded compared to GPU0-2 even at the working split — a known
characteristic of exllamav3's naive sequential-fill autosplit, not a
functional problem.

## Boot topology notes

Getting to `gpu_split [48, 48, 48, 48]` took iteration:

- `[64, 64, 64, 64]` OOM'd at first inference — an uneven fill left no
  KV-cache headroom on GPU0/1 while GPU3 stayed empty.
- `[40, 40, 40, 40]` was too tight — insufficient VRAM for weights plus
  cache.
- `[48, 48, 48, 48]` is the value that worked.

The full attempt-by-attempt failure ladder, with exact commands and error
text, lives in the sibling evidence repository, not here — see below.

## Artifacts

- **Evidence repository (full attempt history, including failed boot
  attempts):** [PixelML/GLM-5.3-Flash-CMP-170HX](https://github.com/PixelML/GLM-5.3-Flash-CMP-170HX).
- **Checkpoint:** [turboderp/GLM-5.3-Flash-exl3](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3), branch `4.05bpw`, revision `2a30229e67012798ba9f0cd832bb78abf4c363d5`.
- **Runtime:** [turboderp/exllamav3](https://github.com/turboderp/exllamav3) 1.4.6+cu128.torch2.10.0, served via [theroyallab/tabbyAPI](https://github.com/theroyallab/tabbyAPI).
- **Superseded fallback:** unsloth/GLM-5.3-Flash-GGUF UD-IQ4_XS on the unslothai llama.cpp sm_80 fork (17.73 tok/s c=1) — see [docs/BENCHMARKS.md](../BENCHMARKS.md) and the [compatibility notebook](../../notebooks/2026-08-31-glm-5.3-flash-compatibility-cmp170hx.ipynb).

## Changelog

- **2026-09-02** — EXL3 4.05bpw on exllamav3 1.4.6 + TabbyAPI measured working across 4 cards: 26.9-44.8 tok/s ladder, 20/20 golden corpus, reasoning-parsing and Q8/DSA context-limit findings documented. This recipe replaces the GGUF fallback as the recommended lane.
- **2026-08-31** — Compatibility review: every vLLM-served checkpoint blocked on SM80; GGUF UD-IQ4_XS on llama.cpp sm_80 fork measured as the only working fallback (17.73 tok/s).
