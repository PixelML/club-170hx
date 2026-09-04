# GLM-5.3-Flash on CMP 170HX

GLM-5.3-Flash serves on 4x CMP 170HX using the EXL3 4.05bpw quantization on
exllamav3 1.4.6, behind TabbyAPI's OpenAI-compatible server. This is the
club's first working, benchmarked lane for this model. An earlier
[compatibility review](../../notebooks/2026-08-31-glm-5.3-flash-compatibility-cmp170hx.ipynb)
found every vLLM-served checkpoint blocked on SM80 (no `glm5_next` support
upstream, sparse-MLA attention backends targeting SM90+/SM12x only) and
fell back to a slow llama.cpp GGUF path at 17.73 tok/s. This EXL3 recipe
replaces that fallback: 25.2-44.6 tok/s across the concurrency ladder at
the club's standing 180 W per-card power cap, on the same hardware.

> **Power cap correction (2026-09-03).** The first ladder (26.9-44.8 tok/s)
> ran at the vBIOS default 250 W by accident — no cap had been set that
> session. A re-measure at the verified 180 W cap found no consistent
> throughput difference (within run-to-run noise at every level). **180 W
> is the canonical cap and the numbers in this document are the 180 W
> re-measure**, except where a table explicitly says otherwise. See
> [Power](#power) below for the full comparison.

> **Context-length update (2026-09-03, resolved).** The `cache_mode: Q8`
> config caps a single request's context at about 2,048 tokens, because
> GLM-5.3-Flash's DeepSeek Sparse Attention (DSA) mechanism activates past
> that point and exllamav3's sparse-attention path does not support a
> quantized MLA cache. **This is fixed by switching to `cache_mode:
> FP16`**, which has now been validated up to 262,144 tokens of context
> (250,000 prompt tokens actually tested, no OOM, no crash). `cache_mode:
> FP16` / 262k context is the recommended default for long-context use;
> `cache_mode: Q8` / 32k remains a valid lower-VRAM alternative for
> short-context chat. See
> [Context limit: Q8 cache and DSA](#context-limit-q8-cache-and-dsa) below
> for the root cause and the 262k-context validation data.

## Run on CMP 170HX

| Cards | Format | Runtime | gpu_split | Cache mode | Power cap | Max context (single request) | Measured decode | Status |
|---|---|---|---|---|---|---:|---|---|
| 4 | EXL3, 4.05 bits/weight | exllamav3 1.4.6 + TabbyAPI | `[48, 48, 48, 48]` GB (manual, not tensor_parallel) | **FP16 (recommended default)** | 250 W (accidental default; 180 W re-measure pending) | 262,144 tokens configured; 250,000 prompt tokens validated, no OOM/crash | 25.2-44.6 tok/s ladder not re-run at 262k; short-context ladder below still applies | Measured (context length), throughput ladder not re-run |
| 4 | EXL3, 4.05 bits/weight | exllamav3 1.4.6 + TabbyAPI | `[48, 48, 48, 48]` GB (manual, not tensor_parallel) | Q8 (lower-VRAM, short-context alternative) | **180 W (verified, canonical)** | ~2,048 tokens before DSA/Q8 fails the request | 25.2 tok/s (c=1) to 44.6 tok/s (c=8) | Measured |
| 4 | AWQ W4A16 (compressed-tensors, wtdcode) | vLLM sm80 (`glm53-sm80` branch, TP4 + MTP-3) | `--tensor-parallel-size 4` (TP4) | KV auto (fp8 KV rejected by the Triton MLA backend) | **180 W (canonical)** | 524,288 | **56.4 tok/s median c=1 (peak 56.9)**; c=8 aggregate 37.0 (below EXL3, see caveats) | Measured 2026-09-03 — see [results/2026-09-03-glm-5.3-flash-vllm-sm80-4gpu](../../results/2026-09-03-glm-5.3-flash-vllm-sm80-4gpu/README.md) |

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

**Recommended default (long context, up to 262,144 tokens):**

```yaml
model_dir: <weights>
gpu_split: [48, 48, 48, 48]   # GB per card, manual split — unchanged from the short-context config
max_seq_len: 262144
cache_size: 262144
chunk_size: 4096
cache_mode: FP16                # required for context past ~2,048 tokens; see below
reasoning: true                 # required, see below
```

**Lower-VRAM short-context alternative (<= ~2,048 tokens usable per request):**

```yaml
model_dir: <weights>
gpu_split: [48, 48, 48, 48]   # GB per card, manual split
max_seq_len: 32768
cache_size: 32768
chunk_size: 2048
cache_mode: Q8                 # caps single-request context at ~2,048 tokens; see below
reasoning: true                 # required, see below
```

`tensor_parallel` raises `NotImplementedError` for
`Glm5NextForConditionalGeneration` in exllamav3 1.4.6. `gpu_split` (manual,
per-card GB) is the working topology, not TP — the same split works
unchanged for both configs above.

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

## Context limit: Q8 cache and DSA (resolved 2026-09-03)

GLM-5.3-Flash uses DeepSeek Sparse Attention (DSA), an indexer-based sparse
attention mechanism. Per the model's own `config.json`, `index_topk: 2048`.
In exllamav3 1.4.6 (`exllamav3/modules/mla_attn.py`):

- Sparse attention activates once `max(host_seqlens) + seqlen >
  self.index_topk` (line 765) — intended DSA behavior, not a bug, on a
  model that natively supports up to 1,048,576 tokens
  (`max_position_embeddings` in `config.json`).
- That sparse-attention path carries an explicit assertion that it does
  not support a quantized MLA cache (`_attend_sparse`, ~line 906):

  ```
  assert qc is None, "sparse DSA over a quantized MLA cache is not supported yet; use an fp16 cache"
  ```

- `qc` is only non-`None` when `cache_mode` is `Q8`/`Q6`/`Q4`
  (`CacheLayer_MLA_quant`). With `cache_mode: FP16`
  (`CacheLayer_MLA_fp16`), `qc = None` and the sparse path works fine.

**Conclusion: the cap comes from `cache_mode: Q8` colliding with the DSA
indexer window — not from `max_seq_len` or TabbyAPI's chunk/max-input
settings.** With `cache_mode: Q8` (the VRAM-efficient short-context
config), any single request whose context exceeds about 2,048 tokens fails
that request with a 503; the server process itself stays up and keeps
serving shorter requests normally.

### Fix and 262k-context validation

Switching to `cache_mode: FP16` (with `max_seq_len`/`cache_size: 262144`
and `chunk_size: 4096`; `gpu_split` unchanged) lifts the cap. Boot
succeeded on the first attempt, about 2.5 minutes from local NVMe.
Per-card memory after load: GPU0=46,230, GPU1=45,456, GPU2=45,488,
GPU3=23,638 MiB — 19-42 GiB headroom per card under the 64 GiB cap.

**Prefill ladder** (max_tokens=1, tokenizer-exact prompt lengths, one
continuously-booted server, no restart between steps):

| Prompt tokens | Result | Prefill time | Peak mem (GPU0/1/2/3, MiB) |
|---:|---|---:|---|
| 2,941 | OK | 14.27 s | 48,308 / 47,518 / 47,550 / 25,662 |
| 16,000 | OK | 20.79 s | 49,076 / 48,322 / 48,354 / 26,496 |
| 32,000 | OK | 26.92 s | 49,076 / 48,356 / 48,356 / 26,496 |
| 65,000 | OK | 50.80 s | 49,142 / 48,388 / 48,420 / 26,624 |
| 131,000 | OK | 102.04 s | 49,144 / 48,390 / 48,422 / 26,626 |
| 200,000 | OK | 48.21 s | 49,176 / 48,390 / 48,422 / 26,626 |
| 250,000 | OK | 35.35 s | 49,176 / 48,390 / 48,422 / 26,626 |

**Largest verified prompt: 250,000 tokens.** No OOM, no crash at any
tested length. Prefill time drops noticeably past 131k tokens, because
DSA's indexer caps attention cost at a fixed top-k=2048 token selection
regardless of total context length — the 131k point looks like a
transitional/less-optimized case rather than a regression, but this is
flagged as an open question, not a settled explanation. Memory stayed
essentially flat from 16k tokens onward: going from 16k to 250k tokens
cost only about 2 GiB total across all four cards, because the MLA/DSA
cache is cheap per token and this model's hybrid linear-attention layers
(KDA / gated-delta-net) carry O(1) state that does not grow with context.

**Needle-in-haystack retrieval:**

| Context length | Result |
|---|---|
| 32k tokens | PASS — correctly retrieved a planted unique fact |
| 250k tokens | PASS — correctly retrieved a planted unique fact |

**Health:** no Xid or ECC (including double-bit) events at any point in
the ladder (`nvidia-smi -q` and `dmesg`); peak temperature 51 degrees C;
server process stayed up throughout, no driver reload needed.

**Scope note:** only prefill/context-length behavior was re-tested. The
full C1/C2/C4/C8 throughput ladder was **not** re-run at 262k context —
the 25.2-44.6 tok/s figures below (180 W, canonical) remain the
short-context (`Q8` cache) measurement, and no throughput regression was
observed at short context during this update, but that claim does not
extend to a re-run 262k ladder.

`cache_mode: FP16` / 262,144-token context is now the recommended default
for this recipe whenever long context matters. Keep `cache_mode: Q8` /
32,768-token context only when the lower VRAM footprint matters more than
long context.

## Recommended settings

- **Sampling for throughput benches:** greedy decoding, exactly 400
  completion tokens, tokens counted from the final usage object.
- **Speculative decoding:** n-gram drafting was tested and gave no
  benefit — 26.9 tok/s (draft off) vs. 25.0 tok/s (draft n-gram), a 7.2%
  regression within noise. The shipped config has drafting disabled.
- **KV cache:** `FP16` is the recommended default (removes the DSA/Q8 cap,
  validated to 262,144-token context, costs only about 2 GiB of extra VRAM
  across all four cards versus Q8 at short context). `Q8` remains a valid
  lower-VRAM alternative for short-context chat, but caps single-request
  context at about 2,048 tokens because of the DSA assertion above.
- **Context:** with `cache_mode: FP16`, `max_seq_len`/`cache_size: 262144`
  is validated up to 250,000 prompt tokens. With `cache_mode: Q8`,
  `max_seq_len`/`cache_size` are configured at 32,768, but the effective
  per-request ceiling is about 2,048 tokens once DSA activates.
- **Concurrency:** measured clean through c=8 with no OOM.
- **Reasoning:** `reasoning: true` is required; see above.

## Benchmarks

Measured 2026-09-02. Full receipts:
[results/2026-09-03-glm-5.3-flash-exl3-4gpu-tabbyapi/](../../results/2026-09-03-glm-5.3-flash-exl3-4gpu-tabbyapi/README.md).
Executed notebook:
[notebooks/2026-09-03-glm-5.3-flash-exl3-4gpu-tabbyapi.ipynb](../../notebooks/2026-09-03-glm-5.3-flash-exl3-4gpu-tabbyapi.ipynb).

### Concurrency ladder (greedy, exactly 400 completion tokens, 1 warmup + 3 measured reps)

**180 W (2026-09-03, verified cap — canonical):**

| Concurrency | Aggregate tok/s (mean of 3 reps) | Mean per-request tok/s |
|---|---:|---:|
| C1 | 25.2 | 25.2 |
| C2 | 35.3 | 18.0 |
| C4 | 43.2 | 11.0 |
| C8 | 44.6 | 8.2 |

**250 W (2026-09-02, accidental default — retained for comparison):**

| Concurrency | Aggregate tok/s (mean of 3 reps) | Mean per-request tok/s |
|---|---:|---:|
| C1 | 26.9 | 26.9 |
| C2 | 31.1 | 15.6 |
| C4 | 41.7 | 10.5 |
| C8 | 44.8 | 8.3 |

**Delta (180 W vs 250 W):** C1 -6.3%, C2 +13.5%, C4 +3.6%, C8 -0.4%. Each
cap has one run of three reps, not five; the C2 gap is best read as
run-to-run noise, not a real power effect. No level shows a directional,
monotonic speed penalty from the lower cap. 180 W is kept as the standing
default; this data does not support raising it for a throughput gain.

No OOM through C8 at either cap. Memory and temperature watched
continuously through both ladders: no growth trend, peak 51 °C, no
Xid/ECC events.

### Prefill / TTFT (2,941-token prompt, exact token count verified against the tokenizer)

Measured with `cache_mode` temporarily set to FP16 to avoid the Q8/DSA
assertion above; reverted to Q8 for standing service afterward (the
2026-09-03 180 W re-measure ran with the standing config already on
FP16 — see the power-cap-correction note above).

**180 W (2026-09-03, canonical), prompt re-tokenized to 2,954 tokens post
chat-template:**

| Rep | Prompt time (s) | Prefill tok/s |
|---|---:|---:|
| 0 (cold) | 0.44 | 313.6 |
| 1 (warm) | 0.39 | 353.9 |
| 2 (warm) | 0.38 | 363.2 |

Warm mean (reps 1-2): 358.5 tok/s. TTFT (same prompt, streaming, wall
time to first content-bearing chunk) ranged 0.73-1.78 s across three reps;
treat as a range, not a point estimate — no other load ran against the
server during the measurement window.

**250 W (2026-09-02):**

| Item | Value |
|---|---:|
| Cold (first request post-boot) | 5.57 s prompt time |
| Warm (reps 2-3, usage-reported) | mean prompt_time 0.39 s -> ~354 tok/s prefill throughput |

Prefill throughput is not power-cap sensitive at this prompt length: the
180 W warm mean (358.5 tok/s) matches the 250 W warm figure (~354 tok/s)
within noise.

### Power

**180 W (2026-09-03, canonical), 1 Hz samples across the full ladder +
prefill/TTFT window (~10.5 min):**

| GPU | Mean W | Peak W | Peak temp (C) |
|---|---:|---:|---:|
| 0 | 56.7 | 139.5 | 51 |
| 1 | 58.8 | 172.6 | 49 |
| 2 | 56.4 | 168.4 | 49 |
| 3 | 49.8 | 102.5 | 45 |
| **Total** | **221.6** | **352.4** | — |

Peak per-card power (172.6 W) stayed under the 180 W cap at every sample.
The total peak (352.4 W) is a coincident-peak artifact of summing each
card's own peak moment, not a real 4-card simultaneous draw — no single
1 s sample summed above 302 W across all four cards.

**250 W (2026-09-02, accidental default, measured during a C4 load run,
1s samples over 60s):**

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

## vLLM sm80 lane (`glm53-sm80` branch) — 2026-09-04

The compatibility-review blocker ("every vLLM-served checkpoint blocked
on SM80") is resolved: the `glm53-sm80` branch on
[PixelML/sm80vllm](https://github.com/PixelML/sm80vllm) (wtdcode
GLM enablement vendored, provenance in its `docs/SM80.md`) serves
GLM-5.3-Flash on all four cards with MTP speculative decoding.

**Headline: 56.4 tok/s median c=1 (peak 56.9), 2.1x the EXL3 lane,
at 524,288-token context** (EXL3: 25.2 tok/s c=1, ~2k-262k context).

- **Recipe:** TP4, compressed-tensors AWQ W4A16 checkpoint
  ([wtdcode/GLM-5.3-Flash-AWQ-W4A16](https://huggingface.co/wtdcode/GLM-5.3-Flash-AWQ-W4A16)),
  `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`,
  Triton sparse-MLA + Triton fp8 MQA-logits fallbacks, Marlin WNA16 MoE.
  Full launch command in the
  [receipts](../../results/2026-09-03-glm-5.3-flash-vllm-sm80-4gpu/README.md).
- **MTP depth sweep (c=1, 5 reps):** k=2 → 51.1, **k=3 → 56.4**,
  k=5 → 47.1 tok/s median. k=3 is optimal, matching the DSpark
  acceptance-cliff pattern in `docs/LESSONS.md` §d.
- **Caveat — aggregate:** c=8 aggregate is 37.0 tok/s, below the EXL3
  lane's 44.8. TP4 all-reduce over PCIe Gen1 is the bottleneck
  (`docs/LESSONS.md` §c). The aggregate fix (PP4 + MTP) needs the
  vLLM MTP-under-PP patch set (upstream PR #46994) ported; not yet done.
- **Do not combine** `--no-enable-flashinfer-autotune` with MTP: it
  crashes at engine startup (`cudaErrorLaunchFailure`, reproduced on a
  clean boot) and the crash wedges the GPU at the PCIe level (VM reboot
  required). Autotune at its default is part of the measured recipe.

## Artifacts

- **Evidence repository (full attempt history, including failed boot
  attempts):** [PixelML/GLM-5.3-Flash-CMP-170HX](https://github.com/PixelML/GLM-5.3-Flash-CMP-170HX).
- **Checkpoint:** [turboderp/GLM-5.3-Flash-exl3](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3), branch `4.05bpw`, revision `2a30229e67012798ba9f0cd832bb78abf4c363d5`.
- **Runtime:** [turboderp/exllamav3](https://github.com/turboderp/exllamav3) 1.4.6+cu128.torch2.10.0, served via [theroyallab/tabbyAPI](https://github.com/theroyallab/tabbyAPI).
- **Superseded fallback:** unsloth/GLM-5.3-Flash-GGUF UD-IQ4_XS on the unslothai llama.cpp sm_80 fork (17.73 tok/s c=1) — see [docs/BENCHMARKS.md](../BENCHMARKS.md) and the [compatibility notebook](../../notebooks/2026-08-31-glm-5.3-flash-compatibility-cmp170hx.ipynb).

## Changelog

- **2026-09-04** — vLLM sm80 lane measured: `glm53-sm80` branch serves GLM-5.3-Flash with MTP-3 at 56.4 tok/s median c=1 (peak 56.9), 2.1x EXL3, at 524,288-token context. c=8 aggregate (37.0) remains below EXL3 pending a PP4+MTP patch port. Receipts: [results/2026-09-03-glm-5.3-flash-vllm-sm80-4gpu](../../results/2026-09-03-glm-5.3-flash-vllm-sm80-4gpu/README.md).
- **2026-09-03 (follow-up)** — Resolved the ~2,048-token context cap: root-caused to `cache_mode: Q8` colliding with GLM-5.3-Flash's DSA sparse-attention indexer (not `max_seq_len` or TabbyAPI settings). Validated `cache_mode: FP16` up to 262,144-token context (250,000 prompt tokens tested, no OOM/crash, needle-in-haystack PASS at 32k and 250k tokens). `cache_mode: FP16` / 262k is now the recommended default for long-context use; `cache_mode: Q8` / 32k remains documented as the lower-VRAM short-context alternative. The C1/C2/C4/C8 throughput ladder was not re-run at 262k context.
- **2026-09-03** — Power cap correction: the 2026-09-02 ladder ran at the vBIOS default 250 W by accident. Re-measured the identical protocol at the verified 180 W club-standard cap: 25.2-44.6 tok/s, no consistent throughput difference from the 250 W run outside noise. 180 W values are now canonical; 250 W values are retained and labeled for comparison. See [Concurrency ladder](#concurrency-ladder-greedy-exactly-400-completion-tokens-1-warmup--3-measured-reps) and [Power](#power) above.
- **2026-09-02** — EXL3 4.05bpw on exllamav3 1.4.6 + TabbyAPI measured working across 4 cards: 26.9-44.8 tok/s ladder (later found to have run at the vBIOS default 250 W — see 2026-09-03 entry), 20/20 golden corpus, reasoning-parsing and Q8/DSA context-limit findings documented. This recipe replaces the GGUF fallback as the recommended lane.
- **2026-08-31** — Compatibility review: every vLLM-served checkpoint blocked on SM80; GGUF UD-IQ4_XS on llama.cpp sm_80 fork measured as the only working fallback (17.73 tok/s).
