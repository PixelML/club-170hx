# Attempt — AWQ INT4 on upstream vLLM

Status: static-fit-only, failed
Date: 2026-08-30

## Checkpoint

- [cyankiwi/GLM-5.3-Flash-AWQ-INT4](https://huggingface.co/cyankiwi/GLM-5.3-Flash-AWQ-INT4) @ `3999f9bf2c3e3064790af5a2d12d19090fc97f4d` (last modified 2026-08-29)
- **212,721,952,636 bytes = 198.1 GiB = 212.7 GB** across 43 safetensors shards
  (measured: HF API `?blobs=true` response, summed per-file)
- AWQ INT4, pack-quantized, group size 32, asymmetric, visual tower exclusions
- License: base-model terms; checkpoint license file present in repo

## Runtime

- Upstream vLLM would be the natural engine for an AWQ checkpoint
- SM80 support: **no** — moot here, because `glm5_next` is not in upstream's
  registry at all (measured 2026-08-30). Recorded anyway because the checkpoint
  is the only SM80-*format* candidate and the fit math is what rules it out
  even on hardware where a runtime exists.

## Static fit calculation

Arithmetic, from the measured byte total:

- Three-card era, TP=3: 212,721,952,636 / 3 = 70,907,317,545 B = **66.04 GiB per
  card** — 2.04 GiB over budget; this is what failed the attempt on 2026-08-30.
- Current four-card node, TP=4: 212,721,952,636 / 4 = 53,180,488,159 B = **49.53
  GiB per card** — fits the 64 GiB budget with ~14.5 GiB/card headroom before
  context/KV. The size blocker is cleared by the topology change; the runtime
  blocker (no `glm5_next` in upstream vLLM) is unchanged.
- Per-card budget: 64 GiB = 68,719,476,736 B.
- **Three-card margin: -2.04 GiB before CUDA context (~0.5-1 GiB),
  activations, or any KV cache.** The checkpoint could not serve even a single
  token on the 3 x 64 GiB node. On the current 4 x 64 GiB node the margin is
  +14.5 GiB/card, but the runtime blocker below still applies.
- Pipeline parallelism does not rescue this: PP splits are layer-wise, not
  byte-proportional, and the dense first stage (embedding + visual tower +
  early blocks) is over budget on its own.
- CPU offload or disk-mapped weights would change the memory story but not the
  Gen2-x4 interconnect story; not attempted, out of scope for this benchmark
  family.

## Execution status and outcome

Not executed. Killed by arithmetic before download on the three-card node: the
checkpoint was 6.1 GiB too large in total for 192 GiB of aggregate VRAM, before
any overhead. The current four-card node (256 GiB aggregate) has room for it;
the attempt remains blocked on runtime support, not size.

## Blocker

Static fit on the three-card node (historical). On the four-card node the
binding blocker is the missing `glm5_next` runtime support, not size.

## Evidence

- Byte total: HF API response, 2026-08-30 (43 shards; exact per-file sizes in
  the API, sum stated above)
- Registry check: [vllm registry.py on main](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py), no `glm5` match on 2026-08-30

## Re-run instructions

Blocked until a checkpoint meeting the budget exists. When one does, use this
methodology (usage-token-counted, per the club's evidence rules):

1. Environment: four-card CMP node, 180 W per-card policy, forced airflow,
   driver/kernel pins from the club baseline, weights staged in shared model
   storage only.
2. Serve with TP=4, conservative context first (e.g. 8192), then raise.
3. Bench harness: streaming with `include_usage: true`, derive decode tokens
   from the final usage object's `completion_tokens` — never SSE event counts.
   Fixed-output 256- and 900-token single-stream runs; prefill at ~1K/4K/16K
   unique prefixes; concurrency sweep 1/2/4 while stable.
4. Record cold load time, TTFT, peak VRAM/power/temps per card; abort at 80 C
   core / 85 C memory / any Xid / any GPU disappearance.
5. Publish per [results template](../../templates/ATTEMPT-TEMPLATE.md) with raw
   redacted JSONL evidence.
