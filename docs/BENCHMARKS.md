# Benchmarks

Results here are a publication-status registry for the tested CMP 170HX setup. Entries are measured results, withheld results pending publication-safe evidence, or compatibility findings. They are not vendor specifications or theoretical estimates.

## Withheld pending publication-safe evidence (no club-published claims)

**Qwen3.8-27B W4A16 + DFlash2, three single-card runs:** a three-card result exists, but its current evidence revision contains prohibited infrastructure identifiers and requires an owner-approved history repair before a sanitized re-pin. Numbers and evidence links return here only after that repair.

**DeepSeek-V4-Flash-0731, three-card pipeline:** a three-card result exists, but the current evidence pin has no redacted raw JSONL receipts, run manifest, or complete environment metadata. Numbers return here only after sanitized receipts land at an exact revision.

## Negative results matter

### GLM-5.3-Flash (all quantizations)

**Compatibility result, not a performance result.** As of the stable summary at [GLM-5.3-Flash-CMP-170HX @ `0eab34e`](https://github.com/PixelML/GLM-5.3-Flash-CMP-170HX/blob/0eab34e173bee43d9cf8a48d546db609c8f469d3/README.md), no completed serving run has been published:

- NVFP4 targets an SM121 runtime path — incompatible with SM80 (**measured** registry check). Upstream vLLM support ([PR 53906](https://github.com/vllm-project/vllm/pull/53906)) is open and SM90+.
- AWQ INT4 is 198.1 GiB — over the 192 GiB three-card total before KV (**measured** blob sizes).
- EXL3/TR3 4 bpw is 175,642,157,752 bytes = 163.58 GiB, or 54.53 GiB/card at TP=3 — static weights fit the 64 GiB/card budget with about 9.47 GiB/card of headroom before overhead (**measured** bytes; fit **inferred**). Runtime and overhead feasibility are **untested**, and its kernels ship SM121-only.

Do not advertise a throughput number until a compatible model format, runtime path, and memory plan are demonstrated.

## Measurement rules

Every submitted result must include:

- exact model repository + revision and quantization;
- runtime image/commit, CUDA, driver, kernel, and launch command;
- card count, parallelism, PCIe topology, power limit, peak draw, and temperatures;
- input/output tokens, concurrency, batch size, context length, warmup, and sample count;
- raw redacted output and an explanation of the metric calculation.

For server-sent-event APIs, count generated tokens from the final `usage.completion_tokens`. Counting stream events produced incorrect results in an earlier harness because events are not tokens.

Use the template in [results/README.md](../results/README.md).
