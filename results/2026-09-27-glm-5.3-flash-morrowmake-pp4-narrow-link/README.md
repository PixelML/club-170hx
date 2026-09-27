# GLM-5.3-Flash — Morrowmake PP4 recipe on a narrow-link rig

Status: measured
Date: 2026-09-27

Replication of [Morrowmake/glm53-flash-cmp170hx-recipe](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe) release [v1.4.1](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe/tree/v1.4.1) (MIT), `LAYOUT=pp4`, all recipe defaults. Their published PP4 figures were measured on four x16 links and they list narrow-link PP4 as unmeasured. This rig has three cards at Gen2 x16 and one at Gen2 **x1**, which puts the x1 card on one pipeline hop.

**Verdict (measured):** single-user decode lands 7–12% under their x16 figures, 8-user aggregate decode matches or beats them (−4% to +10%), cold prefill is 41% lower, and the KV pool is identical. Decode holds ~95 tok/s from 1k to 156k prompt tokens.

## Hardware

- Cards: 4 × CMP 170HX, 65,536 MiB each, labels gpu0–gpu3 in pipeline-stage order
- Topology / PCIe links: gpu0–gpu2 Gen2 x16, gpu3 Gen2 x1 (riser); no P2P (recipe default)
- Power limit: 180 W per card (the recipe's measurement cap). Peak `power.draw` sampled at 1 s during run 2: 271.4 W (reported by `nvidia-smi` above the 180 W limit; recorded as measured)
- Cooling: forced air. Peak core 66 °C, peak memory 75 °C during run 2 (`receipts/run2/telemetry.csv`)

## Software

- OS / kernel: Ubuntu 22.04.5, 6.8.0-138-generic (test VM, GPUs passed through; memory unlock is host-side)
- NVIDIA driver / CUDA: 610.43.03 / 13.3
- Recipe: `Morrowmake/glm53-flash-cmp170hx-recipe` @ `1a516cc71643ff708761c299d73454698d15ea3c` (tag v1.4.1)
- Engine image: `ghcr.io/morrowmake/vllm-cmp170hx@sha256:14d7b380cc623eb9145db06307c0e432024f1060de1460bf14f893abd9792a97` (vLLM fork `Morrowmake/vllm-cmp170hx` @ `378c37b0098a41a5cd25b3bf8b56d158e33a6cbf`)
- Target: [`canada-quant/GLM-5.3-Flash-W4A16-MTP`](https://huggingface.co/canada-quant/GLM-5.3-Flash-W4A16-MTP) @ `5723f4d02af36366c23ace8668866ca7775855c1`, W4A16
- Drafter: [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) @ `bf582e4eacc1810f76656d1811693ff6c6737d2a`, k=3 (CC BY-NC-ND 4.0: research and evaluation only)

## Command

```bash
git clone https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe && cd glm53-flash-cmp170hx-recipe
git checkout v1.4.1
printf 'LAYOUT=pp4\nMODELS_DIR=/path/to/models\n' > .env
./install.sh && ./download.sh
./start.sh && ./start.sh smoke
```

Boot: 1,225 s cold (first start, empty kernel caches), 645 s with warm caches. The x1 stage loads its 44 GiB about 100 s slower than the x16 stages and finishes CUDA-graph capture about 4.5 min after them.

## Method

- **Decode** (`run_decode.sh`): MiaAI-Lab's [`tests/bench_decode.py`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/943912cdcda25f4b7e02f4626656e873c6f14847/tests/bench_decode.py) @ `943912cd` (AGPL-3.0, not vendored here; only `BASE` and `MODEL` changed), the prompt source Morrowmake credits. Structured (count 1→200), code (`clamp_range`), prose (hash map). Temperature 0, thinking off, 400 max tokens, one 32-token warmup, median of 5. Stream tok/s = `(completion_tokens − 1) / (end − first token)`, tokens from the final `usage`, reasoning deltas counted. 8-user aggregate = `sum(completion_tokens) / wall`.
  - Difference from Morrowmake's protocol: this script sends no per-request nonce, so repeated prompts can hit the prefix cache. That affects TTFT, not the decode rate.
- **Cold prefill** (`bench_long.py prefill`): unique nonce first, real text (Python stdlib source + markdown docs), `max_tokens=1`, `prompt_tokens / TTFT`, 2 reps at each of 3 sizes. The first request (750 tok/s) was JIT warmup and is excluded. Prompts landed at 19.2k–28.7k tokens, shorter than Morrowmake's 23.9k–37.9k.
- **Long-context decode** (`bench_long.py decode`): one cold request per size, 400 tokens, same stream metric.
- Runs: run 1 (all three benches) and run 2 (decode matrix again, with 1 s `nvidia-smi` telemetry), on separate server boots.

## Results

| PP4 | Run 1 | Run 2 | Morrowmake v1.4.1 (4 × x16) | Δ vs Morrowmake (run 2) |
|---|---:|---:|---:|---:|
| Decode, 1 user, structured / code / prose (tok/s) | 130.7 / 121.4 / 88.7 | 131.3 / 121.7 / 89.0 | 141.7 / 138.4 / 100.3 | −7% / −12% / −11% |
| Decode, 8 users aggregate (tok/s) | 589.7 / 490.2 / 400.9 | 595.2 / 489.0 / 395.2 | 542.3 / 509.8 / 383.6 | +10% / −4% / +3% |
| Cold prefill (tok/s) | 3,710 (median of 5, 19–29k prompts) | — | 6,254 (24–38k prompts) | −41% (run 1) |
| KV pool @ 262,144 ctx (tokens) | 2,320,328 | 2,320,328 | 2,320,328 | 0 |

DFlash2 accepted tokens per step at 1 user (run 1): 2.94 / 2.75 / 1.67 of 3.

Long-context decode, run 1 (prompt tokens → tok/s): 864 → 94.4 · 6,117 → 83.7 · 25,359 → 85.8 · 52,446 → 102.7 · 106,353 → 95.6 · 155,934 → 94.8. The prompt is code and docs with a "explain the above" task, so acceptance (and speed) differs from the three fixed prompts.

**Inferred:** the prefill gap comes from the x1 hop, since every prefill chunk's activations cross it into stage 3, while decode moves one small activation per step and loses little. Not isolated here: a layer re-partition that shrinks the x1 stage, or the same run on four x16 cards, would test it.

## Correctness and failures

- Smoke (`./start.sh smoke`): chat reply correct, tool call parsed (`get_weather({"city": "Reykjavik", "unit": "celsius"})`), KV line matches.
- No NaN outputs in any decode run; server healthy after each.
- Xid scan (`journalctl -k`): 0 across both boots and all runs.
- Caveat for serving (measured): with `enable_thinking: false`, GLM-5.3-Flash still reasons, inline in `content`, and the reasoning parser cannot split it out. Serve with thinking on (the recipe default) for agent harnesses.

## Evidence

- `receipts/run1/`: decode matrix JSON + stdout, `prefill-cold.json`, `longctx-decode.json`
- `receipts/run2/`: decode matrix JSON + stdout, `telemetry.csv` (1 s), `xid.txt`
- `run_decode.sh`, `bench_long.py`: the exact harness used
- Notebook: [`notebooks/2026-09-27-glm-5.3-flash-morrowmake-4card-pp4-vllm.ipynb`](../../notebooks/2026-09-27-glm-5.3-flash-morrowmake-4card-pp4-vllm.ipynb)
- Upstream figures: [Morrowmake v1.4.1 README](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe/blob/v1.4.1/README.md#results)

Credit: Morrowmake for the recipe and engine, canada-quant for the W4A16 checkpoint, incoai for the DFlash2 drafter, MiaAI-Lab for the benchmark prompts.
