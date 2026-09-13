# LLKVApprox share-asset evidence mapping

The share card, video, and notebook use the same committed receipts. Greedy token agreement is a continuation diagnostic, not task accuracy.

| Visible claim | Receipt and field/calculation |
| --- | --- |
| Full prefill arm: 3.3293 s / 1,983.3 tok/s at 6,603 tokens | `bench-prefill.json`: `rows[mode=full,tokens=6603]` |
| Zero-shot suffix-256 arm: 1.9444 s / 3,395.9 tok/s | Separate file `bench-prefill-zs-s256.json`: `rows[tokens=6603]` |
| 1.71x measured prefill speedup | `3.3293 / 1.9444 = 1.7122505657` |
| Timing protocol boundary | Three samples are recorded per arm. The notebook reports one warmup, but raw timing receipts contain no warmup record, prompt/runtime hashes, timestamps, or evidence of interleaved paired trials. |
| Zero-shot pooled agreement: 69.8% | `quality-eval-zs-s256.json`: `2056 / 2944 = 69.8369565%`; includes exact-path controls |
| Trained pooled agreement: 70.8% | `quality-trained-s256.json`: `2084 / 2944 = 70.7880435%`; includes exact-path controls |
| Zero-shot active agreement: 13.3% | `quality-eval-zs-s256.json`: prompts with `prompt_tokens > 256`; `136 / 1024 = 13.28125%` |
| Trained active agreement: 16.0% | `quality-trained-s256.json`: prompts with `prompt_tokens > 256`; `164 / 1024 = 16.015625%` |
| 15/23 prompts took the exact path | Both suffix-256 quality receipts: `prompt_tokens <= 256`; those controls contribute `1920 / 1920` matching tokens per mode |
| 0/8 active prompts fully matched | Both suffix-256 quality receipts: no active row has `match_rate == 1` |
| Oracle speed ceiling: 2.01x at 6,603 tokens | `bench-prefill.json`: full 3.3293 s / oracle 1.6589 s; teacher captures make this diagnostic only |
| Oracle fidelity at 1,024 prompt tokens | `oracle-test.json`: `prompt_tokens = 1024`, teacher-forced argmax `31/32`, last-position mean absolute logit difference `0.01340116` |
| Oracle free-running negative diagnostic | `oracle-test.json`: `18/64` greedy token agreement; separate from teacher-forced fidelity and not task accuracy |
| Stage 1 completed | `stage1_cached_mlp.json`: `args.steps = 2000`, `wall_hours = 1.59` |
| Trained throughput unverified | No successful trained-projector timing receipt is committed; do not transfer zero-shot speed to trained weights |

Receipt SHA-256 pins are stored in `results/2026-09-12-qwen3.8-27b-llkvapprox-1card-hf/publication-evidence.json`.

Limitations:

- The speed and quality measurements use separate prompt cohorts.
- No quality result exists for the 6,603-token speed cohort.
- The timing arms live in separate receipts and cannot establish interleaved paired trials.
- The public artifact replays committed receipts; it does not contain the full live GPU harness or complete immutable runtime/model/projector revisions.
- No end-to-end TTFT, task accuracy, or video-understanding claim is supported.

Mechanism credit: [kishida's LLKVApprox demo](https://kishida.github.io/webdemos/llkvapprox/), [reference engine](https://github.com/kishida/webdemos/blob/main/llkvapprox/engine.js), and [write-up](https://nowokay.hatenablog.com/entry/2026/09/11/120001).
