# GLM-5.3-Flash v1.7.0: RecoverSSM loop fix, receipts

Notebook: [`notebooks/2026-10-04-glm-5.3-flash-v1.7.0-recoverssm-loop-fix-4card-tp4-vllm.ipynb`](../../notebooks/2026-10-04-glm-5.3-flash-v1.7.0-recoverssm-loop-fix-4card-tp4-vllm.ipynb)

| Pin | Value |
|---|---|
| Fixed engine image | `ghcr.io/pixelml/club-170hx@sha256:f9fc947bb4f1fdacd5e146948cb64cdca01da0942809989d32a6241886defaf0` |
| Fix source | [PixelML/sm80vllm@9522a0efb60972858811a0883d0c001c6b7087b2](https://github.com/PixelML/sm80vllm/commit/9522a0efb60972858811a0883d0c001c6b7087b2) (parent: Morrowmake/vllm-cmp170hx `c1ce6491efe53934119d306d0a0501b475458e9b`) |
| Stock engine image | `ghcr.io/morrowmake/vllm-cmp170hx@sha256:158627a705b6ae24fa63ba6eb454982454c5e6b94b1e32981faf33eb81a7ea3e` |
| Recipe | Morrowmake/glm53-flash-cmp170hx-recipe `e61ba680cd6f9497291cafa4ca13ac3ffbdab844` (v1.7.0) |
| Model / drafter | canada-quant/GLM-5.3-Flash-W4A16-MTP `5723f4d02af36366c23ace8668866ca7775855c1` (MIT) / incoai/GLM-5.3-Flash-DFlash2 `bf582e4eacc1810f76656d1811693ff6c6737d2a` (CC BY-NC-ND 4.0) |

| Path | What |
|---|---|
| `build/Dockerfile` | the fixed image: stock image by digest + 2 files from the fix commit (sha256-checked) |
| `glm_repro.py` | one round: 4 concurrent ~30k-token prompts, 4,000 output tokens, prints `BROKEN n of 4` |
| `run_ab.sh` | all A/B arms (restart, wait for `/health`, 2 rounds each) |
| `receipts/repro/*.out` | every round (stock arms, `fix-r1..4` = fixed image) |
| `receipts/repro/run-log-*.txt` | run order and per-round results (UTC) |
| `receipts/decode-fixed/` | `bench_decode.py` receipts for the fixed image |
| `receipts/launch-fixed.txt` | boot check, KV pool, RecoverSSM-active and block-size lines from the fixed launch |

Measured: stock v1.7.0 with spec decode + RecoverSSM 13 / 24 broken; drafter off or RecoverSSM off 0 / 12; fixed 0 / 16. Cause first reported in [Morrowmake/glm53-flash-cmp170hx-recipe#7](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe/issues/7); upstream: [vllm-project/vllm#59933](https://github.com/vllm-project/vllm/issues/59933).
