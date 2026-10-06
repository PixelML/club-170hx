# MiniMax H3 video + audio on CMP 170HX — receipts (2026-10-06)

Executed notebook: [`notebooks/2026-10-06-minimax-h3-video-1card-comfyui.ipynb`](../../notebooks/2026-10-06-minimax-h3-video-1card-comfyui.ipynb).

**Credits.** Model: MiniMax H3, authored by **MiniMax** ([MiniMax-AI/MiniMax-H3](https://github.com/MiniMax-AI/MiniMax-H3), weights [MiniMaxAI/MiniMax-H3 @ 42ed227e](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/42ed227ee7df40d41602854ae760620d6eb651fe), MiniMax H3 Community License). Quantized ComfyUI files: authored by **Comfy-Org** ([Comfy-Org/MiniMax-H3 @ e5eb578a](https://huggingface.co/Comfy-Org/MiniMax-H3/tree/e5eb578a89295337b8ff433a035929ce0279e0b6)). Turbo LoRA and sampler: authored by **Larryvrh** ([@larryvrh](https://github.com/Larryvrh), [LoRA @ 43a74557](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/tree/43a74557ac3f6539db8e0f2a959d03feb7a81480), [nodes @ 4274783a](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo/tree/4274783a23afcfdbea3b4876cb79effd6c510785)). Community prompts P1–P3 authored by **cocktail peanut** ([@cocktailpeanut](https://x.com/cocktailpeanut/status/2084799914765341083)), **Noor** ([@noorlewisx](https://x.com/noorlewisx/status/2088844629093601702)) and **Zara** ([@ZaraIrahh](https://x.com/ZaraIrahh/status/2083011066800242986)), collected via [videotoprompt.app](https://www.videotoprompt.app/trending-prompts/minimax). Engines: [ComfyUI](https://github.com/comfyanonymous/ComfyUI) (comfyanonymous and Comfy Org), [SGLang Diffusion](https://github.com/sgl-project/sglang) (sgl-project), [diffusers](https://github.com/huggingface/diffusers) (Hugging Face; H3 integration PR #14355 authored by apolinario), [SageAttention](https://github.com/thu-ml/SageAttention) (thu-ml). QA: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (SYSTRAN), Whisper (OpenAI), [jiwer](https://github.com/jitsi/jiwer) (jitsi). Card tuning: 170tune by Cachenetics.

## What is here

| Path | Content |
|---|---|
| `receipts/runs.json` | One record per generated clip (82): task, engine, card, power cap, config, seed, geometry, phase timings (encode / sample / VAE decode / audio decode / mux), per-step sampler times, peak VRAM, mean power and SM clock, GPU energy, ASR WER / script recall / inserted words, audio levels. `cold` = first job after a model load; `tripped` = the thermal watchdog lowered the cap during the run. |
| `receipts/runs/<run_id>/telemetry.csv` | `nvidia-smi` inside the engine container every 500 ms: power, SM and memory clock, memory used, utilisation, core temperature. |
| `receipts/power-sweep/telemetry.csv` | Host `nvidia-smi` every 2 s during the power sweep: core and HBM temperature, power limit in force, power draw, per card. `watchdog-events.json` = every trip. |
| `receipts/graphs/` | The exact ComfyUI API graphs of every public run (P1–P3, R1, T5); submit to `POST /prompt`. |
| `receipts/inputs/` | Public prompts (verbatim), the R1 reference image (a frame of this bench's own P3 output) and `community_sources.json` (authors, links, likes at collection time). |
| `receipts/env/` | `engines.json` (image digests, package versions, GPU/driver/VBIOS per engine pod), `comfyui_setup.py` (pinned sources), `comfyui-venv-pip-freeze.txt`, `run_diffusers.py`, `WEIGHTS.md` + `SHA256SUMS` (served files matched to the pinned Hugging Face LFS hashes). |

Cards: CMP-1 and CMP-3 have VBIOS 92.00.67.00.01 (250 W max), CMP-2 has 92.00.6D.00.0A (300 W max); all four cards in the host run 170tune HBM NDIV 64, and the .6D cards an SM VF offset of +200 MHz at a 1,410 MHz ceiling. Default cap 140 W. DGX Spark = one GB10 running the same ComfyUI commits.

## Labels

- **measured**: everything in `runs.json` and the telemetry files.
- **inferred**: the 200 W episode-time column (140 W shot times × the measured 200 W ratio on the close-up).
- **private inputs**: tasks T1–T4 come from a private short-drama project. Their prompts, reference images, voice references and transcripts are withheld; their timings, VRAM, power and WER are. R1 has the same geometry as T1 with public inputs and reproduces its timing (84 vs 83 s, 15.8 vs 15.9 s/step).
- **not measured**: sustained 200 W soak, lip-sync scores, identity similarity, human review, multi-card inference, SGLang/diffusers on public inputs.

## Safety

The power sweep used a host watchdog at 83 °C core / 85 °C HBM, above this repository's 80 °C stop rule. The 200 W runs peaked at 79–81 °C core; five runs tripped the watchdog at 83–84 °C. All runs stayed below the card's 85 °C max operating temperature (`nvidia-smi -q -d TEMPERATURE`), and all cards were returned to 140 W and verified.

## Licence of the outputs

The sample video and contact sheets are MiniMax H3 Outputs. The MiniMax H3 Community License licenses use and distribution in its Applicable Territory (worldwide except the European Union, the United Kingdom, the Republic of Korea and the United States of America); readers in those territories should read the licence before reusing them.
