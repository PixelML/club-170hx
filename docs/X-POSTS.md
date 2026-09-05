# X post drafts

One draft post per published notebook. Each post has three lines: headline
number with units, a link, and the chart path. Copy the text as is or trim
for length. Do not add claims that are not in the linked notebook.

## DeepSeek-V4-Flash-Vision-Exp, 4x CMP 170HX (2026-09-02)

```
4x CMP 170HX (SM80, 64 GiB each) now serve DeepSeek-V4-Flash-Vision-Exp at 220.2 tok/s aggregate decode at c=8 (median of 3).
Notebook: https://github.com/PixelML/club-170hx/blob/main/notebooks/2026-09-02-deepseek-v4-flash-vision-exp-4card-pp4-vllm.ipynb
Release: https://github.com/PixelML/club-170hx/releases/tag/dsv4-vision-exp-4card-2026-09-02
Chart: assets/charts/2026-09-02-dsv4-vision-exp-4card-ladder.png
```

## DeepSeek-V4-Flash-Vision-Exp, vision path (2026-09-02, in progress)

```
First real-image inference of DeepSeek-V4-Flash-Vision-Exp on 4x CMP 170HX SM80 hardware, reference runtime, 0.9 tok/s decode. A correctness result, not a speed result.
Notebook: https://github.com/PixelML/club-170hx/blob/main/notebooks/2026-09-02-deepseek-v4-flash-vision-exp-4card-vision-pp4-vllm.ipynb
Release: https://github.com/PixelML/club-170hx/releases/tag/dsv4-vision-exp-4card-2026-09-02
```

## DeepSeek-V4-Flash-0731, 3x CMP 170HX (2026-08-30)

```
3x CMP 170HX at 180 W each serve DeepSeek-V4-Flash-0731 at 83.3 tok/s aggregate decode, DSpark speculative decoding at k=5.
Notebook: https://github.com/PixelML/club-170hx/blob/main/notebooks/2026-08-30-deepseek-v4-flash-0731-3card-pp3-vllm.ipynb
Chart: assets/charts/2026-08-30-deepseek-v4-flash-0731-3card-pp3-vllm.png
```

## Qwen3.8-27B W4A16 + DFlash2, 1x CMP 170HX (2026-08-30)

```
One CMP 170HX at 180 W runs Qwen3.8-27B W4A16 with DFlash2 speculative decoding at 140.3 tok/s decode, 95% of a 255 W rented card's speed at 71% of its power cap.
Notebook: https://github.com/PixelML/club-170hx/blob/main/notebooks/2026-08-30-qwen3.8-27b-w4a16-dflash2-1card-vllm.ipynb
Chart: assets/charts/2026-08-30-qwen3.8-27b-w4a16-dflash2-1card-vllm.png
```

## GLM-5.3-Flash AWQ W4A16, 4x CMP 170HX (2026-09-03) — superseded, do not post

Superseded on 2026-09-05 by the PP4 + native MTP k=3 recipe of record. Kept as
a drafting record only.

```
4x CMP 170HX (SM80, 64 GiB each) run GLM-5.3-Flash AWQ W4A16 via vLLM TP4 + native MTP-3 at 56.4 tok/s median c=1 (56.9 peak). C8 aggregate is 37.0 tok/s and TP4 communication-bound.
Notebook: https://github.com/PixelML/club-170hx/blob/main/notebooks/2026-09-03-glm-5.3-flash-4card-tp4-vllm.ipynb
Chart: assets/charts/2026-09-03-glm-5.3-flash-vllm-sm80-4gpu-sweep.png
Video: assets/video/glm53-vllm-sm80-motion/glm53-vllm-sm80-motion-1080x1080.mp4
```

## GLM-5.3-Flash on CMP 170HX, negative result (2026-08-31)

```
GLM-5.3-Flash NVFP4 does not run on CMP 170HX (SM80 lacks native NVFP4 execution). The llama.cpp UD-IQ4_XS fallback runs at 17.73 tok/s decode at c=1.
Notebook: https://github.com/PixelML/club-170hx/blob/main/notebooks/2026-08-31-glm-5.3-flash-compatibility-cmp170hx.ipynb
Chart: assets/charts/2026-08-31-glm-5.3-flash-compatibility-cmp170hx.png
```
