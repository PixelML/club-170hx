# GLM-5.3-Flash vLLM motion card (2026-09-03, superseded)

Publication renders for the 2026-09-03 GLM-5.3-Flash AWQ W4A16 TP4 run on four
CMP 170HX (SM80) cards. Kept as a dated artifact: the TP4 lane it summarises was
superseded on 2026-09-05 by the PP4 + native MTP k=3 recipe of record
(`assets/video/glm53-pp4-motion/`). Do not reuse these renders as current
figures.

## Renders

- `glm53-vllm-sm80-motion-poster.png` — final summary frame for previews.
- `glm53-vllm-sm80-motion-1080x1080.mp4` — square master.
- `glm53-vllm-sm80-motion-1920x1080.mp4` — landscape version.
- `glm53-vllm-sm80-motion-1080x1920.mp4` — vertical mobile version.

All renders are 8 seconds, 30 fps, H.264, and carry the measured headline of
that run: 56.4 tok/s median c=1, 56.9 tok/s peak c=1, 37.0 tok/s aggregate c=8.
The c=8 caveat and the 180 W/card cap are on screen.

Composition sources are not committed to this public repository, matching every
other directory under `assets/video/`.
