# GLM-5.3-Flash vLLM motion card

This directory contains the local HyperFrames-style source and publication
renders for the 2026-09-03 GLM-5.3-Flash AWQ W4A16 run on four CMP 170HX
(SM80) cards.

## Source

- `index.html` — a paused, deterministic 8-second composition with a
  `window.__timelines["pixelml-benchmark"]` GSAP timeline and `?t=` preview
  seeking.
- `hyperframes.json`, `index.motion.json`, `shot-plan.json` — composition
  contract, timing, and motion assertions.
- `.media/gsap.min.js` and `.media/images/pixelml-logo.svg` — local runtime
  and PixelML wordmark; no remote asset is required. The GSAP file retains
  GreenSock's upstream license header.

## Renders

- `glm53-vllm-sm80-motion-poster.png` — final summary frame for previews.
- `glm53-vllm-sm80-motion-1080x1080.mp4` — square master.
- `glm53-vllm-sm80-motion-1920x1080.mp4` — landscape X-friendly version.
- `glm53-vllm-sm80-motion-1080x1920.mp4` — vertical mobile version.

All renders are 8 seconds, 30 fps, H.264, and carry the same measured
headline: 56.4 tok/s median C1, 56.9 tok/s peak C1, and 37.0 tok/s aggregate
C8. The C8 caveat and the 180 W/card cap remain on screen.
