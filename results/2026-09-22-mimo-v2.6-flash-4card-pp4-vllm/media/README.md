# "MiMo on the 170HX" — song + animated explainer

A 35-second, 1080×1080 music video explaining this lane: `mimo-on-the-170hx.mp4`.

- **Song:** original lyrics (`song/lyrics.txt`), generated with **YuE2** (`yue2_infer` 0.1.5) on our own
  hardware. Six takes (3 styles × 2 seeds, `song/generate.sh`); the chosen take is `mimo_synthpop`,
  seed 11 (`song/mimo-on-the-170hx-full.mp3`, 42 s). The video uses 6.6 s → 42 s of it.
- **Animation:** hand-painted in p5.js + p5.brush with the Claude Animation Base starter kit (MIT; based on
  [JohnHeibel/PDoomVideo](https://github.com/JohnHeibel/PDoomVideo)). Scene: `animation/mimo.js`;
  plan: `animation/STORYBOARD.md`; settings: `animation/config.js`. Shot starts and karaoke lyrics are
  snapped to the take's word timestamps (Whisper large-v3-turbo).
- **Character:** Mitu, the Mi Bunny — Xiaomi's mascot, redrawn here as fan art, without the logo. This
  project is not affiliated with or endorsed by Xiaomi.

## The numbers in the video

All measured in this lane (`../receipts/x16/`): decode **117 tok/s** (1 stream, MTP k=2, greedy; 90 at
T=1.0) · prefill **~4,100 tok/s** (1 request, 21.5k-token prompt) · **561 tok/s** aggregate at 32 streams.
The model is MiMo-V2.6-Flash (309B MoE) on 3× CMP 170HX, vLLM, pipeline-parallel 3.

## Re-render

1. Get the starter kit (p5.js + p5.brush + `render.mjs`), `npm install`.
2. Square frame: set `const W = 1080, H = 1080` in `src/core.js`, the `<canvas>` size in `studio.html`,
   and `--window-size=1080,1080` in `render.mjs`.
3. Copy `animation/mimo.js` to `src/scenes/`, point `studio.html` at it, and use `animation/config.js`.
4. Trim the song: `ffmpeg -ss 6.6 -t 35.4 -i song/mimo-on-the-170hx-full.mp3 -af "afade=t=in:d=0.25,afade=t=out:st=33.9:d=1.5" assets/song.wav`
5. `node render.mjs --clip --out=out/video.mp4`
