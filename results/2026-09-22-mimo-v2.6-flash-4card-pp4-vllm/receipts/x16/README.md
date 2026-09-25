# x16 re-run (2026-09-24)

Same image and patches as `receipts/bench/`, re-measured after the rig moved to three cards on
PCIe x16 each (Gen1 at measurement time; the x1-riser card removed). All runs PP3, 180 W cap.

| config | P1 greedy | P2 sampled T=1.0 | prefill c=1 (21.5k-token prompt) | 32 streams aggregate |
|---|---|---|---|---|
| PP3 | 75.5 tok/s | 74.9 | 4,161 tok/s | 454.6 tok/s |
| PP3 + MTP k=2 | **117.7** | **90.1** | **4,121** | **561.3** |
| PP3 + DFlash k=7 | 154.4 | 72.0 | 3,818 | 339.5 (max 16 concurrent) |

Files per config: `p1.json`, `p2.json`, `gate.json` (4/4 correct), `conc.json` (c = 1/4/8/16/32,
256 output tokens, T=1.0, `ignore_eos`), `prefill.jsonl` (uncached unique prompts, `max_tokens=1`),
`launch.json`. Single-stream decode barely moves vs the mixed-link run; aggregate throughput under
load gains most (MTP k=2: 495 → 561 tok/s at 32 streams). DFlash ran with `--max-num-seqs 16` and
22.5k context to fit its drafter, so its c=32 cell is queue-limited.
