# X draft — do not post automatically

## Ready-to-paste single caption

Qwen3.8-27B on 1x CMP 170HX: 1.71x faster zero-shot prefill at 6,603 tokens. Separate 8-prompt cohort: 13.3% greedy-token agreement, 0/8 full matches—not task accuracy. Trained speed unverified. Based on kishida’s LLKVApprox.

## Source URL / version caveat — not part of the caption

Mechanism source: https://kishida.github.io/webdemos/llkvapprox/

The corrected notebook and media are a private review candidate and are not yet present on the public repository's default branch. The timing arms are stored in separate receipts; prompt/runtime hashes, timestamps, and interleaved pairing are not recorded.

## Optional concise thread

1/3 Qwen3.8-27B + LLKVApprox on 1x CMP 170HX: recorded 6,603-token prefill arms were 3.329s full vs 1.944s zero-shot suffix 256, a 1.712x ratio. Three samples are recorded per arm; the receipts do not establish interleaved paired trials.

2/3 Quality is a separate 8-prompt cohort that actually exercised approximation: 13.3% zero-shot and 16.0% stage-1 greedy token agreement, with 0/8 full matches. The pooled 23-prompt result is dominated by 15 exact-path controls. Not task accuracy.

3/3 Oracle speed is a teacher-capture diagnostic ceiling, not deployable speed or TTFT. Trained-projector throughput is unverified. Stage 1 completed 2,000 steps in 1.59h; stage 2 is unvalidated. Credit: kishida's LLKVApprox work.
