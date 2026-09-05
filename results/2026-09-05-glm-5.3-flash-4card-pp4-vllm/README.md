# GLM-5.3-Flash, PP4 + MTP k=3, vLLM sm80 — 4x CMP 170HX

Measured 2026-09-05. This is the **recipe of record** for GLM-5.3-Flash on this
node; the 2026-09-03 TP4 result is superseded and kept as history in the
notebook appendix.

Executed notebook:
[`notebooks/2026-09-05-glm-5.3-flash-4card-pp4-vllm.ipynb`](../../notebooks/2026-09-05-glm-5.3-flash-4card-pp4-vllm.ipynb).
Guide page: [`docs/models/glm-5.3-flash.md`](../../docs/models/glm-5.3-flash.md).

## Pins

| Field | Value |
|---|---|
| Checkpoint | `wtdcode/GLM-5.3-Flash-AWQ-W4A16` @ `abd7b07719111f137e1de8a0c1b7e01c11b74d1a` (AWQ W4A16, compressed-tensors, 190,843,146,533 bytes, 24 files) |
| Image | `ghcr.io/pixelml/club-170hx:vllm-glm53-sm80-pp-20260905` |
| Image index digest | `sha256:62f612b49614523e6a46e1493d35d3efd1f363917129d38cc923a31053693bfb` |
| Runtime source | `PixelML/sm80vllm` branch `pp-dflash2/glm53-flash-487ecf187-20260905` — an orphan overlay over `vllm/vllm-openai:glm53-flash` @ `487ecf187` plus 24 patches ported from [promisezackr/glm53-flash-170hx-pp8](https://github.com/promisezackr/glm53-flash-170hx-pp8) (Apache-2.0, per-patch attribution trailers) |
| Recipe script | `recipes/glm53-flash-4x170hx-pp4.sh` |
| Topology | PP4, `VLLM_PP_LAYER_PARTITION=14,12,12,7` (45 hidden layers) |
| Speculation | native MTP, `num_speculative_tokens=3` |
| Context | `--max-model-len 393216`; KV pool 1,194,627 tokens (3.04x) |
| Other flags | `--no-enable-prefix-caching`, `--gpu-memory-utilization 0.90`, `--max-num-seqs 8`, `--max-num-batched-tokens 4096`, `VLLM_PP_MAX_DECODE_REQS_PER_BATCH=2`, `VLLM_GLM5N_SIDECAR_BLOCK_SIZE=256`, `--limit-mm-per-prompt image:0,video:0` |
| Hardware | 4x CMP 170HX (SM80, 64 GiB each), 180 W per-card cap, no NVLink, no P2P |
| Boot | 6/6 boots served for this recipe (873, 995, 1,029, 1,077, 1,140, 1,263 s to `Application startup complete`), plus 2/2 on the earlier port run; the 873 s boot is post-recovery with a warm compile cache; idle memory 51.6-54.2 GiB per card of 64; 180 W enforced limit on all four throughout |

## Hardware caveat that applies to every link-bound number here

Measured on a degraded link: **GPU1 PCIe Gen1 x1 (slot ceiling x8), GPU0 x8,
GPU2/3 x16, no NVLink.** Aggregate, prefill and TTFT cells are lower bounds.
c=1 decode is link-insensitive under PP4 — one hidden-state hop of roughly
50 KB per decode step — and carries the headline.

## Protocols

- **P1** — the upstream author's `scripts/bench.py`: temperature 0, 512 output
  tokens, three repetitions, five workloads (code, json, counting, math, prose)
  plus a repetition diagnostic. His repeat guard flags a repetition-collapsed
  completion; flagged cells are inflated and are not read as throughput.
- **P2** — this club's protocol, unchanged since the TP4 record: temperature
  0.7, `ignore_eos`, 512 output tokens, five repetitions, median reported, first
  repetition cold.

Token counts come from the final `usage` object of each response, never from
counting stream events.

## Headline

| Metric | Value | Label |
|---|---|---|
| Decode, c=1 | 87.6 tok/s (P1 math, median of 3, clean) | measured |
| Decode, c=1 | 67.9 tok/s (P2 median of 5; peak 92.6, cold rep 56.3) | measured |
| Best aggregate | 78.4 tok/s at c=16 (4k prompt, 256 out) | measured on degraded link |
| Prefill | 1,752 tok/s at 16,384 tokens | measured on degraded link |
| TTFT | 9.44 s at 16,384 tokens, 3.67 s at 4,096 tokens (warm, streaming) | measured on degraded link |
| Longest context measured | 131,042 prompt tokens, 78.6 tok/s generation | measured |
| Quality | GSM8K 49/50, HumanEval 19/20 pass@1, structured output 10/10 | measured at the checkpoint's sampling defaults |
| Sustained stability | 3/3 rounds of c=8, 24/24 requests, zero Xid, 48-53 C | measured |
| Drafter value | MTP k=3 is 1.62x speculation off on P2, up to 2.06x per workload | measured |
| Lossless | **no verdict available** — greedy is not reproducible with speculation on or off | measured finding |

## Receipt map

| Path | Cell |
|---|---|
| `receipts/cells.json` | sanity prompts and the k=3 gate battery |
| `receipts/k3/gate.json` | deterministic greedy repeat |
| `receipts/{k2,k3,k5,k7}/p1.json` | P1 per-workload decode, one boot per draft depth |
| `receipts/{k2,k3,k5,k7}/decode_c1.json` | P2 c=1 decode, one boot per draft depth |
| `receipts/accept_{before,after}.json` | engine `spec_decode` counters for the **k=3** window |
| `receipts/{k2,k5,k7}/accept_{before,after}.json` | the same counters for the other depths |
| `receipts/k3/sweep/context_sweep.json` | decode vs context, 327 to 131k tokens, two thinking arms |
| `receipts/k3/conc_sweep.json` | concurrency c=1,2,4,8,16 at 4k prompt |
| `receipts/k3/prefill_{4096,16384}/{prefill,ttft}.json` | uncached prefill and warm streaming TTFT |
| `receipts/k3/thinking_probe.json` | thinking-switch probe (see limitations) |
| `receipts/k3/quality.json` | held-out quality battery, every item |
| `receipts/k3/c8_stability.json` | 3 rounds of c=8 with health, power and temperature after each |
| `receipts/lossless/` | self-consistency controls and the on-versus-off comparison |
| `receipts/nospec/` | speculation-off throughput baseline |
| `receipts/awq-dflash7/` | negative cell: the community block drafter on our AWQ checkpoint |

Every JSON here is the harness output with endpoints, filesystem paths, IP
literals and container names replaced by `<endpoint>`, `<path>`, `<addr>` and
`<container>`. Numbers, prompts and model text are verbatim.

## Negative results preserved

- **DFlash2 k=7 on the AWQ W4A16 checkpoint is a net loss** (`receipts/awq-dflash7/`).
  Draft acceptance 41.6% against MTP k=3's 65.4%; the code completion opens with
  repeated, broken think tags; code and prose collapse to roughly half of MTP
  k=3; the drafter's sidecar takes 56% of the KV pool. The fast-looking counting
  and json cells trip the repeat guard. This — not the card count, and not PP4 —
  is the main reason the upstream headline figure does not reproduce here.
- **PP4 pre-port**: MTP k=5 on the stock build ran at 3.35 tok/s with degenerate
  text; speculation off ran at 6.11 tok/s with clean text. Two separate faults,
  both fixed by the port.
- **DFlash2 under either topology on the stock build** never reached a
  measurement: under PP it is refused because the drafter's auxiliary
  hidden-state layers cannot all sit on the last stage, and under TP it is
  refused at KV-cache setup because MLA indexer page sizes cannot be padded.

## Untested (pending)

The long-context-retrieval and tool-use quality buckets, integrated energy, the
258k context point, and accepted-tokens-per-pass versus context length.

Four cells came back as findings rather than gaps:

- **no lossless verdict is available** — greedy output on this stack is not
  reproducible with speculation on *or* off, so no token-for-token comparison
  can clear a lossless bar;
- the **NVFP4** checkpoint is `not claim-ready` (attempted, out of memory in the
  mixture-of-experts kernel-format conversion; a retry is queued);
- the **thinking switch** is not switchable on this checkpoint under any key or
  the server default;
- **speculation off is also non-deterministic**, which is what rules the drafter
  out as the cause of the first finding.

Reasons for all of them are in section 2.12 of the notebook.
