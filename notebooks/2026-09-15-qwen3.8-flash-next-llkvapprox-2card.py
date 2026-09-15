# %% [markdown] cell:0
# Qwen3.8-Flash-Next + LLKVApprox v2 (late-layer KV/state approximation) on 2x CMP 170HX (SM80), packed int4 experts + HF eager bf16

| Metric | Value |
|---|---|
| Mechanism validated (oracle fills, single-pass capture) | bit-exact: FA prior K bit-identical (0.0), GDN grid-boundary replay exact-by-construction; residual = MoE routing flips (see gate note) |
| Prefill speed ceiling (oracle, trained-fill cost excluded) | **2.19-2.27x** @1,024-6,603 tok (e.g. 6,603 tok: 119.1 vs 53.4 tok/s) |
| Ridge linear ceiling (24.5k teacher tok) | GDN qkv **0.949**, b 0.981, a 0.976 — FA k 0.62 / v 0.71 / index_k 0.63 (per-layer own-weights + MLP heads required for FA) |
| Stage-1 distillation | **completed**: 2,000 steps / 0.88 h cached-target (zero teacher kernels), final FA k 0.953 / v 0.974 / index_k 0.960, GDN k 0.982 / v 0.991; holdout matches train |
| Trained student greedy match (suffix 1, 48-tok chains) | **69-83% per prompt** (v1 27B trained suffix-1: 7.4%) |

Reference mechanism: [LLKVApprox demo](https://kishida.github.io/webdemos/llkvapprox/) (Qwen3-8B, dense) and
[the author's write-up](https://nowokay.hatenablog.com/entry/2026/09/11/120001). Projector:
[PixelML/Qwen3.8-Flash-Next-KVA-Projector](https://huggingface.co/PixelML/Qwen3.8-Flash-Next-KVA-Projector).
# %% [markdown] cell:1
# Qwen3.8-Flash-Next + LLKVApprox v2 — 2x CMP 170HX, packed int4 experts + HF eager

Port of the v1 LLKVApprox port (Qwen3.8-27B, dense, 1 card) to **Qwen3.8-Flash-Next**
(`qwen4_exp`): 48 layers = 36 gated-delta-net + 12 full attention, **512-expert MoE in every
layer** (top-10), 4-stream **hyper-connections**, a **lightning indexer** per FA layer, PLE
n-gram tables, and an MTP head (unused here). v1's dense model had none of these; each one
changed the port:

- **MoE everywhere**: bf16-resident experts are impossible (121B expert params = 242G), so the
  engine keeps experts int4-packed on GPU (~35G/card) and dequantizes routed experts per
  forward; PLE n-gram tables (~102G) stay CPU-mmap'd. Layer split 24 = CED boundary = card
  boundary (encoder on GPU0, decoder on GPU1).
- **Hyper-connections**: the residual stream is 4x2560; the projector's boundary state is
  10240-dim and the own-weights init becomes the block-average of the layer's in_proj per stream.
- **Lightning indexer (issue open question, resolved: yes)**: every FA layer caches raw per-token
  indexer keys and the block selection consumes them — approximated FA layers need **K, V, and
  index_k** filled (a third projector target v1 did not have).
- **Chunked delta-rule non-associativity**: splitting 256 as 255+1 perturbs the GDN state by
  ~10% rel; the oracle injects at the last 64-chunk boundary and replays the tail through the
  teacher's own chunked kernel (bit-exact by construction).
- **MoE routing flips set the oracle noise floor**: bf16-ulp GEMM differences between the CED
  suffix path (M=1) and the full prefill (M=T) flip near-tie router decisions and amplify ~100x
  through the stack. The model is otherwise fully deterministic (shape-noise floor = 0.0), so
  v1's dense-model 0.013 gate is unreachable on this architecture; the gate is recalibrated to
  bit-exact fills + bounded drift + argmax agreement (see the oracle cell).

Evidence source: committed `receipts/` JSON recorded 2026-09-14/15. Every number below is read
from those receipts; nothing is re-measured by this notebook (LIVE = False).
# %% [code] cell:2
# --- Status cell -------------------------------------------------------
import os

EXPERIMENT = "2026-09-15-qwen3.8-flash-next-llkvapprox-2card"
RESULTS_DIR = os.path.join("..", "receipts", EXPERIMENT)
LIVE = False

print(f"experiment   : {EXPERIMENT}")
print(f"results_dir  : {RESULTS_DIR}")
print(f"LIVE         : {LIVE}")
# %% [code] cell:3
# --- Helpers ------------------------------------------------------------
import json, os

def load_receipt(name, results_dir=RESULTS_DIR):
    with open(os.path.join(results_dir, name)) as f:
        return json.load(f)

def render_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    from IPython.display import display, Markdown
    display(Markdown("\n".join(lines)))
# %% [code] cell:4
# --- Feasibility: packed-resident load, 2x 64GB cards -------------------
smoke = load_receipt("load-smoke.json")
render_table(
    ["item", "value"],
    [
        ["GPU0 / GPU1 resident", f"{smoke['gpu0_gb']} / {smoke['gpu1_gb']} GiB"],
        ["layer split", "0..23 cuda:0 (encoder) | 24..47 cuda:1 (decoder)"],
        ["load time", f"{smoke['load_seconds']} s"],
        ["expert dequant matches compressed-tensors reference", smoke["dequant_matches_ct"]],
        ["512-token full prefill", f"{smoke['prefill_512_s']} s, finite logits: {smoke['prefill_512_finite']}"],
    ],
)
# %% [code] cell:5
# --- Speed ceiling: full vs oracle CED prefill --------------------------
bench = load_receipt("bench-prefill.json")
rows = [[r["tokens"], f"{r['full_s']} / {r['full_tok_s']}", f"{r['oracle_s']} / {r['oracle_tok_s']}",
         f"**{r['speedup']}x**"] for r in bench["rows"]]
render_table(["tokens", "full s / tok/s", "oracle s / tok/s", "speedup"], rows)
print("timed region = encoder + grid-replay fills + suffix; fills come from the pass bundle\n"
      "(in deployment: the trained projector). Both arms share every eager path.")
# %% [code] cell:6
# --- Oracle gate (MoE-aware) --------------------------------------------
o = load_receipt("oracle-test.json")
print("last position:", o["last_position"])
print("drift:", o["teacher_forced_drift"])
print("gate note:", o["gate_note"])
# %% [code] cell:7
# --- Why the dense-model gate is unreachable: chunked-kernel + MoE flips -
probe = load_receipt("gdn-kernel-probe.json")
print("chunked delta-rule: single-call chunk(0..255) vs stored state:",
      probe["B_chunked256_vs_true"])
print("split 255 + chunk(1):", probe["A_fused_from_S254_vs_true"])
floor = load_receipt("shape-noise-floor.json")
print("shape-noise floor (same rows, longer GEMM batch):", floor["shape_noise_last_position"],
      floor["shape_noise_drift"])
# %% [code] cell:8
# --- Ridge linear ceiling (closed form, 24.5k teacher tokens) ------------
ridge = load_receipt("ridge-ceiling.json")
rows = [[name, v["dims"], v["heldout_cosine"], f"lambda x{v['best_lambda_mul']}"]
        for name, v in ridge["ridge"].items()]
render_table(["target", "dims", "held-out cosine", "lambda"], rows)
print("FA targets need per-layer own-weights + MLP heads; GDN is linear-predictable.")
# %% [code] cell:9
# --- Stage-1 cached training (zero teacher kernels) ----------------------
tr = load_receipt("stage1_cached_v2.json")
print("args:", {k: tr["args"][k] for k in ("steps", "lr", "tag")})
print("history tail:")
for rec in tr["history_tail"][-3:]:
    print(" ", rec)
print("holdout eval:")
for e in tr["holdout_eval"]:
    print(" ", e)
print("wall hours:", tr["wall_hours"])
# %% [code] cell:10
# --- Trained student quality (greedy match vs baseline, diagnostic) ------
q = load_receipt("quality-trained.json")
print("summary:", json.dumps(q["summary"], indent=1))
print(q["note"])
# %% [markdown] cell:11
# ### Reproduce
#
| Pin | Value |
|---|---|
| Model | Qwen3.8-Flash-Next (qwen4_exp), packaged from `Qwen3.8-Flash-Next-AWQ-INT4` weights (176G, compressed-tensors pack-quantized) |
| Hardware | 2x CMP 170HX 64GB @250W (SM80), host-offloaded PLE |
| Runtime | torch 2.14.0+cu130, transformers 5.18.0.dev0 (`qwen4_exp`), fla 0.6.0, compressed-tensors 0.18.0 (reference conv fallback; `causal_conv1d` not buildable on this host) |
| Projector | [PixelML/Qwen3.8-Flash-Next-KVA-Projector](https://huggingface.co/PixelML/Qwen3.8-Flash-Next-KVA-Projector) (385MB bf16) |
| Engine | issue #139 progress comments + `~/WIP/llkvapprox-flash-next/` (src + scripts + receipts) |
| Date | receipts recorded 2026-09-14/15 |
#
# Every number above reads the committed receipts; this notebook re-measures nothing.
# %% [markdown] cell:12
# ### Appendix: layer-input divergence profile (oracle vs baseline, single-pass capture)
#
# From `oracle-debug.json`: layer 24 input bit-exact (0.0), FA prior K bit-exact (0.0);
# divergence enters at the GDN row-255 decode step (bf16-ulp GEMM-shape noise, M=1 vs M=T),
# then jumps at the first near-tie router flip (layer-input rel_fro 0.0066 -> 0.72 at layer 28)
# and stays bounded thereafter. Full per-layer profile in the receipt.
