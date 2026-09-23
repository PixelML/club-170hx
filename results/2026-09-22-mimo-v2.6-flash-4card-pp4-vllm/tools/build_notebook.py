import json

REPO = "/home/ubuntu/repos/club-170hx"
EXP = "2026-09-22-mimo-v2.6-flash-4card-pp4-vllm"

def md(src): return {"cell_type": "markdown", "id": "m%08d" % (abs(hash(src)) % 10**8), "metadata": {}, "source": src.splitlines(keepends=True)}
def code(src): return {"cell_type": "code", "id": "c%08d" % (abs(hash(src)) % 10**8), "metadata": {}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}

cells = []

cells.append(md("\n".join([
    "# MiMo-V2.6-Flash-RL on 4x CMP 170HX (SM80) — 4card-pp4, SGLang",
    "",
    "| Metric | Value |",
    "|---|---|",
    "| Verdict | **CLASSIFIED NEGATIVE — not servable on SM80 with stock runtimes (2026-09-22 builds)** |",
    "| Blocker chain | vLLM: FA3-sink + DiffKV pin (SM90+) · SGLang: PP bug (patched) → Triton MoE runner cannot read mxfp4 experts (capture: shape assert; eager: CUDA fault) |",
    "| PP3 | Memory-infeasible (53.7 GiB/rank + transients OOM at 62.0/63.5 GiB) |",
    "| PP4 | Boots past our PP patch; dies at the Triton fused-MoE runner in every mode |",
    "",
    "**The lane's headline finding is a runtime-compatibility chain, not a throughput number.**",
    "The 161 GiB MoE stores its routed experts as mxfp4 — a format whose serving kernels",
    "upstream are deep_gemm/FA3, both SM90+. On Ampere every stock path is excluded by",
    "measured evidence, receipts A1–A5. Porting estimate: an mxfp4→fp8/bf16 expert repack",
    "at load (club autoround-repack precedent exists) or an Ampere mxfp4 kernel.",
    "",
    "Runs today on this hardware instead: **MiMo-V2.6-Distill-Qwen-9B** (dense `qwen3_5`",
    "finetune, standard runtime support) — the family member these cards can serve.",
    "",
    "```bash",
    "docker pull lmsysorg/sglang:latest",
    "```",
])))

cells.append(code("\n".join([
    "# --- Status cell ---",
    'EXPERIMENT = "%s"' % EXP,
    'RESULTS_DIR = "../results/" + EXPERIMENT',
    'RECEIPTS = RESULTS_DIR + "/receipts"',
    "LIVE = False",
    "",
    'print(f"experiment : {EXPERIMENT}")',
    'print(f"LIVE       : {LIVE}")',
    'print("status     : CLASSIFIED NEGATIVE - A1-A5 receipts complete; no throughput number exists to replay.")',
])))

helpers = "\n".join([
    "# --- Helpers: receipt loader and table renderer ---",
    "import json, os",
    "from IPython.display import display, Markdown",
    "",
    "",
    "def receipt(*parts):",
    '    path = os.path.join(RECEIPTS, *parts)',
    "    with open(path) as fh:",
    "        return json.load(fh)",
    "",
    "",
    "def render_table(headers, rows):",
    '    lines = ["| " + " | ".join(str(h) for h in headers) + " |",',
    '             "|" + "|".join(["---"] * len(headers)) + "|"]',
    "    for row in rows:",
    '        lines.append("| " + " | ".join(str(c) for c in row) + " |")',
    '    display(Markdown(chr(10).join(lines)))',
    "",
    "",
    "def fmt(v, nd=2):",
    '    return "untested (pending)" if v is None else f"{v:,.{nd}f}"',
    "",
    "",
    'assert not LIVE, "commit this notebook with LIVE = False"',
    'print("receipts:", len(os.listdir(RECEIPTS)) if os.path.isdir(RECEIPTS) else 0, "entries")',
])
cells.append(code(helpers))

cells.append(md("\n".join([
    "## 1. TL;DR",
    "",
    "Five bring-up attempts across both official runtimes, each reproduced and",
    "receipted: the model family requires SM90+ serving kernels at three independent",
    "points (attention sinks, DiffKV attention, mxfp4 MoE runners). The one-line PP",
    "patch this lane contributed is real and reusable, but it only moves the failure",
    "to the next SM90-only dependency.",
])))

pins_src = "\n".join([
    "pins = {",
    '    "model": "MiMo-V2.6-Flash-RL",',
    '    "checkpoint": "XiaomiMiMo/MiMo-V2.6-Flash-RL",',
    '    "checkpoint_revision": "5711b268169967567844e1e560e8a3966da959b1",',
    '    "checkpoint_bytes": 172932505264,  # 65 shards + MTP head, verified via tools/verify_checkpoint.py',
    '    "quantization": "FP8 e4m3 block 128x128, routed experts stored as MXFP4 (store_dtype=mxfp4)",',
    '    "vllm_image": "vllm/vllm-openai:mimov25-cu129 @ sha256:39b1392da77b36187fde12858cbd1f9e2a9a0732b804b82d871a3446b28031e6",',
    '    "sglang_image": "lmsysorg/sglang:latest (52 GB, pulled 2026-09-22)",',
    '    "patch": "patches/swa_memory_pool.py - SWAKVPool.start_layer from rank-sliced layers_mapping (PP KeyError fix)",',
    '    "topology_tested": "PP4 (all four cards); PP3 measured memory-infeasible",',
    '    "attention_backend": "triton (sinks supported); vLLM path is FA3-pinned (SM90+)",',
    '    "cards": "4x CMP 170HX (SM80, 64 GiB each), Gen1 links x8/x1/x16/x16, 180 W cap, no NVLink, no P2P",',
    '    "power_cap": "180 W bench cap (matches the 2026-09-05 GLM protocol)",',
    '    "measured_utc": None,',
    "}",
    "for k, v in pins.items():",
    '    print(f"{k:20s} {v if v is not None else chr(39)+chr(39)}")',
])
cells.append(code(pins_src))

cells.append(md("\n".join([
    "## 2. The compatibility chain (all measured)",
    "",
    "### 2.1 vLLM official image: blocked at attention init, before any weight loads",
    "",
    "1. **A1** — the checkpoint trains attention sinks into its SWA layers",
    "   (`add_swa_attention_sink_bias: true`). The FlashAttention backend asserts",
    "   `flash_attn_supports_sinks()` — FlashAttention 3, SM90+ only.",
    "2. **A2** — `VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1` cannot escape:",
    "   `mimo_v2.py` hard-pins `FlashAttentionDiffKVBackend` whenever",
    "   `v_head_dim != head_dim` (QK 192 vs V 128 — always true for this arch), and",
    "   that backend subclasses the FA backend and inherits the same assert.",
    "",
    "**Classified: the official vLLM path cannot serve MiMo-V2.6 on Ampere with correct",
    "semantics.** The import probe was otherwise green (`MiMoV2ForCausalLM` registered,",
    "`fp8_per_block`/`mxfp4` quant methods present, device cap (8,0)).",
    "",
    "### 2.2 SGLang: three bugs deep, each receipted",
    "",
    "SGLang natively detects the `MiMo-V2 mxfp4 routed-expert layout`, plumbs attention",
    "sinks through its Triton backend, and ships `mimo_v2_nextn.py` for the shipped",
    "DFlash/MTP drafter — the right architecture. What we hit:",
    "",
    "1. **A4a — PP KeyError (fixed, `patches/swa_memory_pool.py`)**:",
    "   `TritonAttnBackend` reads `token_to_kv_pool.get_value_buffer(start_layer)`, but",
    "   `SWAKVPool` hardcoded `start_layer = 0` while PP ranks hold rank-sliced layer-id",
    "   lists — every rank with `start_layer > 0` raised KeyError. One-line fix: derive",
    "   `start_layer` from `min(layers_mapping)`.",
    "2. **A3 — PP3 is memory-infeasible**: 161 GiB / 3 = 53.7 GiB/rank of weights plus",
    "   loader transients OOMed at 62.0/63.5 GiB (the last rank also carries the",
    "   lm_head). PP3's one-fewer-hop advantage is unreachable on 64 GiB cards.",
    "3. **A4b/A5 — Triton fused-MoE vs mxfp4-packed experts**: capture dies in",
    "   `fused_experts_impl` asserting `hidden_states.shape[1] == w1.shape[2]`. The",
    "   mxfp4-packed expert weights are consumed upstream by the deep_gemm runner",
    "   (SM90+); the SM80 fallback Triton runner cannot read packed weights. No",
    "   dequant-at-load switch exists for MiMo (only DSV4 has one). A5 tests whether",
    "   eager mode passes the same call.",
])))

attempts_src = "\n".join([
    "attempts = receipt('runtime_attempts.json')",
    "rows = []",
    "for a in attempts['attempts']:",
    '    rows.append([a["id"], a.get("topology", "-"), a["result"][:60], a["error"][:70]])',
    'render_table(["Attempt", "Topology", "Result", "Error"], rows)',
])
cells.append(code(attempts_src))

cells.append(md("\n".join([
    "### 2.3 A5 — the decisive eager-mode run",
    "",
    "`--disable-cuda-graph` removes the capture path entirely. Result: the schedulers",
    "hit a raw CUDA fault in the same fused-MoE forward during warmup (coredump",
    "attempted, SIGQUIT) — no shape guard trips in eager because the kernel simply",
    "reads packed mxfp4 weights as fp8 and faults on memory. The incompatibility is",
    "static, not capture-specific.",
    "",
    "**What would change this verdict:** a load-time expert repack (mxfp4 → fp8",
    "W8A8-block or bf16) feeding the existing Triton runner — the same class of repack",
    "this club already ran for DeepSeek via autoround W4A16 — or an upstream Ampere",
    "mxfp4 kernel. Neither exists in any official image as of this date.",
])))

reproduce = "\n".join([
    "## 3. Reproduce",
    "",
    "### Download and verify the weights",
    "",
    "```bash",
    "pip install -U huggingface_hub",
    "hf download XiaomiMiMo/MiMo-V2.6-Flash-RL --local-dir /library/models/mimo-v2.6-flash-rl",
    "python3 results/" + EXP + "/tools/verify_checkpoint.py /library/models/mimo-v2.6-flash-rl",
    "```",
    "",
    "### Launch (SGLang, PP4, Triton attention, PP patch applied)",
    "",
    "```bash",
    "docker run --gpus all --shm-size 32g \\",
    "  -v /library/models:/models \\",
    "  -v $PWD/results/" + EXP + "/patches/swa_memory_pool.py:/sgl-workspace/sglang/python/sglang/srt/mem_cache/swa_memory_pool.py:ro \\",
    "  -p 8000:8000 lmsysorg/sglang:latest \\",
    "  python3 -m sglang.launch_server \\",
    "    --model-path /models/mimo-v2.6-flash-rl \\",
    "    --served-model-name mimo-v2.6-flash \\",
    "    --pipeline-parallel-size 4 \\",
    "    --trust-remote-code \\",
    "    --mem-fraction-static 0.88 \\",
    "    --context-length 32768 \\",
    "    --attention-backend triton \\",
    "    --reasoning-parser mimo \\",
    "    --host 0.0.0.0 --port 8000",
    "```",
    "",
    "PP, never TP: no NVLink, no P2P, Gen1 links; TP measured 6.6x worse on this fabric",
    "in the 2026-09 GLM lane. The patch mount is mandatory on multi-card runs until an",
    "upstream PP fix lands.",
    "",
    "### First request",
    "",
    "```bash",
    "curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{",
    '  "model": "mimo-v2.6-flash",',
    '  "messages": [{"role": "user", "content": "Edit me before sending."}],',
    '  "max_tokens": 256,',
    '  "temperature": 0',
    "}' | jq '{content: .choices[0].message.content, usage: .usage}'",
    "```",
    "",
    "### Speculative decode (blocked behind a working base server)",
    "",
    "The checkpoint ships a DFlash-style drafter (`dflash/`, 5 SWA layers, 7-token",
    "prediction) plus a 3-layer MTP head (`model_mtp.safetensors`). This sglang build",
    "carries `mimo_v2_nextn.py` and a registered `dflash` speculative method with a",
    "matching `DFlashQwen3ForCausalLM` draft arch — the k=7 recipe is the first",
    "follow-up once the MoE runner question resolves.",
    "",
    "### Attribution",
    "",
    "- Model: Xiaomi MiMo team, MiMo-V2.6 release (MIT), 2026-09-22.",
    "- PP-on-170HX serving shape and protocol: this club's 2026-08/09 DeepSeek-V4-Flash",
    "  and GLM-5.3-Flash lanes; community 4x 170HX DeepSeek recipe for PP-not-TP.",
    "- Power cap, QC gates: club CMP 170HX serving strategy notes.",
    "- PP3/PP4 topology question from the PR thread: answered in the attempt table.",
])
cells.append(md(reproduce))

cells.append(md("\n".join([
    "## 4. Appendix",
    "",
    "### Boot-time reference (PP4, this run)",
    "",
    "- Weight load: 2,199 s wall (NFS + mxfp4 unpack), 46.9-50.5 GiB per rank.",
    "- Full-VRAM QC: 4/4 PWRBRK# Not Active, zero Xid, links Gen1 x8/x1/x16/x16.",
    "- Runtime attempts ledger: `receipts/runtime_attempts.json`.",
    "- PP patch: `patches/swa_memory_pool.py` (applied via read-only bind mount).",
])))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
path = REPO + "/notebooks/" + EXP + ".ipynb"
with open(path, "w") as f:
    json.dump(nb, f, indent=1)
print("notebook rewritten:", path, len(cells), "cells")
