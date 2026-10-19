#!/usr/bin/env python3
"""Overlay view of the 1017 routed-experts-only EXL3 pack for the 4x CMP 170HX TP4+EP4 stack. CPU only.

Input pack: PixelML/DeepSeek-V4.1-Flash-1017-EXL3-3bpw (made with sfxnz's tools/quantize_experts_exl3.py,
github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark @ 3de914682d0880c8e478712a60ada930339c22e4):
routed experts = EXL3 K3 MUL1 (layers.L.ffn.experts.E.w{1,2,3}.{trellis,suh,svh,mul1}), everything else
as stored by DeepSeek (MXFP8 + ue8m0 scales, BF16), quantization_config = sfxnz form
(quant_method exl3, non_routed_quantization deepseek_v4_fp8, mtp_experts source, weight_block_size).

Unlike 0xSero's pack/prepare_pack.py (written for Mia-AiLab's all-EXL3 pack) nothing is dequantized:
no wo_a / compressor / indexer expansion, no non_routed_exl3 table, no bf16_as_stored policy.
This script only (1) symlinks every file, (2) points the Engram shards 47-48 at --engram (FP8 from
deepseek-ai/DeepSeek-V4.1-Flash-1017 @ 1bae0e1088993b9b82de08f59de5be5a34e42170, or the int4 rebuild by
engram_int4.py, same tensor names), (3) rewrites config.json for the text-only registry entry
(DeepseekV41LLMForCausalLM, added by 0xSero runtime/patch_runtime.py) and (4) drops vision keys from the index.
The source pack is never edited.

usage: python3 -I prepare_1017.py --source PACK --engram ENGRAM_DIR --out OUT
"""
import argparse
import json
import os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--source", type=Path, required=True)
ap.add_argument("--engram", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
a.out.mkdir(exist_ok=False)

idx = json.loads((a.source / "model.safetensors.index.json").read_text())
wm = idx["weight_map"]
cfg = json.loads((a.source / "config.json").read_text())
q = cfg.get("quantization_config") or {}

# Fail closed if this is not the expected pack form.
assert q.get("quant_method") == "exl3", q
assert (q.get("non_routed_quantization") or {}).get("quant_method") == "deepseek_v4_fp8", q
assert "non_routed_exl3" not in q, "all-EXL3 (Mia) pack: use 0xSero pack/prepare_pack.py instead"
trellis = [k for k in wm if k.endswith(".trellis")]
assert trellis and all(".ffn.experts." in k for k in trellis), "EXL3 tensors outside routed experts"
assert any(k.endswith(".wo_a.scale") for k in wm), "expected FP8 wo_a with ue8m0 scale"
if q.get("weight_block_size") is None:  # build1017/exl3.py also reads it from non_routed_quantization
    q["weight_block_size"] = q["non_routed_quantization"]["weight_block_size"]

engram = {k: v for k, v in wm.items() if ".engram.embed." in k}
assert len(engram) == 4, engram
engram_files = sorted(set(engram.values()))
for fn in engram_files:
    assert (a.engram / fn).is_file(), f"missing Engram shard {a.engram / fn}"

skip = {"config.json", "model.safetensors.index.json"} | set(engram_files)
for f in sorted(a.source.iterdir()):
    if f.name in skip:
        continue
    (a.out / f.name).symlink_to(os.path.relpath(f.resolve(), a.out))
for fn in engram_files:
    (a.out / fn).symlink_to(os.path.relpath((a.engram / fn).resolve(), a.out))

cfg["architectures"] = ["DeepseekV41LLMForCausalLM"]
cfg["vision_config"] = {}
cfg["quantization_config"] = q
if "text_config" in cfg:
    cfg["text_config"]["quantization_config"] = q
(a.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

wm2 = {k: v for k, v in wm.items() if not k.startswith(("vision.", "aligner.", "image_"))}
(a.out / "model.safetensors.index.json").write_text(
    json.dumps({"metadata": idx.get("metadata", {}), "weight_map": wm2}, indent=2) + "\n"
)
receipt = {
    "source": str(a.source),
    "engram": str(a.engram),
    "exl3_tensors": len(trellis),
    "weight_names": len(wm2),
    "dropped_vision": len(wm) - len(wm2),
    "engram_files": engram_files,
    "quantization_config": q,
}
(a.out / "overlay-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({k: v for k, v in receipt.items() if k != "quantization_config"}, indent=2))
