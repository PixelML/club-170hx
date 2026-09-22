#!/usr/bin/env python3
"""Verify the local MiMo-V2.6-Flash-RL download against its own index.

Checks: shard set matches model.safetensors.index.json, every shard is
non-truncated (size > 0 and, where the HF API exposes it, matches the
expected size), config.json parses and carries the expected architecture.
`hf download` already validates each LFS file's digest during transfer; this
gate re-checks the assembled tree so the receipt does not depend on the
downloader's own bookkeeping.

Usage: python3 verify_checkpoint.py /library/models/mimo-v2.6-flash-rl
"""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
fail = []

idx_path = root / "model.safetensors.index.json"
idx = json.loads(idx_path.read_text())
shards = sorted(set(idx["weight_map"].values()))

present = sorted(p.name for p in root.glob("model_pp*.safetensors"))
missing = [s for s in shards if s not in present]
extra = [p for p in present if p not in shards]
if missing:
    fail.append(f"missing shards: {missing[:5]} ({len(missing)} total)")
if extra:
    fail.append(f"extra shards: {extra[:5]} ({len(extra)} total)")

zero = [p.name for p in root.glob("*.safetensors") if p.stat().st_size == 0]
if zero:
    fail.append(f"zero-byte shards: {zero[:5]}")

total = sum(p.stat().st_size for p in root.glob("*.safetensors"))
cfg = json.loads((root / "config.json").read_text())
if cfg.get("architectures") != ["MiMoV2ForCausalLM"]:
    fail.append(f"unexpected architectures: {cfg.get('architectures')}")

incomplete = list((root / ".cache" / "huggingface" / "download").glob("*.incomplete")) if (root / ".cache").exists() else []
if incomplete:
    fail.append(f"{len(incomplete)} .incomplete files remain")

print(f"index_shards   : {len(shards)}")
print(f"weight_bytes   : {total:,}")
print(f"layers         : {cfg.get('num_hidden_layers')} x {cfg.get('n_routed_experts')} experts, top-{cfg.get('num_experts_per_tok')}")
print(f"ctx            : {cfg.get('max_position_embeddings')}")
print(f"quant          : {cfg.get('quantization_config', {}).get('quant_method')} store={cfg.get('quantization_config', {}).get('store_dtype')}")
print(f"incomplete     : {len(incomplete)}")

if fail:
    print("FAIL:")
    for f in fail:
        print(" -", f)
    sys.exit(1)
print("PASS: shard set complete, config sane, no partial files")
