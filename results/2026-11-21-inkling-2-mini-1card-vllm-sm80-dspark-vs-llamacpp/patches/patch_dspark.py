"""PixelML sm80 patch step 6: DSpark drafter KV write without the compiled
_C_cache_ops.reshape_and_cache_flash (no sm_80 kernel in this image). Usage: patch_dspark.py <dspark.py>"""
import sys

p = sys.argv[1]
s = open(p).read()
old = """        impl = cast(FlashAttentionImpl, self.attn.impl)
        impl.do_kv_cache_update(self.attn, k, v, self.attn.kv_cache, slot_mapping)
"""
new = """        if not current_platform.has_device_capability(90):  # sm80 patch: torch KV write
            kv = self.attn.kv_cache
            if kv.numel() == 0:
                return
            hd = self.head_dim
            key_cache, value_cache = kv.transpose(1, 2).split(hd, dim=-1)  # (B, N, H, D) views
            n = slot_mapping.shape[0]
            bs = key_cache.shape[1]
            slots = slot_mapping.to(torch.int64).clamp(min=0)  # padded (-1) -> null block 0
            blk, off = slots // bs, slots % bs
            heads = key_cache.shape[2]
            key_cache[blk, off] = k.reshape(-1, heads, hd)[:n].to(key_cache.dtype)
            value_cache[blk, off] = v.reshape(-1, heads, hd)[:n].to(value_cache.dtype)
            return
        impl = cast(FlashAttentionImpl, self.attn.impl)
        impl.do_kv_cache_update(self.attn, k, v, self.attn.kv_cache, slot_mapping)
"""
assert s.count(old) == 1, "anchor not found"
s = s.replace(old, new)
if "from vllm.platforms import current_platform" not in s:
    s = s.replace("import torch\n", "import torch\nfrom vllm.platforms import current_platform\n", 1)
open(p, "w").write(s)
print("patched", p)
