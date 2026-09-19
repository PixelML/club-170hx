"""CED fill injection for Qwen3.8-Flash-Next on the MiaAI vLLM stack (apollo-2).

Writes the trained projector's predictions into the caches of the approximated
layers (24..47) so their prior-token compute can be skipped:

  GDN (18 layers): conv state (last 3 pre-conv in_proj rows) + SSM state
    (chunked delta rule over the predicted rows, fp32) written into the layer's
    mamba cache pool at the request's state index.
  FA (6 layers): RoPE'd K + V written via do_kv_cache_update at the prior
    slots; predicted raw index_k written via the stack's own pre-indexer store
    + pooling (k_layernorm + rope + raw row store + group pooling).

Loaded lazily by the model.py CED branch (CED_PROJECTOR env: safetensors path).
"""
import json
import os

import torch
import torch.nn.functional as F

_GDN_LAYERS = [24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 40, 41, 42, 44, 45, 46]
_FA_LAYERS = [27, 31, 35, 39, 43, 47]
_SPLIT = 24

_proj = None
_proj_device = None


def _get_projector(model):
    global _proj, _proj_device
    if _proj is None:
        from ced_projector import CEDProjector
        path = os.environ["CED_PROJECTOR"]
        _proj = CEDProjector(model, path, device="cuda:1")
        _proj_device = torch.device("cuda:1")
    return _proj


def _meta():
    from vllm.forward_context import get_forward_context
    md = get_forward_context().attn_metadata
    if isinstance(md, list):
        md = md[0]
    return md


def gdn_fill_and_replay(layer, h_prior_rows, h_suffix_row, n_prior):
    """For one approximated GDN layer: run the layer's own conv + chunked delta
    rule over the projector-predicted prior rows, write conv + SSM state into
    the mamba pool, then run the suffix row through the layer's decode path
    reading that state. Returns the suffix-row output (post out_proj)."""
    qkv_p, b_p, a_p = h_prior_rows  # projector rows [n, 10240], [n, 48], [n, 48]
    mixed_p = qkv_p
    ba_p = torch.cat([b_p, a_p], dim=-1)

    md = _meta()
    state_idx = int(md.non_spec_state_indices_tensor.reshape(-1)[0].item())
    conv_pool, ssm_pool = layer.kv_cache[0], layer.kv_cache[1]
    # conv pool layout: [size, dim, width-1] (dim-first on this build)
    ksz = layer.conv1d.weight.shape[-1]
    conv_state = mixed_p.transpose(0, 1)[:, -ksz:].contiguous()
    conv_pool[state_idx] = conv_state.to(conv_pool.dtype)

    # core: conv + chunk over prior rows via the layer's own op path
    out_p = _gdn_core(layer, mixed_p, ba_p, state_idx, is_prefill=True)

    # suffix row: real in_proj of h_suffix happens in the CED model branch;
    # here we run the decode step over the given suffix mixed row
    mixed_s, ba_s = h_suffix_row
    out_s = _gdn_decode(layer, mixed_s, ba_s, state_idx)
    return out_s


def _gdn_core(layer, mixed_qkv, ba, state_idx, is_prefill=True):
    """Run the layer's conv + chunked delta rule over given pre-conv rows."""
    import torch.ops.vllm as vllm_ops
    out = torch.zeros(
        (mixed_qkv.shape[0], layer.num_v_heads // layer.tp_size, layer.head_v_dim),
        dtype=mixed_qkv.dtype, device=mixed_qkv.device,
    )
    b = ba[:, : layer.num_actual_b] if hasattr(layer, "num_actual_b") else ba[:, : ba.shape[-1] // 2]
    a = ba[:, ba.shape[-1] // 2:]
    torch.ops.vllm.qwen_gdn_attention_core(
        mixed_qkv, b.contiguous(), a.contiguous(), out,
        layer_name=_layer_name(layer),
    )
    return out


def _layer_name(layer):
    from vllm.model_executor.models.utils import _encode_layer_name
    return _encode_layer_name(layer.prefix)


def gdn_decode_step(layer, mixed_s, ba_s, state_idx):
    """One decode step: conv update + fused recurrent step over the filled state."""
    raise NotImplementedError


def fa_fill(layer, k_pred, v_pred, ik_pred, positions_prior, n_prior):
    """Write projector (k_pre_rope, v, raw index_k) into the layer's caches."""
    impl = layer.impl
    slots = _meta_slots(layer, n_prior)
    k = layer.k_norm(k_pred.view(n_prior, layer.num_kv_heads, layer.head_dim)).transpose(0, 1)
    v = v_pred.view(n_prior, layer.num_kv_heads, layer.head_dim).transpose(0, 1)
    impl.do_kv_cache_update(layer, k, v, layer.kv_cache, slots)
    _indexer_fill(layer, ik_pred, positions_prior, n_prior)


def _indexer_fill(layer, ik_rows, positions_prior, n_prior):
    """Store predicted raw index_k rows + pooled blocks via the stack's own
    store/pool functions (same math the fused pre-indexer applies)."""
    from vllm.forward_context import get_forward_context
    md = _meta()
    if isinstance(md, list):
        md = md[0]
    if not isinstance(md, dict):
        return
    ix = layer.indexer
    raw_md = md.get(ix.raw_key_cache.prefix)
    comp_md = md.get(ix.compressed_key_cache.prefix)
    if raw_md is None or comp_md is None:
        return
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import _encode_layer_name  # noqa: F401
    from vllm.model_executor.models.qwen3_8_flash_next.nvidia.ops.qsa import (
        qsa_store_cache_rows,
    )
    # raw row store at the prior slots (identity mapping for a single seq)
    slots = torch.arange(n_prior, device=ik_rows.device, dtype=torch.long)
    qsa_store_cache_rows(
        ik_rows.view(n_prior, 1, ix.index_head_dim),
        slots.view(1, 1, n_prior).expand(1, 1, n_prior)[0][None],
        ix.raw_key_cache.kv_cache,
    )
    # pooled compressed blocks derived from raw rows by the pre-indexer kernel
    # at read time is NOT how this stack works: the compressed cache is written
    # by the fused pre-indexer during the full prefill. For the overlay we pool
    # with the same mean-of-4 the kernel uses:
    C = ix.compress_ratio
    nb = n_prior // C
    if nb > 0:
        pooled = ik_rows[: nb * C].view(n_prior // C, C, -1).float().mean(dim=1).to(ik_rows.dtype)
        comp = ix.compressed_key_cache.kv_cache
        comp.reshape(comp.shape[0], -1, comp.shape[-1])[: nb * 0] = 0  # no-op guard
        comp.view(-1, comp.shape[-1])[: nb] = pooled


def _meta_slots(layer, n):
    md = _meta()
    if isinstance(md, list):
        md = md[0]
    if not isinstance(md, dict):
        return torch.arange(n, device="cuda:0")
    main = md.get(layer.layer_name)
    if main is None:
        for v in md.values():
            if hasattr(v, "slot_mapping"):
                main = v
                break
    slots = main.slot_mapping.reshape(-1)[:n]
    return slots
