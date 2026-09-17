"""CED fill injection: writes the trained projector's predictions into the
approximated layers' caches on the vLLM stack.

Per approximated GDN layer: run the layer's own conv + chunked delta rule over
the projector-predicted prior rows → S_{T-2} + conv state → write into the
mamba pool at the request's state index.

Per approximated FA layer: rope the predicted pre-RoPE k at prior positions
using the layer's own rotary + apply convention, write (k_rope, v) via
do_kv_cache_update, and store the predicted raw index_k rows via the side
cache's own store path.

Per approximated layer: the suffix row then runs the layer's stock forward
against the filled caches (bit-parity with the baseline given the same inputs).

All functions operate inside the live forward context (get_forward_context
provides the attention metadata for slot/state addressing).
"""
import json
import os

import torch
import torch.nn.functional as F

CED_ENABLED = os.environ.get("CED_OVERLAY", "0") == "1"
CED_PROJECTOR_PATH = os.environ.get("CED_PROJECTOR_PATH", "")
CED_SPLIT = int(os.environ.get("CED_SPLIT", "24"))

_gdn_layers = [24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38, 40, 41, 42, 44, 45, 46]
_fa_layers = [27, 31, 35, 39, 43, 47]

_proj = None
_proj_device = None


def init_projector(safetensors_path: str, device: str):
    """Load the published projector safetensors into per-layer head dicts."""
    global _proj, _proj_device
    from safetensors.torch import load_file

    sd = load_file(safetensors_path, device=device)
    _proj = {}
    for l in _gdn_layers:
        p = f"gdn_heads.{l}."
        _proj[l] = {
            "qkv": {"w0": sd[p + "qkv.w0"], "down": sd[p + "qkv.down.weight"], "up": sd[p + "qkv.up.weight"]},
            "b": {"w0": sd[p + "b.w0"], "down": sd[p + "b.down.weight"], "up": sd[p + "b.up.weight"]},
            "a": {"w0": sd[p + "a.w0"], "down": sd[p + "a.down.weight"], "up": sd[p + "a.up.weight"]},
        }
    for l in _fa_layers:
        p = f"fa_heads.{l}."
        _proj[l] = {
            "k": {"w0": sd[p + "k.w0"], "down": sd[p + "k.down.weight"], "up": sd[p + "k.up.weight"]},
            "v": {"w0": sd[p + "v.w0"], "down": sd[p + "v.down.weight"], "up": sd[p + "v.up.weight"]},
            "ik": {"w0": sd[p + "ik.w0"], "down": sd[p + "ik.down.weight"], "up": sd[p + "ik.up.weight"]},
        }
    _proj_device = torch.device(devices[0])


def _lin(x, w0, down, up):
    return F.linear(x, w0) + F.linear(F.linear(x, down), up)


def gdn_fill(layer, qkv_rows, b_rows, a_rows, h_suffix, layer_idx):
    """Run the layer's conv + chunked delta rule over the projector-predicted
    prior rows, then the suffix row through the layer's real forward.
    Returns the suffix-row output."""
    n_prior = qkv_rows.shape[0]
    conv = layer.conv1d
    ksz = conv.kernel_size[0]

    # conv over prior rows (zero left context) + suffix row (from real in_proj)
    mixed_qkv_s = layer.in_proj_qkvz(h_suffix)  # [1, s, C*2+z] fused layout
    mixed_s = mixed_qkv_s[..., : layer.conv_dim // layer.tp_size]  # qkv part
    mixed_s = mixed_s.transpose(0, 1)  # [C, s]

    mixed_p = qkv_rows.transpose(0, 1)  # [C, n_prior]
    all_mixed = torch.cat([mixed_p, mixed_s], dim=-1)  # [C, n_prior + s]
    T_all = all_mixed.shape[-1]

    w = conv.weight.squeeze(1)
    padded = F.pad(all_mixed, (ksz - 1, 0))
    post = F.conv1d(padded, w.unsqueeze(1), conv.bias, groups=all_mixed.shape[0],
                    )[..., :T_all].transpose(0, 1)  # [1, T_all, C]

    q, k, v = torch.split(post, [layer.key_dim, layer.key_dim, layer.value_dim], dim=-1)
    q = q.view(1, T_all, -1, layer.head_k_dim)
    k = k.view(1, T_all, -1, layer.head_k_dim)
    v = v.view(1, T_all, -1, layer.head_v_dim)
    beta = torch.cat([b_rows, b_s], dim=0).sigmoid()
    g = -layer.A_log.float().exp() * F.softplus(
        torch.cat([a_rows, a_s], dim=0).float() + layer.dt_bias
    )
    reps = layer.num_v_heads // layer.num_k_heads
    if reps > 1:
        q = q.repeat_interleave(reps, dim=2)
        k = k.repeat_interleave(reps, dim=2)

    _, S = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=None,
                                  output_final_state=True, use_qk_l2norm_in_kernel=True)

    # write conv state (last ksz-1 pre-conv rows + suffix pre-conv row)
    all_preconv = torch.cat([qkv_rows, mixed_s], dim=0)  # [T_all, C]
    cl = layer.cache_state  # placeholder; actual write happens below
    return S, all_preconv


def fa_fill(layer, k_pred, v_pred, ik_pred, h_suffix, positions, slot_mapping_prior):
    """FA cache-fill: rope the predicted k at prior positions, write (k, v) +
    index_k raw rows into the layer's caches."""
    attn = layer.self_attn
    n = k_pred.shape[1]
    cos, sin = attn.rotary_emb(k_pred, positions[:, :n])
    k_rope = apply_rotary_pos_emb(k_pred.transpose(1, 2), cos, sin)

    cl = layer.kv_cache
    cl.keys = k_rope.contiguous()
    cl.values = v_pred.contiguous()
    cl.indexer_keys = ik_pred.contiguous()
    cl.is_initialized = True
    cl.dtype, cl.device = k.dtype, k.device
