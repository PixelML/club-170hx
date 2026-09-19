# CED overlay design — vLLM `qwen3_8_flash_next` (apollo-2 stack)

Status: designed, not yet implemented. Source of truth: `apollo2_vllm_model.py`
(extracted from `vllm/vllm-openai:qwen38-flash-next`, vllm 0.1.dev20073+g8e685d198,
file `vllm/models/qwen3_8_flash_next/nvidia/model.py`, 1047 lines).

## Stack map (extracted source)

- `Qwen3_8FlashNextModel.forward` (model.py L464): embed → `repeat(1, hc_count)` →
  layer loop `for layer_idx, layer in islice(enumerate(self.layers), start_layer, end_layer)`
  (L491) → final `hyper_connection_mixer.combine_and_mix` → sample hidden.
- `Qwen3_8FlashNextDecoderLayer.forward` (L171 area): takes
  `(hidden_states, prev_block_output, prev_injection, positions, input_ids,
  query_start_loc, ngram_context)`; returns `(hidden_states, block_output, injection)`.
  **Delayed HC combine**: each layer consumes the previous layer's pending
  `block_output`/`injection` and emits its own pending pair; materialization happens via
  `mlp_hyper_connection.combine(...)` only at explicit points (deepstack, PP boundary,
  final mixer).
- Children per layer: `linear_attn` (`QwenGatedDeltaNetAttention`, vllm mamba layers —
  GDN), `self_attn` (`Qwen3_8FlashNextQSAAttention` — FA + lightning indexer), `mlp`
  (`Qwen3_8FlashNextSparseMoeBlock`), `ple` (layer 1 only).
- GDN/FA layer inventory in the approximated half (24..47): 18 GDN + 6 FA
  (module files: `nvidia/qsa.py`, `nvidia/ops/qsa.py`, `nvidia/ops/hc.py`;
  GDN base: `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`).
- GDN recurrent state lives in **vLLM V1's mamba cache pool** (`MambaSpec`,
  `MambaStateCopyFuncs`), scheduler-managed — not a plain KV tensor.

## Injection design

Wrap `Qwen3_8FlashNextModel.forward` (subclass or monkeypatch) with a CED mode,
active only when `enforce_eager` and the prompt fits one chunk (disable chunked prefill
and prefix caching for the A/B; `MAX_NUM_BATCHED_TOKENS >= T`):

1. Run the layer loop normally for `layer_idx in 0..23` (they write their own KV /
   consume PLE at layer 1). Keep the pending `(block_output, injection)` threading
   exactly as the stock loop does.
2. At `layer_idx == 24` (first approximated layer), freeze the boundary state:
   - `h24_priors = hidden_states[:, :-1]` (all but the last prompt row),
     `h24_last = hidden_states[:, -1:]`.
   - Per approximated layer, call the projector on `h24_priors` (single batched
     forward, fp32 → bf16): 18×(qkv,b,a) + 6×(k_pre,v,ik).
3. FA layers (6): apply RoPE analytically at the prior positions (vLLM `positions`
   tensor rows 0..n-2) to the predicted k, then write `(k, v)` into the layer's KV
   cache for slots 0..n-2 **through the same cache-op path the layer's attention
   forward uses** (QSA attention writes via its backend ops with the attention
   metadata's slot mapping — replicate that call with a synthetic metadata view), and
   write the predicted `index_k` rows into the indexer cache the same way.
4. GDN layers (18): from the predicted `qkv/b/a` rows, run the layer's own conv +
   `chunk_gated_delta_rule`-equivalent (the `QwenGatedDeltaNetAttention` internals)
   over rows 0..n-2 → `S_{n-2}` + conv state, and write both into the layer's mamba
   cache pool entry for the sequence (the `MambaStateCopyFuncs`/update path the layer
   uses between chunks).
5. Then run layers 24..47 **only on `h24_last`** (the standard per-layer call), each
   consuming its now-filled cache/state and appending its own last row; keep the
   pending-combine threading intact. Final mixer → sample hidden as stock.

Correctness anchors:
- Oracle mode (fills from a same-pass teacher capture instead of the projector) must
  reproduce the full prefill's last-row hidden state to bf16 noise — same discipline as
  the eager engine's gate.
- The eager-engine implementation of steps 2–4 (`engine.py::_fill_fa`,
  `_fill_gdn`, `_run_gdn_replay`) is the reference for the transforms; the vLLM delta
  is only the cache-write plumbing (paged KV slots + mamba pool) and the delayed-HC
  carry.

Known complications (ranked):
1. Mamba pool write API for a mid-request state replacement (V1 scheduler owns
   per-seq state lifetime).
2. QSA indexer cache write path (compressed keys) — replicate from `qsa.py`.
3. `MAX_NUM_BATCHED_TOKENS` chunking must be single-chunk for the overlay (prompt
   <= chunk width) or the fills must be staged per chunk.
4. MTP hidden buffer (`_mtp_hidden_buffer`) — MTP must stay off for the A/B.

## Bench protocol (matched stack)

- Server: apollo-2, same image, `--enforce-eager`, `--max-num-batched-tokens 8192`,
  `--max-model-len 8192`, MTP off, prefix caching off for the A/B runs.
- Grid: 512/1024/2048/6603 (token-trimmed prompts), median of 3, streaming TTFT +
  prompt tok/s, temp 0, max_tokens 1; identical client for baseline and CED arms.
- Receipts: `vllm-ced-ab.json` + the uplift verdict vs `vllm-baseline.json`
  (4,203 tok/s @6.5k baseline).
