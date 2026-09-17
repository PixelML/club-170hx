# CED overlay patch — Qwen3.8-Flash-Next on the MiaAI vLLM stack (apollo-2)
#
# Three surgical changes, applied on top of the MiaAI image:
#
# 1. qwen_gdn_linear_attn.py (GDN layer): forward_cuda gains an optional
#    `precomputed` kwarg — (mixed_qkv [T, 10240], b [T, 48], a [T, 48]) in the
#    checkpoint/eager layout, which equals the vLLM in_proj output layout for
#    qkv (the checkpoint's in_proj_qkv fuses [q|k|v] in the same order the
#    vLLM forward splits). When set, in_proj is skipped and the given rows flow
#    through conv + chunked delta rule + state writes via the layer's own
#    machinery (bit-parity with the normal path given the same rows).
#
# 2. model.py (Qwen3_8FlashNextModel.forward): CED branch. When
#    CED_OVERLAY=1 and the prompt fits one chunk:
#      a. run layers 0..23 over all T rows (stock);
#      b. boundary hidden h = hidden_states;
#      c. per approximated layer, projector predicts prior rows (0..T-2):
#         GDN -> run the patched forward_cuda over prior rows with the
#         precomputed rows (writes conv + S_{T-2} into the mamba pool via the
#         layer's own addressing; outputs discarded);
#         FA  -> run the layer's pre-indexer write for the predicted index_k
#         rows (k_layernorm + rope + raw-row store + pooled blocks via
#         qsa_compress_groups_with_ratio) and do_kv_cache_update for the
#         rope'd (k, v) fills over slots 0..T-2;
#      d. run layers 24..47 on the LAST row only (each reads its filled
#         cache/state; writes its own last-row entry);
#      e. final mixer -> logits.
#
# 3. ced_projector.py: standalone projector loader (mirrors
#    KVAProjector's head structure; loads the published safetensors).
#
# Deployment: docker cp the three files into the running container at the
# paths below, then docker restart vllm-fn-tp1. CED_OVERLAY=1 + the
# projector path go into the container env (compose or docker run -e).
