#!/bin/bash
# make-sm80-patch.sh — run on the GPU host. Builds /work/patches/inkling2 from the image's
# vllm/models/inkling2 package with sm_80 (CMP 170HX) edits:
#   1. quant_config.get_min_capability 90 -> 80
#   2. Triton PDL removed: gdc_wait()/gdc_launch_dependents() -> no-op, launch_pdl=True -> False
#      (sm_80 has no griddepcontrol; without PDL, same-stream kernels already run in order)
#   4. linear.k_group_partials: Triton TMA path only on sm_90+ (sm_80 has no TMA; the failed
#      descriptor leaves a sticky 'no kernel image' error)
#   5. moe_align_block_size: pure-torch version (sm80_moe_align.py); _moe_C build has no sm_80 kernel
#   6. DSpark drafter KV write in torch (patch_dspark.py); _C_cache_ops.reshape_and_cache_flash has no sm_80 kernel
#   3. conv_mix: sm_80 takes the Triton short_conv + attn_res fallback (native conv_mix.cu has no sm_80 cubin)
set -e
IMG=vllm/vllm-openai@sha256:c2dc759f5984f69c1e2418c6af7c26eb756f5ed2a8934c768ae1253645d1449e
P=/work/patches
rm -rf $P/inkling2 $P/inkling2.orig; mkdir -p $P
cid=$(docker create $IMG)
docker cp $cid:/usr/local/lib/python3.14/dist-packages/vllm/models/inkling2 $P/inkling2.orig
docker rm $cid >/dev/null
cp -a $P/inkling2.orig $P/inkling2
find $P/inkling2 -name __pycache__ -prune -exec rm -rf {} +
sed -i 's/^\(\s*\)return 90$/\1return 80  # PixelML sm80 patch/' $P/inkling2/quant_config.py
for f in $(grep -rl -E 'gdc_wait|gdc_launch_dependents|launch_pdl=True' $P/inkling2 --include=*.py); do
  sed -i -E 's/tl\.extra\.cuda\.gdc_wait\(\)/pass  # sm80: no PDL/; s/tl\.extra\.cuda\.gdc_launch_dependents\(\)/pass  # sm80: no PDL/; s/launch_pdl=True/launch_pdl=False/' "$f"
done
sed -i 's/^\(\s*\)if current_platform.is_device_capability_family(120):$/\1if current_platform.is_device_capability_family(120) or not current_platform.has_device_capability(90):  # sm80 patch/' $P/inkling2/nvidia/ops/conv_mix.py
sed -i 's/^\(\s*\)if tokens >= TMA_MIN_TOKENS and not sm120:$/\1if tokens >= TMA_MIN_TOKENS and not sm120 and current_platform.has_device_capability(90):  # sm80 patch/' $P/inkling2/nvidia/ops/linear.py
cp /work/sm80_moe_align.py $P/inkling2/sm80_moe_align.py
sed -i 's/from vllm.model_executor.layers.fused_moe.moe_align_block_size import (/from vllm.models.inkling2.sm80_moe_align import (  # sm80 patch/' $P/inkling2/nvidia/ops/moe_sm90.py $P/inkling2/nvidia/ops/moe_humming.py
python3 /work/patch_dspark.py $P/inkling2/nvidia/dspark.py
diff -ru -N $P/inkling2.orig $P/inkling2 -x __pycache__ > $P/inkling2-sm80.diff || true
grep -c '^[-+][^-+]' $P/inkling2-sm80.diff
