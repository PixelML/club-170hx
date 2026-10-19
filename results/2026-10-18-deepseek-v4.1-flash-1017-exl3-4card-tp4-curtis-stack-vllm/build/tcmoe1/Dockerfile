# dsv41-1017-exl3:sm80-tcmoe1 -- TC-MoE port on top of our Oct 17 serving image (Pixel ML, Oct 18 2026).
# Credits: tensor-core MoE hook/row dispatch/DSV4.1 shapes by Curtis White (@CurtisAccelerate),
#   github.com/CurtisAccelerate/DeepSeekv41-Flash-EXL3-TC-MoE @ cd83ade14c43a42babad39895d75bdd0e3453d49 (Apache-2.0; NOTICE.curtis.md);
#   kernels ported by him from Morrowmake/vllm-cmp170hx @ 18305948 (Apache-2.0); EXL3 codec + exl3_moe_coop: turboderp
#   exllamav3 @ d3739fd3 (MIT); TP-align/rotate/coop routing: vcruz305/vllm-exl3 PRs #27 #36 #40 (@kaka86mm, @fattchris,
#   @vcruz305; AGPL-3.0); base image: 0xSero dsv41-flash-offload @2e9da5e (MIT) on lazymio/wtdcode vllm-backport, Tokha233 sm80 patches.
# BASE must be dsv41-1017-exl3:sm80-v3p1 = image id sha256:67c73854bc5315082f6b23aaeef281c9cc4e7fa7aaddd19dbfb2452a9031eabd
# (checked by build.sh before and after the build).
ARG BASE=dsv41-1017-exl3:sm80-v3p1
FROM ${BASE}
COPY . /opt/dsv41-tcmoe/src
ARG MAX_JOBS=8
RUN set -eux; cd /opt/dsv41-tcmoe/src && MAX_JOBS=${MAX_JOBS} python3 build_coop.py /usr/local/lib/python3.12/dist-packages \
 && python3 -c "import torch, exllamav3_coop_ext as c; print('coop ok', c.exl3_moe_coop)"
RUN set -eux; cd /opt/dsv41-tcmoe/src && MAX_JOBS=${MAX_JOBS} python3 tc_moe/build_standalone.py --out /opt/dsv41-tc-moe --build-dir /tmp/tc_build \
 && python3 -c "import torch; torch.ops.load_library('/opt/dsv41-tc-moe/_dsv41_tc_moe_C.abi3.so'); print('tc ws v2', torch.ops._dsv41_tc_moe_C.moe_workspace_size_v2(64,5120,640,6,384))" \
 && rm -rf /tmp/tc_build
RUN set -eux; SITE=/usr/local/lib/python3.12/dist-packages; cd /opt/dsv41-tcmoe/src \
 && cp exl3_tc_hook.py /opt/dsv41-tc-moe/exl3_tc_hook.py \
 && cp $SITE/vllm_exl3/exl3.py /opt/dsv41-tcmoe/exl3.v3p1.py \
 && cp exl3.py $SITE/vllm_exl3/exl3.py \
 && python3 -c "import ast; ast.parse(open('$SITE/vllm_exl3/exl3.py').read())" \
 && sha256sum /opt/dsv41-tc-moe/* $SITE/exllamav3_coop_ext.so $SITE/vllm_exl3/exl3.py /opt/dsv41-tcmoe/exl3.v3p1.py > /opt/dsv41-tcmoe/SHA256SUMS
ENV DSV41_TC_HOOK=/opt/dsv41-tc-moe/exl3_tc_hook.py VLLM_EXL3_TC_MOE_SO=/opt/dsv41-tc-moe/_dsv41_tc_moe_C.abi3.so
