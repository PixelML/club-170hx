# DSV4.1-Flash-1017 routed-experts-only EXL3 pack (sfxnz quantize_experts_exl3.py form) on 4x CMP 170HX.
# Base = Oct 4 served image v3 (0xSero/dsv41-flash-offload 2e9da5e + Pixel ML TP4/EP4 + int4 Engram patches),
# recipe: results/2026-10-03-deepseek-v4.1-flash-exl3-4card-tp4ep4-dspark-vllm/build/.
FROM ghcr.io/pixelml/club-170hx@sha256:b8b7a1699bf272855a9b8c8b316f796b3ef917696302baf2a62c856707d69932
COPY exl3-1017.patch prepare_1017.py engram_int4.py /opt/dsv41-1017/
# vllm-exl3 (vcruz305, AGPL-3.0): Exl3Config exposes weight_block_size (MXFP8 scale names) + opt-in delegate attrs.
RUN F=/usr/local/lib/python3.12/dist-packages/vllm_exl3/exl3.py  && sha256sum $F  && patch --forward --no-backup-if-mismatch $F < /opt/dsv41-1017/exl3-1017.patch  && python3 -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" $F  && grep -q EXL3_DELEGATE_ATTRS $F && sha256sum $F
