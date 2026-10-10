#!/bin/bash
# Run the engine with the standard paths; extra args pass through.
#   MODELS = folder holding UD-Q3_K_XL/ (hf download, see README)
#   LOCAL  = local (non-network) disk holding ple.bin from extract_ple.py
#   CACHE  = repacked-weight cache, written on the first run (~57 GiB)
: "${MODELS:?set MODELS}" "${LOCAL:?set LOCAL}" "${CACHE:=$LOCAL/weights.blob}"
M=$MODELS/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL
exec "$(dirname "$0")/fnx_mk" -m $M-00001-of-00003.gguf -m $M-00002-of-00003.gguf -m $M-00003-of-00003.gguf \
  --cache "$CACHE" --ple-file "$LOCAL/ple.bin" "$@"
