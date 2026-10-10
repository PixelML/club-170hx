#!/bin/bash
# v2 launch line: main GGUF + MTP head; paths as environment variables
: "${MODELS:?set MODELS}" "${LOCAL:?set LOCAL}" "${CACHE:=$LOCAL/weights.blob}" "${CACHE2:=$LOCAL/weights_v2.blob}"
M=$MODELS/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL
exec "$(dirname "$0")/fnx_spec" -m $M-00001-of-00003.gguf -m $M-00002-of-00003.gguf -m $M-00003-of-00003.gguf \
  --mtp $MODELS/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf --cache "$CACHE" --cache2 "$CACHE2" --ple-file "$LOCAL/ple.bin" "$@"
