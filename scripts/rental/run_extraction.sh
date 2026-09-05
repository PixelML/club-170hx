#!/usr/bin/env bash
# Direct-exec translation of the drafter lane's slice-C (2M token) PP4 extraction command
# (seanphan/pixelml#108). Only run this if >=4 spare cards exist AND the time budget allows:
# measured 120 tok/s on 4x170HX PP4 -> 2,000,000/120 =~ 4h40m plus ~15-20min weight load.
# Requires the 178 GB AWQ checkpoint at /model — download it first with run_extraction.sh
# download-only, check the ETA gate, THEN run extraction (abort extraction, not training,
# if checkpoint ETA > 60 min).
set -euo pipefail

MODE="${1:?usage: run_extraction.sh [download|eta-check|extract]}"

MODEL_DIR=/model
CKPT_REPO="wtdcode/GLM-5.3-Flash-AWQ-W4A16"
CKPT_BYTES_EXPECTED=190843146533   # 178 GiB
DATA=/data
TOOLS=/data/tools
OUT=/out/sliceC
RAW=/raw
LOGDIR=/workspace/logs/extraction
mkdir -p "$LOGDIR" "$OUT" "$RAW" "$MODEL_DIR"

case "$MODE" in
  eta-check)
    echo "[extract] downloading first shard only to estimate rate..."
    t0=$(date +%s)
    hf download "$CKPT_REPO" --local-dir "$MODEL_DIR" --max-workers 16 \
      --include "*.json" --include "*00001-of-*" 2>&1 | tee "$LOGDIR/eta-shard.log"
    t1=$(date +%s)
    shard_bytes=$(find "$MODEL_DIR" -type f -exec stat -c '%s' {} + | awk '{s+=$1} END {print s+0}')
    elapsed=$((t1 - t0))
    rate=$(python3 -c "print(${shard_bytes}/${elapsed}) if ${elapsed} else print(0)")
    eta_min=$(python3 -c "print(round((${CKPT_BYTES_EXPECTED}-${shard_bytes})/max(${rate},1)/60,1))")
    echo "[extract] shard: ${shard_bytes} bytes in ${elapsed}s => ${rate} B/s, full-download ETA ~${eta_min} min"
    echo "${eta_min}" > "$LOGDIR/eta-minutes.txt"
    if (( $(python3 -c "print(1 if ${eta_min} > 60 else 0)") )); then
      echo "[extract] ABORT: ETA ${eta_min} min > 60 min cap. Do NOT proceed with slice C."
      echo "[extract] Training continues regardless — this only gates the extraction lane."
      exit 1
    fi
    echo "[extract] ETA within 60 min cap — proceed to 'download' then 'extract'"
    ;;
  download)
    t0=$(date +%s)
    hf download "$CKPT_REPO" --local-dir "$MODEL_DIR" --max-workers 16 | tee "$LOGDIR/download.log"
    t1=$(date +%s)
    actual=$(find "$MODEL_DIR" -type f -exec stat -c '%s' {} + | awk '{s+=$1} END {print s+0}')
    echo "[extract] download took $((t1-t0))s, ${actual} bytes (expected ${CKPT_BYTES_EXPECTED})"
    [[ "$actual" -ge "$((CKPT_BYTES_EXPECTED - 1048576))" ]] || { echo "[extract] ABORT: checkpoint incomplete"; exit 1; }
    ;;
  extract)
    echo "[extract] CORPUS: point --corpus-jsonl at /data/corpus.jsonl if present, else this pulls"
    echo "[extract] ultrachat_200k+tulu-3-sft-mixture which needs 'datasets' + network (not in image)."
    CORPUS_ARGS=()
    if [[ -f "$DATA/corpus.jsonl" ]]; then
      CORPUS_ARGS=(--corpus-jsonl "$DATA/corpus.jsonl")
    fi
    PYTHONPATH="$TOOLS" python3 "$TOOLS/extract_hidden_states.py" \
      --model "$MODEL_DIR" --out "$OUT" --raw-dir "$RAW" \
      --tokens 2000000 --tp 4 --max-len 4096 --shard-tokens 50000 \
      --no-batch --resume "${CORPUS_ARGS[@]}" \
      2>&1 | tee "$LOGDIR/extract.log"
    echo "[extract] preflight gate check (should already have printed and been eyeballed live):"
    grep -c "drain/pack test OK" "$LOGDIR/extract.log" || true
    grep -c "batched split/verify OK" "$LOGDIR/extract.log" || true
    grep -c "segment resolution OK" "$LOGDIR/extract.log" || true
    grep -c "\[preflight\] ok" "$LOGDIR/extract.log" || true
    echo "[extract] hook check: must NOT be 'hooked 0 layers' or 'skipped-nonzero-rank' on every rank:"
    grep "hooked .* layers" "$LOGDIR/extract.log" || true
    echo "[extract] five exit checks:"
    python3 "$TOOLS/verify_manifest.py" "$OUT" 1800000
    ls "$OUT"/shard-*.npz 2>/dev/null | wc -l
    du -sh "$OUT"
    grep -c "id verification FAILED" "$LOGDIR/extract.log" || true
    grep -c "skip\]" "$LOGDIR/extract.log" || true
    echo "[extract] MUST confirm: aux_tap == hc_post-materialized+stream-mean, aux_layers == [5,14,24,33,42],"
    echo "[extract]   tokens >= 1,800,000, ~40 shards ~82GB, zero FAILED, zero skips."
    ;;
  *) echo "usage: run_extraction.sh [eta-check|download|extract]"; exit 1 ;;
esac
