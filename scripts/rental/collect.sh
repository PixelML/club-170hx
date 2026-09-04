#!/usr/bin/env bash
# Tar receipts + trimmed logs, ready to scp back and commit to the evidence
# branch. Does not push anything itself. Direct-exec: logs are already
# plain files under /workspace/logs (no docker logs to pull).
set -euo pipefail
RECEIPTS_ROOT="${RECEIPTS_ROOT:-/workspace/receipts}"
LOGDIR="${LOGDIR:-/workspace/logs}"
OUT_TAR="${OUT_TAR:-/workspace/glm53-8x170hx-receipts-$(date -u +%Y%m%dT%H%M%SZ).tar.gz}"

mkdir -p "$LOGDIR"
for f in "$LOGDIR"/*.log; do
  [[ -f "$f" ]] || continue
  tail -c 2000000 "$f" > "${f%.log}.trimmed.log" 2>/dev/null || true
done

tar -czf "$OUT_TAR" -C /workspace receipts logs
echo "[collect] wrote $OUT_TAR"
du -h "$OUT_TAR"
echo "[collect] scp back with:"
echo "  scp -P <port> root@<host>:$OUT_TAR ."
