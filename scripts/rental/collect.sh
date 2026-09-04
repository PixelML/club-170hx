#!/usr/bin/env bash
# Tar receipts + trimmed container logs, ready to scp back and commit to the
# evidence branch. Does not push anything itself.
set -euo pipefail
RECEIPTS_ROOT="${RECEIPTS_ROOT:-/workspace/receipts}"
OUT_TAR="${OUT_TAR:-/workspace/glm53-8x170hx-receipts-$(date -u +%Y%m%dT%H%M%SZ).tar.gz}"

mkdir -p /workspace/logs
for c in glm53-tp4 glm53-tp4a glm53-tp4b glm53-pp4 glm53-tp8; do
  docker logs "$c" > "/workspace/logs/${c}.log" 2>&1 || true
  tail -c 2000000 "/workspace/logs/${c}.log" > "/workspace/logs/${c}.trimmed.log" 2>/dev/null || true
done

tar -czf "$OUT_TAR" -C /workspace receipts logs
echo "[collect] wrote $OUT_TAR"
du -h "$OUT_TAR"
echo "[collect] scp back with:"
echo "  scp -P <port> root@<host>:$OUT_TAR ."
