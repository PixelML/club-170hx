#!/bin/bash
# matrix-curtis-1017.sh ARM... -- run bench-curtis-1017.sh for each arm in order (one boot per arm), log to /root/bench/matrix-cu.log.
# Start detached inside the bench container:
#   systemd-run --unit=dsv41-cu --collect /root/dsv41-tp4/matrix-curtis-1017.sh tcep epcoop tpcoop tpcoop-cf tcfull-cf tcfullhar-cf
# Watch: tail /root/bench/matrix-cu.log   Stop: systemctl stop dsv41-cu; docker rm -f dsv41-1017
# A failed arm is logged and the next arm runs. Arms with a DONE file are skipped (FORCE=1 to redo).
cd /root/dsv41-tp4
mkdir -p /root/bench
for a in "$@"; do
  echo "[$(date '+%F %T')] === $a" >> /root/bench/matrix-cu.log
  ./bench-curtis-1017.sh "$a" >> /root/bench/matrix-cu.log 2>&1 || echo "[$(date '+%F %T')] === $a FAILED rc=$?" >> /root/bench/matrix-cu.log
  docker rm -f dsv41-1017 >/dev/null 2>&1 || true   # never let a failed arm hold the GPUs for the next one
done
echo "[$(date '+%F %T')] === matrix done: $*" >> /root/bench/matrix-cu.log
