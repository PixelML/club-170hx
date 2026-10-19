#!/bin/bash
# matrix-1017.sh ARM... -- run bench-1017.sh for each arm in order (one boot per arm), log to /root/bench/matrix.log.
# Start detached inside the bench container:  systemd-run --unit=dsv41-bench --collect /root/dsv41-tp4/matrix-1017.sh p2poff p2psys p2pauto car128k
# Watch: tail /root/bench/matrix.log   Stop: systemctl stop dsv41-bench; docker rm -f dsv41-1017
# A failed arm is logged and the next arm runs. Arms with a DONE file are skipped (FORCE=1 to redo).
cd /root/dsv41-tp4
mkdir -p /root/bench
for a in "$@"; do
  echo "[$(date '+%F %T')] === $a" >> /root/bench/matrix.log
  ./bench-1017.sh "$a" >> /root/bench/matrix.log 2>&1 || echo "[$(date '+%F %T')] === $a FAILED rc=$?" >> /root/bench/matrix.log
done
echo "[$(date '+%F %T')] === matrix done: $*" >> /root/bench/matrix.log
