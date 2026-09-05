#!/usr/bin/env bash
# Runs LOCALLY (on the orchestrating machine), not on the rental. Relays data from the
# internal source host to the rented Vast instance in two hops (source has no route to
# the rental; the orchestrating machine has ssh access to both).
#
# Required env (no defaults — do not hardcode private hosts into this file):
#   SPECDEC_SOURCE      ssh target for the internal source host, e.g. user@100.x.x.x
#   SPECDEC_DEST_HOST   ssh host/user for the rented instance, e.g. root@<vast-ip>
#   SPECDEC_DEST_PORT   ssh port for the rented instance, e.g. 41234
#
# Usage:
#   SPECDEC_SOURCE=user@<internal-host> \
#   SPECDEC_DEST_HOST=root@<vast-ip> SPECDEC_DEST_PORT=<port> \
#     scripts/rental/transfer_specdec.sh
set -euo pipefail

: "${SPECDEC_SOURCE:?set SPECDEC_SOURCE=user@host (internal source, not committed here)}"
: "${SPECDEC_DEST_HOST:?set SPECDEC_DEST_HOST=root@<vast-ip>}"
: "${SPECDEC_DEST_PORT:?set SPECDEC_DEST_PORT=<vast-ssh-port>}"

SRC_DATA_DIR=/library/models/specdec-data
DEST_SSH="ssh -o StrictHostKeyChecking=no -p ${SPECDEC_DEST_PORT}"
LOCAL_STAGE=$(mktemp -d)
trap 'rm -rf "$LOCAL_STAGE"' EXIT

rate_mb_s() { # rate_mb_s <bytes> <seconds>
  python3 -c "b=$1; s=$2 or 1; print(round(b/1e6/s,1))"
}

echo "[transfer] 1/5: 1 GB rate test, source -> local stage"
t0=$(date +%s)
ssh "$SPECDEC_SOURCE" "head -c 1073741824 /dev/urandom > /tmp/specdec-1g.bin"
scp -q "$SPECDEC_SOURCE:/tmp/specdec-1g.bin" "$LOCAL_STAGE/1g.bin"
t1=$(date +%s)
echo "[transfer] source->local: $((t1 - t0))s, $(rate_mb_s 1073741824 $((t1 - t0))) MB/s"
ssh "$SPECDEC_SOURCE" "rm -f /tmp/specdec-1g.bin"

echo "[transfer] 2/5: 1 GB rate test, local stage -> rental"
t0=$(date +%s)
scp -q -P "$SPECDEC_DEST_PORT" -o StrictHostKeyChecking=no \
  "$LOCAL_STAGE/1g.bin" "${SPECDEC_DEST_HOST}:/tmp/specdec-1g.bin"
t1=$(date +%s)
echo "[transfer] local->rental: $((t1 - t0))s, $(rate_mb_s 1073741824 $((t1 - t0))) MB/s"
ssh -p "$SPECDEC_DEST_PORT" -o StrictHostKeyChecking=no "$SPECDEC_DEST_HOST" "rm -f /tmp/specdec-1g.bin; mkdir -p /data"
rm -f "$LOCAL_STAGE/1g.bin"

echo "[transfer] 3/5: stage small files locally (tools tarball, target-shared, checksums)"
t0=$(date +%s)
rsync -az -e "ssh -o StrictHostKeyChecking=no" \
  "${SPECDEC_SOURCE}:${SRC_DATA_DIR}/specdec-tools-e03679f1.tar.gz" \
  "${SPECDEC_SOURCE}:${SRC_DATA_DIR}/target-shared.safetensors" \
  "${SPECDEC_SOURCE}:${SRC_DATA_DIR}/SHARED.sha256" \
  "${SPECDEC_SOURCE}:${SRC_DATA_DIR}/REFDRAFTER.sha256" \
  "$LOCAL_STAGE/"
mkdir -p "$LOCAL_STAGE/ref-drafter"
rsync -az -e "ssh -o StrictHostKeyChecking=no" \
  "${SPECDEC_SOURCE}:${SRC_DATA_DIR}/ref-drafter/" "$LOCAL_STAGE/ref-drafter/" 2>&1 || \
  echo "[transfer] WARN: ref-drafter/ not found at that path on source — locate it before training"
t1=$(date +%s)
echo "[transfer] small files staged in $((t1 - t0))s"

echo "[transfer] 4/5: sliceB (14 GB), source -> local stage"
t0=$(date +%s)
rsync -az --info=progress2 -e "ssh -o StrictHostKeyChecking=no" \
  "${SPECDEC_SOURCE}:${SRC_DATA_DIR}/sliceB" "$LOCAL_STAGE/"
t1=$(date +%s)
echo "[transfer] sliceB staged in $((t1 - t0))s, $(rate_mb_s 15032385536 $((t1 - t0))) MB/s"

echo "[transfer] 5/5: push staged data to rental /data"
t0=$(date +%s)
rsync -az --info=progress2 -e "$DEST_SSH" "$LOCAL_STAGE/" "${SPECDEC_DEST_HOST}:/data/"
t1=$(date +%s)
echo "[transfer] pushed to rental in $((t1 - t0))s"

echo "[transfer] unpack + verify on rental"
$DEST_SSH "$SPECDEC_DEST_HOST" '
  set -e
  cd /data && tar xzf specdec-tools-e03679f1.tar.gz
  test -d specdec-wt/tools/specdec && mkdir -p tools && cp -r specdec-wt/tools/specdec tools/ || true
  find /data -maxdepth 2 -iname "*.py" -path "*specdec*" | head -3
  sha256sum specdec-tools-e03679f1.tar.gz
  cd /data/sliceB && sha256sum -c SHA256SUMS --quiet && echo SLICEB_OK
  python3 /data/tools/verify_manifest.py /data/sliceB 400000
'
echo "[transfer] gate: the line above MUST read"
echo "[transfer]   tap hc_post-materialized+stream-mean tokens 455367 shards 9 files 9"
echo "[transfer]   OK"
echo "[transfer] If it differs, STOP — do not start training."
