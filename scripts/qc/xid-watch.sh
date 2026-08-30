#!/usr/bin/env bash
set -euo pipefail

if (( EUID != 0 )); then
  printf 'Run as root so kernel messages are visible: sudo %s\n' "$0" >&2
  exit 1
fi

printf 'Watching for NVIDIA Xid, PCIe, and reset errors. Press Ctrl-C to stop.\n'
dmesg --follow --human | grep --line-buffered -Ei 'NVRM|Xid|fallen off|RmInitAdapter|SEC2|AER|PCIe Bus Error'
