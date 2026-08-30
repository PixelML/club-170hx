#!/usr/bin/env bash
set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf 'nvidia-smi not found\n' >&2
  exit 1
fi

printf 'GPU, Bus ID, VRAM, Current Gen, Max Gen, Current Width, Max Width\n'
nvidia-smi \
  --query-gpu=index,pci.bus_id,memory.total,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max \
  --format=csv,noheader

printf '\nPower-brake state:\n'
nvidia-smi -q | grep -A1 'HW Power Brake Slowdown' || true

printf '\nInterpret links in context: virtualized CMP paths may negotiate below bare-metal capability.\n'
