#!/usr/bin/env bash
set -uo pipefail

# Local diagnostic report. Review and redact its output before publication.

section() {
  printf '\n## %s\n' "$1"
}

section "Operating system"
uname -srmo
if [[ -r /etc/os-release ]]; then
  grep -E '^(NAME|VERSION)=' /etc/os-release
fi

section "NVIDIA module"
modinfo nvidia 2>/dev/null | grep -E '^(filename|version):' || true

section "GPU inventory"
nvidia-smi --query-gpu=index,name,memory.total,pstate,power.draw,power.limit,utilization.gpu,temperature.gpu,temperature.memory --format=csv 2>/dev/null || true

section "PCIe links"
nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv 2>/dev/null || true

section "Power brake"
nvidia-smi -q 2>/dev/null | grep -A1 'HW Power Brake Slowdown' || true

section "Recent NVIDIA/kernel errors"
dmesg --level=err,warn 2>/dev/null | grep -Ei 'NVRM|Xid|fallen off|RmInitAdapter|SEC2|AER|RCU' | tail -n 100 || true

printf '\nReview this report for private identifiers before sharing it.\n'
