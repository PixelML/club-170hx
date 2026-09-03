#!/usr/bin/env bash
# Re-apply the club-170hx 180 W cap. Run after any nvidia module reload; the cap does not survive rmmod/modprobe or a reboot.
set -euo pipefail
sudo -n nvidia-smi -pl "${1:-180}"
nvidia-smi --query-gpu=index,power.limit --format=csv,noheader
