#!/usr/bin/env bash
# Re-check the Vast.ai market for CMP 170HX offers, sorted by PCIe bandwidth
# (gen x width, best first). Requires the vastai CLI to be installed and
# authenticated (vastai set api-key ...).
#
# IMPORTANT: verified=any is required — vastai hides unverified hosts by
# default, and the only CMP 170HX 8x hosts seen so far are unverified.
# Without this flag, 8x offers silently disappear from the listing.
#
# Usage: scripts/rental/find_offer.sh [min_gpus]
set -euo pipefail
MIN_GPUS="${1:-4}"

command -v vastai >/dev/null 2>&1 || { echo "vastai CLI not found; pip install vastai" >&2; exit 1; }

echo "# CMP 170HX offers, num_gpus>=${MIN_GPUS}, verified=any, sorted by PCIe bandwidth (desc)"
vastai search offers "gpu_name=CMP_170HX num_gpus>=${MIN_GPUS} verified=any" \
  -o 'pcie_bw-' \
  --raw \
| python3 -c '
import json, sys
rows = json.load(sys.stdin)
if not rows:
    print("(no offers currently match)")
    sys.exit(0)
cols = ["id", "num_gpus", "gpu_ram", "pci_gen", "gpu_lanes", "pcie_bw", "disk_space", "dph_total", "reliability2"]
labels = ["id", "gpus", "gpu_ram_gb", "pci_gen", "lanes", "pcie_bw_gbs", "disk_gb", "usd_per_hr", "reliability"]
print("  ".join(l.rjust(11) for l in labels))
for r in rows:
    vals = []
    for c in cols:
        v = r.get(c)
        if isinstance(v, float):
            v = round(v, 2)
        vals.append(str(v))
    print("  ".join(v.rjust(11) for v in vals))
'
