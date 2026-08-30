# Hardware guide

## What this card is

The CMP 170HX is a passive, compute-only NVIDIA mining card. The cards tested by this project enumerate as PCI device `10de:20c2`, expose CUDA compute capability 8.0, and report 65,536 MiB after the community unlock path is working.

That makes the card interesting for memory-heavy CUDA and AI work. It does not make it equivalent to a supported A100:

| Property | Practical effect |
|---|---|
| SM80 compute capability | A large body of CUDA software can compile for the card |
| 64 GiB HBM exposed after patching | Models that do not fit consumer 24 GiB cards may fit |
| No NVLink | Multi-card collectives cross PCIe and can dominate runtime |
| Passive heatsink | Forced front-to-back airflow is mandatory under load |
| Compute-only device | No display output; do not configure it as a primary VGA card |
| Unsupported unlock path | Version pinning and recovery procedures are part of normal operation |

## Before buying or installing

Confirm these items first:

1. The chassis can deliver directed airflow through the heatsink, not merely move air around an open frame.
2. The PSU and cabling can supply every card without splitters or mixed modular-PSU cables.
3. The motherboard can enumerate the desired slot/riser topology and has Above 4G Decoding support.
4. The host can tolerate a full cold power cycle when PCIe recovery fails.
5. The workload benefits from capacity enough to offset slow inter-card communication.

Never mix modular PSU cables between PSU models, even when the connector fits. Keep each riser and its GPU power on a deliberate, documented power domain.

## First inventory

With no workload running:

```bash
lspci -nn | grep -iE 'NVIDIA|3D controller'
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,power.limit,temperature.gpu --format=csv
nvidia-smi -q | grep -A1 'HW Power Brake Slowdown'
```

Save a redacted baseline outside the public repository. A card missing from `lspci` is a hardware/firmware/power/enumeration problem; reinstalling a guest driver will not make a non-enumerated PCI device appear.

## Known-good measured configuration

**Measured:** three cards passed CUDA enumeration, a 16 GiB cross-window write/read smoke test, an SM kernel, and post-load health checks under Ubuntu 22.04, Linux 6.8, driver 610.43.03, and the pinned unlocker described in [Installation](INSTALLATION.md).

This validates that exact configuration. Other card revisions and software versions remain untested until evidence is added.
