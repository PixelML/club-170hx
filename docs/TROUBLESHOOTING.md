# Troubleshooting

Start at the lowest layer where the card is missing. Do not reinstall a driver when the host itself cannot enumerate the device.

For runtime- and workload-level failures that are not hardware faults —
NCCL store timeouts from storage-read skew, ranks stuck in D state during an
NFS-backed weight load, `--gpus all` losing devices after a crash, NVRM
VA-space corruption after an OOM kill storm, or a container seccomp policy
blocking `pidfd_getfd` — see [What we learned: failure modes and
recovery](LESSONS.md#g-failure-modes-and-recovery).

| Symptom | First check | Likely direction |
|---|---|---|
| No fans/board boot, standby LEDs flash | Minimal motherboard configuration and PSU cabling | PSU protection, short, board power, wrong modular cable |
| Card absent from host `lspci` | Cold power-off, slot/riser/power isolation | Hardware enumeration, riser, slot, power, card |
| Present on host, absent in guest | VM passthrough config and IOMMU binding | Host/VM assignment |
| `RmInitAdapter` or SEC2 failure | Firmware mode and kernel logs | Use measured SeaBIOS path; avoid OVMF baseline |
| Card reports only 8 GiB | `modinfo -n nvidia`, `dkms status` | Stock module shadowing patched module |
| Guest hangs/RCU stalls | Experimental Gen2 patch/service | Return to stable Gen1 baseline |
| `Xid 79`, “fallen off the bus” | Kernel log, host PCI config | Link/power/riser/card fault; cold cycle may be required |
| Low performance | power limit, PWRBRK, utilization, topology | Throttling, link/collective bottleneck, runtime settings |
| Hot while idle | power draw, clocks, processes, airflow | lingering workload/context, persistence behavior, inadequate flow |

## Layered inventory

```bash
# Host or bare metal
lspci -nn | grep -iE 'NVIDIA|3D controller'

# Guest/driver
nvidia-smi -L
nvidia-smi --query-gpu=index,pci.bus_id,memory.total,pstate,power.draw,power.limit,utilization.gpu,temperature.gpu,temperature.memory --format=csv

# Kernel/module
modinfo -n nvidia
sudo dmesg --level=err,warn | grep -Ei 'NVRM|Xid|fallen off|RmInitAdapter|SEC2|AER|RCU'
```

## Xid 79 and D3cold

**Measured:** after Xid 79, a card remained unavailable through function-level reset, secondary-bus reset, runtime-power changes, and remove/rescan. Its PCI configuration read as an invalid header state until a true cold power cycle.

Recovery order:

1. Stop workloads and shut down the guest and host cleanly.
2. Remove main power long enough for standby rails to discharge; 30 seconds worked in the measured case.
3. Restore power and check host `lspci` before starting the VM.
4. If still missing, isolate card/riser/power using the matrix in [QC](QC.md).

Repeated warm reboots can waste time when the root port or endpoint no longer enumerates.

## A card is physically cold

A cold card during an expected workload usually means it is not drawing workload power. Check, in order:

1. Does the host enumerate it?
2. Does the guest own it?
3. Does `nvidia-smi` list it and show a process/utilization?
4. Does swapping in a known-good riser and power lead change the result?
5. Does the suspect card fail in a known-good slot/power path?

Change one variable per shutdown. Label every cable and riser before the next swap.
