# Health watch

This page tracks the ongoing health signals for the four-card CMP 170HX node: what to check, how often, and what the current baseline looks like. It complements [QC and acceptance testing](QC.md), which covers a new or suspect card. This page covers a card the fleet already trusts.

## What to watch

| Signal | Check | Concern threshold |
|---|---|---|
| Core / memory temperature | `nvidia-smi --query-gpu=temperature.gpu,temperature.memory` | Stop a running job at 80 °C core or 85 °C memory |
| Power draw at idle | `nvidia-smi --query-gpu=power.draw` | A jump of more than a few watts over the recorded idle baseline for that card |
| Xid events | `journalctl -k --since <window> \| grep "NVRM: Xid"` | Any hardware-class Xid (for example 48, 79, 94, 95); a software-class Xid (see below) is lower urgency but still worth a note |
| PCIe replay / AER counters | `nvidia-smi -i <n> -q \| grep -A1 Replay`, `lspci -vv -s <bus> \| grep -E "DevSta\|UESta\|CESta"` | Any nonzero uncorrectable count, or a correctable count that climbs between checks |
| PCIe link width/generation | `nvidia-smi --query-gpu=pcie.link.width.current,pcie.link.gen.current` | A width or generation drop from the card's known baseline on a slot that previously trained higher |
| Compute-process leaks | `nvidia-smi --query-compute-apps=pid,used_memory` | A process still holding VRAM after its job should have exited |

## Xid severity, as observed on this node

Not every Xid means a dying card. Two classes have shown up here and read very differently:

- **Software-class (recoverable).** Xid 43 on 2026-09-02, tied to a described compute wedge. The driver re-initialized the GPU (a GSP boot sequence re-logged in the kernel log) and the card kept working — this is the driver recovering a stuck context, not the card leaving the bus.
- **Fatal / needs-reboot class.** Xid 31 (MMU fault) followed by Xid 154 (UVM global fatal error, recovery action "OS Reboot") during an out-of-memory event on 2026-09-02. A full module reload did not recover the fleet; a VM reboot was required. See the incident timeline in [results/2026-09-02-health-gate/README.md](../results/2026-09-02-health-gate/README.md).

Treat a software-class Xid as a note-and-continue event. Treat a UVM fatal error or a hardware-class Xid as a stop-and-verify event: run the [tensor-core correctness gate](QC.md#stage-3b-tensor-core-correctness-gate) on the affected card(s) before trusting them with a new benchmark.

## Idle baseline

Reference idle numbers, 180 W cap, four-card node, collected read-only (no workload disturbed):

| Snapshot | Core temp | Memory temp | Power draw |
|---|---|---|---|
| 2026-08-30 (per-card range) | 37–38 °C | 41–51 °C | 33.7–37.7 W |
| 2026-09-02, day 2 (per-card) | 37–38 °C | 41–51 °C | 33.6–41.4 W |

The two snapshots agree. A future idle reading well outside these ranges, with no workload running, is worth investigating before it is treated as normal drift.

## PCIe: one card runs at x8, and that is expected

One card in this rig trains at x8 instead of its advertised x16. This board is a fully populated seven-slot dual-socket workstation board that shares lanes between neighboring slots — occupying the slot next to this card drops it to x8. This was confirmed board topology, not a riser fault, card defect, or passthrough bug (an earlier note treated it as a riser suspicion; the 2026-09-02 health gate corrected that). Do not re-open this as a hardware ticket without new evidence — a link-width change on a *different* card, or a width regression on this same card after moving hardware, would be new evidence; the existing x8 reading on the same slot is not.

## Cadence

- **Every session before a new benchmark publication:** confirm no unresolved Xid since the last gate run, and that idle numbers are in the ranges above.
- **After any Xid, UVM fatal error, or unexplained crash:** run the tensor-core correctness gate on the affected card(s) before publishing a result that used them.
- **Routinely, independent of incidents:** re-run the full four-card gate periodically as the node accumulates hours, so a slow-developing fault is caught before it shows up mid-benchmark. No fixed interval is set yet; treat "after a major incident" as the current trigger and revisit this once the node has more running hours behind it.

## Known operational gotchas that look like health problems but are not

- **`memtest_vulkan` device selection is an interactive prompt.** Piped stdin is silently ignored, so a naive non-interactive call always tests the same first-listed device regardless of which card you meant to target — it will report a clean pass on the wrong card, not an error. Use the `memtest-select.py` wrapper in [`scripts/qc`](../scripts/qc/README.md), which drives the prompt over a real PTY and matches the card's PCI bus ID to the tool's own listed index.
- **Orphaned `memtest_vulkan` processes pin VRAM after a kill.** The binary re-execs itself into a detached grandchild once a device is selected, so a plain `SIGKILL` on the parent process does not reach it. A leftover process shows up as "failed determining memory budget" or an unexpectedly full VRAM reading on a card that should be idle — check `nvidia-smi --query-compute-apps=pid,used_memory` and `pkill -f memtest_vulkan` before assuming a driver or hardware fault.
- **The 180 W power cap does not survive a guest reboot or an nvidia module reload.** Both actions reset every card to the vBIOS default (250 W on this fleet). A benchmark run after either event, with no cap check first, silently measures at the wrong power envelope — this happened on 2026-09-02 and required a follow-up re-measure. Run `scripts/qc/set-powercap.sh` (or confirm the `nvidia-powercap` systemd unit re-applied it) and verify `nvidia-smi --query-gpu=power.limit --format=csv` shows 180.00 W on all four cards before starting any measurement.
