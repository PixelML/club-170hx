# Failure risk

What can go wrong on this node, what it looks like, and what it costs to recover. This is a risk register, not a benchmark page — read [Health watch](HEALTH-WATCH.md) for ongoing monitoring and [QC](QC.md) for the acceptance gate.

## Fleet-wide OOM can escalate to a required VM reboot

**What happened (2026-09-02).** A weight-load out-of-memory event during a load test hit all four pipeline-parallel ranks at once. The kernel log then showed Xid 31 MMU faults on two cards, followed by a UVM global fatal error (Xid 154, recovery action "OS Reboot") on all four cards. A full `nvidia`/`nvidia_uvm` module reload did not recover the fleet; `nvidia-smi` and CUDA initialization hung. A VM reboot was required. All four cards passed the full tensor-core correctness gate afterward — see [results/2026-09-02-health-gate](../results/2026-09-02-health-gate/README.md).

**Cost.** A cold VM reboot, plus re-verifying the power cap and a clean kernel log, plus re-running the correctness gate before trusting the fleet with a new benchmark. Budget on the order of tens of minutes for the reboot and recovery, more for the full gate.

**Mitigation.** Size a load test's memory footprint before running it on all four cards at once; an OOM on one rank of a pipeline-parallel job can take down every rank. After any Xid 31/154 event, do not resume benchmarking on unverified cards — run the correctness gate first.

**Residual risk.** The correctness gate does not rule out a fault that only reproduces under a workload larger, longer, or more memory-pressured than `gpu-burn`/`memtest_vulkan`/matmul exercise. A clean gate after this incident is evidence of no persisting damage under this gate's own profile, not a blanket guarantee.

## A software-class Xid does not mean a dying card

Xid 43 appeared on one card on 2026-09-02, tied to a described compute wedge. The driver re-initialized the GPU rather than the card dropping off the bus, and the card passed the correctness gate afterward with results identical to the other three cards. Treat this Xid class as a note-and-continue event, distinct from the fatal UVM class above. See [Health watch](HEALTH-WATCH.md#xid-severity-as-observed-on-this-node) for the distinction.

## Board topology, not a card fault: one slot trains at x8

One card in this seven-slot, dual-socket board trains at PCIe x8 instead of its advertised x16, because the board shares lanes between neighboring slots and this card's neighbor is occupied. This was corrected from an earlier riser-fault suspicion. It is expected behavior for this board, not a risk to track or a symptom to chase — recorded here so it is not rediscovered as a false alarm.

## Tooling gaps that can hide a real fault or manufacture a false one

Two gaps in the QC tooling itself were found while building the correctness gate, both capable of producing a misleading result if unaddressed:

- **`memtest_vulkan` has no device-select flag and no `--top-fraction` flag.** Device choice is an interactive prompt that silently ignores piped stdin — a naive non-interactive invocation tests the same first-listed device every time, regardless of which card the operator intended. A batch script that assumes it tested four different cards may have tested one card four times. Fixed with a PTY-driving wrapper (`memtest-select.py`) that matches the intended card's PCI bus ID to the tool's own listed index.
- **Orphaned `memtest_vulkan` processes pin VRAM after a kill,** because the binary re-execs itself into a detached grandchild on device selection that a plain `SIGKILL` on the parent does not reach. This produced a "failed determining memory budget" error on two cards during gate development, which looked like a hardware problem and was actually leftover state from a previous test run. Fixed by signaling the whole process group plus a `pkill -f` backstop.

Both fixes are in the copy of `tensor-gate.sh` under [`scripts/qc`](../scripts/qc/tensor-gate.sh).

## Unsupported unlock path

The 64 GiB VRAM exposed on these cards depends on a community-maintained, NVIDIA-unsupported unlock (`amoghmunikote/cmpunlocker`). A driver update, kernel update, or module rebuild can silently fall back to the card's stock mining-era VRAM window instead of failing loudly. Confirm the module resolves to the patched copy (see the day-2 baseline check in [results/2026-09-02-health-gate](../results/2026-09-02-health-gate/README.md)) after any driver or kernel change, not only at first install.
