# Four-card tensor-core correctness gate — 2026-09-02

Status: measured
Date: 2026-09-02

## Hardware

- Cards: 4 x CMP 170HX (64 GiB each, 256 GiB aggregate)
- Topology: fully populated seven-slot dual-socket workstation board; one card (index 0) shares lanes with a neighboring slot and trains at PCIe Gen1 x8, the other three at Gen1 x16 — expected board behavior, not a fault
- Power limit: 180 W cap, enforced per card before its stage started
- Cooling: forced airflow, all four cards; live 30-second core/memory temperature sampler running throughout

## Software

- OS / kernel: Ubuntu 22.04, Linux 6.8.0-138-generic
- NVIDIA driver: 610.43.03, `updates/cmpunlocker` patched module (unlock intact, confirmed at day-2 baseline)
- Gate script: `scripts/qc/tensor-gate.sh`, this run's fixed version (see "Script bugs found and fixed" below)
- Compute burn: `gpu-burn -tc`, 10 minutes per card
- Matmul correctness: PyTorch 2.13.0+cu130, seed 170170, 4096x4096 BF16/FP16/TF32/INT8 vs a CPU float64 (INT8: int64) reference
- Memory: `memtest_vulkan` v0.5.0, full-VRAM default pass, via the `memtest-select.py` PTY wrapper

## Command

```text
scripts/qc/tensor-gate.sh 0 1 2 3
```

## Method

- Cards run one at a time; no two cards under load simultaneously.
- Each card: 10-minute `gpu-burn -tc`, then the matmul correctness check, then a full-VRAM `memtest_vulkan` pass, then a PCIe replay/AER snapshot compared before vs. after.
- Stop conditions: 80 °C core, 85 °C memory, any Xid, GPU disappearance, or a matmul/memtest mismatch. The live temperature sampler kills the active stage's process within one 30-second sample period of a breach, instead of relying only on a post-hoc check after the stage finishes.
- Preflight refuses to start unless the fleet is fully clear: no resident serving container running, no coordination lock file, zero compute processes fleet-wide.

## Incident timeline

The gate ran as recovery verification after an incident earlier the same day, not as routine QC. In order:

1. **~14:50 UTC — fleet-wide OOM.** A weight-load out-of-memory event during a load test hit all four pipeline-parallel ranks at once.
2. **Xid 31 on two cards.** The kernel log showed MMU faults (Xid 31) on the two cards holding the affected pipeline-parallel workers.
3. **Xid 154 on all four cards.** A UVM global fatal error (`recovery action: OS Reboot`) followed on every card. A full `nvidia`/`nvidia_uvm` module reload did not recover the fleet; `nvidia-smi` and CUDA initialization hung.
4. **VM reboot.** Required to restore the fleet to a working state.
5. **Post-reboot day-2 baseline.** Read-only checks across all four cards showed a clean kernel log window except one recoverable event: Xid 43 (software-class, driver-level GPU re-initialization, not a card dropout) on card 3, tied to a described earlier wedge, hours before the OOM incident. No hardware-class Xid, no PCIe replay or AER counter, on any card.
6. **This gate.** Run sequentially on cards 0-3 to verify no persisting damage from the OOM cascade before trusting the fleet with a new benchmark publication.

## Results

| GPU | gpu-burn -tc (10 min) | Xid during gate | Core / memory temp (peak) | `memtest_vulkan` | BF16 / FP16 / TF32 / INT8 max abs err vs. CPU float64 | Verdict |
|---|---|---:|---|---|---|---|
| 0 | PASS, 0 errors, 77.7 TFLOP/s | 0 | 62 °C / 64 °C | PASS | 1.348 / 0.166 / 0.098 / 0 | PASS |
| 1 | PASS, 0 errors, 76.4 TFLOP/s | 0 | 56 °C / 64 °C | PASS | 1.348 / 0.166 / 0.098 / 0 | PASS |
| 2 | PASS, 0 errors, 74.6 TFLOP/s | 0 | 63 °C / 71 °C | PASS | 1.348 / 0.166 / 0.098 / 0 | PASS |
| 3 | PASS, 0 errors, 78.5 TFLOP/s | 0 | 58 °C / 65 °C | PASS | 1.348 / 0.166 / 0.098 / 0 | PASS |

All four cards produced byte-identical matmul results at every dtype (BF16, FP16, TF32, INT8), cross-checked against each other, not only against the CPU reference — this is stronger evidence of correct, consistent tensor-core arithmetic than any single card's self-consistency. BF16 (1.348) and TF32 (0.098) sit above the gate script's originally set tolerances (1e-1 and 5e-3); at 4096x4096 scale this reflects those formats' native mantissa precision, not a defect. A real hardware fault would make one card's result diverge from the other three; none did.

PCIe replay and AER (`DevSta`/`UESta`/`CESta`) snapshots were unchanged before vs. after, on all four cards: zero replays, zero uncorrectable, zero correctable.

**Verdict: all four cards pass.** No Xid during any stage, zero compute/memory errors, zero PCIe replay/AER deltas, and identical matmul correctness across all four cards, including the two that took the Xid 31 MMU fault earlier the same day. This gate finds no evidence of persisting damage from the OOM cascade; driver-level recovery through the reboot appears complete for the workload this gate exercises. It does not rule out a fault that only reproduces under a different (larger-model, longer-duration, or memory-pressure) workload than `gpu-burn`/`memtest_vulkan`/matmul.

## Script bugs found and fixed

1. **`gpu-burn` binary path.** The gate's original path guess did not match the built binary's actual name, and the compare-kernel fatbin was not found from the script's working directory. Fixed with an explicit `-c <path>` argument.
2. **`memtest_vulkan` has no `--top-fraction` flag** (silently ignored by the binary) **and no CLI device-select flag**; device choice is an interactive prompt read from a real TTY that ignores piped stdin, so a naive non-interactive call silently tests the same first-listed device every time regardless of which card was intended. Fixed with `memtest-select.py`, a PTY wrapper that matches the target card's PCI bus ID to the tool's own listed index and drives the prompt. This runs the tool's default full-VRAM pass — the only mode this binary version has — which is a superset of the originally planned "top 56 GiB" weighting.
3. **Matmul stage's Python had no `torch`.** The system interpreter lacked `torch`; reran with a venv interpreter that had `torch 2.13.0+cu130` and confirmed CUDA availability.

Also found during setup, not a hardware finding: earlier manual `memtest_vulkan` invocations left four orphaned processes pinning full VRAM on all four cards. The binary re-execs itself into a detached grandchild on device selection, which a plain `SIGKILL` on the parent does not reach — this produced a "failed determining memory budget" error on two cards before cleanup, which read like a hardware problem and was leftover process state. Fixed by signaling the whole process group plus a `pkill -f` backstop; VRAM now releases cleanly.

## Correctness and failures

- Output validation: matmul results cross-checked against a CPU float64 (INT8: int64) reference and against every other card, all four dtypes.
- Xid/ECC/AER scan: zero Xid during any card's gate; PCIe replay and AER counters unchanged before vs. after on all four cards.
- Known caveats: this gate does not exercise sustained multi-hour operation, mixed-workload memory pressure, or the exact conditions of the OOM cascade that preceded it.

## Evidence

- `receipts/day2-baseline.json` — read-only fleet baseline, collected before the gate ran, sanitized (PCI bus IDs and PIDs removed per this repository's publication rules).
- `receipts/gpu{0,1,2,3}/receipt.json` — per-card gate summary (stage return codes, Xid count, final temperatures).
- `receipts/gpu{0,1,2,3}/matmul.json` — per-card matmul correctness output.
- `receipts/matmul_check.py` — the matmul correctness harness used by the gate.
- Full command log, temperature time series, and burn/memtest raw logs stay on the host under the gate's own timestamped log directory; the JSON receipts above are the sanitized summary.
