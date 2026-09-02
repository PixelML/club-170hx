# QC and acceptance testing

`nvidia-smi` is an inventory check, not a card-quality verdict. A useful gate exercises the upper HBM region, compute units, PCIe path, cooling, and recovery behavior.

## Safety gate

Before any load:

1. Confirm forced airflow and live core/memory temperature telemetry.
2. Confirm no production process is using the selected GPU.
3. Start an Xid watcher.
4. Set an intentional power limit.
5. Stop at 80 °C core, 85 °C memory, any Xid, GPU disappearance, or verification mismatch.

The scripts in [`scripts/qc`](../scripts/qc/README.md) are building blocks, not a substitute for supervision.

## Acceptance ladder

| Stage | Test | Pass condition |
|---|---|---|
| 0 | PCI and driver inventory | Expected device count, ID, 64 GiB map, patched module |
| 1 | Cross-window smoke test | Allocate/write/read at least 16 GiB per card |
| 2 | Near-full HBM test | PRNG write/read/verify roughly 62 GiB, sequentially per card |
| 3 | Compute burn | Ten minutes without mismatch, Xid, disappearance, or thermal stop |
| 4 | Warm rerun | Repeat memory and compute checks without reboot |
| 5 | Real workload | Model or generation job reaches upper VRAM and completes correctly |

Run cards sequentially during diagnosis. Parallel testing makes it harder to separate a bad card from a PSU, riser, motherboard, or cooling limit.

## Stage 3b: tensor-core correctness gate

Stage 3's compute burn confirms a card survives sustained load. It does not
confirm the card's tensor cores compute the right answer. `scripts/qc/tensor-gate.sh`
adds a deterministic correctness check on top of the burn, one card at a
time:

1. **Compute burn.** `gpu-burn -tc` for 10 minutes, checked for 0 errors.
2. **Deterministic tensor-core correctness.** A same-seed BF16, FP16, TF32,
   and INT8 matmul (4096x4096, PyTorch + cuBLAS) run on the card and compared
   against a CPU float64 reference, and against every other card's result on
   the same seed. A real hardware fault would diverge between cards; an
   identical result across every card, even one above the format's naive
   tolerance, is the actual pass signal (see the result table below).
3. **Full-VRAM `memtest_vulkan` pass.**
4. **PCIe replay and AER snapshot**, taken before and after each card.

A live 30-second temperature sampler runs throughout stages 1 and 3 and
kills the active burn or memtest process the moment a card crosses 80 °C
core or 85 °C memory, instead of relying only on a post-hoc check after each
stage finishes.

```bash
scripts/qc/tensor-gate.sh 0 1 2 3     # cards to test; defaults to every detected GPU
```

The script refuses to start if a coordination lock file is present, a
resident serving container is running, or any GPU fleet-wide still has a
compute process attached — read the header comment in the script for the
exact refusal conditions before running it. Every stage writes a JSON
receipt under `logs/tensor-gate-<timestamp>/receipts/gpu<N>/`, so a run that
stops partway still leaves usable evidence for the stages that completed.

`gpu-burn` and `memtest_vulkan` are external tools (see
[scripts/qc/README.md](../scripts/qc/README.md#what-is-still-external));
build or place them next to the script before running it. The matmul
correctness check needs a Python interpreter with `torch` and CUDA
available — the system interpreter on a fresh guest usually does not have
this; point `TG_MATMUL_PYTHON` at one that does.

## Isolation matrix

When a card is cold, missing, or fails load, change one variable at a time:

| Card | Slot/riser | Power lead | Result | Interpretation |
|---|---|---|---|---|
| suspect | known-good | known-good | fails | card becomes primary suspect |
| known-good | suspect | known-good | fails | slot/riser path becomes primary suspect |
| known-good | known-good | suspect | fails | power path becomes primary suspect |

Power down safely before moving PCIe or power cables. A warm reboot does not clear every PCIe fault.

## Evidence to retain

Keep redacted raw output for:

- GPU UUID suffix or stable anonymous card label;
- driver/kernel/module path and card count;
- memory size, allocation size, pattern/seed, duration, and verdict;
- power limit, peak power, peak core/memory temperatures;
- PCIe negotiated/max width and generation;
- Xid/ECC/AER scan before and after;
- warm/cold state and the exact workload revision.

Do not publish hostnames, private addresses, serial numbers, complete UUIDs, or unrelated logs.
