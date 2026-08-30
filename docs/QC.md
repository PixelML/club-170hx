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
