# QC scripts

These scripts support a supervised card-acceptance workflow. They do not certify a card and must not run on a GPU that holds a serving or training context.

## Near-full HBM test

Build for SM80:

```bash
nvcc -O3 -arch=sm_80 scripts/qc/compare_vram.cu -o /tmp/compare-vram
```

Test one idle card with a 62 GiB allocation:

```bash
/tmp/compare-vram --device 0 --gib 62
```

The program fills and verifies every 64-bit word with an index-derived pattern. It checks allocation, kernel-launch, synchronization, and verification errors.

## Supporting checks

```bash
scripts/qc/pcie-check.sh
sudo scripts/qc/xid-watch.sh
```

Run `xid-watch.sh` in another terminal during the memory and compute tests. Keep `nvidia-smi` temperature monitoring visible and apply the stop thresholds from [QC](../../docs/QC.md).

## Tensor-core correctness gate

```bash
scripts/qc/tensor-gate.sh 0 1 2 3
```

Chains a `gpu-burn -tc` compute burn, a deterministic BF16/FP16/TF32/INT8
matmul check, a full-VRAM `memtest_vulkan` pass, and a PCIe replay/AER
snapshot, one card at a time. `scripts/qc/memtest-select.py` is a PTY
wrapper this script uses to drive `memtest_vulkan`'s interactive
device-select prompt, since that tool has no CLI device flag. Full stage
description: [docs/QC.md](../../docs/QC.md#stage-3b-tensor-core-correctness-gate).

## What is still external

A sustained compute test such as `gpu-burn` is not bundled because compiled third-party binaries should not be copied into this repository. Pin, review, and build the selected tool from its original source; record its commit and command in the result. `memtest_vulkan` is the same case: build or place the binary next to `tensor-gate.sh` before running it.
