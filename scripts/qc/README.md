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

## What is still external

A sustained compute test such as `gpu-burn` is not bundled because compiled third-party binaries should not be copied into this repository. Pin, review, and build the selected tool from its original source; record its commit and command in the result.
