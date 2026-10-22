# GEMM microbenchmark — CMP 170HX throughput by dtype and power cap, and what the 6D vBIOS changes

Status: measured
Date: 2026-10-18

Notebook: [`notebooks/2026-10-18-cmp170hx-gemm-dtype-power-6d.ipynb`](../../notebooks/2026-10-18-cmp170hx-gemm-dtype-power-6d.ipynb)

**Question.** Which number formats run at full speed on a CMP 170HX, what does the power cap buy, and what does flashing 92.00.67 → 92.00.6D change?

Background on the two vBIOS revisions (HBM NDIV 54 vs 64, 1,458 vs 1,728 MHz, 250 vs 300 W): [CMP 170HX understanding §4](../../docs/CMP170HX-UNDERSTANDING.md#4-hbm-clock-two-vbios-revisions-one-slow-pair). This result adds GEMM rates by dtype, the INT8 layout trap, a 5 W efficiency sweep, a before/after on the same cards, and the flashing procedure.

## Answer (measured)

1. **No FP32/TF32/FP16/BF16 nerf.** Square 8192–16384 GEMMs run at 92–98 % of a clock-adjusted A100-per-SM peak (70 SMs × measured SM clock) in bf16, fp16, tf32 and fp32. bf16 is the fastest float path.
2. **INT8 is real but not 2x.** `torch._int_mm` with a column-major B reaches 57–62 % of full INT8 rate: 1.6–1.7x bf16 at 140 W for N = 4096–8192 (1.38x at 16384), 1.34–1.42x at 225 W. With a row-major B it silently takes a ~4x slower path (40–49 TOPS, 12 % of peak).
3. **The power cap moves compute, not bandwidth.** 140 → 225 W: bf16 8192² 105 → 161 TFLOP/s (+53 %), int8 168 → 225 TOPS (+34 %). HBM copy bandwidth is the same at every cap.
4. **Efficiency sweet spots (5 W sweep, 100–300 W):** bf16 peaks at **170 W** (0.82 TFLOP/s per W, 138 TFLOP/s; plateau ~0.80 at 150–180 W, 0.67 at 250 W); int8 peaks at **135–140 W** (1.36 TOPS/W, ~175–179 TOPS). Below ~130 W both lose efficiency.
5. **6D vBIOS = +21 % HBM bandwidth, same compute.** On the same two cards before and after the flash: HBM copy 1.32 → 1.59–1.60 TB/s at every cap (HBM 1,458 → 1,728 MHz); bf16 8192² within ±2 % and every large GEMM within ±3 % at the same cap; the decode-like bf16 M=32 GEMM +20–23 % at 225 and 250 W, +10 % / +25 % at 175 W, and −4 % / 0 % at 140 W, where the SM clock limits first. 6D unlocks a 300 W cap, worth only +6 % bf16 over 250 W.
6. **Fleet check, all 7 cards on 6D (5 flashed from 67, 2 shipped with 6D), 170 W each:** HBM copy 1.59 TB/s, bf16 8192² 138–142 TFLOP/s, int8 202–205 TOPS on every card; card-to-card spread ≤ 3.2 % on every test.

## Hardware

- Cards: 7 × CMP 170HX (GA100, SM80), 70 SMs, 65,536 MiB unlocked; every measurement runs on one card at a time
- Anonymous card labels:
  - CMP-A, CMP-B: shipped with vBIOS 92.00.67.00.01, flashed to 92.00.6D.00.0A; measured before and after the flash on the same cap sweep. CMP-B also ran the 5 W sweep before and after.
  - CMP-C: vBIOS 67 when it ran the full dtype table (140 / 225 W); flashed to 6D later.
  - CMP-D, CMP-F: shipped with vBIOS 6D. CMP-F ran the 140 / 300 W table, the 5 W sweep and the cap sweep.
  - CMP-E, CMP-G: shipped with vBIOS 67, flashed to 6D.
  - Fleet check: all seven, on 6D.
- vBIOS: 92.00.67.00.01 (SM max 1,410 MHz, HBM 1,458 MHz) and 92.00.6D.00.0A (SM max 1,695 MHz, HBM 1,728 MHz)
- Topology / PCIe links: Gen2 x4 per card (CMP-D at x2); single-card tests, so the link does not enter the GEMM numbers
- Power limit and measured peak draw: caps set with `nvidia-smi -pl` (140/225 W; 140/300 W; 140–300 W; 100–300 W in 5 W steps; 170 W fleet). The sampler overshoots the cap: max 290 W samples on the 225 W cap and 313 W on the 300 W cap in the table runs; the cap sweeps show brief single-sample spikes up to 415 W (per-set maxima in notebook §2.6)
- Cooling and peak core/memory temperatures: directed airflow; 83 °C core / 90 °C HBM kill switch in the run10, run11, run13 (in-process, plus an 85 / 92 °C external watchdog) and fleet drivers, never hit; run9 (the dtype table) had no kill switch. Peaks: CMP-C (67) 78 °C core at 225 W (HBM temperature not logged in that run); CMP-F (6D) 72 / 71 °C core/HBM at 300 W in the table run and 82 / 77 °C at the 300 W step of the continuous 5 W sweep; before/after sweeps ≤ 67 / 71 °C; fleet check 46–49 / 56–61 °C

## Software

- OS / kernel: Ubuntu, Linux 6.8, bare metal
- NVIDIA driver / CUDA: NVIDIA open kernel modules 610.43.02 with the [cmpunlocker](https://github.com/amoghmunikote/cmpunlocker) memory unlock at `50a7c5e83cf02002b77d39a953ca2e6812c4d0f6`, plus the WPR2 fix of upstream `f51da030ef4260656288edf07e95ae45f339aea8` backported into `0003-late-pma.patch`; installed with `sudo env CMPUNLOCKER_DRIVER_VERSION=610.43.02 ./install.sh --profile=8gb --no-iommu` and a cold boot. CUDA 12.8 runtime from the torch wheel
- Runtime repository + exact commit or image digest: no container; `pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128` (torch 2.11.0+cu128, Python 3.11)
- Model repository + exact revision: none (synthetic GEMMs, random inputs)
- Quantization / dtype: int8 (`torch._int_mm`, int32 out), bf16, fp16, tf32, fp32
- Flash tool: NVIDIA nvflash 5.867 (Linux x64). No ROM files or nvflash binaries are in this repository

## Command

```text
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
cd results/2026-10-18-cmp170hx-gemm-dtype-power-6d/tools
IDX=<nvidia-smi index> GPU_UUID=<GPU-uuid> PY=python OUT=. bash run9.sh          # full dtype table, 225 W then 140 W
IDX=<idx> GPU_UUID=<GPU-uuid> PY=python OUT=. bash run10.sh                       # subset, 140 W then 300 W
CARDS="<idx>:<GPU-uuid>:cmp-x:300" RESTORE=140 PY=python OUT=. bash run13.sh      # 5 W sweep, 100 .. 300 W
CARDS="<idx>:<GPU-uuid>:cmp-x" CAPS="140 175 225 250 300" PY=python OUT=. bash run11.sh   # before/after cap sweep
CARDS="<idx>:cmp-a <idx>:cmp-b" PY=python OUT=. bash run_fleet.sh                 # fleet check at the current (170 W) cap
python an9.py .                                                                  # Markdown table from run9 output
```

## Method

- Warmup: 3 calls before every timed run.
- Samples: CUDA-event timing of back-to-back calls sized to ~2 s; each cell = median of 3 such runs (`mm9.py`, `mm10.py`). The before/after and fleet sweeps use `mm11.py`: 2 runs of ~1 s and a 1 GiB copy. The 5 W sweep (`mm13.py`) settles 3 s under load at each cap, then takes 3 timed runs per dtype. Short bursts run hotter clocks and overstate low caps, so every cell is sustained.
- Shapes: square N×N GEMMs (N = 4096, 8192, 16384) and model-shaped M×K×N (M = 32 … 2048, K×N = 4096×4096 and 2048×1024). int8 B is column-major unless labelled row-major. HBM copy = 2 GiB (1 GiB in `mm11.py`) device-to-device, read + write bytes counted.
- Metric calculation: power and clocks from a 250 ms (5 W sweep: 100 ms) `nvidia-smi` sampler, averaged over each cell's time window. '% of peak' = rate / (A100 dense per-SM rate × 70 SMs × measured mean SM clock).

## Results

One vBIOS 67 card (CMP-C), 140 W vs 225 W; rate = TFLOP/s or TOPS. Full table: [`receipts/dtype_table_67/`](receipts/dtype_table_67/TABLE_cmp-c_67_140W_225W.md).

| test | dtype | 140 W | SM MHz | % peak | 225 W | SM MHz | % peak |
|---|---|---:|---:|---:|---:|---:|---:|
| 8192² | int8 (B col-major) | 168.3 | 965 | 61 % | 224.7 | 1307 | 60 % |
| 8192² | bf16 | 105.1 | 784 | 94 % | 160.7 | 1211 | 93 % |
| 8192² | fp16 | 85.4 | 626 | 95 % | 147.4 | 1097 | 94 % |
| 8192² | tf32 | 49.3 | 746 | 92 % | 77.4 | 1174 | 92 % |
| 8192² | fp32 | 10.0 | 1132 | 98 % | 12.1 | 1393 | 97 % |
| 8192² | int8 (B row-major) | 40.0 | 1153 | 12 % | 48.5 | 1399 | 12 % |
| 32×4096×4096 | bf16 | 25.5 | 831 | 21 % | 31.1 | 1367 | 16 % |
| 2048×4096×4096 | bf16 | 102.9 | 855 | 84 % | 143.4 | 1204 | 83 % |
| HBM copy | — | 1.3 TB/s | | | 1.3 TB/s | | |

Note the fp16 vs bf16 gap at 8192²: fp16 runs at a lower clock at the same cap (626 vs 784 MHz), so bf16 is faster per watt on this card.

The 6D card table, the 5 W sweep, the per-cap before/after table and the fleet table are rendered from the receipts in the notebook (§2.2–2.5).

## Flashing 67 → 6D: what worked

Flashing firmware is irreversible without a saved ROM and can brick a card. This is what worked for us, not a guarantee.

- `nvflash` (5.867) refuses to save or write once the NVIDIA driver has touched the card ("A system restart might be required"), even after unloading the module. Unloading is not enough.
- `modprobe.blacklist=` is NOT sufficient: services such as `nvidia-persistenced`, `nvidia-cdi-refresh` and a power-limit unit load the module by name. What worked: a one-time boot entry (`grub-reboot`) with the kernel-enforced `module_blacklist=nvidia,nvidia_drm,nvidia_modeset,nvidia_uvm`, `systemd.mask=` for those services and the display manager, and `systemd.unit=multi-user.target`. After that boot, `nvflash --save` produced a ROM byte-identical to an earlier dump, and the flashes went through.
- Order we used per card: save the original ROM → flash → confirm 92.00.6D.00.0A, 10DE:20C2 / subsystem 10DE:1585 on nvflash's own confirmation screen → cold power cycle (box + any external PSU off 30 s) → confirm 64 GiB + 70 SMs still unlocked, HBM 1,728 MHz live → `memtest` 2-pass + 4-pass (62 GiB, 0 mismatches, no Xid) before returning the card to service. (Our memtest build wanted a lowercase PCI bus id.)
- Have a second machine that can ssh into the box before you start: a failed no-driver boot left us with a desktop that would not open apps, recovered over ssh.

## Correctness and failures

- Output validation: none; throughput only, GEMM outputs were not compared with a reference.
- Xid/ECC/AER scan: not recorded during the GEMM runs. Each flashed card passed memtest 2-pass + 4-pass (62 GiB, 0 mismatches, no Xid).
- Known caveats: power per point in the 5 W sweep comes from 1–4 sampler readings, so single points are noisy (int8 especially). One card per configuration except the before/after pair and the fleet check.
- Not measured: sustained soaks above 250 W; 74-SM builds (our driver predates cmpunlocker's SM override); P2P builds; non-GEMM kernels (attention, MoE routing).

## Evidence

- [`receipts/dtype_table_67/`](receipts/dtype_table_67/) — CMP-C (67), 140 / 225 W: `res9_*.jsonl`, sampler `smi9_*.csv`, rendered table
- [`receipts/card_6d_140_300/`](receipts/card_6d_140_300/) — CMP-F (6D), 140 / 300 W: `res10_*.jsonl`, `smi10_*.csv`, rendered table
- [`receipts/cap_sweep_5w/`](receipts/cap_sweep_5w/) — 5 W sweep, CMP-B (67) 100–250 W and CMP-F (6D) 100–300 W: `res13_*.jsonl`, `smi13_*.csv`
- [`receipts/pre6d/`](receipts/pre6d/) and [`receipts/post6d/`](receipts/post6d/) — CMP-A and CMP-B before / after the flash (and CMP-F before, for reference): `res11_<card>_<cap>.jsonl`, `smi11_*.csv`, run logs; `post6d/res13_cmp-b.jsonl` = CMP-B 5 W sweep on 6D
- [`receipts/fleet6d_170w/`](receipts/fleet6d_170w/) — all seven cards on 6D at 170 W: `res_<card>.jsonl`, `smi_<card>.csv`
- [`tools/`](tools/) — bench scripts `mm9.py`, `mm10.py`, `mm11.py`, `mm13.py`, drivers `run9.sh`, `run10.sh`, `run11.sh`, `run13.sh`, `run_fleet.sh`, table builders `an9.py`, `an10.py`
- Sampler CSV columns (no header): timestamp, power.draw W, clocks.sm MHz, clocks.mem MHz, temperature.gpu °C, then memory.used MiB (`smi9_*`) or temperature.memory °C (all others), plus pcie.link.width.current in the fleet `smi_*` files. JSON rows carry `t0`/`t1` wall times for alignment.
- Host identifiers (bus addresses, UUIDs, host names, card indices) are replaced by CMP-A … CMP-G. The original runs set caps through a local wrapper around `nvidia-smi -pl`; the published drivers call `nvidia-smi -pl` directly.

## Credits

- **Amogh Munikote** ([@amoghmunikote](https://github.com/amoghmunikote)) authored [cmpunlocker](https://github.com/amoghmunikote/cmpunlocker), the unlock driver these cards run on (64 GiB), including the WPR2 fix backported here.
- **PyTorch** ([@pytorch](https://github.com/pytorch)) authored [PyTorch](https://github.com/pytorch/pytorch), which runs every GEMM and copy here.
- **NVIDIA** ([@NVIDIA](https://github.com/NVIDIA)) authored nvflash, the vBIOS flash tool, and the driver. No NVIDIA binaries or ROMs are redistributed here.
- vBIOS background: [CMP 170HX understanding §4](../../docs/CMP170HX-UNDERSTANDING.md#4-hbm-clock-two-vbios-revisions-one-slow-pair).
