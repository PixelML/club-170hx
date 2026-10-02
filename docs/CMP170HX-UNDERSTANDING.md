# What we know about the CMP 170HX (as of 2026-10-02)

One page that pulls together what this club has measured across its notebooks, what the community has reported, and what is still out of reach. It supersedes older statements in other pages where they disagree; [§11](#11-older-statements-this-page-supersedes) lists those. Labels follow [AGENTS.md](../AGENTS.md): **measured** (by this club, with a linked receipt), **community-reported**, **inferred**, **untested**.

Hosts named below:

- **4-card PLX host**: bare-metal Proxmox VE 9.2, dual-socket Broadwell Xeon (E5-2686 v4), four cards on one socket behind two PLX PEX 8747 switches (two cards per switch), serving from a GPU LXC that shares the host driver. Most results from 2026-10-01 onward.
- **Earlier test node**: Proxmox VFIO guest, four cards, used for the August and September notebooks.
- **Second host**: three cards unlocked host-side and passed to VMs that run the stock driver.

## 1. The card in one table

| Property | Value | Status |
|---|---|---|
| Silicon | GA100-105F (A100 family), compute capability 8.0 | community-reported, corroborated |
| SMs | 70 stock; **74** with cmpunlocker `6c442ee` | measured (torch `multi_processor_count` = 74 on all four cards) |
| Memory | 8 GB stock; **64 GB** after the memory unlock on dual-rank cards (single-rank cards stop at 32 GB) | measured 64 GB on every card we own; single-rank: community-reported ([#48](https://github.com/PixelML/club-170hx/issues/48)) |
| HBM2e clock | VBIOS-dependent: **1,458 MHz (NDIV 54)** on VBIOS `92.00.67.00.01`, **1,728 MHz (NDIV 64)** on `92.00.6D.00.0A`, same board part number, identical timings | measured |
| PCIe | Gen1 stock; **Gen2** with the cmpunlocker link patch; **Gen3 blocked by a fuse** on every card we read | measured |
| PCIe width | x16 or x4 depending on the board (AC-coupling capacitors on lanes 4–15 populated or not) | community-reported; our cards train x16 |
| BAR1 | small stock; **64 GiB** with the unlock; static-BAR1 peer-to-peer works with extra patches | measured |
| NVLink | not present (bridge components unpopulated) | community-reported |
| FP8 / FP4 tensor cores | none (GA100 tops out at FP16/BF16/TF32 tensor math) | vendor architecture |
| Board power | 250 W or 300 W depending on VBIOS (community-reported naming, see §4); we run 140 W | measured cap |
| Cooling | passive heatsink, forced air required | measured |

## 2. The lock layers, and what software can move

| Layer | Stock | Unlocked state we run | Mechanism | Status |
|---|---|---|---|---|
| Memory size | 8 GB | 64 GB | cmpunlocker: Falcon BootROM signature-load bug → geometry straps reprogrammed; patched open kernel modules | measured |
| Compute throttle | FP32 FMA and tensor issue rate gated | restored | same unlock (SEC2 post-boot PLM writes) | measured: tensor-core gate 74.6–78.5 TFLOP/s per card ([2026-09-02 health gate](../notebooks/2026-09-02-cmp170hx-health-gate.ipynb)) |
| SM count | 70 | 74 | upstream `6c442ee` opens `RECONFIG_PLM` and writes the GPC reconfiguration overrides | measured |
| PCIe speed | Gen1 (2.5 GT/s) | Gen2 (5 GT/s) | cmpunlocker link patch: fuse-override register + retrain at probe | measured |
| PCIe Gen3 | — | not reachable | `OPT_DISABLE_GEN3_SPEED` fuse (BAR0 `0x820250` bit 0) reads 1 | measured on 4/4 cards — see §3 |
| HBM clock | VBIOS NDIV 54 or 64 | NDIV 64 on all four | PR #60 opens the FBPA PLL / FBPA_MEM privilege masks; [170tune](https://github.com/cachenetics/170tune) writes the PLL live after a hot gate | measured — see §4 |
| SM voltage/frequency offset | 0 | +200 at a 1,410 MHz ceiling on the 300 W VBIOS cards only | NVML VF offset (undervolt) via 170tune | measured: +0.6% / −6 W at 140 W, within noise; the 250 W VBIOS exposes no offset range ([2026-10-02 notebook §2.6](../notebooks/2026-10-02-glm-5.3-flash-4card-tp4-sm74-hbm-power-vllm.ipynb)) |
| BAR1 peer-to-peer | not supported | 12/12 card pairs at 5.79 GB/s | static BAR1 + 4 P2P patches + hand-placed PCIe layout | measured — see §5 |
| ECC | off | off | upstream work in progress ([cmpunlocker #56](https://github.com/amoghmunikote/cmpunlocker/pull/56)) | untested |
| NVLink | none | none | hardware absent | community-reported |

The memory and link unlocks are volatile: they are reapplied by the patched driver on every boot. A cold power cycle returns the card to stock.

## 3. PCIe Gen3: blocked by a fuse, and the risk of trying

**Measured (2026-10-01):** `OPT_DISABLE_GEN3_SPEED` (BAR0 `0x820250`, bit 0) reads **1** on all four cards on the 4-card PLX host. Every link trains Gen2 x16 (`LnkCap` and `LnkSta` 5 GT/s x16).

Upstream [cmpunlocker PR #37 "PCIe Gen 3"](https://github.com/amoghmunikote/cmpunlocker/pull/37) (open, unmerged) raises the link to 8 GT/s **only when that fuse is clear**. Its own commit notes (**community-reported**) measured what happens on a fuse-set card: writing the fuse override to 0xA raises `LNKCAP` only to 5 GT/s, and asking for 8 GT/s anyway takes the card off the bus, recoverable only by a chassis power cycle (a PCI remove/rescan hung). The follow-up commits clamp the target to the fuse and fall back one generation at a time. A YouTube video titled "The 170HX Can Now PCIe Gen3.0 x16?!" links that repository; we found no report of a fuse-set card reaching Gen3.

What this means in practice:

- Read the fuse before trying anything: if it reads 1, Gen2 is the ceiling in software.
- Gen2 x16 measured 6.68 GB/s device-to-host on one card (~84% of the 8 GB/s theoretical), and 6.2–6.3 GB/s for host-staged card-to-card copies ([3-card P2P notebook](../notebooks/2026-10-01-cmp170hx-3card-p2p-plx-cuda.ipynb)).
- For pipeline-parallel decode the link barely matters: MiMo-V2.6-Flash PP3 decoded at the same speed at Gen1 x16 and Gen2 x16 ([notebook](../notebooks/2026-10-01-mimo-v2.6-flash-3card-pp3-vllm.ipynb)).

## 4. HBM clock: two VBIOS revisions, one slow pair

**Measured:** the four cards on the 4-card PLX host carry two VBIOS revisions on the same board part number (`900-11001-0108-000`):

| VBIOS | Stock HBM NDIV | Clock (NDIV × 27 MHz) | Cards |
|---|---:|---:|---:|
| `92.00.67.00.01` | 54 | 1,458 MHz | 2 |
| `92.00.6D.00.0A` | 64 | 1,728 MHz | 2 |

The two 8 GB VBIOS variants are **250 W / NDIV 54** (`92.00.67.00.01`) and **300 W / NDIV 64** (`92.00.6D.00.0A`); the 250 W / 300 W naming is from 170tune's source notes (community-reported). Only the 300 W variant exposes a GPC VF offset: on the 250 W variant NVML reports the offset range as [0..+0] MHz and refuses a set (measured).

Memory timings read identical on all four (`RC 67, RFC 657, RAS 43, RP 24, RD_RCD 27, WR_RCD 18, WR 25, FAW 22, RRD 5`, refresh 6). The 1,728 MHz cards run that timing table at the higher clock from the factory.

In tensor parallel every step waits for the slowest card, so a mixed set runs at the slow pair's memory speed. With the HBM privilege masks opened (cmpunlocker PR #60, closed upstream because the maintainer does not want tuning features) 170tune can rewrite the PLL live. After a hot gate on each slow card (`GATE_TEMP=75`, 12 full-VRAM sweeps plus a bit-exact compute check, peak HBM 75 °C, 0 errors) both run NDIV 64, persisted by 170tune's boot service.

**Measured effect** (GLM-5.3-Flash TP4, 150 W): decode step 19.5 → 18.4 ms; single-user decode +5.8 to +6.3%, eight-user +3.1 to +3.8%, identical draft acceptance ([2026-10-02 notebook](../notebooks/2026-10-02-glm-5.3-flash-4card-tp4-sm74-hbm-power-vllm.ipynb)). 418.0 tok/s structured single-user is the best this host has measured.

Going past 1,728 MHz is a real overclock. 170tune's reference card (**community-reported**) serves at NDIV 70 (+10% read bandwidth) only with HBM at or below 76 °C; NDIV 72 corrupted as it heated and 74–76 crashed under load. Our cards serve at HBM 72–75 °C at 150 W and reach 82–83 °C at 165 W, so we have not tried above 64. **Note:** `nvidia-smi` keeps showing the clock cached at driver load; `170tune -i N status` shows the PLL.

## 5. Interconnect, peer-to-peer and parallelism

Consolidated from the 2026-10-01 notebooks (all **measured** on the 4-card PLX host unless noted):

| Finding | Number | Source |
|---|---|---|
| No peer access with the plain unlock; copies stage through host memory | 6.21–6.30 GB/s at 256 MiB, 22 µs at 4 KiB; same-switch = cross-switch | [3-card notebook](../notebooks/2026-10-01-cmp170hx-3card-p2p-plx-cuda.ipynb) |
| Two cards behind one switch corrupt the kernel at boot unless the 64 GB BAR1 resize is serialized | crash on every boot → 0 with `rebar-serialize.patch` (same fix as upstream open [PR #59](https://github.com/amoghmunikote/cmpunlocker/pull/59)) | same |
| Static-BAR1 P2P works with 4 extra patches and a hand-programmed PCIe layout adopted via kexec | 12/12 pairs byte-exact, 5.79 GB/s; latency 16 / 194 µs at 4 KiB / 1 MiB vs 22 / 343 µs staged | [4-card BAR1 P2P notebook](../notebooks/2026-10-01-cmp170hx-4card-bar1-p2p-plx-cuda.ipynb) |
| P2P helps two-card TP | Qwen3.8-27B TP2 decode +11% (+14% with MTP k=3) | same |
| P2P hurts four-card TP behind PLX on Broadwell | GLM-5.3-Flash TP4 decode −20 to −26% (1 user), −38 to −48% (8 users); step 19.4 → 26.1 ms. Morrowmake measured +11% from P2P on EPYC root ports | [GLM TP4 notebook](../notebooks/2026-10-01-glm-5.3-flash-morrowmake-1.6.0-4card-tp4-vllm.ipynb) |
| TP2 without P2P still lifts single-stream decode | Qwen3.8-27B 54.0 → 72.0 tok/s; no gain under load (c=16) | [TP2 notebook](../notebooks/2026-10-01-qwen3.8-27b-w4a16-2card-tp2-vllm.ipynb) |
| Pipeline-parallel decode is HBM-bound, not link-bound | MiMo PP3 74.6 vs 75.5 tok/s at Gen2 vs Gen1 x16 | [MiMo notebook](../notebooks/2026-10-01-mimo-v2.6-flash-3card-pp3-vllm.ipynb) |
| Under VFIO passthrough there is no P2P; the unlock can be applied host-side so the guest runs the stock driver | Gen2 x16, 64 GB in the guest (second host) | measured, no notebook |

Older guidance that "PP beats TP on this card" ([Topology and parallelism](TOPOLOGY-AND-PARALLELISM.md)) was measured on a slower link and an older stack (GLM PP4 87.6 tok/s, TP4 56.4 tok/s). The current TP4 recipe with a DFlash2 drafter reaches 396–418 tok/s at Gen2 x16. That is a different drafter and engine, not a like-for-like PP vs TP rematch; PP4 on the current stack is untested.

## 6. Power and heat

**Measured (2026-10-02), GLM-5.3-Flash TP4, HBM 1,728 MHz on all cards, cap set live:**

| Cap per card | 1 user structured tok/s | 8 users structured tok/s | Busy GPU power (4 cards) | 8-user tok/s per W |
|---:|---:|---:|---:|---:|
| 165 W | 418.6 | 822.4 | 650 W | 1.27 |
| 150 W | 418.0 | 816.5 | 582 W | 1.40 |
| **140 W** (our default) | 412.7 | 814.2 | 537 W | 1.52 |
| 130 W | 394.9 | 785.9 | 517 W | 1.52 |
| 120 W | 374.4 | 756.5 | 476 W | 1.59 |
| 110 W | 340.1 | 706.8 | 436 W | **1.62** |
| 100 W | 291.9 | 627.4 | 398 W | 1.58 |

- At 150 W every card sits on the cap during decode (median ~149 W) with the core near 1,320 MHz, yet raising the cap to 165 W adds only 0.1–1.1%. Decode here waits on memory and on the four-way synchronization, not on core clock.
- 140 W costs ~1% for 8% less GPU power and noticeably less heat. Below 130 W the core clock falls fast (1,290 → 1,050 MHz) and decode follows.
- Earlier single-card data agrees: Qwen3.8-27B at 180 W reached 95% of its 255 W decode rate ([LESSONS §f](LESSONS.md#f-power-and-thermal)).
- Heat: passive cards need forced, directed airflow. On the 4-card PLX host the card with the worst airflow reached 80–85 °C core on long-context prefill at 180 W, which is why the cap came down. Our stop rule stays 80 °C core / 85 °C memory.
- The old "125 W quiet / 180 W benchmark" policy in [Cooling and power](COOLING-AND-POWER.md) predates the four-card TP4 workload; this table replaces it for that workload.

**Fan control lesson (measured, Supermicro BMC):** if a fan controller switches the BMC to "Full" mode and then writes zone duties once, a cold boot can leave every fan at 100%: the BMC applies the mode change after the first duty writes and the controller never notices. Read the duty back periodically and rewrite it if it changed.

## 7. Tuning with 170tune: what it is and its sharp edges

[cachenetics/170tune](https://github.com/cachenetics/170tune) is a per-card tuning and qualification harness: SM VF offset (undervolt) and clock ceiling, HBM NDIV, DRAM timings and refresh. Its central rule is that a benchmark finishing proves nothing; only a hot gate (soak, full-VRAM pattern sweeps, bit-exact compute) qualifies a setting, and `persist` refuses a setting without that card's gate receipt. Lessons from using it on four cards:

- **One card under test at a time.** A second gate refuses while another card's run is in flight.
- **Gate at the card's real serving temperature.** The default `GATE_TEMP` is 60 °C; we serve at HBM 72–75 °C, so we gated at 75 °C. A cooler card that cannot reach the target waits out the whole soak (`GATE_SOAK_MAX`) before every sweep: 8 min per sweep instead of 3. Gate such a card at its own measured serving peak.
- **Power-limit side effects.** Gates soak at 300 W; `recover` and the crash-revert boot path set **250 W**. Re-apply your cap afterwards (we use a systemd drop-in after 170tune's services).
- **The card boots stock first.** A persisted profile is applied after the driver is up; if a tuning run was in flight when the machine went down, the next boot stays stock.
- **A passed SM-offset gate does not prove the offset applied.** On the 250 W VBIOS NVML exposes a VF offset range of [0..0]; the set fails and reads back 0, and the gate still reports GATED because it tests whatever the card runs. Check the range (`nvml_oc`) and the readback first (measured).
- **Do not combine with a driver that bakes a memory clock** (`--mclk-ndiv` in some cmpunlocker forks): 170tune refuses, because the stock snapshot would be wrong.
- **What it bought here:** the HBM equalization (+6% single-user). The SM offset (+200 at 1,410 MHz on the two cards that accept it) gave +0.6% and −6 W at 140 W, within noise: in TP4 the two cards without an offset still set the pace. Its reference card's own data says the offset mainly saves power at a fixed clock (same throughput at −24% power).

## 8. The driver stack we run

On the 4-card PLX host (NVIDIA 615.71.09 open kernel modules, kernel 7.0.2-6-pve):

1. [amoghmunikote/cmpunlocker](https://github.com/amoghmunikote/cmpunlocker) @ `88e39ce` — memory unlock, compute throttle, Gen2 link, 64 GB BAR1, late-PMA fix
2. `6c442ee` — +4 SMs (70 → 74)
3. PR #60 `hbm-control-plm.patch` — HBM clock privilege masks (opens control only)
4. `rebar-serialize.patch` — one mutex around the per-card BAR1 resize (cards behind one PLX switch otherwise race and corrupt the kernel resource tree)
5. Four BAR1 peer-to-peer patches from [admunch888/cmpunlocker](https://github.com/admunch888/cmpunlocker) (ported from bayley's work), loaded with `RMForceStaticBar1=1;RMPcieP2PType=1`, after a pre-driver step that places each card's BARs and kexecs once so the kernel adopts the layout. If any card does not show a 64 GB BAR1, the driver loads without P2P.

Build check before installing: apply the full patch series to a clean source tree (`patch --dry-run` per patch); all 17 applied cleanly on 615.71.09. Install, then a **cold** boot.

## 9. Software: engines, the sm80vllm fork and its container image

- **Serving engine of record for GLM-5.3-Flash TP4:** [Morrowmake recipe v1.6.0](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe/tree/v1.6.0), image `ghcr.io/morrowmake/vllm-cmp170hx@sha256:cc26c8abb63953a37c6c1861cadc9188e99c1cceab403600b5822740f3a82ae0`, P2P off, replicated embedding. Drafter `incoai/GLM-5.3-Flash-DFlash2` is CC BY-NC-ND 4.0 (benchmarks only). The Apache-2.0 `canada-quant/GLM-5.3-Flash-DFlash2-G` **does not load** on this engine (`indexer.k_cache: page size is not divisible by the maximum page size`) — negative result, open.
- **PixelML/sm80vllm, branch [`glm53-sm80-tf-learnings`](https://github.com/PixelML/sm80vllm/tree/glm53-sm80-tf-learnings):** our fork's lane for porting ideas from TensorFold's GLM-5.3 recipe for DGX Spark.
  - **Why not run TensorFold itself:** it needs compute capability ≥ 9.0 features (thread-block clusters, FP8 MMA) and hard-wires GLM to two ranks; CMP 170HX is sm80 and serves on four. **Inferred** from its source; not attempted.
  - `127c6f076` merges Morrowmake's `mm/ampere` at `3a2bf16da` (the engine the recipe serves) into the fork. With copy drafts off it decodes 0.5–1.9% below the Morrowmake image's 2026-10-01 numbers (**measured**; 150 W vs that run's 180 W, so parity within the cap difference).
  - `c93c274c8` adds **copy drafts** (`VLLM_GLM5_COPY_DRAFTS=1`, model runner V2, `vllm/v1/worker/gpu/spec_decode/copy_drafts.py`): when the reply repeats a span of the prompt or of itself, the next tokens are proposed from that span. **Measured** on the 70-SM driver, 150 W: an edit task that returns a file with a rename runs 256 → 351 tok/s (+37%, byte-identical reply); a second edit task runs 254 → 327 tok/s (+28%, reply differs); the general decode matrix is 0.3–2.2% slower. The engine is not run-to-run reproducible at temperature 0, so exactness checks need pinned cache state ([2026-10-02 notebook §2.5](../notebooks/2026-10-02-glm-5.3-flash-4card-tp4-sm74-hbm-power-vllm.ipynb)).
  - **Container image (published):** `ghcr.io/pixelml/sm80vllm@sha256:704cbb8841be7c99bd1d5d6a8136fc118ca22cfc85457ef4a669a07ef957e8ff` (`tf-learnings-127c6f076`, full build of `127c6f0761b2c81d2c0ba7e2d8a5cf71e3247903`) and `ghcr.io/pixelml/sm80vllm@sha256:f0dbb483b000f85ac66adb6f190c11d1af395914df67a0d9b7e54dfe2258fa86` (`tf-learnings-c93c274c8`, the four-file copy-drafts overlay of `c93c274c86543d513c67243859480aeec46bc951`). Pull by digest. Pinned build recipe (CUDA base image by digest, engine and wheel by full commit): `docker/cmp170hx/` in [PixelML/sm80vllm](https://github.com/PixelML/sm80vllm).
  - **Next on that lane (untested):** cheaper per-step kernels, wider verify windows, a noise-aware stop rule, dense q4, FP8 KV storage on sm80.

## 10. What is still out of reach, or open

| Item | Status |
|---|---|
| PCIe Gen3/Gen4 on fuse-set cards | blocked in software (§3) |
| NVLink | hardware absent |
| ECC | upstream in progress; untested |
| HBM above NDIV 64 | needs HBM ≤ ~76 °C under load; not tried |
| SM VF offset | measured: within noise in TP4 here; impossible on the 250 W VBIOS |
| Single-user code decode gap vs Morrowmake (310 vs 377 tok/s) | open; same acceptance, so engine/kernel |
| An Apache-2.0 drafter that loads (DFlash2-G) | open |
| 500 tok/s structured single-user (now 418) | needs ~15.3 ms per step or ~8 accepted tokens per step; hardware levers above add a few percent at most, so this sits with the engine/drafter work (inferred) |

## 11. Older statements this page supersedes

- "PCIe Gen1 is the card's advertised maximum" / "Gen2 software claim contested" ([Hardware](HARDWARE.md#pcie-link-status), [Topology §1](TOPOLOGY-AND-PARALLELISM.md#1-what-the-link-really-is), [Research digest §3](RESEARCH-CMP170HX-UNLOCKS.md#3-pcie)): true for the stock driver; the cmpunlocker link patch gives Gen2 x16 on our cards (measured). Gen3 is fuse-blocked.
- "70 SMs": 74 with `6c442ee`.
- "No P2P over PCIe": true without the extra patches; with them, static-BAR1 P2P works on the 4-card PLX host (it helps TP2 and hurts TP4 there).
- "125 W quiet / 180 W benchmark" ([Cooling and power](COOLING-AND-POWER.md)): for the four-card TP4 workload we now run 140 W (§6).

## See also

- [Notebooks](../notebooks/README.md) — every executed experiment
- [Lessons](LESSONS.md) · [Operator lessons](OPERATOR-LESSONS.md) · [Troubleshooting](TROUBLESHOOTING.md)
- [Unlock and mod research digest](RESEARCH-CMP170HX-UNLOCKS.md) · [Card and quant research](CARD-AND-QUANT-RESEARCH.md)
