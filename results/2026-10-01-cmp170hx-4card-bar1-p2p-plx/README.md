# Interconnect — static-BAR1 P2P on 4× CMP 170HX behind two PLX switches (Broadwell host)

Status: measured
Date: 2026-10-01

Notebook: [`notebooks/2026-10-01-cmp170hx-4card-bar1-p2p-plx-cuda.ipynb`](../../notebooks/2026-10-01-cmp170hx-4card-bar1-p2p-plx-cuda.ipynb)
Earlier attempt on the same host (no P2P): [`results/2026-10-01-cmp170hx-3card-p2p-plx/`](../2026-10-01-cmp170hx-3card-p2p-plx/README.md)

## Verdict

Direct GPU-to-GPU access works on all 12 ordered pairs of four cards (every byte verified,
5.79 GB/s each, 16 µs at 4 KiB). It needs the memory unlock's 64 GiB BAR1 on **every** card,
and on this board two cards behind one PLX switch only get that with a hand-programmed PCIe
layout applied before the driver loads. P2P cuts small-copy latency by up to 1.77× and lifts
Qwen3.8-27B TP2 single-user decode by 11–14 %, but it **slows** GLM-5.3-Flash TP4 decode by
25–44 %: the host-staged all-reduce vLLM picks without P2P is faster on this PLX + Broadwell path.

## Hardware

- 4 × CMP 170HX (GA100, SM80), 65,536 MiB each, Gen2 x16; GPU0+GPU1 behind PLX PEX 8747 switch A, GPU2+GPU3 behind switch B, both switches on CPU2
- Host: dual Xeon E5-2686 v4, bare metal (one verify run and the GLM runs inside an LXC container on the same host)
- Power limit 180 W; forced airflow

## What it took

1. **Crash fix.** The unlock's driver resizes every card's BAR1 to 64 GiB in parallel per-GPU probe threads; two cards sharing a switch corrupted the kernel's resource tree (slab faults, `Bad swap file entry`). [`patches/rebar-serialize.patch`](patches/rebar-serialize.patch) serializes the resize.
2. **Why stock static-BAR1 P2P failed.** The BIOS places each switch's 64-bit window at the top of CPU2's 1 TiB aperture (224 MiB for switch A), so it cannot grow; the second card on a switch gets no BAR1 and static BAR1 refuses to start it (`-EIO`). A kernel remove/rescan with 64 GiB BARs still fails: its bridge sizing puts each 32 MiB BAR3 right after BAR1, leaving the second 64 GiB-aligned slot 32 MiB short ([`receipts/layout-before.txt`](receipts/layout-before.txt)).
3. **Fix.** With no driver bound: unlock BAR1 through config space (the same XVE writes the driver does), select the 64 GiB ReBAR size, program every BAR and bridge window by hand with each BAR3 directly **below** its 64 GiB-aligned BAR1, then kexec into the same kernel so it adopts the layout as firmware assignments ([`tools/cmp-bar-layout.py`](tools/cmp-bar-layout.py), [`receipts/layout-after.txt`](receipts/layout-after.txt)). [`tools/boot/`](tools/boot/) does this on every boot with a safe-mode fallback.
4. **Driver.** Upstream cmpunlocker + rebar-serialize + the four BAR1-P2P patches from admunch888/cmpunlocker (`p2p-caps-override`, `p2p-bar1`, `p2p-skip-mailbox-preinit`, `p2p-readcap-override`), `RMForceStaticBar1=1;RMPcieP2PType=1`.

## Receipts

| File | What |
|---|---|
| `receipts/p2p-latency-stock.json` | 3 cards, stock driver (host-staged), from the earlier attempt |
| `receipts/p2p-latency-bar1.json` | P2P driver, peer access **not** enabled (still staged) |
| `receipts/p2p-latency-bar1-peer.json` | P2P driver, peer access enabled, 3 cards |
| `receipts/p2p-latency-4card-peer.json` | P2P driver, peer access enabled, 4 cards |
| `receipts/p2p-verify-4card-{host,lxc}.txt` | byte-verified copies, 12 pairs, bare metal and inside an LXC container |
| `receipts/nccl/` | NCCL all-reduce logs, `nccl-*` = SHM (no P2P), `nccl-p2p-*` = P2P/CUMEM |
| `receipts/qwen/` | Qwen3.8-27B W4A16 TP2 suite + concurrency, with and without P2P (`*-p2p`) |
| `receipts/glm/` | GLM-5.3-Flash TP4 decode matrix (Morrowmake recipe 1.6.0 protocol), P2P on/off and variants |
| `receipts/layout-*.txt` | 64-bit window layouts before and after |

## Commands

```text
python3 tools/cmp-bar-layout.py --unlock --dry-run   # host-specific BDFs; edit CARDS/UPSTREAM first
python3 tools/p2p_latency.py --peer > p2p-latency.json
python3 tools/charts.py                               # re-render the nine charts from receipts/
```

## Limits

- The layout script hard-codes this board's bus numbers and four card positions; it refuses to write if any card or parent port differs.
- Cards in the CPU1 slots were not tested (that switch board's link was down during these runs).
- GLM numbers use Morrowmake's default DFlash2 drafter (CC BY-NC-ND 4.0) as a benchmark only.
