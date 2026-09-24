# 2026-09-22-mimo-v2.6-flash-4card-pp4-vllm

MiMo-V2.6-Flash-RL (XiaomiMiMo, revision 5711b268) on CMP 170HX cards (SM80,
64 GiB each) behind one vLLM server with pipeline parallelism. Serves correctly;
PP3 on three cards with the native MTP head (k=2) is the recommended shape:
117 tok/s greedy, 91 tok/s sampled single-stream decode (75 tok/s without MTP),
495 tok/s aggregate at 32 streams. The shipped DFlash drafter reaches 162 tok/s greedy.
TP4 measured 4.6x slower (no P2P, Gen1 links).
Notebook: `notebooks/2026-09-22-mimo-v2.6-flash-4card-pp4-vllm.ipynb`.

- `receipts/bench/{pp3-dflash7,pp3-dflash4,pp3-mtp2,pp3-mtp3,pp3,pp4}/` — gate.json, p1.json, p2.json, launch.json
- `receipts/conc/` — aggregate-throughput sweep (PP3, PP3+MTP2, PP4, TP4)
- `receipts/runtime_attempts.json` — every attempt A1-A16 with evidence
- `receipts/fork_resolution.json` — how the fork image covers the SM90-only features
- `patches/mimo_v2.py` — upstream vLLM #57508 + #57784 backport + PP aux-hidden relay for DFlash (required)
- `patches/mimo_v2_mtp.py` — upstream #57508 backport for the MTP head (MTP only)
- `patches/mimo_v2_omni.py`, `patches/qwen3_dflash.py` — upstream #57784 backports (DFlash only)
- `patches/cmpunlocker-late-pma-wprfix.diff` — driver fix for unlocked cards (Xid 31)
- `patches/swa_memory_pool.py` — SGLang PP fix from the earlier SGLang attempts
- `tools/` — bench_mimo.py, gate.py, verify_checkpoint.py, probe_sm80.sh, build_notebook.py
