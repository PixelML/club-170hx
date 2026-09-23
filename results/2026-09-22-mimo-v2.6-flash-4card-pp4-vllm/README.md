# 2026-09-22-mimo-v2.6-flash-4card-pp4-vllm

MiMo-V2.6-Flash-RL (XiaomiMiMo, revision 5711b268) on CMP 170HX cards (SM80,
64 GiB each) behind one vLLM server with pipeline parallelism. Serves correctly;
PP3 on three cards is the recommended shape (75 tok/s single-stream decode).
Notebook: `notebooks/2026-09-22-mimo-v2.6-flash-4card-pp4-vllm.ipynb`.

- `receipts/bench/{pp3,pp4}/` — gate.json, p1.json, p2.json, launch.json
- `receipts/runtime_attempts.json` — every attempt A1-A13 with evidence
- `receipts/fork_resolution.json` — how the fork image covers the SM90-only features
- `patches/mimo_v2.py` — the only runtime patch (upstream vLLM #57508 + #57784 backport)
- `patches/cmpunlocker-late-pma-wprfix.diff` — driver fix for unlocked cards (Xid 31)
- `patches/swa_memory_pool.py` — SGLang PP fix from the earlier SGLang attempts
- `tools/` — bench_mimo.py, gate.py, verify_checkpoint.py, probe_sm80.sh, build_notebook.py
