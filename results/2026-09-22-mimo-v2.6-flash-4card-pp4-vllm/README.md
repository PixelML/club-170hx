# 2026-09-22-mimo-v2.6-flash-4card-pp4-vllm

Run in progress. MiMo-V2.6-Flash-RL (XiaomiMiMo, revision 5711b268) on four
CMP 170HX cards (SM80, 64 GiB each) behind one vLLM server, pipeline parallel 4.

Scope and authorization: owner-directed run, committed before the first
expensive load. Receipts land here as gates pass:

- `gate.json` — boot + functional + deterministic-greedy gate
- `p1.json` — greedy per-workload decode, median of 3
- `p2.json` — sampled decode, median of 5
- `conc_sweep.json` — concurrency scaling
- `prefill.json` — prefill/TTFT sweep
- `stability.json` — sustained rounds with power + thermals

Bench scripts are committed beside the receipts they produced.
