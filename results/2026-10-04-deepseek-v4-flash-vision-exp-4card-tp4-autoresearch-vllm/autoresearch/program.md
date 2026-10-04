# autoresearch: DeepSeek-V4-Flash-Vision on 4x CMP 170HX (sm_80)

Goal: make DeepSeek-V4-Flash-Vision-Exp fast and correct on this rig. You run an endless research loop,
in the style of karpathy/autoresearch: edit → run one experiment → read the score → keep or reset → repeat.

## Machine
- Run everything on the server container host over SSH: `ssh <host> '<cmd>'`.
- Repo (the only place you edit): `<repo>` (git). Fixed harness: `<bench>` (never edit).
- 4x CMP 170HX: sm_80, 64 GiB each, no FP8 tensor cores, PCIe Gen2 behind 2 PLX switches, verified P2P, 140 W cap.
- The image is pinned in `bench/run_experiment.sh`. `./vllm` (the image's editable `/vllm/vllm` tree) is mounted over it.
  Python and Triton edits take effect on the next boot. There is no nvcc build step: do not edit `.so` files.

## What you may edit
- `vllm/**` Python and Triton source (DeepSeek-V4 code: `vllm/models/deepseek_v4/**`, MoE, attention, spec decode, all-reduce).
- `serve.env` (container env, one `VAR=value` per line) and `serve.args` (vLLM serve arguments, one line).

## What you must not do
- Do not edit `<bench>/*`, the image, the weights, the driver, the host, the LXC config or the power limit.
- Do not start, stop or touch any other container (for example `dsv41-tp4`). Only `run_experiment.sh` starts `dsv4v`.
- Do not delete anything under `/models`. Do not pin large host RAM (the host has 251 GB; this container is capped at 192 GiB).
- Stop the loop and report if a card reaches 80 °C core, if `dmesg` shows a new Xid other than 43/31 from your own crash,
  or if a GPU disappears from `nvidia-smi`.

## One experiment
```
ssh <host> 'cd <repo> && bash <bench>/run_experiment.sh > run.log 2>&1; grep -E "^(boot_s|prefill_tok_s|c1_tok_s|c4_agg_tok_s|c4_pass|gates_passed|status|out):" run.log'
```
About 12–15 minutes (boot is 8–11 of them). The script stops the server at the end. Receipts are in the `out:` directory.

Gates (all must pass for a "keep" once the baseline passes them): 1 deterministic greedy check, 2 five known answers,
3 five short-prompt probes on-topic, 4 image gradient answer, 5 four concurrent requests without failure,
6 server still alive at the end, 7 no new Xid.

## Objective (lexicographic)
1. `gates_passed` (higher is better). The baseline fails gates 2–3 (bare short prompts drift off-topic) and sometimes 5
   (illegal memory access at 4 concurrent requests). Fix correctness first.
2. Then `c1_tok_s` (single-user decode, median of 3, 400 tokens, DSpark drafter): higher is better.
   Constraints: `c4_agg_tok_s` and `prefill_tok_s` must not drop more than 10% below the best kept run.
- Noise: c1 varies ±20% with draft acceptance. A gain under 8% counts only after a second run of the same commit
  confirms it (use the mean of the two).
- Simplicity: a small gain with ugly code is not worth it. Equal speed with less code is a win.

## Setup (once)
1. `git checkout -b autoresearch/<date>` in the repo.
2. Create `results.tsv` (untracked) with header: `commit	gates	c1_tok_s	c4_agg_tok_s	prefill_tok_s	status	description`.
3. Wait until `<bench>/chain.log` contains `CHAIN-DONE` (another bench owns the GPUs until then).
4. Run the unmodified baseline first.

## Loop (do not stop to ask the human)
1. Pick one idea. Edit. `git commit -am "<idea>"`.
2. Run one experiment. If `status: crash_boot`, read `out:/container.log` (tail), fix a trivial bug or drop the idea.
3. Append a row to `results.tsv`: status `keep`, `discard` or `crash`.
4. Keep the commit if the objective improved. Else `git reset --hard HEAD~1`.
5. Every 5 experiments, append a 3-line summary to `<repo>/NOTES.md` (what worked, what failed, next).

## Ideas to start from (measured facts first)
- Measured: experts use Marlin MXFP4 (`Using 'MARLIN' Mxfp4 MoE backend`); the FP8 spine uses weight-only Marlin FP8.
  The log suggests `VLLM_MARLIN_USE_ATOMIC_ADD=1` for small `size_n`.
- Measured: TP4 prefill 1,292 tok/s with P2P on vs 1,529 with P2P off: `NCCL_P2P_LEVEL=SYS` may slow large NCCL messages.
  Try leaving NCCL at its default while keeping the custom all-reduce for small messages.
- Correctness suspects: `vllm/models/deepseek_v4/eager_scratch.py`, `sparse_mla.py`, the DSpark drafter (`nvidia/dspark.py`),
  first-token state on short prompts, sliding-window or compressor state reuse between requests.
  Try without the drafter (remove `--speculative-config`) to locate the bug, then fix the real cause.
- Speed: GLM-5.3-Flash on the same cards does 396–418 tok/s single-user with Morrowmake's fused sm_80 decode kernels
  (github.com/Morrowmake/vllm-cmp170hx, branch ampere): fused mHC post+pre, MoE gate+top-k+align in one kernel,
  thin-M BF16 GEMM, indexer decode glue, shared-expert overlap on an aux stream. DeepSeek-V4 has the same blocks (mHC, MoE, sparse MLA).
  Port them as Triton kernels where the profile shows time.
- Profile before big work: vLLM's torch profiler (`VLLM_TORCH_PROFILER_DIR` + `/start_profile` and `/stop_profile`) to find the top decode kernels.
- Also try: cudagraph capture sizes, `--max-num-batched-tokens`, the custom all-reduce size cap, DSpark `num_speculative_tokens`, EP (`--enable-expert-parallel`).
