# Qwen3.8-Flash-Next on one CMP: guarded reproduction

**Goal:** Reproduce the licensed Flash-Next CLI recipe and its same-quant llama.cpp baseline on exactly one CMP 170HX.
**Done when:** Either record a clear route/safety stop before load, or commit the receipts, executed replay notebook, and inspected comparison below.
**Where it runs:** Trusted coordination in this worktree; all third-party code through `yx run qwen38-flash-1cmp …`.
**Budget:** Download at most 120 minutes; build, smoke, and benchmark at most 60 additional minutes. Model storage at most 200 GiB. No paid model calls. At most two retries per changed hypothesis; no identical blind retry.
**Push allowed:** `worktree-qwen38-flash-1cmp-20261109` only, after review; never force or main. The planner does not push.

## Identity, credits, and licences

Use club source at `c375fe520147635ec2f81f3535095e0aa6acf24f`, not the older 27B recipe. The branch already matches this source; no fast-forward is required.

| Input | Author and immutable source | Licence |
|---|---|---|
| Model | Qwen ([@QwenLM](https://github.com/QwenLM)), [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540) | qwen-community-1.0 |
| GGUF | Unsloth ([@unslothai](https://github.com/unslothai)), [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/766911a6b7369840a91dbcd95f9f997acaab6cd6), UD-Q3_K_XL | qwen-community-1.0 |
| Kernel design | Louis Forster ([@L-Forster](https://github.com/L-Forster)), [open-jet](https://github.com/L-Forster/open-jet/tree/93c2b9abee50ea2ea41981c406cfd201989e4d58/megakernel) | AGPL-3.0 |
| Graph, tokenizer, tables, baseline | ggml-org ([@ggml-org](https://github.com/ggml-org)), [llama.cpp](https://github.com/ggml-org/llama.cpp/tree/aa94f2086147cd4bfbcff818ce998dffce5fcb39) | MIT |
| Engine implementation | Sean Phan ([@seanphan](https://github.com/seanphan)), [club-170hx source](https://github.com/PixelML/club-170hx/tree/c375fe520147635ec2f81f3535095e0aa6acf24f/results/2026-10-10-qwen3.8-flash-next-megakernel-1card/src) | `fnx_mk.cu`: AGPL-3.0-or-later; tables: MIT |

These works are authored by the named authors. Preserve their notices and source access obligations.
The four prompts are the existing repository fixtures, not a new dataset. No external evaluation dataset or judge is used.
The model licence permits internal use but restricts some commercial services. Read the pinned LICENSE before use; no public endpoint or third-party model access.

**Pin validation, 2026-11-09:** Public HF metadata resolved both exact revisions. GitHub resolved both exact runtime/design commits. The base LICENSE states `Qwen Community License 1.0`; the quant card names the same licence. This validates identity, not downloaded weights or a runtime build.

Expected shard bytes and SHA-256, from pinned HF LFS metadata:

| Shard suffix | Bytes | SHA-256 |
|---|---:|---|
| `00001-of-00003.gguf` | 10946624 | `f2ef4328929d8b8c8930e2856eef52128dd4ce3425302f04bc3c657431cc4c49` |
| `00002-of-00003.gguf` | 49983253824 | `7d230e7c9421d868b89eebaf23033af0ea1a4e046956df00fb156814fb62346e` |
| `00003-of-00003.gguf` | 39992153376 | `21d4f90f9cd7b7c3a1582667c20cb22f7b03de895b88a23bb20aaeaa44f2c199` |

## Contract and hypothesis

Source notebook: `notebooks/2026-10-10-qwen3.8-flash-next-megakernel-1card.ipynb`.
Source receipts: `results/2026-10-10-qwen3.8-flash-next-megakernel-1card/`.

**Previously measured, not a new result:** 70.64 versus 53.36 decode tok/s, 1.324x, using the same GGUF. Teacher-forced agreement was 1870/1890 positions. The baseline reached 82 °C core; that run violated the present stop rule. Do not reproduce that violation.
**Hypothesis:** The megakernel has higher warm decode throughput on these four prompts, without a material functional regression.
**Prediction:** The speed ratio exceeds 1.0; the historical 1.324x is a reference, not a promised target.
**Decisive baseline:** llama.cpp at the exact pin, same GPU, GGUF, prompt bytes, greedy settings, output cap, and unchanged power limit.
**Change the decision if:** The median ratio is at most 1.0, the guard trips, outputs fail inspection, or teacher-forced agreement is below 98% overall.

Batch 1, concurrency 1, no speculation, thinking disabled by the existing template, prompt plus output at most 2048 tokens. Each generation stops at EOS or 512 tokens. Keep actual token counts; do not pad EOS results or assume 512 tokens were produced.
The four fixed prompts cover code, explanation, narrative, and arithmetic. They are reproduction fixtures, not a broad quality evaluation. Math must yield 80 km/h; code must terminate and express a valid quicksort; chat and story must answer their prompts without loops or corrupt text. Inspect all four outputs per engine and every failure.
No tuning or winner selection on these prompts. Mark held-out quality, vision, tools, long context, concurrency, speculation, energy cost, and other topologies `untested: outside the one-card reproduction contract`.
Report speed only if both arms pass the declared functional floor. Do not claim max-quality or balanced winners from this test.

## Global constraints and stop conditions

1. No host, VM, bridge, driver, passthrough, cluster, harness, or agent configuration changes. No power or clock changes. No workloads outside the selected device.
2. Do not run third-party code on the workbench or directly on another GPU machine. Do not forward credentials or substitute `YX_HOST` to bypass isolation.
3. Stop if the authorized isolated GPU route is absent, the selected GPU is busy, airflow lacks confirmation, telemetry is unavailable, or device health is unsafe.
4. Stop at core >=80 °C or memory >=85 °C, Xid, GPU loss, memory errors, process failure, storage path mismatch, budget expiry, or no progress. Preserve partial receipts.
5. Use `/models` only for weights, HF cache, PLE, and repacked cache. Never use WIP, root filesystem, or a PVC for these files. Do not delete or overwrite pre-existing model files.

## Task 1: Prove the execution route before any download or load

Read `lab-kit/README.md`, `lab-kit/bin/yx`, and the route documentation supplied by the infrastructure owner. `yx` executes in an isolated job directory; it does not provide GPU forwarding. The documented VM creation recipe says **no GPU**. A repaired network or more RAM on another guest does not establish a GPU route.

The planner's read-only probe returned exit 124 after 15 seconds, with no output:

```bash
timeout 15s yx run qwen38-flash-1cmp 'command -v nvidia-smi && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
```

**Current status: blocked; no verified GPU execution route.** Infrastructure restoration is separate work. Do not perform it under this plan.

After the infrastructure owner supplies an existing authorized route, rerun exactly:

```bash
timeout 20s yx run qwen38-flash-1cmp 'set -eu; command -v nvidia-smi; nvidia-smi --query-gpu=index,name,memory.total,memory.free,temperature.gpu,power.limit --format=csv,noheader; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; findmnt -T /models -o TARGET,FSTYPE; df -B1 /models; free -b'
```

**Verify:** The isolated environment can see one authorized, unused CMP 170HX with about 64 GiB, and the owner confirms forced airflow. If several devices are visible, obtain a private selected index and restrict every child with `CUDA_VISIBLE_DEVICES`; never publish device identifiers. `/models` must be configured model storage, not a root/PVC fallback. The PLE/cache subdirectory must use local storage; stop if only network storage is available. Require 200 GiB free for a fresh run, or a verified accounting of existing files plus sufficient free space within the same 200 GiB allocation. Require at least 32 GiB total RAM and sufficient available RAM for the 26.8 GiB PLE mapping plus runtime overhead; stop if uncertain. Extra RAM elsewhere is irrelevant.

**Produces:** Private preflight receipt and a sanitized `route.json`, with `passed` or `blocked`, reason, card count/class, memory, storage capacity, and no host identifiers. A blocked receipt is a valid bounded result; do not continue.

## Task 2: Prepare the guarded workspace and exact source

Only after Task 1 passes, create the isolated workspace:

```bash
yx run qwen38-flash-1cmp 'set -eu; mkdir -p work out; test ! -e work/club; git clone --no-checkout https://github.com/PixelML/club-170hx.git work/club; git -C work/club checkout --detach c375fe520147635ec2f81f3535095e0aa6acf24f; git -C work/club rev-parse HEAD'
```

Use a fresh source directory for each upstream download. Keep trusted guard code in `work/control`, separate from cloned code. Use `python3 -I` for Python helpers. Capture environment versions, dependency versions, exact build commands, and source SHA-256 in the manifest. Prefer an existing authorized CUDA environment; the source notebook's image pin is `docker.io/nvidia/cuda@sha256:020bc241a628776338f4d4053fed4c38f6f7f3d7eb5919fecb8de313bb8ba47c`. Do not invent a replacement digest. No container or package setup on the workbench. If the authorized environment lacks dependencies and safe installation would exceed budget, stop.

Implement `work/control/guard.py` before model load. Its CLI is `python3 -I work/control/guard.py --seconds 3600 -- command args…`. It must start a new process group, sample the selected GPU once per second, and terminate then kill the complete child group on any stop condition. Capture core and memory temperatures, power, memory, utilization, health errors, timestamp, and exit reason. Missing/invalid samples fail closed; do not substitute zero for missing memory temperature. Check storage growth and the shared deadline. The guard must cover the server's entire lifetime, not only HTTP requests. Record current power limit; do not change it.

Add a fake-telemetry self-test mode. It must prove normal exit, core=80 stop, memory=85 stop, missing sample stop, child-group termination, and deadline stop without a GPU workload.

**Verify:** `yx run qwen38-flash-1cmp 'python3 -I work/control/guard.py --self-test'` prints `6/6 guard tests passed`. Review its code before use. Confirm source HEAD is the exact club pin and record the selected environment digest or source/tool versions. No load if this gate fails.
**Produces:** Guard, test receipt, environment manifest, and source hashes. Start one shared 60-minute execution deadline after the separate download phase, before build. Enforce remaining time on build, PLE extraction, and every guarded command; do not reset this budget between commands.

## Task 3: Acquire only the pinned GGUF and build the two engines

Run within the authorized environment, through `yx`. Install dependencies there only if needed. Record exact installed versions. Set `HF_HOME`, `TMPDIR`, and every model output path under `/models/qwen38-flash-1cmp`. Avoid a second weight copy. No base checkpoint or vision projector download.

```bash
yx run qwen38-flash-1cmp 'set -eu; mkdir -p /models/qwen38-flash-1cmp/hf /models/qwen38-flash-1cmp/tmp /models/qwen38-flash-1cmp/local; export HF_HOME=/models/qwen38-flash-1cmp/hf TMPDIR=/models/qwen38-flash-1cmp/tmp; timeout 7200s hf download unsloth/Qwen3.8-Flash-Next-GGUF --revision 766911a6b7369840a91dbcd95f9f997acaab6cd6 --include "UD-Q3_K_XL/*" --local-dir /models/qwen38-flash-1cmp; sha256sum /models/qwen38-flash-1cmp/UD-Q3_K_XL/*.gguf'
```

Compare all three sizes and digests with the table above. Stop on any mismatch. Keep any partial file within the 200 GiB budget; do not repeat a full download into another location.

```bash
yx run qwen38-flash-1cmp 'set -eu; git clone https://github.com/ggml-org/llama.cpp.git work/llama.cpp; git -C work/llama.cpp checkout --detach aa94f2086147cd4bfbcff818ce998dffce5fcb39; cmake -S work/llama.cpp -B work/llama.cpp/build-sm80 -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release; cmake --build work/llama.cpp/build-sm80 -j4 --target llama-server llama-bench; make -C work/club/results/2026-10-10-qwen3.8-flash-next-megakernel-1card/src LLAMA="$PWD/work/llama.cpp" BUILD="$PWD/work/llama.cpp/build-sm80" ARCH=sm_80'
yx run qwen38-flash-1cmp 'python3 -I work/club/results/2026-10-10-qwen3.8-flash-next-megakernel-1card/src/extract_ple.py /models/qwen38-flash-1cmp/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00002-of-00003.gguf /models/qwen38-flash-1cmp/local/ple.bin'
```

Do not overwrite an existing PLE/cache file. Reuse only after source/model provenance and size validation; otherwise stop. Rough storage is 83.8 GiB shards + 26.8 GiB PLE + 57 GiB repack = 168 GiB. The cache writer uses a temporary file then rename; preserve headroom and do not create a second cache.

**Verify:** Three shard digests match; both binaries exist; source HEAD matches; the PLE is about 26.8 GiB; total model storage remains below 200 GiB. Record actual toolchain/driver/kernel versions, not historical versions.
**Produces:** Binaries, local PLE, weight manifest, build logs, dependency versions. Preserve failure class and exact sanitized error if build fails.

## Task 4: Smoke, warm, then measure under the guard

Use the source `src/prompts.tsv` unchanged. Define `SRC` within each remote shell as shown. Use environment variables for the selected GPU; never assume a global GPU index. The engine has no sampling flag because it is greedy only.

```bash
yx run qwen38-flash-1cmp 'SRC=work/club/results/2026-10-10-qwen3.8-flash-next-megakernel-1card/src; python3 -I work/control/guard.py --seconds 3600 -- env MODELS=/models/qwen38-flash-1cmp LOCAL=/models/qwen38-flash-1cmp/local CACHE=/models/qwen38-flash-1cmp/local/weights.blob bash "$SRC/run.sh" -p "Explain how a hash map works, in simple terms." -n 32 >out/mk-smoke.txt 2>out/mk-smoke.log'
yx run qwen38-flash-1cmp 'SRC=work/club/results/2026-10-10-qwen3.8-flash-next-megakernel-1card/src; python3 -I work/control/guard.py --seconds 3600 -- env MODELS=/models/qwen38-flash-1cmp LOCAL=/models/qwen38-flash-1cmp/local CACHE=/models/qwen38-flash-1cmp/local/weights.blob bash "$SRC/run.sh" --bench "$SRC/prompts.tsv" --reps 1 -n 512 >out/mk-warmup.jsonl 2>out/mk-warmup.log'
yx run qwen38-flash-1cmp 'SRC=work/club/results/2026-10-10-qwen3.8-flash-next-megakernel-1card/src; python3 -I work/control/guard.py --seconds 3600 -- env MODELS=/models/qwen38-flash-1cmp LOCAL=/models/qwen38-flash-1cmp/local CACHE=/models/qwen38-flash-1cmp/local/weights.blob bash "$SRC/run.sh" --bench "$SRC/prompts.tsv" --reps 3 -n 512 >out/mk-bench.jsonl 2>out/mk-bench.log'
```

The shared deadline overrides each command's upper bound. Record cold repack/load separately from warm measurements. Inspect smoke before either subsequent command. Stop if the engine emits errors, a loop, or invalid token output.

Start the same-quant baseline under the same guard. Use an unused loopback port; the reference helper requires 8090. If 8090 is occupied, stop rather than kill or reconfigure an existing service.

```bash
yx run qwen38-flash-1cmp 'python3 -I work/control/guard.py --seconds 3600 -- work/llama.cpp/build-sm80/bin/llama-server -m /models/qwen38-flash-1cmp/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf -ngl 99 -ot "per_layer_token_embd=CPU" -fa on -c 2048 -np 1 --port 8090 --host 127.0.0.1 >out/llama-server.log 2>&1'
```

The executor runs that foreground command in its managed background task. It retains the child process-group identity for cleanup. Do not use an unmanaged detached server. Wait for `/health` with a maximum 300-second deadline; preserve the guard's monitoring across requests.

```bash
yx run qwen38-flash-1cmp 'curl --fail --silent --max-time 10 http://127.0.0.1:8090/health; curl --fail --silent --max-time 60 http://127.0.0.1:8090/completion -H "Content-Type: application/json" -d '\''{"prompt":"Explain a hash map.","n_predict":32,"temperature":0,"top_k":1,"cache_prompt":false}'\'' >out/llama-smoke.json'
yx run qwen38-flash-1cmp 'set -eu; SRC=work/club/results/2026-10-10-qwen3.8-flash-next-megakernel-1card/src; for r in warmup r0 r1 r2; do mkdir -p "out/ref-$r"; python3 -I "$SRC/ref_llamacpp.py" "out/ref-$r" 512 >"out/ref-$r.log"; done'
```

Check tokenized prompt plus output <=2048. Retain all warmup receipts but exclude them from the primary table. This yields 12 measured generations per arm. Stop the baseline process group after its requests, before another GPU load. Never stop another service.

For each reference prompt, read its `prompt_tokens` from the JSON. Run teacher-forced comparison on the reference IDs with that count, not the story's historical constant 29:

```bash
yx run qwen38-flash-1cmp 'set -eu; SRC=work/club/results/2026-10-10-qwen3.8-flash-next-megakernel-1card/src; for p in code chat story math; do P=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))[\"prompt_tokens\"])" "out/ref-r0/$p.json"); python3 -I work/control/guard.py --seconds 3600 -- env MODELS=/models/qwen38-flash-1cmp LOCAL=/models/qwen38-flash-1cmp/local CACHE=/models/qwen38-flash-1cmp/local/weights.blob bash "$SRC/run.sh" --score "out/ref-r0/$p.ids" --score-prompt "$P" >"out/score-$p.txt" 2>"out/score-$p.log"; done'
```

**Verify:** 12 valid measured rows per arm; each has actual token counts and timings. All guard exits are normal. Inspect all fixed-prompt outputs for both arms. Decode megakernel IDs with the pinned tokenizer if needed; do not treat IDs alone as inspected text. Record exact agreement counts and all divergences. A score below the floor is a quality failure, not an isolation workaround or reason to change prompts.
**Produces:** Smoke, warmup, measured JSON/JSONL, generated text, teacher-forced scores, telemetry, and cold/warm load times. A thermal stop means `stability failure`; no automatic power change or resume.

## Task 5: Private usability, receipts, and review

The megakernel is a CLI program with **no HTTP endpoint**. State this explicitly. The separately tested llama.cpp endpoint is not the megakernel. Do not add a custom HTTP wrapper, change provider configuration, restart the harness, or change `agent-default-model`.
If an existing supported private baseline provider is already selectable, send one fixed text request without configuration changes and record provider/model names privately. Otherwise mark harness selection `not verified: no approved existing provider`. The loopback baseline smoke above proves only the baseline endpoint. Full harness delivery remains incomplete without a real supported request; do not claim otherwise.

Pull receipts after each phase:

```bash
yx pull qwen38-flash-1cmp out
```

Review and sanitize before copying anything into the repository. Keep private raw receipts outside git. Do not publish hostnames, device IDs, PCI addresses, process IDs, mount sources, private paths, or full infrastructure logs.

Create only these experiment outputs:

- `results/2026-11-09-qwen38-flash-1cmp/`: sanitized manifest, guard and self-test, harness, all pass/fail receipts, `summary.json`, `summary.csv`, output inspection, and methodology.
- `notebooks/2026-11-09-qwen3.8-flash-next-1card-megakernel.ipynb`: the four sections from `notebooks/README.md`, all outputs committed, `LIVE=False` receipt replay. CLI live mode must disclose the absent HTTP API.
- `assets/charts/2026-11-09-qwen38-flash-1cmp.{png,svg}` plus chart source. Load the dataviz skill before chart code. Show per-prompt medians, sample counts, ranges, actual output lengths, and clear source files.
- Add links to `README.md` and `docs/BENCHMARKS.md` only if a measured notebook passes. Preserve old results and all upstream credits.

For a blocked route, commit only sanitized status evidence and a short explicit stop reason; do not create fake notebook metrics or charts. For a passing route, aggregate three measured repetitions per prompt. Report median throughput, per-prompt range, actual lengths, end-to-end latency, success count, and core/memory peaks. Do not present three repetitions as strong uncertainty estimates. Report cold/warm values separately. Mark power comparison invalid if limits differ. Tokens/s alone do not prove quality or cost.

**Verify, from the isolated worktree:**

```bash
git diff --check
python3 scripts/validate_recipe_notebooks.py
jupyter nbconvert --to notebook --execute --inplace notebooks/2026-11-09-qwen3.8-flash-next-1card-megakernel.ipynb
git diff --stat
git diff --cached
```

The validator must pass; replay needs no GPU/network. Skip notebook execution only for the explicit blocked-route result. Review tracked files for secret values and infrastructure identifiers; inspect large files and symlinks. Review the full staged diff before commit. Preserve source hashes and source notices. Do not commit weights, generated caches, binaries, or unsanitized logs.

**Done receipt:** State `blocked` or `measured`; list proof commands and their actual outputs, current commit, artifact paths, limitations, and endpoint/harness status. If measured, quote both arm medians with sample counts and the quality checks. If blocked, quote the failed route check and stop without a download or workload.
