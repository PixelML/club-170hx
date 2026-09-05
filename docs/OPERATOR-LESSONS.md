# Operator lessons: running a two-lane DeepSeek Vision experiment

Mined from the operator transcripts of two parallel DeepSeek-V4-Flash-Vision-Exp
lanes run over several days: one on a single four-card CMP 170HX box, one on
a two-node setup with pooled memory across nodes. This page is the
operator/process layer: coordination, launch discipline, and multi-agent
process failure modes. For kernel, runtime, topology, and hardware
failure-mode lessons, see [LESSONS.md](LESSONS.md) and
[TROUBLESHOOTING.md](TROUBLESHOOTING.md); this page does not repeat those.

## 1. Repeated mistakes

| # | Mistake | What went wrong | Fix | Time lost |
|---|---|---|---|---|
| 1 | Host-vs-container path mismatch (3x) | A patch applied on the host was not bind-mounted into the running container, so the container kept executing stale code; a research-path bind mount also pointed at the wrong directory once | Add or correct the explicit `-v host:container:ro` mount, then verify the file content from inside the container as well as on the host | ~15-30 min per occurrence |
| 2 | `--gpus all` assigns zero devices after a crash | Stale cgroup device state left behind by an earlier OOM/crash makes `--gpus all` see no devices on relaunch | Use an explicit device list, or set the visible-devices environment variable explicitly, instead of `all` | ~10 min per occurrence |
| 3 | NVRM VA-space corruption after an OOM kill storm | `torch.cuda.is_available()` returns False host-wide even though device count reads correctly | Unload and reload the NVIDIA kernel modules (`nvidia_uvm`, then `nvidia`), which restores every GPU with no VM reboot; confirmed working twice | ~10-15 min recovery vs. hours for a reboot |
| 4 | Background processes die silently when their spawning shell session is reaped | A plain `nohup … &` inside a short-lived exec session gets killed when that session's process group is torn down (~35s window); hit at least 3 separate times even after being root-caused once | Use `setsid` or a detached, persistent terminal session (`tmux`/`screen`) for anything meant to outlive the launching command | 3 incidents, ~10-20 min diagnosis each |
| 5 | Duplicate parallel work from assuming a background job was dead | A build/conversion job, a hash/verify job, and a rebuild were each relaunched a second time while the first copy was still running, fighting the new copy for GPU/CPU/storage bandwidth | Check process/container state directly before relaunching; "no output for a while" is not evidence a job died | ~30+ min wasted each of 3 times (90+ min total) |
| 6 | Commit-hash transcription error | A reported commit hash did not match the actual head on the remote; a retyped/paraphrased hash silently diverged from the real one | Copy-paste hashes from command output; never retype from memory | Extra review round on the same change |
| 7 | Package-presence check used instead of an architecture-aware capability check | Hardware without native fast-path tensor-core support, but with the accelerator package installed, would select an unsupported code path purely because the package was present | Gate fast paths on device compute capability, not on whether a package is importable | Caught in review before merge |
| 8 | Missing required keyword argument in a patched constructor | A patched attention-indexer class was missing a required constructor argument; only surfaces at runtime | Run an import smoke test plus an undefined-name/lint scan on every patched file, in addition to a syntax check | One extra review round-trip |
| 9 | A weight-loader fix landed in the model's loader but not in its speculative draft model's separate loader | The draft/speculative model loads weights through its own code path that silently missed the same fix | Apply loader fixes to every code path that touches the same tensors, in addition to the primary one | One follow-up patch cycle |
| 10 | A multi-attempt boot cycle surfaced a new architecture-specific gap on each attempt | Six consecutive boot attempts each hit a different unsupported-on-this-architecture code path in sequence | Fix one gap at a time with a real fix, not a retry, but do a full compatibility sweep before the first launch rather than discovering gaps one boot at a time | 2+ hours across the worst two attempts alone |
| 11 | Shell/script string-escaping mangled multiline content | Markdown backticks collided with template-literal quoting; a custom text encoder was broken, corrupting non-ASCII output; recurred 6+ times | Build multiline content with escaping-safe quoting; prefer ASCII-safe substitutions across any shell/script/markdown boundary; verify by round-tripping instead of assuming | Multiple 5-15 min fixes, one full document rewrite |
| 12 | Storage-authorization ambiguity blocked a lane for about two hours | Fast local storage was policy-forbidden for model weights, but no approved non-root storage path had been documented for the assigned nodes before the lane was handed off | The lane correctly refused to violate policy and reported a blocker, then had to wait for a live policy correction from the owner. Resolve storage/authorization boundaries before delegating a lane, not mid-run | ~2 hours wall-clock, mostly idle |
| 13 | A non-default KV-cache layout was launched with the default cache dtype | The checkpoint's attention configuration required an explicit, non-default KV-cache dtype; the default silently failed at first request | Set the exact KV-cache dtype/layout flag a checkpoint's attention config requires; never rely on an automatic default for a non-standard layout | One full launch-diagnose-relaunch cycle |
| 14 | Intermittent remote-shell connectivity was repeatedly mistaken for a hung load | A large weight transfer over the network coincided with a period of unreliable remote-shell connectivity, and each timeout was re-diagnosed as a possible hang | Bind the server so it can be checked directly over its own network port, and use that as the liveness check when the remote shell itself is degraded | ~35 min of investigation |
| 15 | Compressed image/video byte sequences false-triggered a private-identifier scanner | Rendered chart labels created byte sequences inside compressed PNG/MP4 data that coincidentally matched the pattern the sanitizer used to catch leaked private identifiers | Regenerate with varied rendering parameters, or scan pre-compression source content instead of the compressed byte stream | Several regenerate cycles, ~7 min each |
| 16 | Roughly a dozen restart attempts before a working boot configuration was found, independently on both lanes | Each restart re-paid the full weight-load time before the next configuration error surfaced | Fix the compatibility/config sweep (partition, KV dtype, gating checks) before the first launch attempt, not through live iteration | Repays the full weight-load time on every restart; see time sinks below |
| 17 | A second serving runtime hit the same host-memory-exhaustion bug class that had already been fixed once in the first runtime | A streaming-load fix applied to one serving stack's loader did not carry over to a different serving stack used for the same checkpoint | Apply a load-strategy fix per runtime; do not assume a fix in one serving stack propagates to another one used for the same checkpoint | New investigation cycle needed |
| 18 | Two independent workstreams posting to one shared tracking thread got their comments mixed | An operator had to be told explicitly which comments in a shared thread were in scope for its own lane | Use a separate tracking thread per lane, or tag every comment with the lane it belongs to | One full re-read of the thread history to re-scope |

## 2. Time sinks

Ranked by rough wall-clock cost:

1. **A storage-authorization deadlock, roughly two hours.** Mostly idle time spent polling the same tracking thread for an owner decision that had not yet arrived.
2. **A multi-attempt boot cycle, 2+ hours across its worst two attempts.** Each attempt paid the full weight-load time before failing on a new architecture-specific gap.
3. **Cumulative network-loaded weight shards, roughly 50-70 minutes across the day.** Several minutes per full shard load, repeated across at least six boot attempts on one lane alone.
4. **Duplicate parallel work from believing a process was dead, roughly 90+ minutes total.** Three separate incidents, each burning 30+ minutes of redundant compute/storage bandwidth.
5. **Manual integrity narrowing done piece-by-piece, roughly 11 minutes.** More than a dozen sequential runs sweeping slices of a checkpoint by hand instead of one full scan with complete per-item reporting.
6. **Investigating intermittent remote-shell connectivity, roughly 35 minutes.** Time spent distinguishing "still loading" from "actually hung" without a reliable remote-shell channel.
7. **Coordination and acknowledgement overhead.** Fixed-interval progress updates posted continuously throughout long attempts, plus multiple review hold-and-resume cycles: individually cheap, but continuous across the whole window and added real waiting time before each rebuild could proceed.

## 3. Things that worked (reuse verbatim)

- **Stable 4-card pipeline partition:** an uneven split (`11,11,11,10`) was the only one that didn't crash on the first request; two even-leaning alternatives both failed immediately with a device-side assert.
- **A concurrency-deadlock fix for a custom serving front-end:** concurrent requests were racing the same shared inference call from separate handler threads (confirmed with a stack-sampling profiler showing three simultaneous stacks in the same critical section). Fix: add a module-scope lock and wrap the shared call site with it to serialize requests across handler threads. Restart with fault-handler and blocking-collective-wait environment flags enabled for future diagnosability.
- **Module-reload recovery sequence for NVRM VA-space corruption:** unload the `nvidia_uvm` and `nvidia` kernel modules, then reload them, which restores every GPU without a VM reboot. Confirmed working twice.
- **Architecture-capability-gated dispatch** (compute capability check, not package-presence check) as the correct fallback pattern on this hardware.
- **A local NVMe staging pipeline:** hash the source, copy, verify the destination against a frozen manifest; that cut cold-start weight availability from 30+ minutes over the network to about two minutes.
- **A container-scoped read-only network mount with no host-level privilege change:** when a target host has no passwordless elevated access but its container runtime needs none, run a privileged container from the runtime image, install the network-filesystem client tools inside that container only, then mount the canonical export read-only inside the container's own mount namespace. The host itself is unchanged, no bytes are copied, and the checkpoint is reachable read-only.
- **A byte-and-header integrity gate run before spending any GPU time:** verify exact shard count, exact total bytes, and parsed shard headers against a frozen manifest before ever launching a serve attempt.
- **A bounded read-only preflight gate run before every major action:** per-node checks for a clean fault-log window, idle memory/GPU state, zero running containers, free storage space, interconnect reachability, and current ownership; cheap, and it is what catches a storage-authorization gap or a node's fault history before a launch is wasted.
- **Setting the exact KV-cache dtype/layout flag explicitly** whenever a checkpoint's attention configuration names a non-default layout; never rely on an automatic default.
- **An HTTP port-probe liveness check** as a fallback when the remote shell is degraded: a server bound on the host network can be checked directly on its port even when the shell control channel is stalling.
- **A disciplined A/B protocol:** a written pre-run checkpoint (frozen pins, hypothesis, source-lock file hashes, rollback plan, numeric stop condition) committed before changing a single variable, a written decision gate, and an executed rollback the moment a treatment measured worse than control. This produced a clean, trustworthy negative result instead of an ambiguous one.
- **An extended process-group startup timeout** to survive the default collective-communication timeout under network-load-induced rank skew (root cause documented in [LESSONS.md](LESSONS.md)).

## 4. Anti-patterns in the multi-agent process

- **Status "receipts" substituting for results.** Fixed-interval progress updates gave the appearance of forward motion while the same class of boot failure recurred attempt after attempt. A status update that doesn't change the next action is overhead, not communication.
- **Retries without a changed hypothesis.** Consecutive boot attempts re-ran the same fix-and-boot cycle framed as a "single-variable retry," while a third, still-unaddressed architecture gap was waiting behind the two just-fixed ones. A retry should target one specific, named hypothesis: not "try again with the last fixes applied."
- **Ownership ambiguity resolved only by direct, blunt owner intervention.** The owner had to step in more than once to surface real gaps: a required benchmark ladder that had never actually been run, and a direct instruction to skip a review step that had already caught two real defects in the same change. Bypassing review for speed removes the one mechanism shown to catch real bugs.
- **Over-scoped goal-setting mid-stream.** A course correction toward the actual deliverable (vision capability, not text-only throughput) arrived only after significant effort had already gone into the wrong track. State the actual deliverable before committing GPU time to it.
- **Two lanes sharing one coordination thread without a lane tag.** An operator had to be told explicitly which comments in a shared thread belonged to its own lane, wasting a full re-read of the thread history to re-scope. Separate coordination threads per lane, or an explicit lane tag on every update, avoids this.
- **Storage and authorization policy decided mid-delegation instead of before it.** A lane was handed off as "own this end to end" without first confirming an approved storage path existed on its target hardware, discovered only when the lane hit the gate live, then sat blocked for about two hours until policy was corrected. Decide storage/authorization boundaries before handing off a lane, not as a live blocker discovered mid-run.
- **A parallel review/approval process ran continuously with unclear value.** A paired review process mirrored nearly every action across thousands of turns and approved nearly all of them; no case was found in this pass of it meaningfully altering an action. Worth auditing whether a lighter-weight review process would catch the same issues before continuing to run a full-mirror review process unconditionally.

## 5. Fifteen-item checklist: before you start a GPU run on this box

1. Confirm which physical node(s) are actually assigned to your lane; do not act on another lane's updates in a shared coordination thread.
2. Read the full coordination thread chronologically and acknowledge the latest processed update before taking any action.
3. Run a read-only preflight gate: clean fault-log window for the last 30 minutes, idle GPU (0% utilization, 0 compute processes), 0 running containers, sufficient free storage, interconnect reachable.
4. Verify directly (container/process listing, GPU utilization query), never by guessing from silence, that no other process already owns the GPU before launching anything.
5. Verify the checkpoint's exact byte count, shard count, and per-shard header/hash against a frozen manifest before spending any GPU time loading it.
6. Confirm the checkpoint fits: compare exact checkpoint bytes to exact usable GPU memory at the intended parallelism degree (per-card and pooled) before launching, not after an out-of-memory failure tells you.
7. Use an explicit device list, never an unqualified "all devices" flag, on any host that has had a prior crash or out-of-memory event.
8. Confirm every host path you plan to bind-mount exists and is non-empty, and re-verify the mount from inside the container after starting it; do not assume a previous session's mount is still valid.
9. Set the exact KV-cache dtype/layout flag a checkpoint's attention configuration requires; never rely on an automatic default for a non-standard layout.
10. Launch any long-running or background process with a detached session mechanism, never a bare background shell job inside a short-lived tool session that can itself be reaped.
11. Check root-disk free percentage against your floor thresholds before any build or serve start; never stage weights or build caches on the root disk.
12. Confirm which runtime is actually being benchmarked and that the patched files actually loaded are the ones you intend, not a stale cached layer.
13. Copy-paste every commit hash and reference used in a report from its source; never retype from memory.
14. Before any A/B change: write down the single variable being changed, the numeric decision gate, and the rollback plan, and commit that pre-run checkpoint before starting.
15. Replace any package-presence check with an architecture-aware capability check before trusting a fast-path branch on hardware that lacks native support for it.

## 6. A card that drops under multimodal-encoder profiling: diagnosing slot power vs. riser vs. card

Mined from a multi-day debugging thread on a four-card node, 2026-09-04.

**vLLM's multimodal-encoder profiling step is the largest power/bus transient
of a boot.** The stage that profiles the vision/video encoder against the
maximum feature size draws a sharper, larger transient than steady-state
decode or even prefill. Treat it as a deliberate stress test for a marginal
card position, not an incidental crash site: if a card is going to fall off
the bus under load, this step is where it will happen first.

**A card that drops (`rev ff`, D3cold-style loss) under that transient, with
a clean link and no correctable PCIe errors right up to the drop, points at
slot power delivery — not the riser and not the card.** The discriminating
signature: a healthy link status (no `CorrErr`/`BadTLP`/timeout counters
climbing) immediately before the device vanishes from the bus is not
consistent with a signal-integrity problem (which usually shows correctable
errors building up first); it is consistent with a power rail sagging under
a sudden current draw and the device browning out.

**Riser swap and card swap are the two discriminating tests, run in that
order.** Swap the riser at the affected position for a known-good one and
reproduce the load: if the drop recurs on the very next boot, the riser is
cleared. Then swap the card at that position with a card from a stable
position: if the failure stays with the position rather than following the
card, the card is cleared too. What is left — the slot's power delivery —
is the remaining hypothesis, and it is the one to test next (power-limit the
card and retry, or move its power feed to a different rail).

**`--limit-mm-per-prompt '{"image":0,"video":0}'` is the text-only
workaround.** Passing zero for both image and video limits skips the
multimodal-encoder profiling stage entirely at boot. On the affected node
this let the server boot clean and serve text-only traffic stably across
repeated concurrency rounds with no further drops, while leaving the actual
vision-serving path unfixed and unmeasured. Use this flag to keep a lane
moving on text while the hardware root cause is chased separately — it is a
workaround, not a fix, and does not clear the card/slot for other
peak-load workloads.

**Recovery ladder for a card that has fallen off the bus, cheapest first,
never skip a step:**

1. Guest-side kernel module reload (unload then reload the GPU driver
   modules) — recovers a device that is still enumerated on the guest's PCI
   bus but wedged at the driver level.
2. A full VM reboot — recovers a guest-level wedge that a module reload
   alone does not clear.
3. A cold power cycle of the host — required when the device has actually
   left the host-side PCI bus (not just the guest's view of it), which a
   guest reboot cannot fix.

**Never run a host-side device "remove" on the passthrough device as a
recovery step.** It turns a recoverable wedge into a card that vanishes
from the host bus entirely and forces a full cold power cycle to bring
back — a strictly worse outcome than doing nothing and going straight to
the VM reboot or cold cycle above.

**A VM reboot can itself land in a stuck power state** — the guest reports
being unable to move the device from a low-power state back to an active
one. Treat that specific failure as equivalent to step 3: it needs a cold
power cycle, not a second reboot attempt.

**Slot power-delivery headroom matters most when several cards share one
bifurcated slot or riser position.** Powered risers, or an onboard
low-voltage power-input header on the riser/backplane feeding a slot that
several cards share, are the durable fix once slot power delivery is
confirmed (not merely suspected) as the root cause — an MCIO-style
bifurcation kit with its own powered device board is the reference
approach for a slot carrying more than one card's worth of peak transient
current.

**Power cap re-apply after any module reload.** A per-card power limit set
with the driver's runtime power-limit tool does not survive a guest reboot
or a kernel module reload. Re-apply the power cap and verify it against
every card's reported power limit after either event, before taking any
measurement — a card silently back at its default (higher) power limit
after a "recovery" is itself a new risk factor for the exact transient this
section describes.

## See also

- [LESSONS.md](LESSONS.md) — kernel, runtime, topology, memory/storage, power/thermal, and hardware failure-mode lessons.
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — symptom-to-cause diagnosis at the hardware/driver layer.
- [MODEL-STATUS.md](MODEL-STATUS.md) — every model attempted on CMP 170HX, one table.
- [TOPOLOGY-AND-PARALLELISM.md](TOPOLOGY-AND-PARALLELISM.md) — PCIe link facts and PP-vs-TP topology choice for this card.
