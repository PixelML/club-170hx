# Building a low-cost SM80 cluster

The goal is a useful, reproducible compute pool—not a claim that CMP 170HX cards become A100s. The design should lean into HBM capacity and work around weak interconnects.

## Topology first

```text
shared model storage
        │
        ▼
  compute node / VM
   ├── CMP card 0 ── model stages 0…N
   ├── CMP card 1 ── model stages N…M
   └── CMP card 2 ── model stages M…end
```

There is no NVLink in the tested configuration. PCIe links may also negotiate conservatively under virtualization. This changes which parallelism strategy wins.

## Parallelism choices

| Strategy | Fit on this hardware | Why |
|---|---|---|
| Pipeline parallel | Preferred starting point | Communicates stage boundaries instead of frequent full-tensor collectives |
| Tensor parallel | Use only after measurement | All-reduce traffic can erase compute gains on slow PCIe |
| Data parallel | Good for independent jobs/throughput | Each replica needs its own weights or a suitable sharding design |
| Expert parallel | Workload-dependent | Routing traffic and runtime support must be measured |

The three-card DeepSeek-V4-Flash result validates pipeline parallelism as a practical path. It does not prove pipeline parallel is optimal for every model.

## Storage layout

Use shared high-capacity storage for model weights and local disks for code, environments, caches that need high IOPS, and transient outputs.

Before a launch, record:

```bash
findmnt -T /path/to/model/library
df -hT /path/to/model/library
du -sh /path/to/model/checkpoint
```

Do not silently fall back to a small root disk when shared storage disappears. Treat a missing mount or unsafe free-space level as a stop condition.

## Scheduling rules

1. Give every card a stable anonymous label and record its physical slot/riser/power mapping privately.
2. Run a preflight inventory and temperature check before each workload.
3. Prevent benchmark and serving jobs from sharing a card unless the experiment explicitly measures contention.
4. Preserve raw results with runtime/model commits and power/thermal telemetry.
5. Cold-cycle only after orderly guest/host shutdown and only when PCIe recovery requires it.

## Scaling beyond one node

Measure the network before distributed serving or training. Model capacity can scale across nodes while latency and throughput regress from communication. Publish link speed, transport, topology, batch size, and failure behavior alongside any multi-node claim.
