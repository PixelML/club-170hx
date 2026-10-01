#!/usr/bin/env python3
"""NCCL all-reduce latency/bandwidth sweep with torch.distributed.

Launch with torchrun, one process per visible GPU:
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node 2 nccl_allreduce.py

Rank 0 prints one JSON line per message size: median time over REPS
all-reduces (bf16 sum) and the nccl-tests "bus bandwidth" convention,
busbw = algbw * 2 * (n - 1) / n.
"""
import json
import os
import statistics
import time

import torch
import torch.distributed as dist

SIZES = [8 << 10, 64 << 10, 256 << 10, 1 << 20, 4 << 20, 16 << 20, 64 << 20, 256 << 20]
WARMUP = 5
REPS = 20


def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    for size in SIZES:
        x = torch.ones(size // 2, dtype=torch.bfloat16, device="cuda")
        for _ in range(WARMUP):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        times = []
        for _ in range(REPS):
            dist.barrier()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            dist.all_reduce(x)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        med = statistics.median(times)
        algbw = size / med / 1e9
        if rank == 0:
            print(json.dumps({
                "world": world,
                "bytes": size,
                "median_us": round(med * 1e6, 1),
                "algbw_GBps": round(algbw, 3),
                "busbw_GBps": round(algbw * 2 * (world - 1) / world, 3),
            }), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
