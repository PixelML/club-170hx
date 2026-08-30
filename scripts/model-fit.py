#!/usr/bin/env python3
"""Estimate whether static model weights fit a GPU topology.

This is a capacity preflight, not a runtime guarantee. It does not model
quantization workspaces, CUDA graphs, activations, KV cache, or imbalance.
"""

from __future__ import annotations

import argparse
import math


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights_gib", type=float, help="total checkpoint size in GiB")
    parser.add_argument("--gpus", type=int, default=3, help="GPU count (default: 3)")
    parser.add_argument("--vram-gib", type=float, default=64.0, help="VRAM per GPU")
    parser.add_argument(
        "--reserve-gib",
        type=float,
        default=8.0,
        help="VRAM reserved per GPU for runtime/KV/activations",
    )
    args = parser.parse_args()

    if args.weights_gib <= 0 or args.gpus <= 0 or args.vram_gib <= 0:
        parser.error("weights, GPU count, and VRAM must be positive")
    if not 0 <= args.reserve_gib < args.vram_gib:
        parser.error("reserve must be non-negative and smaller than per-GPU VRAM")

    usable_per_gpu = args.vram_gib - args.reserve_gib
    static_per_gpu = args.weights_gib / args.gpus
    minimum_gpus = math.ceil(args.weights_gib / usable_per_gpu)
    margin = usable_per_gpu - static_per_gpu

    print(f"weights:          {args.weights_gib:.2f} GiB")
    print(f"topology:         {args.gpus} x {args.vram_gib:.2f} GiB")
    print(f"runtime reserve:  {args.reserve_gib:.2f} GiB/GPU")
    print(f"static shard:     {static_per_gpu:.2f} GiB/GPU")
    print(f"usable capacity:  {usable_per_gpu:.2f} GiB/GPU")
    print(f"static margin:    {margin:.2f} GiB/GPU")
    print(f"minimum GPUs:     {minimum_gpus} at this reserve")
    print(f"verdict:          {'STATIC FIT' if margin >= 0 else 'DOES NOT FIT'}")
    print("warning: runtime overhead and uneven layer/expert placement can still fail")


if __name__ == "__main__":
    main()
