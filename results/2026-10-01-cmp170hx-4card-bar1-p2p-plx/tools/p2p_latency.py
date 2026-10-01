#!/usr/bin/env python3
"""GPU-to-GPU copy time vs message size over the CUDA driver API (ctypes).

Times cuMemcpyPeer (synchronised on both contexts) for sizes from 4 KiB to 256 MiB between every ordered GPU
pair (median of REPS synchronous copies), plus a single-GPU device-to-host
copy of the same size for reference. Output: one JSON document on stdout.
"""
import ctypes
import json
import statistics
import time

SIZES = [4 << 10, 64 << 10, 1 << 20, 4 << 20, 16 << 20, 64 << 20, 256 << 20]
REPS = 30

cu = ctypes.CDLL("libcuda.so.1")


def check(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what}: {rc}")


def main():
    check(cu.cuInit(0), "cuInit")
    n = ctypes.c_int()
    check(cu.cuDeviceGetCount(ctypes.byref(n)), "count")
    ctxs, bufs = [], []
    for i in range(n.value):
        d = ctypes.c_int()
        check(cu.cuDeviceGet(ctypes.byref(d), i), "get")
        c = ctypes.c_void_p()
        check(cu.cuDevicePrimaryCtxRetain(ctypes.byref(c), d), "retain")
        check(cu.cuCtxSetCurrent(c), "set")
        p = ctypes.c_uint64()
        check(cu.cuMemAlloc_v2(ctypes.byref(p), SIZES[-1]), "alloc")
        ctxs.append(c)
        bufs.append(p.value)
    host = ctypes.c_void_p()
    check(cu.cuMemAllocHost_v2(ctypes.byref(host), SIZES[-1]), "allocHost")

    # Without peer access enabled, cuMemcpyPeer silently stages through host
    # memory. --peer turns it on for every ordered pair (needs a P2P driver).
    peer_access = "--peer" in __import__("sys").argv
    if peer_access:
        for a in range(n.value):
            check(cu.cuCtxSetCurrent(ctxs[a]), "set")
            for b in range(n.value):
                if a != b:
                    check(cu.cuCtxEnablePeerAccess(ctxs[b], 0), f"enable-peer {a}->{b}")

    def timed(fn, reps):
        out = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            out.append(time.perf_counter() - t0)
        return statistics.median(out)

    rows = []
    for size in SIZES:
        reps = REPS if size <= (16 << 20) else 8
        for a in range(n.value):
            for b in range(n.value):
                if a == b:
                    continue

                def peer():
                    check(cu.cuMemcpyPeer(ctypes.c_uint64(bufs[b]), ctxs[b], ctypes.c_uint64(bufs[a]), ctxs[a], ctypes.c_size_t(size)), "peer")
                    check(cu.cuCtxSynchronize(), "sync-src")
                    check(cu.cuCtxSetCurrent(ctxs[b]), "set")
                    check(cu.cuCtxSynchronize(), "sync-dst")
                    check(cu.cuCtxSetCurrent(ctxs[a]), "set")

                check(cu.cuCtxSetCurrent(ctxs[a]), "set")
                peer()
                t = timed(peer, reps)
                rows.append({"kind": "peer", "src": a, "dst": b, "bytes": size, "median_us": round(t * 1e6, 1), "GBps": round(size / t / 1e9, 3)})

        def d2h():
            check(cu.cuMemcpyDtoH_v2(host, ctypes.c_uint64(bufs[0]), ctypes.c_size_t(size)), "d2h")

        check(cu.cuCtxSetCurrent(ctxs[0]), "set")
        d2h()
        t = timed(d2h, reps)
        rows.append({"kind": "d2h", "src": 0, "dst": -1, "bytes": size, "median_us": round(t * 1e6, 1), "GBps": round(size / t / 1e9, 3)})
    json.dump({"reps_small": REPS, "peer_access": peer_access, "rows": rows},__import__("sys").stdout, indent=1)


if __name__ == "__main__":
    main()
