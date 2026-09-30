#!/usr/bin/env python3
"""GPU-to-GPU copy probe over the CUDA driver API (ctypes, no toolkit needed).

For every ordered GPU pair it records:
  - cuDeviceCanAccessPeer (the advertised cap),
  - whether cuCtxEnablePeerAccess succeeds,
  - an alias-proof correctness check: the destination is pre-filled with a byte
    pattern that never occurs in the source, then a device-to-device copy is
    issued; the pair passes only if the destination now equals the source,
  - peer-copy bandwidth (cuMemcpyPeer, timed over N reps) and, for reference,
    host-staged bandwidth (D2H into pinned memory + H2D).

cuMemcpyPeer falls back to staging through host memory when direct peer access
is not enabled, so a pair can pass correctness without true P2P; the bandwidth
column and the peer-access column together say which path was used.

Output: one JSON document on stdout.
"""
import ctypes
import json
import sys
import time

SIZE = 256 * 1024 * 1024
REPS = 10

cu = ctypes.CDLL("libcuda.so.1")


def check(rc, what):
    if rc != 0:
        name = ctypes.c_char_p()
        cu.cuGetErrorName(rc, ctypes.byref(name))
        raise RuntimeError(f"{what}: {name.value.decode() if name.value else rc}")


def main():
    check(cu.cuInit(0), "cuInit")
    n = ctypes.c_int()
    check(cu.cuDeviceGetCount(ctypes.byref(n)), "cuDeviceGetCount")
    devs, ctxs, names = [], [], []
    for i in range(n.value):
        d = ctypes.c_int()
        check(cu.cuDeviceGet(ctypes.byref(d), i), "cuDeviceGet")
        buf = ctypes.create_string_buffer(128)
        cu.cuDeviceGetName(buf, 128, d)
        c = ctypes.c_void_p()
        check(cu.cuDevicePrimaryCtxRetain(ctypes.byref(c), d), "ctxRetain")
        devs.append(d)
        ctxs.append(c)
        names.append(buf.value.decode())

    bufs = []
    for i, c in enumerate(ctxs):
        check(cu.cuCtxSetCurrent(c), "setCurrent")
        p = ctypes.c_uint64()
        check(cu.cuMemAlloc_v2(ctypes.byref(p), SIZE), "memAlloc")
        bufs.append(p.value)

    host = ctypes.c_void_p()
    check(cu.cuMemAllocHost_v2(ctypes.byref(host), SIZE), "memAllocHost")
    src_pattern = (ctypes.c_ubyte * SIZE).from_address(host.value)
    # Source pattern never contains 0xFE; destinations are pre-filled with 0xFE.
    for k in range(0, SIZE, 4096):
        v = (k // 4096) % 250
        src_pattern[k] = v if v != 0xFE else 0x01

    def fill(i, byte):
        check(cu.cuCtxSetCurrent(ctxs[i]), "setCurrent")
        check(cu.cuMemsetD8_v2(ctypes.c_uint64(bufs[i]), ctypes.c_ubyte(byte), ctypes.c_size_t(SIZE)), "memset")
        check(cu.cuCtxSynchronize(), "sync")

    def upload(i):
        check(cu.cuCtxSetCurrent(ctxs[i]), "setCurrent")
        check(cu.cuMemcpyHtoD_v2(ctypes.c_uint64(bufs[i]), host, ctypes.c_size_t(SIZE)), "HtoD")

    def sample(i):
        out = (ctypes.c_ubyte * SIZE)()
        check(cu.cuCtxSetCurrent(ctxs[i]), "setCurrent")
        check(cu.cuMemcpyDtoH_v2(out, ctypes.c_uint64(bufs[i]), ctypes.c_size_t(SIZE)), "DtoH")
        return out

    results = []
    for a in range(len(devs)):
        for b in range(len(devs)):
            if a == b:
                continue
            can = ctypes.c_int()
            check(cu.cuDeviceCanAccessPeer(ctypes.byref(can), devs[a], devs[b]), "canAccessPeer")
            check(cu.cuCtxSetCurrent(ctxs[a]), "setCurrent")
            rc = cu.cuCtxEnablePeerAccess(ctxs[b], 0)
            enabled = rc in (0, 704)  # 704 = PEER_ACCESS_ALREADY_ENABLED

            upload(a)
            fill(b, 0xFE)
            check(cu.cuCtxSetCurrent(ctxs[a]), "setCurrent")
            t0 = time.perf_counter()
            rc_copy = cu.cuMemcpyPeer(ctypes.c_uint64(bufs[b]), ctxs[b], ctypes.c_uint64(bufs[a]), ctxs[a], ctypes.c_size_t(SIZE))
            check(cu.cuCtxSynchronize(), "sync")
            first = time.perf_counter() - t0
            out = sample(b)
            ok = rc_copy == 0 and all(out[k] == src_pattern[k] for k in range(0, SIZE, 4096))
            untouched = all(out[k] == 0xFE for k in range(0, SIZE, 4096))

            t0 = time.perf_counter()
            for _ in range(REPS):
                cu.cuMemcpyPeer(ctypes.c_uint64(bufs[b]), ctxs[b], ctypes.c_uint64(bufs[a]), ctxs[a], ctypes.c_size_t(SIZE))
            check(cu.cuCtxSetCurrent(ctxs[b]), "setCurrent")
            check(cu.cuCtxSynchronize(), "sync")
            peer_gbs = SIZE * REPS / (time.perf_counter() - t0) / 1e9

            t0 = time.perf_counter()
            for _ in range(REPS):
                check(cu.cuCtxSetCurrent(ctxs[a]), "setCurrent")
                check(cu.cuMemcpyDtoH_v2(host, ctypes.c_uint64(bufs[a]), ctypes.c_size_t(SIZE)), "DtoH")
                check(cu.cuCtxSetCurrent(ctxs[b]), "setCurrent")
                check(cu.cuMemcpyHtoD_v2(ctypes.c_uint64(bufs[b]), host, ctypes.c_size_t(SIZE)), "HtoD")
            staged_gbs = SIZE * REPS / (time.perf_counter() - t0) / 1e9

            if enabled and rc == 0:
                check(cu.cuCtxSetCurrent(ctxs[a]), "setCurrent")
                cu.cuCtxDisablePeerAccess(ctxs[b])
            results.append({
                "src": a, "dst": b,
                "can_access_peer": bool(can.value),
                "enable_peer_access_rc": rc,
                "copy_rc": rc_copy,
                "alias_proof_pass": bool(ok),
                "dst_untouched": bool(untouched),
                "first_copy_s": round(first, 4),
                "peer_copy_GBps": round(peer_gbs, 3),
                "host_staged_GBps": round(staged_gbs, 3),
            })
            print(json.dumps(results[-1]), file=sys.stderr, flush=True)

    json.dump({"size_bytes": SIZE, "reps": REPS, "devices": names, "pairs": results}, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
