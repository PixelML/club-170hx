"""Convert DeepSeek-V4.1 Engram shards (fp8 e4m3 rows + ue8m0 per-32 scales) to int4 rows +
fp16 per-32 scales, keeping every tensor name, so the pack index resolves unchanged.

Per row of 256 values: 128 bytes of packed nibbles (offset-binary, value = (q - 8) * scale)
+ 8 fp16 scales (16 bytes) = 144 bytes, against 264 for fp8. Other tensors are copied byte for byte.

usage: python3 engram_int4.py SRC_SHARD DST_SHARD [--workers N] [--limit ROWS]
"""
import argparse
import json
import os
import struct
from multiprocessing import Pool

import numpy as np

CHUNK = 262144
GROUP = 32


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return n, json.loads(f.read(n))


def fp8_e4m3_lut():
    lut = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0xF and m == 0x7:
            lut[b] = 0.0  # NaN in e4m3fn; never expected in weights
        elif e == 0:
            lut[b] = s * (m / 8.0) * 2.0 ** -6
        else:
            lut[b] = s * (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return lut


LUT = fp8_e4m3_lut()


def convert_rows(w8, s8):
    """w8: [R, 256] uint8 (e4m3), s8: [R, 8] uint8 (ue8m0) -> packed [R, 128] uint8, scales [R, 8] f16."""
    r, d = w8.shape
    vals = LUT[w8].reshape(r, d // GROUP, GROUP)
    vals *= np.ldexp(np.float32(1.0), s8.astype(np.int32) - 127)[:, :, None]
    amax = np.abs(vals).max(axis=2)
    scale = (amax / 7.0).astype(np.float16)
    sf = scale.astype(np.float32)
    q = np.where(sf[:, :, None] > 0, np.rint(vals / np.where(sf > 0, sf, 1.0)[:, :, None]), 0.0)
    q = (np.clip(q, -8, 7) + 8).astype(np.uint8).reshape(r, d)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    return packed, scale


def work(args):
    src, dst, src_hdr, dst_hdr, wkey, skey, start, stop = args
    sw, ss = src_hdr[wkey], src_hdr[skey]
    dw, ds = dst_hdr[wkey], dst_hdr[skey]
    d = sw["shape"][1]
    g = ss["shape"][1]
    so = 8 + src_hdr["__n"]
    do = 8 + dst_hdr["__n"]
    rows = stop - start
    w8 = np.fromfile(src, dtype=np.uint8, count=rows * d, offset=so + sw["data_offsets"][0] + start * d).reshape(rows, d)
    s8 = np.fromfile(src, dtype=np.uint8, count=rows * g, offset=so + ss["data_offsets"][0] + start * g).reshape(rows, g)
    packed, scale = convert_rows(w8, s8)
    with open(dst, "r+b") as f:
        f.seek(do + dw["data_offsets"][0] + start * (d // 2))
        f.write(packed.tobytes())
        f.seek(do + ds["data_offsets"][0] + start * g * 2)
        f.write(scale.tobytes())
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="convert only the first N rows of each table (test)")
    a = ap.parse_args()

    n, hdr = read_header(a.src)
    tables = [k[: -len(".weight")] for k, v in hdr.items() if k.endswith(".embed.weight") and v["dtype"] == "F8_E4M3"]
    out = {}
    off = 0
    order = sorted((k for k in hdr if k != "__metadata__"), key=lambda k: hdr[k]["data_offsets"][0])
    for k in order:
        v = hdr[k]
        base = k.rsplit(".", 1)[0]
        if base in tables and k.endswith(".weight"):
            rows, d = v["shape"]
            out[k] = {"dtype": "U8", "shape": [rows, d // 2]}
            size = rows * (d // 2)
        elif base in tables and k.endswith(".scale"):
            rows, g = v["shape"]
            out[k] = {"dtype": "F16", "shape": [rows, g]}
            size = rows * g * 2
        else:
            out[k] = {"dtype": v["dtype"], "shape": v["shape"]}
            size = v["data_offsets"][1] - v["data_offsets"][0]
        out[k]["data_offsets"] = [off, off + size]
        off += size
    meta = dict(hdr.get("__metadata__", {}))
    meta["engram_int4"] = "rows: int4 offset-binary (value=(q-8)*scale), packed low nibble first; scale: fp16 per 32"
    out["__metadata__"] = meta
    hb = json.dumps(out, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(a.dst, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.truncate(8 + len(hb) + off)
    src_hdr = dict(hdr, __n=n)
    dst_hdr = dict(out, __n=len(hb))

    # Copy non-table tensors byte for byte.
    with open(a.src, "rb") as fs, open(a.dst, "r+b") as fd:
        for k in order:
            base = k.rsplit(".", 1)[0]
            if base in tables:
                continue
            s0, s1 = hdr[k]["data_offsets"]
            fs.seek(8 + n + s0)
            fd.seek(8 + len(hb) + out[k]["data_offsets"][0])
            fd.write(fs.read(s1 - s0))

    jobs = []
    for t in tables:
        rows = hdr[t + ".weight"]["shape"][0]
        rows = min(rows, a.limit) if a.limit else rows
        for start in range(0, rows, CHUNK):
            jobs.append((a.src, a.dst, src_hdr, dst_hdr, t + ".weight", t + ".scale", start, min(start + CHUNK, rows)))
    done = 0
    total = sum(j[7] - j[6] for j in jobs)
    with Pool(a.workers) as p:
        for r in p.imap_unordered(work, jobs):
            done += r
            if done % (CHUNK * 200) < CHUNK:
                print(f"{done}/{total} rows", flush=True)
    print(f"done: {len(tables)} tables, {total} rows -> {a.dst}", flush=True)


if __name__ == "__main__":
    main()
