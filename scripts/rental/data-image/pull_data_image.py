#!/usr/bin/env python3
"""Pull an OCI image from GHCR and extract its layers to a destination dir,
without needing a docker daemon or CLI on the pulling machine.

Reusable for every rental where the target instance has no docker binary
(direct-exec Vast images typically don't). Uses only the stdlib.

Usage:
    python3 pull_data_image.py --image ghcr.io/pixelml/club-170hx \
        --tag specdec-sliceb-20260905 --dest /data
"""
import argparse
import io
import json
import sys
import tarfile
import time
import urllib.request

MANIFEST_ACCEPT = ", ".join([
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
])


def get_repo(image: str) -> str:
    parts = image.split("/", 1)
    assert len(parts) == 2, f"expected host/repo, got {image!r}"
    return parts[1]


def get_token(registry: str, repo: str) -> str:
    url = f"https://{registry}/token?service={registry}&scope=repository:{repo}:pull"
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.load(r)
    return data.get("token") or data.get("access_token")


def fetch_json(registry: str, repo: str, ref: str, token: str) -> dict:
    url = f"https://{registry}/v2/{repo}/manifests/{ref}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Accept": MANIFEST_ACCEPT,
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def fetch_blob_stream(registry: str, repo: str, digest: str, token: str):
    url = f"https://{registry}/v2/{repo}/blobs/{digest}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    return urllib.request.urlopen(req, timeout=60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="e.g. ghcr.io/pixelml/club-170hx")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--dest", required=True, help="filesystem root to extract layers into")
    ap.add_argument("--arch", default="amd64")
    ap.add_argument("--os", default="linux")
    args = ap.parse_args()

    registry = args.image.split("/", 1)[0]
    repo = get_repo(args.image)

    t0 = time.time()
    token = get_token(registry, repo)
    print(f"[pull] got anonymous pull token for {repo}", file=sys.stderr)

    top = fetch_json(registry, repo, args.tag, token)

    if top.get("mediaType", "").endswith("index.v1+json") or "manifests" in top:
        chosen = None
        for m in top["manifests"]:
            plat = m.get("platform", {})
            if plat.get("architecture") == args.arch and plat.get("os") == args.os:
                chosen = m
                break
        if chosen is None:
            chosen = top["manifests"][0]
            print(f"[pull] no exact {args.os}/{args.arch} match, using first entry", file=sys.stderr)
        manifest = fetch_json(registry, repo, chosen["digest"], token)
    else:
        manifest = top

    layers = manifest["layers"]
    print(f"[pull] {len(layers)} layer(s) to extract", file=sys.stderr)

    total_bytes = 0
    for i, layer in enumerate(layers):
        digest = layer["digest"]
        size = layer.get("size", 0)
        print(f"[pull] layer {i+1}/{len(layers)} {digest} ({size/1e6:.1f} MB)", file=sys.stderr)
        resp = fetch_blob_stream(registry, repo, digest, token)
        raw = resp.read()
        total_bytes += len(raw)
        media = layer.get("mediaType", "")
        fileobj = io.BytesIO(raw)
        mode = "r:gz" if "gzip" in media else "r:*"
        with tarfile.open(fileobj=fileobj, mode=mode) as tf:
            try:
                tf.extractall(path=args.dest, filter="data")
            except TypeError:
                tf.extractall(path=args.dest)

    dt = time.time() - t0
    rate = total_bytes / dt / 1e6 if dt > 0 else 0
    print(f"[pull] done: {total_bytes/1e9:.2f} GB in {dt:.1f}s ({rate:.1f} MB/s) -> {args.dest}", file=sys.stderr)


if __name__ == "__main__":
    main()
