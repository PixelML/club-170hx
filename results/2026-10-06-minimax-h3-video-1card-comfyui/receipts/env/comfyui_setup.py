"""One-time setup for the H3 ComfyUI pod: pinned sources + venv on the app PVC.

Runs as the init container. Skips if the marker for this pin set exists.
"""
import io
import json
import os
import subprocess
import sys
import tarfile
import urllib.request

APP = "/opt/app"
# separate marker for the torch-from-index path, so an image change forces a rebuild
MARK = f"{APP}/.setup-v1" + ("-idx" if os.environ.get("H3_TORCH_INDEX") else "")

# repo, commit, destination (relative to APP)
SOURCES = [
    ("comfyanonymous/ComfyUI", "2f35f4a08176d993cded35dac3332be4f7287f41", "ComfyUI"),
    ("Kosinkadink/ComfyUI-VideoHelperSuite", "4ee72c065db22c9d96c2427954dc69e7b908444b",
     "ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite"),
    ("Lightricks/ComfyUI-LTXVideo", "15d09abb5a187a8dcaea2fc31fe51ee96e6c9d0d",
     "ComfyUI/custom_nodes/ComfyUI-LTXVideo"),
    ("LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler", "6a4b191e8af583b7c097f564690325f91d18c2e2",
     "ComfyUI/custom_nodes/Comfyui_Minimax_h3_latent_Upscaler"),
    ("scraed/LanPaint", "9fe919558f3ff085daca921ee8401dc1832a1258", "ComfyUI/custom_nodes/LanPaint"),
    ("Larryvrh/ComfyUI-MiniMax-H3-Turbo", "4274783a23afcfdbea3b4876cb79effd6c510785",
     "ComfyUI/custom_nodes/ComfyUI-MiniMax-H3-Turbo"),
]


def fetch(repo, sha, dest):
    url = f"https://codeload.github.com/{repo}/tar.gz/{sha}"
    print("fetch", url, flush=True)
    data = urllib.request.urlopen(url, timeout=300).read()
    out = os.path.join(APP, dest)
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        members = []
        for m in t.getmembers():
            parts = m.name.split("/", 1)
            if len(parts) < 2 or not parts[1]:
                continue
            m.name = parts[1]
            members.append(m)
        t.extractall(out, members=members, filter="data")


def run(*cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    if os.path.exists(MARK):
        print("setup done:", open(MARK).read())
        return
    for repo, sha, dest in SOURCES:
        fetch(repo, sha, dest)

    venv = f"{APP}/venv"
    vpy = f"{venv}/bin/python"
    pip = [vpy, "-m", "pip"]
    index = os.environ.get("H3_TORCH_INDEX")
    if index:
        # plain-Python image: install the torch stack into the venv from the given index
        run(sys.executable, "-m", "venv", "--clear", venv)
        run(*pip, "install", "--upgrade", "pip")
        run(*pip, "install", *os.environ["H3_TORCH_PKGS"].split(), "--index-url", index)
    else:
        # CUDA torch image: reuse its torch stack; the image has no ensurepip,
        # so use the system pip through --system-site-packages
        run(sys.executable, "-m", "venv", "--system-site-packages", "--without-pip", venv)

    def dist_version(name):
        out = subprocess.run([vpy, "-c", f"from importlib.metadata import version; print(version({name!r}))"],
                             capture_output=True, text=True)
        return out.stdout.strip() if out.returncode == 0 else None

    # Keep the torch stack fixed. Pin the installed dist-info versions (NGC
    # torch.__version__ differs from its dist-info). The requirement files never install torch.
    skip = ["torch", "torchvision", "torchaudio"]
    pins = [f"{n}=={v}" for n in skip if (v := dist_version(n))]
    # kornia 0.8.3 removed geometry.transform.pyramid.pad, which ComfyUI-LTXVideo imports
    pins.append("kornia<0.8.3")
    torch_version = dist_version("torch") or ""
    with open(f"{APP}/constraints.txt", "w") as f:
        f.write("\n".join(pins) + "\n")

    reqs = []
    cands = [f"{APP}/ComfyUI/requirements.txt"] + [
        f"{APP}/ComfyUI/custom_nodes/{d}/requirements.txt"
        for d in os.listdir(f"{APP}/ComfyUI/custom_nodes")]
    for r in cands:
        if os.path.exists(r):
            # headless OpenCV: the image has no libGL
            txt = open(r).read().replace("opencv-python\n", "opencv-python-headless\n")
            txt = "".join(l for l in txt.splitlines(True) if l.strip() not in skip)
            if txt.rstrip().endswith("opencv-python"):
                txt = txt.rstrip()[: -len("opencv-python")] + "opencv-python-headless\n"
            open(r, "w").write(txt)
            reqs.append(r)
    args = [*pip, "install", "-c", f"{APP}/constraints.txt"]
    if "a0" in torch_version or ".nv" in torch_version:
        # NGC pre-release torch only satisfies transitive torch>=x pins with --pre;
        # constraints.txt still holds torch at the image version
        args.append("--pre")
    for r in reqs:
        args += ["-r", r]
    run(*args)

    os.makedirs(f"{APP}/input", exist_ok=True)
    os.makedirs(f"{APP}/output", exist_ok=True)
    with open(MARK, "w") as f:
        json.dump({"sources": SOURCES, "pins": pins, "torch_index": index}, f)
    print("setup done")


if __name__ == "__main__":
    main()
