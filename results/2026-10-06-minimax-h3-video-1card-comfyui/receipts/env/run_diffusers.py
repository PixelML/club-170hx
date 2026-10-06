"""diffusers engine runner for MiniMax H3 Ref2VA on one 64 GB CMP 170HX (runs inside the bench pod).

Recipe: the doc's torchao int8 weight-only path (DiT + Qwen3-VL), adapted to one 64 GB card:
the DiT loads straight to the GPU (quantized while loading), the text encoder stays on the CPU and
ComponentsManager moves whole components on and off the GPU in turn.
Writes <out>/run.json in the same schema as the ComfyUI runs, plus av.mp4 and audio.flac.

Usage: python run_diffusers.py --task T1_dialogue_closeup --steps 20 --seed 10501 --out /work/runs/<id>
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

import torch

sys.path.insert(0, os.environ.get("H3_BENCH", "."))
from h3bench import tasks  # noqa: E402

HF = os.environ.get("H3_HF", "MiniMax-H3")
EV = []


def mark(name):
    torch.cuda.synchronize()
    EV.append((time.time(), name))


class Tel(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples, self.p = [], None

    def run(self):
        self.p = subprocess.Popen(["nvidia-smi", "--query-gpu=power.draw,clocks.sm,clocks.mem,memory.used,"
                                   "utilization.gpu,temperature.gpu", "--format=csv,noheader,nounits", "-lms", "500"],
                                  stdout=subprocess.PIPE, text=True)
        for line in self.p.stdout:
            try:
                p, sm, mem, used, util, temp = [float(x) for x in line.split(",")]
            except ValueError:
                continue
            self.samples.append(dict(t=time.time(), power_w=p, sm_mhz=sm, mem_mhz=mem, mem_used_mib=used,
                                     util=util, temp_c=temp))


def hook_times(module, label):
    def pre(m, a, k):
        torch.cuda.synchronize()
        EV.append((time.time(), label + "_start"))

    def post(m, a, k, o):
        torch.cuda.synchronize()
        EV.append((time.time(), label + "_end"))
    module.register_forward_pre_hook(pre, with_kwargs=True)
    module.register_forward_hook(post, with_kwargs=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="T1_dialogue_closeup")
    ap.add_argument("--steps", type=int, default=20, help="model evaluations (grid points = steps + 1)")
    ap.add_argument("--seed", type=int, default=10501)
    ap.add_argument("--out", required=True)
    ap.add_argument("--placement", default="auto_offload", choices=["auto_offload", "dit_resident"])
    ap.add_argument("--attn", default=None, help="diffusers attention backend name, e.g. native, sage")
    ap.add_argument("--repeat", type=int, default=1, help="extra warm runs in the same process")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tel = Tel()
    tel.start()
    from diffusers import ComponentsManager, MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3AudioReference, MiniMaxH3ImageReference
    from diffusers.utils.export_utils import encode_video
    from torchao.quantization import Int8WeightOnlyConfig
    from transformers import Qwen3VLForConditionalGeneration
    from transformers import TorchAoConfig as TTorchAoConfig
    import soundfile as sf

    t = tasks.task_spec(a.task)
    t_load0 = time.time()
    keep = ["proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj", "token_refiner",
            "norm_out", "proj_out", "audio_proj_out"]
    dit = MiniMaxH3Transformer3DModel.from_pretrained(
        HF, subfolder="transformer_ref", dtype=torch.bfloat16, device_map="cuda",
        quantization_config=TorchAoConfig(Int8WeightOnlyConfig(version=2), modules_to_not_convert=keep))
    mark("dit_loaded")
    te = Qwen3VLForConditionalGeneration.from_pretrained(
        HF, subfolder="text_encoder", dtype=torch.bfloat16, device_map="cpu",
        quantization_config=TTorchAoConfig(Int8WeightOnlyConfig(version=2), modules_to_not_convert=[
            "model.visual", "model.language_model.embed_tokens", "model.language_model.norm", "lm_head"]))
    mark("te_loaded")
    dit.requires_grad_(False)
    te.requires_grad_(False)
    manager = ComponentsManager()
    pipe = ModularPipeline.from_pretrained(HF, workflow="ref2va", components_manager=manager)
    pipe.update_components(transformer_ref=dit, text_encoder=te)
    pipe.load_components(dtype=torch.bfloat16)  # names=None skips the components set above
    manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin="6GB")
    if a.attn:
        pipe.transformer_ref.set_attention_backend(a.attn)
    mark("pipeline_ready")
    load_s = time.time() - t_load0
    hook_times(pipe.transformer_ref, "dit")
    hook_times(pipe.text_encoder, "te")
    hook_times(pipe.vae, "vae")
    hook_times(pipe.audio_vae, "avae")
    refs = [MiniMaxH3ImageReference.from_file(f"inputs/{im}") for im in t["images"]]
    refs += [MiniMaxH3AudioReference.from_file(f"inputs/{au}") for au in t["audios"]]
    recs = []
    for rep in range(a.repeat):
        EV.clear()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        res = pipe(prompt=t["prompt_text"], references=refs, height=t["height"], width=t["width"],
                   num_frames=t["length"], num_inference_steps=a.steps + 1,
                   generator=torch.Generator("cpu").manual_seed(a.seed + rep),
                   output=["videos", "audio", "sampling_rate"])
        mark("pipe_done")
        t1 = time.time()
        sub = a.out if rep == 0 else os.path.join(a.out, f"rep{rep}")
        os.makedirs(sub, exist_ok=True)
        aud = res["audio"][0]
        sr = res["sampling_rate"]
        mp4 = os.path.join(sub, "av-audio.mp4")
        encode_video(res["videos"][0], fps=24, output_path=mp4, audio=aud, audio_sample_rate=sr)
        an = aud.float().cpu().numpy() if torch.is_tensor(aud) else aud
        if an.ndim == 2 and an.shape[0] in (1, 2):
            an = an.T
        sf.write(os.path.join(sub, "audio.flac"), an, sr)
        recs.append(summarize(t0, t1, list(EV), tel.samples, a, t, sub, mp4, load_s if rep == 0 else 0.0))
    tel.p.terminate()
    for r in recs:
        json.dump(r, open(os.path.join(r["dir"], "run.json"), "w"), indent=1, default=str)
    print(json.dumps({k: recs[0][k] for k in ("wall_s", "phase_s", "sampler_mean_step_s", "peak_mem_mib")},
                     default=str))


def summarize(t0, t1, ev, tel, a, t, sub, mp4, load_s):
    def spans(label):
        st = [x for x, n in ev if n == label + "_start"]
        en = [x for x, n in ev if n == label + "_end"]
        return list(zip(st, en))
    dit = spans("dit")
    step_s = [e - s for s, e in dit]
    phase = {"encode": sum(e - s for s, e in spans("te")), "sample": (dit[-1][1] - dit[0][0]) if dit else 0,
             "vae_decode": sum(e - s for s, e in spans("vae")), "audio_decode": sum(e - s for s, e in spans("avae"))}
    phase["other"] = (t1 - t0) - sum(phase.values())
    if load_s:
        phase["load"] = load_s
    run = [s for s in tel if t0 <= s["t"] <= t1]
    smp = [s for s in tel if dit and dit[0][0] <= s["t"] <= dit[-1][1]]
    en = sum(x["power_w"] * (y["t"] - x["t"]) / 3600 for x, y in zip(run, run[1:]))
    avg = lambda v: sum(v) / len(v) if v else None
    files = [dict(path=mp4, kind="videos"), dict(path=os.path.join(sub, "audio.flac"), kind="audio")]
    return dict(dir=sub, engine="diffusers", hardware="CMP 170HX", wall_s=(t1 - t0) + load_s, exec_s=t1 - t0,
                phase_s=phase, sampler_steps=len(step_s), sampler_step_s=step_s[1:],
                sampler_first_step_s=step_s[0] if step_s else None,
                sampler_mean_step_s=avg(step_s[1:]), peak_mem_mib=max((s["mem_used_mib"] for s in run), default=None),
                torch_peak_alloc_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
                mean_power_sample_w=avg([s["power_w"] for s in smp]), max_power_w=max((s["power_w"] for s in run), default=None),
                mean_util_sample=avg([s["util"] for s in smp]), mean_sm_mhz_sample=avg([s["sm_mhz"] for s in smp]),
                max_temp_c=max((s["temp_c"] for s in run), default=None), energy_wh=en, telemetry=run, files=files,
                job=dict(run_id=os.path.basename(sub), task=t["id"], config=f"base{a.steps}", seed=a.seed,
                         placement=a.placement, attn=a.attn),
                task=dict((k, v) for k, v in t.items() if k != "prompt_text"), error=None,
                versions=versions())


def versions():
    import importlib.metadata as m
    v = {p: m.version(p) for p in ("torch", "diffusers", "transformers", "torchao", "accelerate")}
    v["cuda"] = torch.version.cuda
    v["device"] = torch.cuda.get_device_name()
    return v


if __name__ == "__main__":
    main()
