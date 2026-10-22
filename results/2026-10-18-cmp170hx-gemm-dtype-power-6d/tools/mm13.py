# 2026-10-18: 5 W power-cap sweep. One process per card; sets the cap itself (sudo -n nvidia-smi -pl),
# measures bf16 and int8 8192^3 GEMM (3 x ~1 s each) per step under continuous load; JSON line per measurement.
import torch, sys, time, json, subprocess, statistics
idx, lo, hi, tag = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
d = torch.device('cuda:0'); assert torch.cuda.get_device_name(0) == 'NVIDIA Graphics Device'
def smi(q): return subprocess.run(['nvidia-smi', '-i', idx, f'--query-gpu={q}', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.strip()
def setcap(w): subprocess.run(['sudo', '-n', 'nvidia-smi', '-i', idx, '-pl', str(w)], check=True, capture_output=True)
N = 8192; f = 2 * N**3
a8 = torch.randint(-127, 128, (N, N), dtype=torch.int8, device=d); b8 = torch.randint(-127, 128, (N, N), dtype=torch.int8, device=d).t()
ab = torch.randn(N, N, dtype=torch.bfloat16, device=d); bb = torch.randn(N, N, dtype=torch.bfloat16, device=d)
ops = {'bf16': lambda: ab @ bb, 'int8': lambda: torch._int_mm(a8, b8)}
def run(fn, secs):
    torch.cuda.synchronize(); t = time.time(); n = 0
    while time.time() - t < secs: fn(); n += 1; torch.cuda.synchronize()
def rate(fn, iters):
    s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize(); return f / (s.elapsed_time(e) / iters / 1e3) / 1e12
try:
    for cap in range(lo, hi + 1, 5):
        setcap(cap); run(ops['bf16'], 3.0)          # settle under load at the new cap
        for name, fn in ops.items():
            r = []; t0 = time.time()
            for _ in range(3): r.append(rate(fn, 8 if name == 'bf16' else 12))
            t1 = time.time()
            print(json.dumps(dict(tag=tag, cap=cap, dtype=name, median=statistics.median(r), lo=min(r), hi=max(r), t0=t0, t1=t1)), flush=True)
        c, m = (smi('temperature.gpu,temperature.memory').replace(' ', '') + ',0').split(',')[:2]
        if int(c) >= 83 or (m.isdigit() and int(m) >= 90): print(json.dumps(dict(tag=tag, abort=True, cap=cap, core=c, hbm=m)), flush=True); break
finally:
    setcap(140)
