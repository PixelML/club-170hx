# 2026-10-18: CMP 170HX matmul + HBM bench. One JSON line per measurement with wall times for power/clock alignment.
# mm11 = mm10 with shorter cells (2 x ~1 s per test, 1 GiB copy): used for the before/after-6D and fleet sweeps.
import torch, sys, time, json, statistics
d = torch.device('cuda:0'); assert torch.cuda.get_device_name(0) == 'NVIDIA Graphics Device'
tag = sys.argv[1]
def timed(fn, target_s=1.0):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t = time.time(); fn(); torch.cuda.synchronize(); one = max(time.time() - t, 1e-5)
    iters = max(3, int(target_s / one))
    s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / iters / 1e3
def meas(name, fn, work, unit, **kw):
    t0 = time.time(); secs = [timed(fn) for _ in range(2)]; t1 = time.time()
    rates = [work / s / 1e12 for s in secs]
    print(json.dumps(dict(tag=tag, test=name, unit=unit, median=statistics.median(rates), lo=min(rates), hi=max(rates), t0=t0, t1=t1, **kw)), flush=True)
def mk(dt, m, k, n):
    if dt == 'int8':
        return torch.randint(-127, 128, (m, k), dtype=torch.int8, device=d), torch.randint(-127, 128, (n, k), dtype=torch.int8, device=d).t()
    t = dict(bf16=torch.bfloat16, fp16=torch.float16, tf32=torch.float32, fp32=torch.float32)[dt]
    return torch.randn(m, k, dtype=t, device=d), torch.randn(k, n, dtype=t, device=d)
def op(dt, a, b):
    torch.backends.cuda.matmul.allow_tf32 = (dt == 'tf32')
    return (lambda: torch._int_mm(a, b)) if dt == 'int8' else (lambda: a @ b)
# 1. square GEMMs
for N in (8192,):
    for dt in ('int8', 'bf16', 'fp16'):
        a, b = mk(dt, N, N, N); meas(f'{dt}', op(dt, a, b), 2 * N**3, 'TOPS', shape='square', M=N, K=N, N=N, dtype=dt); del a, b
    if False:
        a = torch.randint(-127, 128, (N, N), dtype=torch.int8, device=d); b = torch.randint(-127, 128, (N, N), dtype=torch.int8, device=d)
        meas('int8 row-major B (trap)', lambda: torch._int_mm(a, b), 2 * N**3, 'TOPS', shape='square', M=N, K=N, N=N, dtype='int8_rowB'); del a, b
    torch.cuda.empty_cache()
# 2. model-shaped GEMMs (rows = tokens in flight)
for (K, N) in ((4096, 4096),):
    for M in (32, 2048):
        for dt in ('int8', 'bf16'):
            a, b = mk(dt, M, K, N); meas(f'{dt}', op(dt, a, b), 2 * M * K * N, 'TOPS', shape='model', M=M, K=K, N=N, dtype=dt); del a, b
# 3. HBM bandwidth: device-to-device copy, read + write counted
x = torch.empty(1024**3, dtype=torch.uint8, device=d); y = torch.empty_like(x)
meas('HBM copy', lambda: y.copy_(x), 2 * x.numel(), 'TB/s', shape='copy', dtype='bytes')
