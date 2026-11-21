"""Device-to-device copy bandwidth (read+write GB/s) on one GPU: 2 GiB buffer, 50 copies, median of 5."""
import statistics, torch
x = torch.empty(2 * 2**30, dtype=torch.uint8, device="cuda"); y = torch.empty_like(x)
for _ in range(5): y.copy_(x)
torch.cuda.synchronize()
res = []
for _ in range(5):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(50): y.copy_(x)
    e.record(); torch.cuda.synchronize()
    res.append(2 * x.numel() * 50 / (s.elapsed_time(e) / 1e3) / 1e9)
print(f"{torch.cuda.get_device_name(0)} copy_bw_GBps={statistics.median(res):.1f}")
