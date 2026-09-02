import json, sys, torch

def run(dtype_name, gpu_index, seed, n=4096):
    torch.manual_seed(seed)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "tf32": torch.float32}[dtype_name]
    a64 = torch.randn(n, n, dtype=torch.float64)
    b64 = torch.randn(n, n, dtype=torch.float64)
    ref = (a64 @ b64).numpy()

    dev = f"cuda:{gpu_index}"
    if dtype_name == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        a = a64.to(dev, dtype=torch.float32)
        b = b64.to(dev, dtype=torch.float32)
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        a = a64.to(dev, dtype=dtype)
        b = b64.to(dev, dtype=dtype)
    out = (a @ b).to(torch.float64).cpu().numpy()

    max_abs_err = float((out - ref).__abs__().max())
    return max_abs_err, out

def run_int8(gpu_index, seed, n=4096):
    torch.manual_seed(seed)
    ai = torch.randint(-127, 127, (n, n), dtype=torch.int8)
    bi = torch.randint(-127, 127, (n, n), dtype=torch.int8)
    ref = (ai.to(torch.int64) @ bi.to(torch.int64)).numpy()
    dev = f"cuda:{gpu_index}"
    out = torch._int_mm(ai.to(dev), bi.to(dev)).to(torch.int64).cpu().numpy()
    max_abs_err = int((out - ref).__abs__().max())
    return max_abs_err, out

if __name__ == "__main__":
    gpu_index = int(sys.argv[1])
    seed = int(sys.argv[2])
    out_path = sys.argv[3]
    result = {"gpu": gpu_index, "seed": seed, "paths": {}}
    for name in ("bf16", "fp16", "tf32"):
        try:
            err, _ = run(name, gpu_index, seed)
            result["paths"][name] = {"max_abs_err_vs_cpu_f64": err}
        except Exception as e:
            result["paths"][name] = {"error": str(e)}
    try:
        err, _ = run_int8(gpu_index, seed)
        result["paths"]["int8"] = {"max_abs_err_vs_cpu_i64": err}
    except Exception as e:
        result["paths"]["int8"] = {"error": str(e)}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))
