# Debug hook (PixelML sm80 port): after every dispatched torch op, check the CUDA last-error state
# and print the first op that leaves cudaErrorNoKernelImageForDevice (209). Enabled by OPROBE=1.
import os
if os.environ.get("OPROBE") == "1":
    import ctypes, sys, traceback
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    _rt = None
    for name in ("libcudart.so.13", "libcudart.so.12", "libcudart.so"):
        try:
            _rt = ctypes.CDLL(name); break
        except OSError:
            pass
    _seen = set()

    class _Probe(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            if _rt is not None:
                err = _rt.cudaPeekAtLastError()
                if err != 0:
                    key = str(func)
                    if key not in _seen:
                        _seen.add(key)
                        sys.stderr.write(f"OPROBE err={err} after op {func}\n")
                        sys.stderr.write("".join(traceback.format_stack(limit=12)[:-1]))
                        sys.stderr.flush()
            return out

    _orig_init = torch.cuda.init
    _mode = _Probe()
    _mode.__enter__()
