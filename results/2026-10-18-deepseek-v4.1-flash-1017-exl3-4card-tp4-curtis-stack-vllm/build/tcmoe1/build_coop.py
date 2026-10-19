# build_coop.py OUT_DIR -- compile exllamav3 @d3739fd3 exl3_moe_coop (turboderp, MIT) as `exllamav3_coop_ext` for sm_80.
# Flags follow exllamav3/util/cuda_flags.py at that commit. No GPU needed. Pixel ML, Oct 18 2026.
import glob, os, shutil, sys
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")
from torch.utils import cpp_extension
here = os.path.dirname(os.path.abspath(__file__))
E = os.path.join(here, "exllamav3_ext")
out = sys.argv[1]
src = [os.path.join(here, "bind_coop.cpp"), f"{E}/quant/exl3_moe_coop.cu", f"{E}/quant/exl3_devctx.cu"]
src += sorted(glob.glob(f"{E}/quant/comp_units/exl3_moe_coop_inst_*.cu"))
extra = sys.argv[2:]  # optional extra sources when the linker asks for them
src += [f"{E}/{x}" for x in extra]
bd = "/tmp/coop_build"; os.makedirs(bd, exist_ok=True); os.makedirs(out, exist_ok=True)
cpp_extension.load(
    name="exllamav3_coop_ext", sources=src, build_directory=bd, verbose=True, is_python_module=False,
    extra_include_paths=[E],
    extra_cflags=["-Ofast"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-Xcudafe", "--diag_suppress=177", "-Xcudafe", "--diag_suppress=20012", "-Xptxas=-v"],
)
shutil.copy2(f"{bd}/exllamav3_coop_ext.so", out)
print("built", os.path.join(out, "exllamav3_coop_ext.so"))
