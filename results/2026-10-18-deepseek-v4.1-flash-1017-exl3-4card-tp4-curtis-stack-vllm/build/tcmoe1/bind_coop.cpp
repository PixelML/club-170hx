// exllamav3_coop_ext: only exl3_moe_coop from turboderp-org/exllamav3 @ d3739fd3 (v1.5.3, MIT; LICENSE.exllamav3),
// built as a side module next to the image's stock exllamav3_ext. Pixel ML, Oct 18 2026.
#include <torch/extension.h>
#include "quant/exl3_moe_coop.cuh"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("exl3_moe_coop", &exl3_moe_coop, "exl3_moe_coop");
}
