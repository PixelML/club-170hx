set -ex
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq git cmake ninja-build python3 python3-pip python3-venv curl libcurl4-openssl-dev >/dev/null
mkdir -p /work && cd /work
[ -d llama.cpp ] || git clone -q https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && git fetch -q --tags && git checkout -q b10246 && git rev-parse HEAD > /work/llama.cpp.sha
cmake -S . -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF >/dev/null
cmake --build build -j8 --target llama llama-server llama-cli llama-bench 2>&1 | tail -3
cd /work
[ -d open-jet ] || git clone -q https://github.com/L-Forster/open-jet.git
cd open-jet && git checkout -q 93c2b9abee50ea2ea41981c406cfd201989e4d58 && git rev-parse HEAD > /work/open-jet.sha
cd megakernel && make ARCH=sm_80 LLAMA=/work/llama.cpp NVCC=/usr/local/cuda/bin/nvcc -j3 2>&1 | tail -20
echo BUILD_DONE
