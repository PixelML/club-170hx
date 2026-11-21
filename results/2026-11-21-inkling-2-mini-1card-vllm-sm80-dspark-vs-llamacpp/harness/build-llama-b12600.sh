#!/bin/bash
# Build llama.cpp release b12600 (ggml-org master, includes PR #32694 Inkling 2 Mini) for sm_80 (CMP 170HX).
set -ex
apt-get update -qq && apt-get install -y -qq git cmake build-essential >/dev/null
mkdir -p /src/llama.cpp && cd /src/llama.cpp
[ -d .git ] || git init -q
git fetch -q --depth 1 https://github.com/ggml-org/llama.cpp 29b52d398b294509746ec0a3f28948a3107c0213
git checkout -q FETCH_HEAD
git log -1 --format=%H
ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
cmake -B build -DGGML_CUDA=ON -DCMAKE_EXE_LINKER_FLAGS='-L/usr/local/cuda/lib64/stubs -Wl,-rpath-link,/usr/local/cuda/lib64/stubs' -DCMAKE_SHARED_LINKER_FLAGS='-L/usr/local/cuda/lib64/stubs' -DCMAKE_CUDA_ARCHITECTURES=80 -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 32 --target llama-server llama-bench
ls -la build/bin
