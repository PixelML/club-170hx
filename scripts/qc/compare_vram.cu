#include <cuda_runtime.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

struct Failure {
  unsigned long long found;
  unsigned long long expected;
  unsigned long long index;
  int mismatch;
};

__device__ __forceinline__ unsigned long long pattern(unsigned long long index) {
  unsigned long long value = index + 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

__global__ void fill(unsigned long long* memory, unsigned long long words) {
  const unsigned long long start =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long stride =
      static_cast<unsigned long long>(blockDim.x) * gridDim.x;
  for (unsigned long long i = start; i < words; i += stride) {
    memory[i] = pattern(i);
  }
}

__global__ void verify(const unsigned long long* memory,
                       unsigned long long words,
                       Failure* failure) {
  const unsigned long long start =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const unsigned long long stride =
      static_cast<unsigned long long>(blockDim.x) * gridDim.x;
  for (unsigned long long i = start; i < words; i += stride) {
    const unsigned long long expected = pattern(i);
    const unsigned long long found = memory[i];
    if (found != expected && atomicCAS(&failure->mismatch, 0, 1) == 0) {
      failure->found = found;
      failure->expected = expected;
      failure->index = i;
    }
  }
}

static bool cuda_ok(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) return true;
  std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
  return false;
}

static unsigned long long parse_ull(const char* text, const char* option) {
  errno = 0;
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0') {
    std::fprintf(stderr, "invalid %s value: %s\n", option, text);
    std::exit(2);
  }
  return value;
}

int main(int argc, char** argv) {
  int device = 0;
  unsigned long long gib = 62;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
      device = static_cast<int>(parse_ull(argv[++i], "--device"));
    } else if (std::strcmp(argv[i], "--gib") == 0 && i + 1 < argc) {
      gib = parse_ull(argv[++i], "--gib");
    } else {
      std::fprintf(stderr, "usage: %s [--device N] [--gib N]\n", argv[0]);
      return 2;
    }
  }

  if (!cuda_ok(cudaSetDevice(device), "cudaSetDevice")) return 1;

  cudaDeviceProp properties{};
  if (!cuda_ok(cudaGetDeviceProperties(&properties, device),
               "cudaGetDeviceProperties")) {
    return 1;
  }

  size_t free_bytes = 0;
  size_t total_bytes = 0;
  if (!cuda_ok(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo")) {
    return 1;
  }

  const unsigned long long bytes = gib * 1024ULL * 1024ULL * 1024ULL;
  const unsigned long long words = bytes / sizeof(unsigned long long);
  std::printf("device=%d name=%s requested=%llu GiB free=%.2f GiB total=%.2f GiB\n",
              device, properties.name, gib,
              static_cast<double>(free_bytes) / (1024.0 * 1024.0 * 1024.0),
              static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0));

  if (bytes > free_bytes) {
    std::fprintf(stderr, "requested allocation exceeds currently free VRAM\n");
    return 1;
  }

  unsigned long long* memory = nullptr;
  Failure* failure = nullptr;
  if (!cuda_ok(cudaMalloc(&memory, static_cast<size_t>(bytes)), "cudaMalloc(data)")) {
    return 1;
  }
  if (!cuda_ok(cudaMalloc(&failure, sizeof(Failure)), "cudaMalloc(failure)")) {
    cudaFree(memory);
    return 1;
  }
  if (!cuda_ok(cudaMemset(failure, 0, sizeof(Failure)), "cudaMemset(failure)")) {
    cudaFree(failure);
    cudaFree(memory);
    return 1;
  }

  constexpr int threads = 256;
  const int blocks = properties.multiProcessorCount * 16;
  fill<<<blocks, threads>>>(memory, words);
  if (!cuda_ok(cudaGetLastError(), "fill launch") ||
      !cuda_ok(cudaDeviceSynchronize(), "fill synchronize")) {
    cudaFree(failure);
    cudaFree(memory);
    return 1;
  }

  verify<<<blocks, threads>>>(memory, words, failure);
  if (!cuda_ok(cudaGetLastError(), "verify launch") ||
      !cuda_ok(cudaDeviceSynchronize(), "verify synchronize")) {
    cudaFree(failure);
    cudaFree(memory);
    return 1;
  }

  Failure host_failure{};
  const bool copied = cuda_ok(
      cudaMemcpy(&host_failure, failure, sizeof(Failure), cudaMemcpyDeviceToHost),
      "cudaMemcpy(failure)");
  cudaFree(failure);
  cudaFree(memory);
  if (!copied) return 1;

  if (host_failure.mismatch != 0) {
    std::fprintf(stderr,
                 "FAIL index=%llu expected=0x%016llx found=0x%016llx\n",
                 host_failure.index, host_failure.expected, host_failure.found);
    return 1;
  }

  std::printf("PASS verified %llu GiB (%llu 64-bit words)\n", gib, words);
  return 0;
}
