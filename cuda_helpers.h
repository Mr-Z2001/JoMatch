#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

struct timespec timespec_diff(struct timespec start, struct timespec end);

#define FULL_MASK 0xffffffff

#define DIV_CEIL(a, b) ((a) / (b) + ((a) % (b) != 0))

#define TO_GB(x) (x / 1024.0 / 1024.0 / 1024.0)

#define cuchk(ans)                        \
  {                                       \
    gpuAssert((ans), __FILE__, __LINE__); \
  }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    std::cerr << "GPU assert: "
              << cudaGetErrorName(code) << " "
              << cudaGetErrorString(code) << " "
              << file << " "
              << line << " " << std::endl;
    if (abort)
      exit(code);
  }
}

#define cuchk_kernel(call)                          \
  {                                                 \
    cudaError_t cucheck_err = (call);               \
    if (cucheck_err != cudaSuccess)                 \
    {                                               \
      std::cerr << __FILE__ << " "                  \
                << __LINE__ << " "                  \
                << cudaGetErrorString(cucheck_err); \
      assert(0);                                    \
    }                                               \
  }

// Recording Time

#define TIME_INIT()                                           \
  cudaEvent_t gpu_start, gpu_end;                             \
  float kernel_time, total_kernel = 0.0, total_host = 0.0;    \
  auto cpu_start = std::chrono::high_resolution_clock::now(); \
  auto cpu_end = std::chrono::high_resolution_clock::now();   \
  std::chrono::duration<double> diff = cpu_end - cpu_start;

#define TIME_START()                                     \
  cpu_start = std::chrono::high_resolution_clock::now(); \
  cudaEventCreate(&gpu_start);                           \
  cudaEventCreate(&gpu_end);                             \
  cudaEventRecord(gpu_start)

#define TIME_END()                                        \
  cpu_end = std::chrono::high_resolution_clock::now();    \
  cudaEventRecord(gpu_end);                               \
  cudaEventSynchronize(gpu_start);                        \
  cudaEventSynchronize(gpu_end);                          \
  cudaEventElapsedTime(&kernel_time, gpu_start, gpu_end); \
  total_kernel += kernel_time;                            \
  diff = cpu_end - cpu_start;                             \
  total_host += diff.count();

#define PRINT_LOCAL_TIME(name)                                 \
  std::cout << name << ", time (ms): "                         \
            << static_cast<unsigned long>(diff.count() * 1000) \
            << "(host), "                                      \
            << static_cast<unsigned long>(kernel_time)         \
            << "(kernel)\n"

#define PRINT_TOTAL_TIME(name)                               \
  std::cout << name << " time (ms): "                        \
            << static_cast<unsigned long>(total_host * 1000) \
            << "(host) "                                     \
            << static_cast<unsigned long>(total_kernel)      \
            << "(kernel)\n";

#define micro_init()       \
  struct timespec time_st; \
  struct timespec time_ed; \
  struct timespec diff_micro;

#define micro_start() \
  clock_gettime(CLOCK_MONOTONIC, &time_st);

#define micro_end() \
  clock_gettime(CLOCK_MONOTONIC, &time_ed);

#define micro_calc_local() \
  diff_micro = timespec_diff(time_st, time_ed);

#define micro_print_local(name)                 \
  diff_micro = timespec_diff(time_st, time_ed); \
  std::cout << name << " time (us): " << diff_micro.tv_nsec / 1000.0 << std::endl;

#define MEM_INIT() size_t mf, ma;

#define PRINT_MEM_INFO(name)              \
  cudaMemGetInfo(&mf, &ma);               \
  std::cout << name << ", Free "          \
            << TO_GB(mf) << "GB, Total: " \
            << TO_GB(ma) << "GB\n";

struct nonZeroOp
{
  __host__ __device__ bool operator()(const uint32_t &x)
  {
    return x != 0;
  }
};

struct nonUintMax
{
  __host__ __device__ bool operator()(const uint32_t &x)
  {
    return x != UINT32_MAX;
  }
};

template <typename T>
__forceinline__ __device__ uint32_t lower_bound(T *array, uint32_t size, const T &v)
{
  if (array == nullptr || size == 0 || array[size - 1] < v)
    return UINT32_MAX;

  uint32_t low = 0u, high = size - 1, mid = (low + high) / 2;
  while (low < high)
  {
    if (array[mid] < v)
    {
      low = mid + 1;
    }
    else
    {
      high = mid;
    }
    mid = (low + high) / 2;
  }
  return mid;
}

#endif // CUDA_HELPERS_H