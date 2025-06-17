#include "memManag.cuh"

size_t getFreeGlobalMemory(int gpu_num)
{
  size_t free_mem = 0;
  size_t total_mem = 0;
  cuchk(cudaSetDevice(gpu_num));
  cuchk(cudaMemGetInfo(&free_mem, &total_mem));
  return free_mem / 1024 / 1024; // in MB
}