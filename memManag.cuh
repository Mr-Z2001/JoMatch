#ifndef MM_CUH
#define MM_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>

#include "cuda_helpers.h"

size_t getFreeGlobalMemory(int gpu_num);

#endif