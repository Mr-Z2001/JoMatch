#ifndef DEFS_H
#define DEFS_H

#include <cstdint>

#define GRID_DIM 504u
// #define GRID_DIM 252u
// #define GRID_DIM 168u // for A6000 252u theoretically
// #define GRID_DIM 1024u
// #define BLOCK_DIM 512u
#define BLOCK_DIM 256u
#define WARP_SIZE 32u
#define WARP_PER_BLOCK (BLOCK_DIM / WARP_SIZE)
#define NWARPS_TOTAL (GRID_DIM * BLOCK_DIM / WARP_SIZE)

#define MAX_VQ 16u
#define MAX_EQ 64u
#define MAX_VLQ 16u

#define MAX_CLUSTERS 150
#define MAX_LAYERS 100
#define MAX_LEVEL 100

#define BLK_SIZE 32

// #define MAX_RES 500000000
#define MAX_RES 5000000

#define MAX_SLOT_NUM 15u
#define MAX_DEGREE 4096u

#define JOB_CHUNK_SIZE 8u
#define DETECT_LEVEL 1u
// #define STOP_LEVEL 2u

#define TIMEOUT 10 // ms
#define TIMEOUT_QUEUE_CAP 100'000'000
#define CLOCK_RATE 1800000
#define ELAPSED_TIME(start) ((clock64() - start) / (1.0 * CLOCK_RATE)) // ms

using vtype = uint32_t;
using etype = uint32_t;
using vltype = uint32_t;
using numtype = uint32_t;
using offtype = uint32_t;
using degtype = uint32_t;
// using eltype = uint32_t;

inline unsigned int calc_grid_dim(int N, int block_size)
{
  if (N == 0)
    ++N;
  return (N - 1) / block_size + 1;
}

#define TID (threadIdx.x)
#define BID (blockIdx.x)
#define IDX (TID + BID * blockDim.x)
#define LID (TID & 31)
#define WID (TID >> 5)
#define WID_G (IDX >> 5)

#endif