#ifndef GLOBALS_H
#define GLOBALS_H

#include <cinttypes>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "defs.h"
#include "cpuGraph.h"

extern int GPU_NUM;
extern uint32_t NUM_VQ;
extern uint32_t NUM_EQ;
extern uint32_t NUM_VLQ;
// extern uint32_t NUM_ELQ;
extern uint32_t NUM_VD;
extern uint32_t NUM_ED;
// extern uint32_t NUM_VLD;
// extern uint32_t NUM_ELD;
// extern uint32_t NUM_VL;
// extern uint32_t NUM_EL;
extern uint32_t COL_LEN;
extern uint32_t NUM_BLOCKS;

extern uint32_t NUM_CAN_UB;
extern __device__ __constant__ uint32_t C_NUM_CAN_UB;

extern uint32_t STOP_LEVEL;

extern double total_match_time_us;
extern double total_match_time_ms;

extern uint32_t MAX_DATA_DEGREE;
extern uint32_t MAX_L_FREQ;
extern uint32_t TABLE_SIZE;

// // edge index given the two endpoints
extern uint8_t EIDX[MAX_VQ * MAX_VQ];
// // the i-th bit is 1 in C_NBRBIT[j] if there is an edge between i and j
extern uint16_t NBRBIT[MAX_VQ];

extern __device__ __constant__ bool C_ADAPTIVE_ORDERING;
extern __device__ __constant__ bool C_LB_ENABLE;

extern __device__ __constant__ uint32_t C_NUM_VQ;
extern __device__ __constant__ uint32_t C_NUM_EQ;
extern __device__ __constant__ uint32_t C_NUM_VLQ;
// extern __device__ __constant__ uint32_t C_NUM_ELQ;
extern __device__ __constant__ uint32_t C_NUM_VD;
extern __device__ __constant__ uint32_t C_NUM_ED;
// extern __device__ __constant__ uint32_t C_NUM_VLD;
// extern __device__ __constant__ uint32_t C_NUM_ELD;
// extern __device__ __constant__ uint32_t C_NUM_VL;
// extern __device__ __constant__ uint32_t C_NUM_EL;
extern __device__ __constant__ uint32_t C_NUM_BLOCKS;
extern __device__ __constant__ uint32_t C_COL_LEN;
extern __device__ __constant__ uint32_t C_STOP_LEVEL;

extern __device__ __constant__ uint32_t C_MAX_DEGREE;
extern __device__ __constant__ uint32_t C_MAX_L_FREQ;
extern __device__ __constant__ uint32_t C_TABLE_SIZE;

extern __device__ __constant__ GraphUtils C_UTILS;

extern __device__ __constant__ uint32_t C_ORDER[MAX_VQ];
extern __device__ __constant__ uint32_t C_ORDER_OFFS[MAX_VQ + 1];

// // constants C0 and C1 for each hash table
extern __device__ __constant__ uint32_t C[MAX_EQ * 2 * 2 * 2];

extern __device__ __constant__ uint32_t C_TABLE_OFFS[MAX_EQ * 2];
extern __device__ __constant__ uint32_t C_NUM_BUCKETS[MAX_EQ * 2];

extern __device__ __constant__ uint32_t C_GRID_DIM;
extern __device__ __constant__ uint32_t C_BLOCK_DIM;
extern __device__ __constant__ uint32_t C_WARP_SIZE;
extern __device__ __constant__ uint32_t C_WARP_PER_BLOCK;

extern __device__ __constant__ uint32_t C_MAX_VQ;
extern __device__ __constant__ uint32_t C_MAX_EQ;
extern __device__ __constant__ uint32_t C_MAX_VLQ;

#endif