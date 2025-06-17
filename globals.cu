#include "globals.cuh"
#include "cpuGraph.h"

int GPU_NUM;
uint32_t NUM_VQ;
uint32_t NUM_EQ;
uint32_t NUM_VLQ;
// uint32_t NUM_ELQ;
uint32_t NUM_VD;
uint32_t NUM_ED;
// uint32_t NUM_VLD;
// uint32_t NUM_ELD;
// uint32_t NUM_VL;
// uint32_t NUM_EL;
uint32_t COL_LEN;
uint32_t NUM_BLOCKS;

uint32_t NUM_CAN_UB;
__device__ __constant__ uint32_t C_NUM_CAN_UB;

uint32_t MAX_L_FREQ;
uint32_t MAX_DATA_DEGREE;
uint32_t TABLE_SIZE;
uint8_t EIDX[MAX_VQ * MAX_VQ];
uint16_t NBRBIT[MAX_VQ];

uint32_t STOP_LEVEL;

double total_match_time_us = 0;
double total_match_time_ms = 0;

__constant__ bool C_ADAPTIVE_ORDERING;
__constant__ bool C_LB_ENABLE;

__device__ __constant__ uint32_t C_NUM_VQ;
__device__ __constant__ uint32_t C_NUM_EQ;
__device__ __constant__ uint32_t C_NUM_VLQ;
// __constant__ uint32_t C_NUM_ELQ;
__device__ __constant__ uint32_t C_NUM_VD;
__device__ __constant__ uint32_t C_NUM_ED;
// __constant__ uint32_t C_NUM_VLD;
// __constant__ uint32_t C_NUM_ELD;
// __constant__ uint32_t C_NUM_VL;
// __constant__ uint32_t C_NUM_EL;
__device__ __constant__ uint32_t C_NUM_BLOCKS;
__device__ __constant__ uint32_t C_COL_LEN;
__device__ __constant__ uint32_t C_STOP_LEVEL;

__device__ __constant__ uint32_t C_MAX_DEGREE;
__device__ __constant__ uint32_t C_MAX_L_FREQ;
__device__ __constant__ uint32_t C_TABLE_SIZE;

__device__ __constant__ GraphUtils C_UTILS;

__device__ __constant__ uint32_t C_ORDER[MAX_VQ];
__device__ __constant__ uint32_t C_ORDER_OFFS[MAX_VQ + 1];

__device__ __constant__ uint32_t C[MAX_EQ * 2 * 2 * 2];

__device__ __constant__ uint32_t C_TABLE_OFFS[MAX_EQ * 2];
__device__ __constant__ uint32_t C_NUM_BUCKETS[MAX_EQ * 2];

// cuda settings

__device__ __constant__ uint32_t C_GRID_DIM = 252;
__device__ __constant__ uint32_t C_BLOCK_DIM = 512;
__device__ __constant__ uint32_t C_WARP_SIZE = 32;
__device__ __constant__ uint32_t C_WARP_PER_BLOCK = 16;

// max values

__device__ __constant__ uint32_t C_MAX_VQ = 16;
__device__ __constant__ uint32_t C_MAX_EQ = 64;
__device__ __constant__ uint32_t C_MAX_VLQ = 16;