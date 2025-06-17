#ifndef JOIN_H
#define JOIN_H

#include "globals.cuh"
#include "cpuGraph.h"
#include "gpuGraph.h"
#include "structure.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void
firstJoinKernel(
    vtype u,
    vtype *d_u_candidate_vs_, numtype num_u_candidate_vs,
    vtype *d_res_table_);

__global__ void
selectPartialMatchingsKernel(
    offtype *offsets_, vtype *nbrs_,
    vtype u, vtype u_matched,
    vtype *d_res_table_old_, unsigned long long num_res_old,
    vtype *d_res_table_, unsigned long long *num_res_new);

__global__ void
joinOneEdgeKernel(
    // structure
    offtype *offsets_, vtype *nbrs_,

    vtype u, vtype u_matched,
    vtype *d_res_table_old_, unsigned long long num_res_old,
    vtype *d_res_table_, unsigned long long *num_res_new,

    uint32_t *encodings_, numtype num_blocks);

void joinOneEdge(
    cpuGraph *hq, cpuGraph *hg,
    gpuGraph *dq, gpuGraph *dg,

    vtype u, vtype u_matched,
    vtype *d_res_table_old_, unsigned long long &num_res_old,
    vtype *d_res_table_, unsigned long long &num_res_new,

    uint32_t *d_encodings_,

    encodingMeta *enc_meta);

#endif