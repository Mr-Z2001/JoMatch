#ifndef GPU_MATCH_H
#define GPU_MATCH_H

#include "gpuGraph.h"
#include "callstack.cuh"
#include "job_queue.cuh"
#include "queue.cuh"
#include "order.h"
#include "defs.h"
#include "res_table.hpp"

class StealingArgs
{
public:
    Queue *queue;
};

class Arg_t
{
public:
    vtype *set1, *set2, *res;
    vtype set1_size, set2_size, *res_size;
};

__device__ void
intersect(Arg_t *__restrict__ arg);

__device__ void
steal(
    CallStack *__restrict__ stk, StealingArgs *__restrict__ stealing_args, bool *__restrict__ __restrict__ ret, long long &start_clk, vtype *v_order,
    int *queue_arr_,
    vtype *__restrict__ cur_res, int *order_id);

__device__ void
get_new_v(
    degtype *__restrict__ d_degs_, offtype *__restrict__ d_offsets_, vtype *__restrict__ d_nbrs_,
    CallStack *__restrict__ stk, int *__restrict__ cur,
    StealingArgs *__restrict__ stealing_args, long long &start_clk,
    OrderGPU *__restrict__ order_obj,

    int start_level,

    uint32_t *__restrict__ compact_encodings_,
    int num_blocks,

    vtype *__restrict__ initial_task_table_, numtype num_initial_task_table_rows,

    bool *__restrict__ ret, vtype *__restrict__ cur_res, int *order_id);

__device__ void match(
    degtype *__restrict__ d_degs_, offtype *__restrict__ d_offsets_, vtype *__restrict__ d_nbrs_,
    CallStack *__restrict__ stk, int *__restrict__ cur, unsigned long long *__restrict__ count, StealingArgs *__restrict__ stealing_args, long long &start_clk,
    OrderGPU *__restrict__ order_obj, int start_level, int *__restrict__ queue_arr_,
    uint32_t *__restrict__ compact_encodings_, int num_blocks,
    vtype *__restrict__ initial_task_table_, numtype num_initial_task_table_rows,

    bool *ret, vtype *cur_res);

__global__ void
parallel_match_kernel(
    degtype *__restrict__ d_degs_, offtype *__restrict__ d_offsets_, vtype *__restrict__ d_nbrs_,
    CallStack *__restrict__ call_stack, int *__restrict__ cur, unsigned long long *__restrict__ res,
    Queue *__restrict__ queue, OrderGPU *__restrict__ order_obj, int num_orders,

    int start_level,

    uint32_t *__restrict__ compact_encodings_,
    int num_blocks,
    vtype *__restrict__ d_u_candidate_vs_, numtype *__restrict__ d_num_u_candidate_vs_,

    vtype *__restrict__ initial_task_table_, numtype num_initial_task_table_rows);

__global__ void
vertexJoinBFS(
    offtype *d_offsets, vtype *d_nbrs, degtype *d_degs,
    OrderGPU *order_obj,
    int level, // size of one row = level + 1
    vtype *intersect_temp_storage, numtype *num_intersect_temp_storage,

    uint32_t *d_encodings_, numtype num_blocks, int *d_enc_pos_u_,
    vtype *d_res_table_old, numtype num_res_old,
    vtype *d_res_table, numtype *num_res_new,
    int *exceed);

void parallelMatch(
    cpuGraph *hq, cpuGraph *hg,
    gpuGraph *dq, gpuGraph *dg,
    OrderCPU *order_obj,

    uint32_t *d_encodings_,
    encodingMeta *enc_meta,
    uint32_t *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_,
    numtype *h_num_u_candidate_vs_,

    ResTable *res_table);

#endif // GPU_MATCH_H