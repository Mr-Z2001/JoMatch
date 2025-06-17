#include "globals.cuh"
#include "cpuGraph.h"
#include "gpuGraph.h"
#include "join.cuh"
#include "structure.cuh"
#include "cuda_helpers.h"
#include "order.h"
#include "defs.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cooperative_groups.h>

#include <cub/cub.cuh>

__global__ void
selectPartialMatchingsKernel(
    offtype *offsets_, vtype *nbrs_,
    vtype u, vtype u_matched,
    vtype *d_res_table_old_, unsigned long long num_res_old,
    vtype *d_res_table_, unsigned long long *num_res_new)
{
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;
  uint32_t idx = tid + bid * blockDim.x;
  uint32_t wid = tid >> 5;
  uint32_t lid = tid & 31;
  uint32_t wid_g = idx >> 5;

  __shared__ int warp_pos[WARP_PER_BLOCK];

  vtype v, v_nbr;

  int block_iter_cnt = 0;
  int grid_size = blockDim.x * gridDim.x;

  while (block_iter_cnt * grid_size < num_res_old)
  {
    idx = tid + bid * blockDim.x + block_iter_cnt * grid_size;
    wid_g = idx >> 5;

    bool keep = false;
    if (idx < num_res_old)
    {
      // int mask = __activemask();
      auto group = cooperative_groups::coalesced_threads();
      v = d_res_table_old_[idx * C_NUM_VQ + u];
      v_nbr = d_res_table_old_[idx * C_NUM_VQ + u_matched];

      // #pragma unroll 8
      // int pos = lower_bound(offsets_[v], offsets_[v + 1], v_nbr, nbrs_);
      int pos = lower_bound(nbrs_ + offsets_[v], offsets_[v + 1] - offsets_[v], v_nbr);
      keep = (pos != UINT32_MAX && nbrs_[offsets_[v] + pos] == v_nbr);
      // __syncwarp(mask);
      group.sync();
      // group.sync();
      // for (offtype v_off = offsets_[v]; !keep && v_off < offsets_[v + 1]; ++v_off)
      // keep = keep || (nbrs_[v_off] == v_nbr);
      // group.sync();

      if (keep)
      {
        auto g = cooperative_groups::coalesced_threads();
        if (g.thread_rank() == 0)
          warp_pos[wid] = atomicAdd(num_res_new, g.size());
        g.sync();
        int my_pos = warp_pos[wid] + g.thread_rank();
        for (int i = 0; i < C_NUM_VQ; ++i)
          d_res_table_[my_pos * C_NUM_VQ + i] = d_res_table_old_[idx * C_NUM_VQ + i];
      }
    }
    __syncthreads();
    block_iter_cnt++;
  }
}
__global__ void
firstJoinKernel(
    vtype u,
    vtype *d_u_candidate_vs_, numtype num_u_candidate_vs,
    vtype *d_res_table_)
{
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;
  uint32_t idx = tid + bid * blockDim.x;

  int block_iter_cnt = 0;
  int grid_size = blockDim.x * gridDim.x;

  while (block_iter_cnt * grid_size < num_u_candidate_vs)
  {
    idx = tid + bid * blockDim.x + block_iter_cnt * grid_size;

    if (idx < num_u_candidate_vs)
    {
      vtype v = d_u_candidate_vs_[u * C_MAX_L_FREQ + idx];
      d_res_table_[idx * C_NUM_VQ + u] = v;
    }
    __syncthreads();

    block_iter_cnt++;
  }
}

__global__ void
joinOneEdgeKernel(
    // structure
    offtype *offsets_, vtype *nbrs_,

    vtype u, vtype u_matched,
    vtype *d_res_table_old_, unsigned long long num_res_old,
    vtype *d_res_table_, unsigned long long *num_res_new,

    uint32_t *compact_encodings_, numtype num_blocks)
{
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;
  uint32_t idx = tid + bid * blockDim.x;
  uint32_t wid = tid >> 5;
  uint32_t lid = tid & 31;
  uint32_t wid_g = idx >> 5;

  __shared__ vtype s_v[WARP_PER_BLOCK];
  __shared__ int warp_pos[WARP_PER_BLOCK];

  int block_iter_cnt = 0;
  int grid_size = blockDim.x * gridDim.x;

  while (block_iter_cnt * (grid_size / warpSize) < num_res_old)
  {
    idx = tid + bid * blockDim.x + block_iter_cnt * grid_size;
    wid_g = idx / warpSize;

    // one warp one row
    if (lid == 0)
      if (wid_g < num_res_old)
        s_v[wid] = d_res_table_old_[wid_g * C_NUM_VQ + u_matched];
      else
        s_v[wid] = UINT32_MAX;
    __syncwarp();

    if (wid_g < num_res_old && s_v[wid] != UINT32_MAX)
    {
      vtype v = s_v[wid];
      int row = wid_g;

      offtype v_nbr_off = offsets_[v] + lid;
      offtype v_nbr_off_end = offsets_[v + 1];
      while (v_nbr_off < v_nbr_off_end)
      {
        auto group = cooperative_groups::coalesced_threads();
        vtype v_nbr = nbrs_[v_nbr_off];
        if (compact_encodings_[u * C_COL_LEN + v_nbr / BLK_SIZE] & (1 << (v_nbr % BLK_SIZE)))
        // if (compact_encodings_[v_nbr * num_blocks + enc_pos / BLK_SIZE] & (1u << (enc_pos % BLK_SIZE)))
        {
          bool same_flag = false;
          for (int i = 0; i < C_NUM_VQ; ++i)
          {
            same_flag = same_flag || ((d_res_table_old_[row * C_NUM_VQ + i] == v_nbr));
          }
          if (!same_flag)
          {
            auto g = cooperative_groups::coalesced_threads();
            if (g.thread_rank() == 0)
              warp_pos[wid] = atomicAdd(num_res_new, g.size());
            g.sync();
            int pos = warp_pos[wid] + g.thread_rank();
            for (int i = 0; i < C_NUM_VQ; ++i)
              d_res_table_[pos * C_NUM_VQ + i] = d_res_table_old_[row * C_NUM_VQ + i];
            d_res_table_[pos * C_NUM_VQ + u] = v_nbr;
          }
        }
        group.sync();
        v_nbr_off += warpSize;
      }
      __syncwarp();
    }
    __syncthreads();

    block_iter_cnt++;
  }
}

void joinOneEdge(
    cpuGraph *hq, cpuGraph *hg,
    gpuGraph *dq, gpuGraph *dg,

    vtype u, vtype u_matched,
    vtype *d_res_table_old_, unsigned long long &num_res_old,
    vtype *d_res_table_, unsigned long long &num_res_new,

    uint32_t *d_compact_encodings_,

    encodingMeta *enc_meta)
{
  num_res_new = 0;

  unsigned long long *d_num_new_res;
  cudaMalloc((void **)&d_num_new_res, sizeof(unsigned long long));
  cudaMemset(d_num_new_res, 0, sizeof(unsigned long long));

  dim3 joe_block = BLOCK_DIM;
  int N = num_res_old * 32;
  dim3 joe_grid = std::min(GRID_DIM, calc_grid_dim(N, joe_block.x));
  joinOneEdgeKernel<<<joe_grid, joe_block>>>(
      dg->offsets_, dg->neighbors_,
      u, u_matched,
      d_res_table_old_, num_res_old,
      d_res_table_, d_num_new_res,
      d_compact_encodings_, enc_meta->num_blocks);
  cuchk(cudaDeviceSynchronize());

  cuchk(cudaMemcpy(&num_res_new, d_num_new_res, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

#ifndef NDEBUG
  std::cout << "After join one edge, num_res_new: " << num_res_new << std::endl;
#endif

  cudaFree(d_num_new_res);
}