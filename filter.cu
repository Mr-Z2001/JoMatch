#include "filter.cuh"
#include "cuda_helpers.h"
#include "structure.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

#include <set>
#include <stack>
#include <algorithm>

#include "order.h"
#include "memManag.cuh"

#define row_size ((NUM_VD - 1) / 32 + 1)

/**
 * TODO: need a lot of optimization.
 */

void dfs_cvc(
    vtype cur_v,
    vtype cur_v_father,
    int **f,
    cpuGraph *hq)
{
  f[cur_v][0] = 0;
  f[cur_v][1] = 1;

  for (offtype v_nbr_off = hq->offsets_[cur_v]; v_nbr_off < hq->offsets_[cur_v + 1]; ++v_nbr_off)
  {
    vtype v_nbr = hq->neighbors_[v_nbr_off];
    if (v_nbr == cur_v_father)
      continue;
    dfs_cvc(v_nbr, cur_v, f, hq);
    f[cur_v][0] += f[v_nbr][1];
    f[cur_v][1] += min(f[v_nbr][0], f[v_nbr][1]);
  }
}

void getVertexCover(
    cpuGraph *hq,
    // return
    vtype *vertex_cover_, numtype *num_vertex_covers)
{
  int **f = new int *[MAX_VQ];
  for (int i = 0; i < MAX_VQ; ++i)
    f[i] = new int[2];

  vtype root = 0;
  for (vtype u = 1; u < hq->num_v; ++u)
  {
    if (hq->outdeg_[u] > hq->outdeg_[root])
      root = u;
  }
  dfs_cvc(root, (uint32_t)-1, f, hq);
  *num_vertex_covers = min(f[root][0], f[root][1]);

  // top-down select vertex cover
  int cover_index = 0;
  std::stack<vtype> st;
  std::stack<vtype> st_father;
  std::stack<bool> stf_selected;
  st.push(root);
  st_father.push((uint32_t)-1);
  stf_selected.push(false);
  while (!st.empty())
  {
    vtype cur_v = st.top();
    st.pop();
    vtype cur_v_father = st_father.top();
    st_father.pop();
    bool father_selected = stf_selected.top();
    stf_selected.pop();

    if (f[cur_v][1] <= f[cur_v][0])
    {
      vertex_cover_[cover_index++] = cur_v;

      for (offtype v_nbr_off = hq->offsets_[cur_v]; v_nbr_off < hq->offsets_[cur_v + 1]; ++v_nbr_off)
      {
        vtype v_nbr = hq->neighbors_[v_nbr_off];
        if (v_nbr == cur_v_father)
          continue;
        st.push(v_nbr);
        st_father.push(cur_v);
        stf_selected.push(true);
      }
    }
    else
    {
      if (father_selected || cur_v_father == (uint32_t)-1)
        father_selected = false;
      else // father NOT selected
      {
        vertex_cover_[cover_index++] = cur_v;
        father_selected = true;
      }
      for (offtype v_nbr_off = hq->offsets_[cur_v]; v_nbr_off < hq->offsets_[cur_v + 1]; ++v_nbr_off)
      {
        vtype v_nbr = hq->neighbors_[v_nbr_off];
        if (v_nbr == cur_v_father)
          continue;
        st.push(v_nbr);
        st_father.push(cur_v);
        stf_selected.push(father_selected);
      }
    }
  }

  for (int i = 0; i < MAX_VQ; ++i)
    delete[] f[i];
  delete[] f;
}

// TODO: change it to an array of `T*`(i.e. T** for an 1-d array), save `new` and `delete` operations.
// make sure all type `T` support operator `=`
// template <typename T>
// inline void extendArray(T *&arr_old, numtype old_length, numtype addition_length = 1)
// {
//   T *arr_new = new T[old_length + addition_length];
//   for (int i = 0; i < old_length; i++)
//     arr_new[i] = arr_old[i];
//   if (arr_old != nullptr)
//     delete[] arr_old;
//   arr_old = arr_new;
// }

void clustering(
    cpuGraph *hq,
    std::vector<cpuCluster> &cpu_clusters_,
    encodingMeta *enc_meta)
{
  // get vertex cover
  vtype *vertex_cover_ = new vtype[hq->num_v];
  memset(vertex_cover_, 0, sizeof(vtype) * hq->num_v);
  numtype num_vertex_covers = 0;
  getVertexCover(hq, vertex_cover_, &num_vertex_covers);
#ifndef NDEBUG
  std::cout << "number of vertex covers: " << num_vertex_covers << std::endl;
  for (int i = 0; i < num_vertex_covers; i++)
    std::cout << vertex_cover_[i] << " ";
  std::cout << std::endl;
#endif

  // construct clusters first layer
  numtype &num_clusters = enc_meta->num_clusters; // use reference for simplicity.
  num_clusters = num_vertex_covers;

  // cpu_clusters_ = new cpuCluster[MAX_CLUSTERS];

  cpu_clusters_.assign(MAX_CLUSTERS, cpuCluster());
  for (int i = 0; i < num_clusters; ++i)
  {
    // aliases for simplicity.
    vtype core_u = vertex_cover_[i];
    cpuCluster &cluster = cpu_clusters_[i];

    cluster.num_query_us = hq->outdeg_[core_u] + 1;
    cluster.query_us_ = new vtype[cluster.num_query_us];
    cluster.query_us_[0] = core_u;
    memcpy(&cluster.query_us_[1],
           hq->neighbors_ + hq->offsets_[core_u],
           sizeof(vtype) * (cluster.num_query_us - 1));
  }

  // join clusters
  vtype connection_vertex;
  numtype num_new_clusters = 0;
  numtype num_actual_new_clusters = 0; // `new_cluster - combined_cluster`, assigned arbitrarily, larger than 1.
  int layer_index = 0;

  enc_meta->num_clusters_per_layer_[0] = num_clusters;

  int i = 0;

  do // while(num_actual_new_clusters > 1)
  {
    num_new_clusters = 0;
    num_actual_new_clusters = 0;
    // join clusters in layer `k-1` to form new clusters in layer `k`
    for (; i < num_clusters; ++i) // get cluster left
    {
      cpuCluster &cluster_i = cpu_clusters_[i];
      if (enc_meta->is_a_valid_cluster_[i] == false)
        continue;
      vtype core_i = cluster_i.query_us_[0];

      for (int j = i + 1; j < num_clusters; ++j) // get cluster right
      {
        cpuCluster &cluster_j = cpu_clusters_[j];
        if (enc_meta->is_a_valid_cluster_[j] == false)
          continue;
        vtype core_j = cluster_j.query_us_[0];

        // join cluster_i and cluster_j
        for (uint32_t i_ptr = 1; i_ptr < cluster_i.num_query_us; ++i_ptr) // iterate on cluster_i query vertices
        {
          vtype u_i = cluster_i.query_us_[i_ptr];

          for (uint32_t j_ptr = 0; j_ptr < cluster_j.num_query_us; ++j_ptr) // iterate on cluster_j query vertices
          {
            vtype u_j = cluster_j.query_us_[j_ptr];
            if (u_i != u_j)
              continue;

            connection_vertex = u_i;
            num_new_clusters++;
            int new_cluster_index = num_clusters + num_new_clusters - 1;

            enc_meta->final_cluster_id = new_cluster_index;

            enc_meta->is_a_valid_cluster_[new_cluster_index] = true;
            // cpu_clusters_.emplace_back(cpuCluster());
            cpuCluster &new_cluster = cpu_clusters_[new_cluster_index];

            if (i_ptr && j_ptr) // both are not core vertex
            {
              new_cluster.num_query_us = 3;
              if (core_i == core_j)
                new_cluster.num_query_us--;
              new_cluster.query_us_ = new vtype[new_cluster.num_query_us];
              new_cluster.query_us_[0] = connection_vertex;
              new_cluster.query_us_[1] = core_i; // core of i-th
              if (core_i != core_j)
                new_cluster.query_us_[2] = core_j; // core of j-th
            }
            else if (!j_ptr) // u_j is the core vertex
            {
              new_cluster.num_query_us = 2;
              new_cluster.query_us_ = new vtype[2];
              new_cluster.query_us_[0] = connection_vertex;
              new_cluster.query_us_[1] = core_i; // core of i-th
            }
            else
            {
              std::cerr << "unexpected case" << std::endl;
              std::cerr << "i: " << i << " j: " << j << " i_ptr: " << i_ptr << " j_ptr: " << j_ptr << std::endl;
              std::cerr << "u_i: " << u_i << " u_j: " << u_j << std::endl;
              exit(1);
            }

            enc_meta->merged_cluster_left_.push_back(i);
            enc_meta->merged_cluster_right_.push_back(j);

            enc_meta->merge_count++;
          }
        }
      }
    }

    // combine

    layer_index++; // the layer of source clusters to combine. So, 'this layer' indeed.
    num_actual_new_clusters = num_new_clusters;
    enc_meta->combine_checkpoints_[layer_index] = enc_meta->num_clusters + num_new_clusters;

    uint32_t &comb_cnt = enc_meta->combine_cnt;

    for (int new_cluster_out_ptr = num_clusters; new_cluster_out_ptr < num_clusters + num_new_clusters; ++new_cluster_out_ptr) // iterate on new clusters out
    {
      if (enc_meta->is_a_valid_cluster_[new_cluster_out_ptr] == false)
        continue;
      cpuCluster &cluster_out = cpu_clusters_[new_cluster_out_ptr];
      vtype core_u_out = cluster_out.query_us_[0];

      bool duplicate = false; // is the `core_u_out` the unique core in this layer?

      // scan for the inner, combine all clusters that have the same core vertex.
      std::set<int> to_combine_cluster_index_set;
      to_combine_cluster_index_set.insert(new_cluster_out_ptr);
      std::set<vtype> core_nbrs_set;
      for (int i = 1; i < cluster_out.num_query_us; ++i)
        core_nbrs_set.insert(cluster_out.query_us_[i]);

      for (int new_cluster_in_ptr = new_cluster_out_ptr + 1; new_cluster_in_ptr < num_clusters + num_new_clusters; ++new_cluster_in_ptr) // iterate on new clusters in
      {
        if (enc_meta->is_a_valid_cluster_[new_cluster_in_ptr] == false)
          continue;

        cpuCluster &cluster_in = cpu_clusters_[new_cluster_in_ptr];
        vtype core_u_in = cluster_in.query_us_[0];

        if (core_u_in == core_u_out)
        {
          duplicate = true;
          to_combine_cluster_index_set.insert(new_cluster_in_ptr);
          for (int i = 1; i < cluster_in.num_query_us; ++i)
            core_nbrs_set.insert(cluster_in.query_us_[i]);
        }
      }

      if (!duplicate)
        continue;

      bool if_create_new = true;
      auto set_iterator = to_combine_cluster_index_set.begin();
      while (if_create_new &&
             (set_iterator != to_combine_cluster_index_set.end()))
      {
        int num_query_us = cpu_clusters_[*set_iterator].num_query_us;
        if (num_query_us == core_nbrs_set.size() + 1)
          if_create_new = false;
        set_iterator++;
      }
      int largest_cluster_index;

      if (!if_create_new) // There exists a cluster contains all.
      {
        set_iterator--;
        largest_cluster_index = *set_iterator;
        to_combine_cluster_index_set.erase(largest_cluster_index);
        enc_meta->combine_type_.push_back(1);
      }
      else // create new cluster. no cluster contains all others.
      {
        num_new_clusters++;
        num_actual_new_clusters++;
        enc_meta->combine_type_.push_back(0);

        int new_index = num_clusters + num_new_clusters - 1;

        enc_meta->is_a_valid_cluster_[new_index] = true;
        // cpu_clusters_.emplace_back(cpuCluster());
        cpuCluster &new_cluster = cpu_clusters_[new_index];
        new_cluster.num_query_us = core_nbrs_set.size() + 1;
        new_cluster.query_us_ = new vtype[new_cluster.num_query_us];
        new_cluster.query_us_[0] = core_u_out;
        int i = 1;
        for (auto core_nbr : core_nbrs_set)
          new_cluster.query_us_[i++] = core_nbr;

        largest_cluster_index = new_index;
      }

      enc_meta->final_cluster_id = largest_cluster_index;

      set_iterator = to_combine_cluster_index_set.begin();
      enc_meta->combine_cluster_out_.push_back(largest_cluster_index);
      enc_meta->combine_clusters_other_.emplace_back(to_combine_cluster_index_set);

      num_actual_new_clusters -= to_combine_cluster_index_set.size();
      while (set_iterator != to_combine_cluster_index_set.end())
      {
        enc_meta->is_a_valid_cluster_[*set_iterator] = false;
        set_iterator++;
      }
      comb_cnt++;
    }

    num_clusters += num_new_clusters;
    enc_meta->num_clusters_per_layer_[layer_index] = num_new_clusters;

    // std::cout << "num_actual_new_clusters: " << num_actual_new_clusters << std::endl;
  } while (num_actual_new_clusters > 1);

  if (enc_meta->combine_checkpoints_[layer_index] == -1)
  {
    std::cout << "triggered: enc_meta->combine_checkpoints_[layer_index] == -1" << std::endl;
    enc_meta->combine_checkpoints_[layer_index] = num_clusters;
  }

  for (int l_index = 1; l_index <= layer_index; ++l_index)
  {
    enc_meta->layer_offsets_[l_index] =
        enc_meta->layer_offsets_[l_index - 1] + enc_meta->num_clusters_per_layer_[l_index - 1];
  }
  enc_meta->layer_offsets_[layer_index + 1] = num_clusters;

#ifndef NDEBUG
  std::cout << "see checkpoints: " << std::endl;
  for (int i = 0; i < layer_index + 1; ++i)
  {
    std::cout << "Layer " << i << " checkpoint: " << enc_meta->combine_checkpoints_[i] << std::endl;
  }

  std::cout << "see layer index: " << std::endl;
  for (int i = 0; i <= layer_index + 1; ++i)
  {
    std::cout << "Layer " << i << " offset: " << enc_meta->layer_offsets_[i] << std::endl;
  }

#endif

  // construct meta
  enc_meta->init(cpu_clusters_);
  enc_meta->num_layers = layer_index + 1;

  delete[] vertex_cover_;
}

__global__ void
oneRoundFilterBidirectionKernel(
    // structure info
    vltype *query_vLabels_, degtype *query_out_degrees_,
    offtype *d_offsets_, vtype *d_nbrs_, vltype *d_v_labels_, degtype *d_v_degrees_,

    vtype *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_,
    vtype *d_v_candidate_us_, numtype *d_num_v_candidate_us_,

    numtype *d_query_nlc_table_)
{
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;
  uint32_t idx = tid + bid * blockDim.x;
  uint32_t wid = tid >> 5;
  uint32_t lid = tid & 31;
  uint32_t wid_g = idx >> 5;

  __shared__ vltype s_q_vlabels[MAX_VQ];
  __shared__ degtype s_q_degs[MAX_VQ];
  __shared__ uint32_t s_bitmap[MAX_VQ][BLOCK_DIM / 32];
  __shared__ uint32_t s_bitmap_reverse[BLOCK_DIM];
  __shared__ uint32_t s_d_nlc_table[WARP_PER_BLOCK][MAX_VLQ];
  __shared__ numtype s_q_nlc_table[MAX_VQ][MAX_VLQ];
  __shared__ uint32_t warp_pos[MAX_VQ][WARP_PER_BLOCK];
  __shared__ vltype s_d_vlabels_part[WARP_PER_BLOCK][WARP_SIZE];

  __shared__ degtype s_d_v_degs[WARP_PER_BLOCK][WARP_SIZE];

  // for optimization
  __shared__ int grid_size;
  __shared__ int block_iter_cnt[WARP_PER_BLOCK];
  __shared__ vtype v[WARP_PER_BLOCK];
  __shared__ vtype v_end[WARP_PER_BLOCK];
  __shared__ offtype v_nbr_off[WARP_PER_BLOCK];
  __shared__ offtype v_nbr_off_end[WARP_PER_BLOCK];

  if (lid == 0)
    block_iter_cnt[wid] = 0;
  __syncwarp();

  if (tid == 0)
    grid_size = gridDim.x * blockDim.x;
  __syncthreads();

  if (tid < C_NUM_VQ)
  {
    s_q_degs[tid] = query_out_degrees_[tid];
    s_q_vlabels[tid] = query_vLabels_[tid];
  }
  if (tid < C_NUM_VQ * C_NUM_VLQ)
    s_q_nlc_table[tid / C_NUM_VLQ][tid % C_NUM_VLQ] = d_query_nlc_table_[tid];
  __syncthreads();

  while (block_iter_cnt[wid] * grid_size < C_NUM_VD)
  {
    idx = tid + bid * blockDim.x + block_iter_cnt[wid] * grid_size;
    wid_g = idx / warpSize;

    s_bitmap[lid][wid] = 0;
    s_bitmap_reverse[tid] = 0;
    if (idx < C_NUM_VD)
    {
      s_d_v_degs[wid][lid] = d_v_degrees_[idx];
      s_d_vlabels_part[wid][lid] = d_v_labels_[idx];
    }
    __syncthreads();

    if (lid == 0)
    {
      v[wid] = wid_g * warpSize;
      v_end[wid] = min(v[wid] + warpSize, C_NUM_VD); // exclusive
    }
    __syncwarp();

    while (v[wid] < v_end[wid])
    {
      if (lid < C_NUM_VLQ)
        s_d_nlc_table[wid][lid] = 0;
      __syncwarp();

      if (s_d_vlabels_part[wid][v[wid] & 31] >= C_NUM_VLQ)
      {
        if (lid == 0)
          ++v[wid];
        __syncwarp();
        continue;
      }

      // build data nlc table
      if (lid == 0)
      {
        v_nbr_off[wid] = d_offsets_[v[wid]];
        v_nbr_off_end[wid] = d_offsets_[v[wid] + 1];
      }
      __syncwarp();
      while (v_nbr_off[wid] + lid < v_nbr_off_end[wid])
      {
        auto group = cooperative_groups::coalesced_threads();
        vtype v_nbr = d_nbrs_[v_nbr_off[wid] + lid];
        vltype v_nbr_label = d_v_labels_[v_nbr];
        if (v_nbr_label < C_NUM_VLQ)
          atomicAdd(&s_d_nlc_table[wid][v_nbr_label], 1); // `wid` is `v`
        // group.sync();
        if (group.thread_rank() == 0)
          v_nbr_off[wid] += warpSize;
        group.sync();
        // v_nbr_off += warpSize;
      }
      __syncwarp();

      for (vtype u = 0; u < C_NUM_VQ; ++u)
      {
        // all lanes take the same branch
        if (s_q_degs[u] <= s_d_v_degs[wid][v[wid] & 31] &&
            s_q_vlabels[u] == s_d_vlabels_part[wid][v[wid] & 31])
        {
          // lid == vLabel
          if (lid < C_NUM_VLQ)
          {
            auto group = cooperative_groups::coalesced_threads();
            uint32_t mask = group.all(s_d_nlc_table[wid][lid] >= s_q_nlc_table[u][lid]);
            //  d_query_nlc_table_[u * C_NUM_VLQ + lid];
            if (mask && group.thread_rank() == 0)
            {
              s_bitmap[u][wid] |= (1u << (v[wid] & 31));
              s_bitmap_reverse[v[wid] % BLOCK_DIM] |= (1u << u);
              // atomicOr(&s_bitmap[u][wid], (1u << (v[wid] & 31)));
              // atomicOr(&s_bitmap_reverse[v[wid] % BLOCK_DIM], (1u << (u & 31)));
            }
            group.sync();
          }
        }
        __syncwarp();
      }
      if (lid == 0)
        ++v[wid];
      __syncwarp();
    }
    __syncwarp();

    // read from shared, write to global memory.
    for (vtype u = 0; u < C_NUM_VQ; ++u)
    {
      // lid: v%warpSize
      if (s_bitmap[u][wid] & (1u << lid)) // if v is a candidate vertex for u
      {
        auto group = cooperative_groups::coalesced_threads();
        int rank = group.thread_rank();
        if (rank == 0)
          warp_pos[u][wid] = atomicAdd(&d_num_u_candidate_vs_[u], group.size());
        group.sync();
        // int my_pos = warp_pos[u][wid] + rank;
        d_u_candidate_vs_[u * C_MAX_L_FREQ + warp_pos[u][wid] + rank] = wid_g * warpSize + lid;
      }
      __syncwarp();
    }
    __syncthreads();
    if (idx < C_NUM_VD)
      d_num_v_candidate_us_[idx] = __popc(s_bitmap_reverse[tid]);
    __syncthreads();
    // __syncwarp();

    if (idx < C_NUM_VD)
      d_v_candidate_us_[idx] = s_bitmap_reverse[tid];
    __syncthreads();
    // __syncwarp();

    if (lid == 0)
      block_iter_cnt[wid]++;
    __syncthreads();
  }
}

void oneRoundFilterBidirection(
    cpuGraph *hq, cpuGraph *hg,
    gpuGraph *dq, gpuGraph *dg,

    vtype *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_,
    vtype *d_v_candidate_us_, numtype *d_num_v_candidate_us_)
{
  numtype *h_query_nlc = nullptr;
  h_query_nlc = new numtype[NUM_VQ * NUM_VLQ];
  memset(h_query_nlc, 0, sizeof(numtype) * NUM_VQ * NUM_VLQ);

  numtype *d_query_nlc = nullptr;
  cuchk(cudaMalloc((void **)&d_query_nlc, sizeof(numtype) * NUM_VQ * NUM_VLQ));

  for (vtype u = 0; u < NUM_VQ; ++u)
  {
    uint32_t NLC_offset = u * NUM_VLQ;
    for (offtype off = hq->offsets_[u]; off < hq->offsets_[u + 1]; ++off)
    {
      vtype nbr = hq->neighbors_[off];
      vltype vlabel = hq->vLabels_[nbr];
      h_query_nlc[NLC_offset + vlabel]++;
    }
  }

  cuchk(cudaMemcpy(d_query_nlc, h_query_nlc, sizeof(numtype) * NUM_VQ * NUM_VLQ, cudaMemcpyHostToDevice));

  dim3 orfb_block = BLOCK_DIM;
  int N = NUM_VD;
  dim3 orfb_grid = std::min(GRID_DIM, calc_grid_dim(N, orfb_block.x));
#ifndef NDEBUG
  std::cout << "orfb_grid: " << orfb_grid.x << std::endl;
#endif
  // max threads: NUM_VD
  // oneRoundFilterBidirectionKernel<<<orfb_grid, orfb_block>>>(
  oneRoundFilterBidirectionKernel<<<GRID_DIM, BLOCK_DIM>>>(
      dq->vLabels_, dq->degree_,
      dg->offsets_, dg->neighbors_, dg->vLabels_, dg->degree_,
      d_u_candidate_vs_, d_num_u_candidate_vs_,
      d_v_candidate_us_, d_num_v_candidate_us_,
      d_query_nlc);
  cuchk(cudaDeviceSynchronize());

  cuchk(cudaFree(d_query_nlc));
  delete[] h_query_nlc;
}

__global__ void
encodeKernel(
    // graph info
    offtype *d_offsets_, vtype *d_nbrs_,

    // candidate vertices
    // vtype core_u, uint32_t cluster_index,
    vtype *d_u_candidate_vs_, numtype num_u_candidate_vs,
    vtype *d_v_candidate_us_,

    // encoding info
    uint32_t *encodings_,
    numtype enc_num_query_us_,
    vtype *enc_query_us_compact_, uint32_t enc_pos)
{
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;
  uint32_t idx = tid + bid * blockDim.x;
  uint32_t wid = tid >> 5;
  uint32_t lid = tid & 31;
  uint32_t wid_g = idx >> 5;

  __shared__ vtype s_core_v[WARP_PER_BLOCK];

  // for optimization
  __shared__ int block_iter_cnt[WARP_PER_BLOCK];
  __shared__ int grid_size;
  __shared__ int num_warps;
  __shared__ offtype nbr_off[WARP_PER_BLOCK];
  __shared__ offtype nbr_off_end[WARP_PER_BLOCK];
  __shared__ vtype s_enc_query_us_compact[MAX_VQ];

  if (lid == 0)
    block_iter_cnt[wid] = 0;
  __syncwarp();

  if (tid == 0)
  {
    grid_size = gridDim.x * blockDim.x;
    num_warps = grid_size / warpSize;
  }
  if (tid < enc_num_query_us_)
    s_enc_query_us_compact[tid] = enc_query_us_compact_[enc_pos + tid];
  __syncthreads();

  while (block_iter_cnt[wid] * num_warps < num_u_candidate_vs) // wid_g < num_can
  {
    idx = tid + bid * blockDim.x + block_iter_cnt[wid] * grid_size;
    wid_g = idx / warpSize;

    if (wid_g < num_u_candidate_vs) // one warp one cluster.
    {
      if (lid == 0)
        // s_core_v[wid] = d_u_candidate_vs_[core_u * C_MAX_L_FREQ + wid_g];
        s_core_v[wid] = d_u_candidate_vs_[wid_g];
    }
    __syncwarp();

    if (s_core_v[wid] >= C_NUM_VD)
    {
      if (lid == 0)
        block_iter_cnt[wid]++;
      __syncwarp();
      continue;
    }

    // encode core vertex

    if (lid == 0 && s_core_v[wid] < C_NUM_VD)
      encodings_[s_core_v[wid] * C_NUM_BLOCKS + enc_pos / BLK_SIZE] |= (1u << (enc_pos % BLK_SIZE));
    __syncwarp();

    // encode core_v's neighbors
    if (lid == 0)
    {
      nbr_off[wid] = d_offsets_[s_core_v[wid]];
      nbr_off_end[wid] = d_offsets_[s_core_v[wid] + 1];
    }
    __syncwarp();

    while (nbr_off[wid] + lid < nbr_off_end[wid])
    {
      auto group = cooperative_groups::coalesced_threads();
      vtype v_nbr = d_nbrs_[nbr_off[wid] + lid];
      uint32_t code = d_v_candidate_us_[v_nbr];
      for (int i = 1; i < enc_num_query_us_; ++i)
      {
        vtype tobe_map_u = s_enc_query_us_compact[i];
        if (code & (1u << tobe_map_u))
          atomicOr(&encodings_[v_nbr * C_NUM_BLOCKS + (enc_pos + i) / BLK_SIZE], 1u << ((enc_pos + i) % BLK_SIZE));
        group.sync();
      }
      if (group.thread_rank() == 0)
        nbr_off[wid] += warpSize;
      group.sync();
    }

    __syncwarp();
    if (lid == 0)
      block_iter_cnt[wid]++;
    __syncwarp();
  }
}

__global__ void
mergeKernel(
    // positions
    int enc_pos,
    int left_pos, int right_pos,
    int left_core_pos, int right_core_pos,

    // graph info
    offtype *d_offsets_, vtype *d_nbrs_,

    // candidate vertices
    vtype *d_u_candidate_vs_, numtype num_u_candidate_vs,

    // encoding info
    uint32_t *encodings_)
{
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;
  uint32_t idx = tid + bid * blockDim.x;
  uint32_t wid = tid >> 5;
  uint32_t lid = tid & 31;
  uint32_t wid_g = idx >> 5;

  __shared__ vtype s_core_v[WARP_PER_BLOCK]; // one warp one cluster.

  int block_iter_cnt = 0;
  int grid_size = gridDim.x * blockDim.x;

  while (block_iter_cnt * (grid_size / warpSize) < num_u_candidate_vs)
  {
    idx = tid + bid * blockDim.x + block_iter_cnt * grid_size;
    wid_g = idx / warpSize;

    if (lid == 0)
      if (wid_g < num_u_candidate_vs)
        s_core_v[wid] = d_u_candidate_vs_[wid_g];
      else
        s_core_v[wid] = UINT32_MAX;
    __syncwarp();

    // encode core vertex
    vtype core_v = s_core_v[wid];

    if (lid == 0 && core_v != UINT32_MAX)
    {
      if ((encodings_[core_v * C_NUM_BLOCKS + left_pos / BLK_SIZE] & (1u << (left_pos % BLK_SIZE))) &&
          (encodings_[core_v * C_NUM_BLOCKS + right_pos / BLK_SIZE] & (1u << (right_pos % BLK_SIZE))))
      {
        encodings_[core_v * C_NUM_BLOCKS + enc_pos / BLK_SIZE] |= (1u << (enc_pos % BLK_SIZE));
      }
      else
      {
        d_u_candidate_vs_[wid_g] = UINT32_MAX;
      }
    }
    __syncwarp();

    // TODO: neighbor encode should refer to core-vertex positions of old clusters.
    if (core_v != UINT32_MAX)
    {
      offtype nbr_off = d_offsets_[core_v] + lid;
      while (nbr_off < d_offsets_[core_v + 1])
      {
        auto group = cooperative_groups::coalesced_threads();
        vtype v_nbr = d_nbrs_[nbr_off];

        // first leaf: left core vertex
        if (encodings_[v_nbr * C_NUM_BLOCKS + left_core_pos / BLK_SIZE] & (1u << (left_core_pos % BLK_SIZE)))
          atomicOr(&encodings_[v_nbr * C_NUM_BLOCKS + (enc_pos + 1) / BLK_SIZE], 1u << ((enc_pos + 1) % BLK_SIZE));

        // second leaf: right core vertex (if exists)
        if (right_core_pos != right_pos)
        {
          if (encodings_[v_nbr * C_NUM_BLOCKS + right_core_pos / BLK_SIZE] & (1u << (right_core_pos % BLK_SIZE)))
            atomicOr(&encodings_[v_nbr * C_NUM_BLOCKS + (enc_pos + 2) / BLK_SIZE], 1u << ((enc_pos + 2) % BLK_SIZE));
        }
        group.sync();
        nbr_off += warpSize;
      }
    }
    __syncthreads();

    block_iter_cnt++;
  }
}

__global__ void
combineMultipleClustersKernel(
    offtype *d_offsets_, vtype *d_nbrs_,
    int combine_type,
    int big_cluster, int *small_clusters_arr_, int num_small_clusters,
    uint32_t *d_encodings_,
    numtype *num_query_us_,
    vtype *query_us_compact_, offtype *cluster_offsets_,
    vtype *d_u_candidate_vs_, numtype num_u_candidate_vs)
{
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;
  uint32_t idx = tid + bid * blockDim.x;
  uint32_t wid = tid >> 5;
  uint32_t lid = tid & 31;
  uint32_t wid_g = idx >> 5;

  // __shared__ int s_combine_type; // 0: new cluster, 1: old cluster.
  __shared__ int s_big_pos;
  __shared__ int s_small_pos[32]; // I guess 32 is enough. Need test it for larger graphs.
  __shared__ int s_small_clusters_[32];
  __shared__ vtype s_warp_v[WARP_PER_BLOCK];

  if (tid == 0)
    s_big_pos = cluster_offsets_[big_cluster];
  if (tid < num_small_clusters)
  {
    s_small_clusters_[tid] = small_clusters_arr_[tid];
    // int small_cluster = small_clusters_arr_[tid];
    int pos = cluster_offsets_[s_small_clusters_[tid]];
    s_small_pos[tid] = pos;
  }
  __syncthreads();

  int block_iter_cnt = 0;
  int grid_size = gridDim.x * blockDim.x;

  while (block_iter_cnt * (grid_size / warpSize) < num_u_candidate_vs)
  {
    idx = tid + bid * blockDim.x + block_iter_cnt * grid_size;
    wid_g = idx / warpSize;

    if (lid == 0)
    {
      s_warp_v[wid] = UINT32_MAX;
      if (wid_g < num_u_candidate_vs)
        s_warp_v[wid] = d_u_candidate_vs_[wid_g];
    }
    __syncwarp();
    vtype v = s_warp_v[wid];

    if (v != UINT32_MAX)
    {
      // combine core_v
      if (lid < num_small_clusters) // one lid, one small cluster.core_u position.
      {
        auto group = cooperative_groups::coalesced_threads();
        int small_pos = s_small_pos[lid];
        int big_pos = s_big_pos;
        uint32_t enc = d_encodings_[v * C_NUM_BLOCKS + small_pos / BLK_SIZE] & (1u << (small_pos % BLK_SIZE));

        uint32_t mask = group.all(enc);
        if (group.thread_rank() == 0)
        {
          if (mask)
          {
            if (combine_type == 1) // old cluster.
              d_encodings_[v * C_NUM_BLOCKS + big_pos / BLK_SIZE] &= FULL_MASK;
            else // new cluster
              d_encodings_[v * C_NUM_BLOCKS + big_pos / BLK_SIZE] |= (1u << (big_pos % BLK_SIZE));
          }
          else
          {
            d_encodings_[v * C_NUM_BLOCKS + big_pos / BLK_SIZE] &= (~(1u << (big_pos % BLK_SIZE)));
            d_u_candidate_vs_[wid_g] = UINT32_MAX;
          }
        }
      }
      __syncwarp();

      // combine core_v_nbrs
      // one lane: one v_nbr, intersect by for loop.

      offtype v_nbr_off = d_offsets_[v] + lid;
      offtype v_nbr_off_end = d_offsets_[v + 1];
      while (v_nbr_off < v_nbr_off_end)
      {
        auto group = cooperative_groups::coalesced_threads();
        vtype v_nbr = d_nbrs_[v_nbr_off];
        for (int i = 1; i < num_query_us_[big_cluster]; ++i)
        {
          int target_pos = s_big_pos + i;
          vtype target_u = query_us_compact_[target_pos];
          int final_value = 1;
          for (int small_cluster_index = 0; small_cluster_index < num_small_clusters; ++small_cluster_index)
          {
            int small_pos = s_small_pos[small_cluster_index] + 1;
            int small_pos_end = s_small_pos[small_cluster_index] + num_query_us_[s_small_clusters_[small_cluster_index]];
            while (small_pos < small_pos_end)
            {
              if (target_u == query_us_compact_[small_pos])
              {
                final_value =
                    final_value &&
                    (d_encodings_[v_nbr * C_NUM_BLOCKS + small_pos / BLK_SIZE] & (1u << (small_pos % BLK_SIZE)));
                break;
              }
              small_pos++;
            }
          }
          group.sync();
          if (final_value)
            final_value = 1;
          // no warp divergence here, all the threads take the same path.
          if (combine_type == 1) // old
            atomicAnd(&d_encodings_[v_nbr * C_NUM_BLOCKS + target_pos / BLK_SIZE],
                      ~((!final_value) << (target_pos % BLK_SIZE)));
          else // new
            atomicOr(&d_encodings_[v_nbr * C_NUM_BLOCKS + target_pos / BLK_SIZE],
                     final_value << (target_pos % BLK_SIZE));
        }

        v_nbr_off += warpSize;
      }
    }
    __syncthreads();

    block_iter_cnt++;
  }
}

void encode(
    gpuGraph *dg,
    std::vector<cpuCluster> &cpu_clusters_,
    uint32_t *h_encodings_, uint32_t *d_encodings_, encodingMeta *enc_meta,
    numtype *h_num_u_candidate_vs_,
    vtype *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_,
    vtype *d_v_candidate_us_, numtype *d_num_v_candidate_us_)
{
  // TODO: organize as a gpu_encoding_meta class.
  // device encoding meta
  uint32_t *enc_meta_buffer_;

  numtype *enc_d_num_query_us_;
  vtype *enc_d_query_us_compact_;
  offtype *enc_d_cluster_offsets_;

  uint32_t tot_mem_cnt = 0;
  tot_mem_cnt += enc_meta->num_clusters;
  tot_mem_cnt += enc_meta->num_total_us;
  tot_mem_cnt += enc_meta->num_clusters + 1;
  tot_mem_cnt *= sizeof(uint32_t);

  cuchk(cudaMalloc((void **)&enc_meta_buffer_, tot_mem_cnt));
  uint32_t *current_ptr = enc_meta_buffer_;

  enc_d_num_query_us_ = current_ptr;
  cuchk(cudaMemcpy(enc_d_num_query_us_, enc_meta->num_query_us_, sizeof(numtype) * enc_meta->num_clusters, cudaMemcpyHostToDevice));
  current_ptr += enc_meta->num_clusters;

  enc_d_query_us_compact_ = current_ptr;
  cuchk(cudaMemcpy(enc_d_query_us_compact_, enc_meta->query_us_compact_, sizeof(vtype) * enc_meta->num_total_us, cudaMemcpyHostToDevice));
  current_ptr += enc_meta->num_total_us;

  enc_d_cluster_offsets_ = current_ptr;
  cuchk(cudaMemcpy(enc_d_cluster_offsets_, enc_meta->cluster_offsets_, sizeof(offtype) * (enc_meta->num_clusters + 1), cudaMemcpyHostToDevice));

  // encode the first layer.
  // std::cout << "first layer num clusters: " << enc_meta->num_clusters_per_layer_[0] << std::endl;
  for (int cluster_index = 0; cluster_index < enc_meta->num_clusters_per_layer_[0]; ++cluster_index)
  {
    vtype core_u = cpu_clusters_[cluster_index].query_us_[0];

    dim3 enc_block = BLOCK_DIM;
    int N = h_num_u_candidate_vs_[core_u] * 32;
    dim3 enc_grid = std::min(GRID_DIM, calc_grid_dim(N, enc_block.x));

#ifndef NDEBUG
    std::cout << "enc_grid: " << enc_grid.x << std::endl;
#endif

    // TODO: change d_num_u_candidate_vs_ into num_u_candidate_vs_[core_u]
    encodeKernel<<<enc_grid, enc_block>>>(
        // encodeKernel<<<GRID_DIM, BLOCK_DIM>>>(
        dg->offsets_, dg->neighbors_,
        // core_u, cluster_index,
        // d_u_candidate_vs_, h_num_u_candidate_vs_[core_u],
        d_u_candidate_vs_ + core_u * MAX_L_FREQ, h_num_u_candidate_vs_[core_u],
        d_v_candidate_us_,

        d_encodings_,
        enc_meta->num_query_us_[cluster_index],
        enc_d_query_us_compact_, enc_meta->cluster_offsets_[cluster_index]);
    cuchk(cudaDeviceSynchronize());
  }

#ifndef NDEBUG
  std::cout << "first layer encoding done" << std::endl;
#endif

  uint32_t layer_index = 1;
  uint32_t cluster_sum = enc_meta->num_clusters_per_layer_[0];
  uint32_t combine_ptr = 0;
  uint32_t merge_ptr = 0;

  int *d_small_clusters_;
  cuchk(cudaMalloc((void **)&d_small_clusters_, sizeof(int) * enc_meta->num_clusters));

  /**
   * from now on, each cluster is
   * (1) obtained by merging
   * (2) obtained by combining and that's a new cluster.
   * if it is obtained by merging,
   *    no need to check if it is still valid.
   *    do a two-way intersection.
   * if it is obtained by combining,
   *    if it is not a new cluster, in-place intersection
   *    if it is a new cluster, combine.
   */

  bool layer_changed = false;

  for (int cluster_index = enc_meta->num_clusters_per_layer_[0]; cluster_index < enc_meta->num_clusters; ++cluster_index)
  {
    layer_changed = false;
    if (cluster_index == cluster_sum + enc_meta->num_clusters_per_layer_[layer_index])
    {
      cluster_sum += enc_meta->num_clusters_per_layer_[layer_index];
      layer_index++;
      layer_changed = true;
    }

    if (cluster_index == enc_meta->combine_checkpoints_[layer_index])
    {
#ifndef NDEBUG
      std::cout << "combining layer: " << layer_index << std::endl;
      std::cout << "combining checkpoint: " << enc_meta->combine_checkpoints_[layer_index] << std::endl;
#endif
      // combine
      if (enc_meta->combine_cnt != 0)
      {
        uint32_t next_layer_start_pos = enc_meta->layer_offsets_[layer_index + 1];
        while (enc_meta->combine_cluster_out_[combine_ptr] < next_layer_start_pos && combine_ptr < enc_meta->combine_cnt)
        {
          int big_cluster = enc_meta->combine_cluster_out_[combine_ptr];
          std::set<int> small_clusters = enc_meta->combine_clusters_other_[combine_ptr];

          std::vector<int> small_clusters_vec = std::vector<int>(small_clusters.begin(), small_clusters.end());
          int *small_clusters_arr = small_clusters_vec.data();
          int num_small_clusters = small_clusters.size();

          if (num_small_clusters > 32)
          {
            std::cerr << "encoding: combing: num small cluster: " << num_small_clusters << std::endl;
            std::cerr << "too many small clusters" << std::endl;
            exit(1);
          }

          cuchk(cudaMemcpy(d_small_clusters_, small_clusters_arr, sizeof(int) * num_small_clusters, cudaMemcpyHostToDevice));

          vtype core_u = cpu_clusters_[big_cluster].query_us_[0];

          dim3 comb_block = BLOCK_DIM;
          int N = h_num_u_candidate_vs_[core_u] * 32;
          dim3 comb_grid = std::min(GRID_DIM, calc_grid_dim(N, comb_block.x));

#ifndef NDEBUG
          std::cout << "comb_grid: " << comb_grid.x << std::endl;
#endif

          combineMultipleClustersKernel<<<comb_grid, comb_block>>>(
              dg->offsets_, dg->neighbors_,
              enc_meta->combine_type_[combine_ptr],
              big_cluster, d_small_clusters_, num_small_clusters,
              d_encodings_,
              enc_d_num_query_us_,
              enc_d_query_us_compact_, enc_d_cluster_offsets_,
              d_u_candidate_vs_ + core_u * MAX_L_FREQ, h_num_u_candidate_vs_[core_u]);
          cuchk(cudaDeviceSynchronize());
          ++combine_ptr;
        }
      }

      // goto next layer

      // TODO: fix it !!!!!!!!!
      if (cluster_index != enc_meta->layer_offsets_[layer_index + 1])
      {
        cluster_index = enc_meta->layer_offsets_[layer_index + 1] - 1;
        continue;
      }
    }

    vtype core_u = cpu_clusters_[cluster_index].query_us_[0];

    // merge clusters.
    // int merge_index = cluster_index - enc_meta->num_clusters_per_layer_[0];
    int left = enc_meta->merged_cluster_left_[merge_ptr];
    int left_position = enc_meta->cluster_offsets_[left];
    while (enc_meta->query_us_compact_[left_position] != core_u)
      left_position++;
    int right = enc_meta->merged_cluster_right_[merge_ptr];
    int right_position = enc_meta->cluster_offsets_[right];
    while (enc_meta->query_us_compact_[right_position] != core_u)
      right_position++;

    dim3 mg_block = BLOCK_DIM;
    int N = h_num_u_candidate_vs_[core_u] * 32;
    dim3 mg_grid = std::min(GRID_DIM, calc_grid_dim(N, mg_block.x));

#ifndef NDEBUG
    std::cout << "mg_grid: " << mg_grid.x << std::endl;
#endif

    mergeKernel<<<mg_grid, mg_block>>>(
        enc_meta->cluster_offsets_[cluster_index],
        left_position, right_position,
        enc_meta->cluster_offsets_[left], enc_meta->cluster_offsets_[right],
        dg->offsets_, dg->neighbors_,
        d_u_candidate_vs_ + core_u * MAX_L_FREQ, h_num_u_candidate_vs_[core_u],

        d_encodings_);
    cuchk(cudaDeviceSynchronize());
    merge_ptr++;
  }
  // additional, final combining.
  // if there are combines in the last layer, but no new ones.
  if (enc_meta->combine_checkpoints_[enc_meta->num_layers - 1] == enc_meta->num_clusters)
  {
    while (combine_ptr < enc_meta->combine_cnt)
    {
      int big_cluster = enc_meta->combine_cluster_out_[combine_ptr];
      std::set<int> small_clusters = enc_meta->combine_clusters_other_[combine_ptr];

      std::vector<int> small_clusters_vec = std::vector<int>(small_clusters.begin(), small_clusters.end());
      int *small_clusters_arr = small_clusters_vec.data();
      int num_small_clusters = small_clusters.size();

      if (num_small_clusters > 32)
      {
        std::cerr << "encoding: combing: num small cluster: " << num_small_clusters << std::endl;
        std::cerr << "too many small clusters" << std::endl;
        exit(1);
      }

      cuchk(cudaMemcpy(d_small_clusters_, small_clusters_arr, sizeof(int) * num_small_clusters, cudaMemcpyHostToDevice));

      vtype core_u = cpu_clusters_[big_cluster].query_us_[0];

      dim3 comb_block = BLOCK_DIM;
      int N = h_num_u_candidate_vs_[core_u] * 32;
      dim3 comb_grid = std::min(GRID_DIM, calc_grid_dim(N, comb_block.x));

#ifndef NDEBUG
      std::cout << "comb_grid: " << comb_grid.x << std::endl;
#endif

      combineMultipleClustersKernel<<<comb_grid, comb_block>>>(
          dg->offsets_, dg->neighbors_,
          enc_meta->combine_type_[combine_ptr],
          big_cluster, d_small_clusters_, num_small_clusters,
          d_encodings_,
          enc_d_num_query_us_,
          enc_d_query_us_compact_, enc_d_cluster_offsets_,
          d_u_candidate_vs_ + core_u * MAX_L_FREQ, h_num_u_candidate_vs_[core_u]);
      cuchk(cudaDeviceSynchronize());
      ++combine_ptr;
    }
  }

  cuchk(cudaFree(enc_meta_buffer_));
  cuchk(cudaFree(d_small_clusters_));
}

void collectCandidates(
    uint32_t start_pos,
    encodingMeta *enc_meta,
    uint32_t *d_encodings_,
    int layer_index)
{
  vtype *d_u_candidate_vs_temp_;
  numtype *d_num_u_candidate_vs_temp_;
  // vtype *d_v_candidate_us_temp_;
  // numtype *d_num_v_candidate_us_temp_;

  cuchk(cudaMalloc((void **)&d_u_candidate_vs_temp_, sizeof(vtype) * NUM_VQ * MAX_L_FREQ));
  cuchk(cudaMalloc((void **)&d_num_u_candidate_vs_temp_, sizeof(numtype) * NUM_VQ));
  // cuchk(cudaMalloc((void **)&d_v_candidate_us_temp_, sizeof(vtype) * NUM_VD));
  // cuchk(cudaMalloc((void **)&d_num_v_candidate_us_temp_, sizeof(numtype) * NUM_VD));

  bool *vis_v = new bool[NUM_VQ];
  memset(vis_v, false, sizeof(bool) * NUM_VQ);
  int *h_pos_array_ = new int[NUM_VQ];
  int *d_pos_array_ = nullptr;
  cuchk(cudaMalloc((void **)&d_pos_array_, sizeof(int) * NUM_VQ));
  int cnt = 0;
  // for (int i = enc_meta->num_total_us - 1; ~i; --i)
  for (int cluster_id = start_pos; ~cluster_id; --cluster_id)
  {
    if (enc_meta->is_a_valid_cluster_[cluster_id] == false)
      continue;
    if (cnt == NUM_VQ)
      break;
    offtype u_off_st = enc_meta->cluster_offsets_[cluster_id];
    offtype u_off_ed = enc_meta->cluster_offsets_[cluster_id + 1];
    while (u_off_st < u_off_ed)
    {
      vtype u = enc_meta->query_us_compact_[u_off_st];
      if (!vis_v[u])
      {
        vis_v[u] = true;
        h_pos_array_[cnt] = u_off_st;
        cnt++;
      }
      u_off_st++;
    }
  }
  cuchk(cudaMemcpy(d_pos_array_, h_pos_array_, sizeof(int) * NUM_VQ, cudaMemcpyHostToDevice));

  cuchk(cudaMemset(d_num_u_candidate_vs_temp_, 0, sizeof(numtype) * NUM_VQ));
  // cuchk(cudaMemset(d_num_v_candidate_us_temp_, 0, sizeof(numtype) * NUM_VD));

  vtype *enc_d_query_us_compact_ = nullptr;
  cuchk(cudaMalloc((void **)&enc_d_query_us_compact_, sizeof(vtype) * enc_meta->num_total_us));
  cuchk(cudaMemcpy(enc_d_query_us_compact_, enc_meta->query_us_compact_, sizeof(vtype) * enc_meta->num_total_us, cudaMemcpyHostToDevice));

  dim3 cc_block = BLOCK_DIM;
  int N = NUM_VD;
  dim3 cc_grid = std::min(GRID_DIM, calc_grid_dim(N, cc_block.x));

#ifndef NDEBUG
  std::cout << "cc_grid: " << cc_grid.x << std::endl;
#endif

  collectCandidatesKernel<<<cc_grid, cc_block>>>(
      d_u_candidate_vs_temp_, d_num_u_candidate_vs_temp_,
      d_encodings_, d_pos_array_,
      enc_d_query_us_compact_,
      enc_meta->num_blocks);
  cuchk(cudaDeviceSynchronize());

#ifndef NDEBUG
  numtype *h_num_u_candidate_vs_temp_ = new numtype[NUM_VQ];
  cuchk(cudaMemcpy(h_num_u_candidate_vs_temp_, d_num_u_candidate_vs_temp_, sizeof(numtype) * NUM_VQ, cudaMemcpyDeviceToHost));
  std::cout << "layer: " << layer_index << std::endl;
  for (int i = 0; i < NUM_VQ; ++i)
    std::cout << h_num_u_candidate_vs_temp_[i] << " ";
  std::cout << std::endl;
  delete[] h_num_u_candidate_vs_temp_;
#endif

  cuchk(cudaFree(d_u_candidate_vs_temp_));
  cuchk(cudaFree(d_num_u_candidate_vs_temp_));
  // cuchk(cudaFree(d_v_candidate_us_temp_));
  // cuchk(cudaFree(d_num_v_candidate_us_temp_));
  cuchk(cudaFree(d_pos_array_));
  cuchk(cudaFree(enc_d_query_us_compact_));
  delete[] vis_v;
  delete[] h_pos_array_;
}

__global__ void
collectCandidatesKernel(
    vtype *d_u_candidate_vs_, vtype *d_num_u_candidate_vs_,
    uint32_t *d_encodings_,
    int *d_pos_array_, vtype *d_query_us_compact_,

    int num_blocks)
{
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;
  uint32_t idx = tid + bid * blockDim.x;
  uint32_t wid = tid >> 5;
  uint32_t lid = tid & 31;
  uint32_t wid_g = idx >> 5;

  __shared__ int pos_array[MAX_VQ];
  __shared__ vtype target_us[MAX_VQ];
  __shared__ int warp_pos[WARP_PER_BLOCK];

  // for optimization
  __shared__ int grid_size;
  __shared__ int block_iter_cnt;
  __shared__ vtype v[WARP_PER_BLOCK][WARP_SIZE];

  if (tid < C_NUM_VQ)
  {
    pos_array[tid] = d_pos_array_[tid];
    target_us[tid] = d_query_us_compact_[pos_array[tid]];
  }
  __syncthreads();

  if (tid == 0)
  {
    block_iter_cnt = 0;
    grid_size = gridDim.x * blockDim.x;
  }
  __syncthreads();

  while (block_iter_cnt * grid_size < C_NUM_VD)
  {
    idx = tid + bid * blockDim.x + block_iter_cnt * grid_size;
    wid_g = idx / warpSize;

    if (idx < C_NUM_VD)
      v[wid][lid] = idx;
    else
      v[wid][lid] = UINT32_MAX;
    __syncwarp();

    for (int i = 0; i < C_NUM_VQ; ++i)
    {
      if (v[wid][lid] < C_NUM_VD)
      {
        if (d_encodings_[v[wid][lid] * num_blocks + pos_array[i] / BLK_SIZE] & (1u << (pos_array[i] % BLK_SIZE)))
        {
          int mask = __activemask();
          int size = __popc(mask);
          int rank = __popc(mask & (FULL_MASK >> (31 - lid))) - 1;

          if (rank == 0)
            warp_pos[wid] = atomicAdd(&d_num_u_candidate_vs_[target_us[i]], size);
          __syncwarp(mask);
          d_u_candidate_vs_[target_us[i] * C_MAX_L_FREQ + warp_pos[wid] + rank] = v[wid][lid];
        }
      }
      __syncwarp();
    }

    __syncthreads();
    if (tid == 0)
      block_iter_cnt++;
    __syncthreads();
  }
}

__global__ void
pruneKernel(
    offtype *offsets_, vtype *nbrs_,
    vtype *u_nbrs_, numtype num_nbrs,
    int *pos_array_,
    vtype *d_u_candidate_vs_, int num_candidates,
    uint32_t *d_encodings_)
{
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;
  uint32_t idx = tid + bid * blockDim.x;
  uint32_t wid = tid >> 5;
  uint32_t lid = tid & 31;
  uint32_t wid_g = idx >> 5;

  __shared__ vtype s_core_v[WARP_PER_BLOCK];
  __shared__ uint32_t s_nbr_found[WARP_PER_BLOCK][WARP_SIZE];
  __shared__ bool s_found[WARP_PER_BLOCK];

  // for optimization
  __shared__ int block_iter_cnt;
  __shared__ int grid_size;
  __shared__ int num_warps;
  __shared__ int s_pos_array[MAX_VQ];
  __shared__ vtype s_nbrs_[MAX_VQ];
  __shared__ offtype v_nbr_off[WARP_PER_BLOCK];
  __shared__ offtype v_nbr_off_end[WARP_PER_BLOCK];

  if (tid < num_nbrs)
  {
    s_nbrs_[tid] = u_nbrs_[tid];
  }
  if (tid < C_NUM_VQ)
  {
    s_pos_array[tid] = pos_array_[tid];
  }
  __syncthreads();

  if (tid == 0)
  {
    block_iter_cnt = 0;
    grid_size = gridDim.x * blockDim.x;
    num_warps = grid_size >> 5;
  }
  __syncthreads();

  while (block_iter_cnt * num_warps < num_candidates)
  {
    idx = tid + bid * blockDim.x + block_iter_cnt * grid_size;
    wid_g = idx >> 5;

    s_nbr_found[wid][lid] = 0;

    if (lid == 0)
    {
      s_found[wid] = true;
      if (wid_g < num_candidates)
        s_core_v[wid] = d_u_candidate_vs_[wid_g];
      else
        s_core_v[wid] = UINT32_MAX;
    }
    __syncwarp();

    if (s_core_v[wid] != UINT32_MAX)
    {
      if (lid == 0)
      {
        v_nbr_off[wid] = offsets_[s_core_v[wid]];
        v_nbr_off_end[wid] = offsets_[s_core_v[wid] + 1];
      }
      __syncwarp();
      while (v_nbr_off[wid] + lid < v_nbr_off_end[wid])
      {
        int mask = __activemask();
        vtype v_nbr = nbrs_[v_nbr_off[wid] + lid];
        for (int i = 0; i < num_nbrs; ++i)
        {
          if (d_encodings_[v_nbr * C_NUM_BLOCKS + s_pos_array[i] / BLK_SIZE] & (1u << (s_pos_array[i] % BLK_SIZE)))
          {
            s_nbr_found[wid][lid] = 1;
            s_nbr_found[wid][lid] |= (1u << s_nbrs_[i]);
          }
        }
        __syncwarp(mask);
        if (lid == 0)
          v_nbr_off[wid] += warpSize;
        __syncwarp(mask);
      }
      __syncwarp();

      for (int i = 0; i < num_nbrs; ++i)
      {
        uint32_t mask = __any_sync(FULL_MASK, (s_nbr_found[wid][i] & (1 << s_nbrs_[i])));
        if (!mask && lid == 0)
          s_found[wid] = false;
        __syncwarp();
        if (!s_found[wid])
          break;
      }

      if (lid == 0 && !s_found[wid])
      {
        d_u_candidate_vs_[wid_g] = UINT32_MAX;
      }
      __syncwarp();
    }
    __syncthreads();

    if (tid == 0)
      block_iter_cnt++;
    __syncthreads();
  }
}

void prune(
    cpuGraph *hq_backup, gpuGraph *dq_backup, gpuGraph *dg, bool *keep,
    vtype *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_,
    numtype *h_num_u_candidate_vs_,
    int *h_pos_array_, int *d_pos_array_,
    uint32_t *d_encodings_,
    encodingMeta *enc_meta)
{
  bool *v_processed = new bool[NUM_VQ];
  memset(v_processed, false, sizeof(bool) * NUM_VQ);

  vtype *u_nbr_array_ = new vtype[NUM_VQ];

  vtype *d_u_nbr_array_;
  cuchk(cudaMalloc((void **)&d_u_nbr_array_, sizeof(vtype) * NUM_VQ));

  for (etype e = 0; e < hq_backup->num_e * 2; e += 2)
  {
    if (keep[e])
      continue;
    vtype u = hq_backup->evv[e].first;
    vtype u_nbr = hq_backup->evv[e].second;

    if (v_processed[u] == false)
    {
      cuchk(cudaMemcpy(
          d_u_nbr_array_, hq_backup->neighbors_ + hq_backup->offsets_[u], sizeof(vtype) * hq_backup->outdeg_[u], cudaMemcpyHostToDevice));

      dim3 p_block = BLOCK_DIM;
      int N = h_num_u_candidate_vs_[u] * 32;
      dim3 p_grid = std::min(GRID_DIM, calc_grid_dim(N, p_block.x));

      // pruneKernel<<<p_grid, p_block>>>(
      pruneKernel<<<GRID_DIM, BLOCK_DIM>>>(
          dg->offsets_, dg->neighbors_,
          d_u_nbr_array_, hq_backup->outdeg_[u],
          d_pos_array_,
          d_u_candidate_vs_ + u * MAX_L_FREQ, h_num_u_candidate_vs_[u],
          d_encodings_);
      cuchk(cudaDeviceSynchronize());

      v_processed[u] = true;
    }

    if (v_processed[u_nbr] == false)
    {
      cuchk(cudaMemcpy(
          d_u_nbr_array_, hq_backup->neighbors_ + hq_backup->offsets_[u_nbr], sizeof(vtype) * hq_backup->outdeg_[u_nbr], cudaMemcpyHostToDevice));

      dim3 p_block = BLOCK_DIM;
      int N = h_num_u_candidate_vs_[u_nbr] * 32;
      dim3 p_grid = std::min(GRID_DIM, calc_grid_dim(N, p_block.x));

      // pruneKernel<<<p_grid, p_block>>>(
      pruneKernel<<<GRID_DIM, BLOCK_DIM>>>(
          dg->offsets_, dg->neighbors_,
          d_u_nbr_array_, hq_backup->outdeg_[u_nbr],
          d_pos_array_,
          d_u_candidate_vs_ + u_nbr * MAX_L_FREQ, h_num_u_candidate_vs_[u_nbr],
          d_encodings_);
      cuchk(cudaDeviceSynchronize());
      v_processed[u_nbr] = true;
    }
  }

  cuchk(cudaFree(d_u_nbr_array_));
  delete[] v_processed;
  delete[] u_nbr_array_;
}

void clusterFilter(
    cpuGraph *hq_backup, gpuGraph *dq_backup,
    cpuGraph *hq, cpuGraph *hg,
    gpuGraph *dq, gpuGraph *dg,

    // cluster related
    std::vector<cpuCluster> &cpu_clusters_,
    uint32_t *&h_encodings_, uint32_t *&d_encodings_,
    encodingMeta *enc_meta,

    // return
    vtype *&h_u_candidate_vs_, numtype *&h_num_u_candidate_vs_,
    vtype *&d_u_candidate_vs_, numtype *&d_num_u_candidate_vs_,
    vtype *&h_v_candidate_us_, numtype *&h_num_v_candidate_us_,
    vtype *&d_v_candidate_us_, numtype *&d_num_v_candidate_us_)
{
  // clustering
  clustering(hq, cpu_clusters_, enc_meta);

  // h_encodings_ = new uint32_t[NUM_VD * enc_meta->num_blocks];
  // memset(h_encodings_, 0, sizeof(uint32_t) * NUM_VD * enc_meta->num_blocks);
  cuchk(cudaMalloc((void **)&d_encodings_, sizeof(uint32_t) * NUM_VD * enc_meta->num_blocks));
  cuchk(cudaMemset(d_encodings_, 0, sizeof(uint32_t) * NUM_VD * enc_meta->num_blocks));

  oneRoundFilterBidirection(
      hq_backup, hg,
      dq_backup, dg,
      d_u_candidate_vs_, d_num_u_candidate_vs_,
      d_v_candidate_us_, d_num_v_candidate_us_);

#ifndef NDEBUG
  std::cout << "one round filter done" << std::endl;
#endif

  cuchk(cudaMemcpy(h_num_u_candidate_vs_, d_num_u_candidate_vs_, sizeof(numtype) * NUM_VQ, cudaMemcpyDeviceToHost));

#ifndef NDEBUG
  std::cout << "LDF&NLF: " << std::endl;
  for (int i = 0; i < NUM_VQ; ++i)
    std::cout << h_num_u_candidate_vs_[i] << " ";
  std::cout << std::endl;
#endif

  encode(
      dg,
      cpu_clusters_,
      h_encodings_, d_encodings_, enc_meta,
      h_num_u_candidate_vs_,
      d_u_candidate_vs_, d_num_u_candidate_vs_,
      d_v_candidate_us_, d_num_v_candidate_us_);

#ifndef NDEBUG
  std::cout << "encode done" << std::endl;
#endif

  // cuchk(cudaMemcpy(h_encodings_, d_encodings_, sizeof(uint32_t) * NUM_VD * enc_meta->num_blocks, cudaMemcpyDeviceToHost));

  bool *vis_v = new bool[NUM_VQ];
  memset(vis_v, false, sizeof(bool) * NUM_VQ);
  // int *h_pos_array_ = new int[NUM_VQ];
  // int *d_pos_array_ = nullptr;
  // cuchk(cudaMalloc((void **)&d_pos_array_, sizeof(int) * NUM_VQ));
  int cnt = 0;
  // for (int i = enc_meta->num_total_us - 1; ~i; --i)
  for (int cluster_id = enc_meta->final_cluster_id; ~cluster_id; --cluster_id)
  {
    if (enc_meta->is_a_valid_cluster_[cluster_id] == false)
      continue;
    if (cnt == NUM_VQ)
      break;
    offtype u_off_st = enc_meta->cluster_offsets_[cluster_id];
    offtype u_off_ed = enc_meta->cluster_offsets_[cluster_id + 1];
    while (u_off_st < u_off_ed)
    {
      vtype u = enc_meta->query_us_compact_[u_off_st];
      if (!vis_v[u])
      {
        vis_v[u] = true;
        enc_meta->enc_pos_of_u_[u] = u_off_st;
        // h_pos_array_[cnt] = u_off_st;
        cnt++;
      }
      u_off_st++;
    }
  }
  // cuchk(cudaMemcpy(d_pos_array_, h_pos_array_, sizeof(int) * NUM_VQ, cudaMemcpyHostToDevice));

  // cuchk(cudaMemset(d_num_u_candidate_vs_, 0, sizeof(numtype) * NUM_VQ));
  // cuchk(cudaMemset(d_num_v_candidate_us_, 0, sizeof(numtype) * NUM_VD));

  // prune(
  //     hq_backup, dq_backup, dg, hq->keep,
  //     d_u_candidate_vs_, d_num_u_candidate_vs_,
  //     h_num_u_candidate_vs_,
  //     h_pos_array_, d_pos_array_,
  //     d_encodings_,
  //     enc_meta);

  //   vtype *enc_d_query_us_compact_ = nullptr;
  //   cuchk(cudaMalloc((void **)&enc_d_query_us_compact_, sizeof(vtype) * enc_meta->num_total_us));
  //   cuchk(cudaMemcpy(enc_d_query_us_compact_, enc_meta->query_us_compact_, sizeof(vtype) * enc_meta->num_total_us, cudaMemcpyHostToDevice));

  //   dim3 cc_block = BLOCK_DIM;
  //   int N = NUM_VD;
  //   dim3 cc_grid = std::min(GRID_DIM, calc_grid_dim(N, cc_block.x));

  // #ifndef NDEBUG
  //   std::cout << "cc_grid: " << cc_grid.x << std::endl;
  // #endif

  //   collectCandidatesKernel<<<cc_grid, cc_block>>>(
  //       d_u_candidate_vs_, d_num_u_candidate_vs_,
  //       d_encodings_, d_pos_array_,
  //       enc_d_query_us_compact_,
  //       enc_meta->num_blocks);
  //   cuchk(cudaDeviceSynchronize());

  //   cuchk(cudaMemcpy(h_num_u_candidate_vs_, d_num_u_candidate_vs_, sizeof(numtype) * NUM_VQ, cudaMemcpyDeviceToHost));

  // #ifndef NDEBUG
  //   for (int i = 0; i < NUM_VQ; ++i)
  //     std::cout << h_num_u_candidate_vs_[i] << " ";
  //   std::cout << std::endl;
  // #endif

  // cuchk(cudaFree(d_pos_array_));
  // cuchk(cudaFree(enc_d_query_us_compact_));
  delete[] vis_v;
  // delete[] h_pos_array_;
}

// __global__ void
// compact(
//     uint32_t *compact_encodings_, uint32_t *encodings_, int *d_enc_pos_u)
// {
//   uint32_t tid = threadIdx.x;
//   uint32_t bid = blockIdx.x;
//   uint32_t idx = tid + bid * blockDim.x;
//   uint32_t wid = tid >> 5;
//   uint32_t lid = tid & 31;
//   uint32_t wid_g = idx >> 5;

//   int grid_size = gridDim.x * blockDim.x;
//   for (vtype u = 0; u < C_NUM_VQ; ++u)
//   {
//     int block_iter = 0;
//     int pos = d_enc_pos_u[u];
//     while (grid_size * block_iter < C_NUM_VD)
//     {
//       vtype v = idx + grid_size * block_iter;
//       if (v - (v & 31) >= C_NUM_VD)
//         break;
//       int pred = 0;
//       if (v < C_NUM_VD)
//         pred = (encodings_[v * C_NUM_BLOCKS + pos / BLK_SIZE] & (1u << (pos % BLK_SIZE)));
//       int mask = __ballot_sync(FULL_MASK, pred);
//       if (lid == 0)
//       {
//         compact_encodings_[u * C_COL_LEN + v / BLK_SIZE] = mask;
//       }
//       __syncwarp();

//       block_iter++;
//     }
//   }
// }

__global__ void
compact(
    uint32_t *compact_encodings_, uint32_t *encodings_, int *d_enc_pos_u)
{
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;
  uint32_t idx = tid + bid * blockDim.x;
  uint32_t wid = tid >> 5;
  uint32_t lid = tid & 31;
  uint32_t wid_g = idx >> 5;

  __shared__ int grid_size;
  __shared__ int block_iter[WARP_PER_BLOCK];
  __shared__ int s_pos[MAX_VQ];

  if (tid == 0)
    grid_size = gridDim.x * blockDim.x;
  __syncthreads();

  if (tid < C_NUM_VQ)
  {
    s_pos[tid] = d_enc_pos_u[tid];
  }
  __syncthreads();

  if (lid == 0)
  {
    block_iter[wid] = 0;
  }
  __syncwarp();

  // int grid_size = gridDim.x * blockDim.x;
  for (vtype u = 0; u < C_NUM_VQ; ++u)
  {
    if (lid == 0)
      block_iter[wid] = 0;
    __syncwarp();
    // int block_iter = 0;
    // int pos = s_pos[u];
    while (grid_size * block_iter[wid] < C_NUM_VD)
    {
      vtype v = idx + grid_size * block_iter[wid];
      if (v - (v & 31) >= C_NUM_VD)
        break;
      int pred = 0;
      if (v < C_NUM_VD)
        pred = (encodings_[v * C_NUM_BLOCKS + s_pos[u] / BLK_SIZE] & (1u << (s_pos[u] % BLK_SIZE)));
      int mask = __ballot_sync(FULL_MASK, pred);
      if (lid == 0)
      {
        compact_encodings_[u * C_COL_LEN + v / BLK_SIZE] = mask;
        block_iter[wid]++;
      }
      __syncwarp();

      // block_iter++;
    }
  }
}