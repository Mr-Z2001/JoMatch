#ifndef STRUCTURE_H
#define STRUCTURE_H

#include "globals.cuh"
#include "cpuGraph.h"
#include "gpuGraph.h"

#include <set>
#include <vector>

struct cpuCluster
{
  numtype num_query_us; // size: 1
  vtype *query_us_;     // `num_query_us` query vertices. 0-th vertex is the root, it has `num_query_us-1` neighbors.

  cpuCluster();
  ~cpuCluster();

  cpuCluster &operator=(const cpuCluster &rhs);
};

struct gpuCluster
{
  numtype num_query_us; // size: 1
  vtype *query_us_;     // `num_query_us` query vertices. 0-th vertex is the root, it has `num_query_us-1` neighbors.

  gpuCluster();
  ~gpuCluster();
};

struct encodingMeta
{
  numtype num_clusters;
  numtype *num_query_us_;    // size: num_clusters
  numtype num_total_us;      // num_total_us = sum(num_query_us_)
  numtype num_blocks;        // num_blocks = ceil(sum(num_query_us_) / 32);
  vtype *query_us_compact_;  // size: sum(num_query_us_)
  offtype *cluster_offsets_; // size: num_clusters
  bool *is_a_valid_cluster_;
  uint32_t *enc_pos_of_u_; // size: num_vq

  // layer info
  numtype num_layers;
  numtype *num_clusters_per_layer_;
  offtype *layer_offsets_;

  // merge
  numtype merge_count; // how many times did the merge happen.
  std::vector<numtype> merged_cluster_left_;
  std::vector<numtype> merged_cluster_right_;

  // combine
  numtype combine_cnt;
  std::vector<numtype> combine_cluster_out_;
  int *combine_checkpoints_;
  std::vector<std::set<int>> combine_clusters_other_;
  std::vector<int> combine_type_; // 0 means new, 1 means use old.

  // final
  numtype final_cluster_id = 0;

  encodingMeta();
  ~encodingMeta();

  void init(std::vector<cpuCluster> &cpu_clusters_);
  void print();
};

#endif // STRUCTURE_H