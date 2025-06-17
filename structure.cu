#include "cpuGraph.h"
#include "gpuGraph.h"
#include "structure.cuh"
#include "cuda_helpers.h"

#include <cmath>

cpuCluster::cpuCluster()
{
  num_query_us = 0;
  query_us_ = nullptr;
}

cpuCluster::~cpuCluster()
{
  if (query_us_)
    delete[] query_us_;
}

cpuCluster &cpuCluster::operator=(const cpuCluster &rhs)
{
  num_query_us = rhs.num_query_us;
  query_us_ = new vtype[num_query_us];
  memcpy(query_us_, rhs.query_us_, num_query_us * sizeof(vtype));
  return *this;
}

gpuCluster::gpuCluster()
{
  num_query_us = 0;
  query_us_ = nullptr;
}

gpuCluster::~gpuCluster()
{
  if (query_us_)
    cuchk(cudaFree(query_us_));
}

encodingMeta::encodingMeta()
{
  num_clusters = 0;
  num_query_us_ = nullptr;
  num_total_us = 0;
  num_blocks = 0;
  query_us_compact_ = nullptr;
  cluster_offsets_ = nullptr;
  is_a_valid_cluster_ = new bool[MAX_CLUSTERS];
  memset(is_a_valid_cluster_, true, sizeof(bool) * MAX_CLUSTERS);
  enc_pos_of_u_ = new uint32_t[NUM_VQ];

  // layer
  num_layers = 0;
  num_clusters_per_layer_ = new numtype[MAX_LAYERS];
  memset(num_clusters_per_layer_, 0, sizeof(numtype) * MAX_LAYERS);
  layer_offsets_ = new offtype[MAX_LAYERS];
  memset(layer_offsets_, 0, sizeof(offtype) * MAX_LAYERS);

  // merge
  merge_count = 0;
  // merged_cluster_left_ = nullptr;
  // merged_cluster_right_ = nullptr;
  // merged_cluster_vertex_ = nullptr;
  // merged_cluster_layer_ = nullptr;

  // combine
  combine_cnt = 0;
  // combine_clusters_other_ = nullptr;
  // combine_cluster_out_ = nullptr;
  // combine_type_ = nullptr;

  combine_checkpoints_ = new int[MAX_LAYERS];
  memset(combine_checkpoints_, -1, sizeof(int) * MAX_LAYERS);
}

encodingMeta::~encodingMeta()
{
  if (num_query_us_ != nullptr)
    delete[] num_query_us_;
  if (query_us_compact_ != nullptr)
    delete[] query_us_compact_;
  if (cluster_offsets_ != nullptr)
    delete[] cluster_offsets_;

  if (is_a_valid_cluster_ != nullptr)
    delete[] is_a_valid_cluster_;
  if (enc_pos_of_u_ != nullptr)
    delete[] enc_pos_of_u_;

  // layer
  if (num_clusters_per_layer_ != nullptr)
    delete[] num_clusters_per_layer_;

  // merge
  // if (merged_cluster_left_ != nullptr)
  // delete[] merged_cluster_left_;
  // if (merged_cluster_right_ != nullptr)
  // delete[] merged_cluster_right_;
  // if (merged_cluster_vertex_ != nullptr)
  // delete[] merged_cluster_vertex_;
  // if (merged_cluster_layer_ != nullptr)
  // delete[] merged_cluster_layer_;

  // combine
  // if (combine_clusters_other_ != nullptr)
  // delete[] combine_clusters_other_;
  // if (combine_cluster_out_ != nullptr)
  // delete[] combine_cluster_out_;
  // if (combine_type_ != nullptr)
  // delete[] combine_type_;
}

void encodingMeta::init(std::vector<cpuCluster> &cpu_clusters_)
{
  num_query_us_ = new numtype[num_clusters];
  cluster_offsets_ = new numtype[num_clusters + 1];
  num_total_us = 0;
  for (int i = 0; i < num_clusters; ++i)
  {
    num_query_us_[i] = cpu_clusters_[i].num_query_us;
    cluster_offsets_[i] = num_total_us;
    num_total_us += num_query_us_[i];
  }
  cluster_offsets_[num_clusters] = num_total_us;
  num_blocks = std::ceil(num_total_us / 32.0);
  NUM_BLOCKS = num_blocks;
  cuchk(cudaMemcpyToSymbol(C_NUM_BLOCKS, &num_blocks, sizeof(numtype)));
  query_us_compact_ = new vtype[num_total_us];
  offtype off = 0;
  for (int i = 0; i < num_clusters; ++i)
  {
    memcpy(query_us_compact_ + off, cpu_clusters_[i].query_us_, num_query_us_[i] * sizeof(vtype));
    off += num_query_us_[i];
  }
}

void encodingMeta::print()
{
  std::cout << "num_clusters: " << num_clusters << std::endl;
  std::cout << "num_total_us: " << num_total_us << std::endl;
  std::cout << "num_blocks: " << num_blocks << std::endl;
  std::cout << "num_query_us_ in each cluster: ";
  for (int i = 0; i < num_clusters; ++i)
    std::cout << num_query_us_[i] << " ";
  std::cout << std::endl;
  std::cout << "cluster_offsets: ";
  for (int i = 0; i < num_clusters + 1; ++i)
    std::cout << cluster_offsets_[i] << " ";
  std::cout << std::endl;
  std::cout << "query_us_compact_: ";
  for (int i = 0; i < num_total_us; ++i)
    std::cout << query_us_compact_[i] << " ";
  std::cout << std::endl;
}