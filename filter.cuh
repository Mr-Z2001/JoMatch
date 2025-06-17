#ifndef FILTER_H
#define FILTER_H

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "gpuGraph.h"
#include "cpuGraph.h"
#include "globals.cuh"
#include "structure.cuh"

void getVertexCover( // DP on spanning tree.
    cpuGraph *hq,
    // return
    vtype *vertex_cover_, numtype *vertex_cover_size);

void clustering(
    cpuGraph *hq,
    std::vector<cpuCluster> &cpu_clusters_,
    encodingMeta *enc_meta);

void oneRoundFilterBidirection(
    cpuGraph *hq, cpuGraph *hg,
    gpuGraph *dq, gpuGraph *dg,

    vtype *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_,
    vtype *d_v_candidate_us_, numtype *d_num_v_candidate_us_);

__global__ void
oneRoundFilterBidirectionKernel(
    // structure info
    vltype *query_vLabels_, degtype *query_out_degrees_,
    offtype *d_offsets_, vtype *d_nbrs_, vltype *d_v_labels_, degtype *d_v_degrees_,

    vtype *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_,
    vtype *d_v_candidate_us_, numtype *d_num_v_candidate_us_,

    numtype *d_query_nlc_table_);

void encode(
    gpuGraph *dg,
    std::vector<cpuCluster> &cpu_clusters_,
    uint32_t *h_encodings_, uint32_t *d_encodings_, encodingMeta *enc_meta,
    numtype *h_num_u_candidate_vs_,
    vtype *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_,
    vtype *d_v_candidate_us_, numtype *d_num_v_candidate_us_);

__global__ void encodeKernel(
    // graph info
    offtype *d_offsets_, vtype *d_nbrs_,

    // candidate vertices
    // vtype core_u, uint32_t cluster_index,
    vtype *d_u_candidate_vs_, numtype num_u_candidate_vs,
    vtype *d_v_candidate_us_,

    // encoding info
    uint32_t *encodings_,
    numtype enc_num_query_us_,
    vtype *enc_query_us_compact_, uint32_t enc_pos);

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
    uint32_t *encodings_);

__global__ void
combineMultipleClustersKernel(
    offtype *d_offsets_, vtype *nbrs_,
    int combine_type,
    int big_cluster, int *small_clusters_arr_, int num_small_clusters,
    uint32_t *d_encodings_,
    numtype *num_query_us_,
    vtype *query_us_compact_, offtype *cluster_offsets_,
    vtype *d_u_candidate_vs_, numtype num_u_candidate_vs);

void collectCandidates(
    uint32_t start_pos,
    encodingMeta *enc_meta,
    uint32_t *d_encodings_,
    int layer_index);

__global__ void
collectCandidatesKernel(
    vtype *d_u_candidate_vs_, vtype *d_num_u_candidate_vs_,
    uint32_t *d_encodings_, int *d_pos_array_, vtype *d_query_us_compact_,
    int num_blocks);

__global__ void
pruneKernel(
    offtype *offsets_, vtype *nbrs_,
    vtype *u_nbrs_, numtype num_nbrs,
    int *pos_array_,
    vtype *d_u_candidate_vs_, int num_candidates,
    uint32_t *d_encodings_);

void prune(
    cpuGraph *hq_backup, gpuGraph *dq_backup, gpuGraph *dg, bool *keep,
    vtype *d_u_candidate_vs_, numtype *d_num_u_candidate_vs_,
    numtype *h_num_u_candidate_vs_,
    int *h_pos_array_, int *d_pos_array_,
    uint32_t *d_encodings_,
    encodingMeta *enc_meta);

void clusterFilter(
    cpuGraph *hq_backup, gpuGraph *dq_backup,
    cpuGraph *hq, cpuGraph *hg,
    gpuGraph *dq, gpuGraph *dg,

    // cluster related
    std::vector<cpuCluster> &cpu_clusters_,
    uint32_t *&h_encodings_, uint32_t *&d_encodings_,
    encodingMeta *encoding_meta,

    // return
    vtype *&h_u_candidate_vs_, numtype *&h_num_u_candidate_vs_,
    vtype *&d_u_candidate_vs_, numtype *&d_num_u_candidate_vs_,
    vtype *&h_v_candidate_us_, numtype *&h_num_v_candidate_us_,
    vtype *&d_v_candidate_us_, numtype *&d_num_v_candidate_us_);

__global__ void
compact(
    uint32_t *compact_encodings_, uint32_t *encodings_, int *d_enc_pos_u);

#endif