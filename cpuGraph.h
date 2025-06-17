#ifndef CPUGRAPH_H
#define CPUGRAPH_H

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
#include <map>
#include <unordered_map>

#include "defs.h"

class change
{
public:
  bool add = true; // false for edge deletion
  vtype src, dst;
};

class cpuGraph
{
public:
  numtype num_v;
  numtype num_e;
  numtype largest_l;
  degtype maxDegree;

  degtype *indeg_;  // in degree. for CSC, unused for now.
  degtype *outdeg_; // outdegree. for CSR
  vltype *vLabels_; // size = num_v
  numtype maxLabelFreq;

  // CSR
  vtype *vertexIDs_; // size = num_v
  offtype *offsets_; // size = num_v + 1
  vtype *neighbors_; // size = num_e * 2
  etype *edgeIDs_;   // size = num_e * 2
  // vtype *src_vtx;    // size = num_e * 2

  bool isQuery;
  bool *keep;

  std::map<std::pair<vtype, vtype>, etype> vve;
  std::map<etype, std::pair<vtype, vtype>> evv; // e is `eid`, for an undirected edge, eids are different.

  // for dynamic graphs
  int num_batches = 0;
  std::vector<int> batch_size;
  std::vector<std::vector<change>> batches;

public:
  cpuGraph();
  ~cpuGraph();

  void Print();

  offtype get_u_off(vtype u);
};

class GraphUtils
{
public:
  uint8_t eidx_[MAX_VQ * MAX_VQ]; // actually it is a 2-d array.
  uint16_t nbrbits_[MAX_VQ];

public:
  void Set(const cpuGraph &g);
};

void updateHostGraph(cpuGraph &graph, std::vector<change> &batch);

#endif