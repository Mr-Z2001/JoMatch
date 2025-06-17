#ifndef GPUGRAPH_H
#define GPUGRAPH_H

#include <cinttypes>

#include "cpuGraph.h"
#include "globals.cuh"

class gpuGraph
{
public:
  uint32_t *degree_; // arr
  vltype *vLabels_;  // arr
  // eltype *eLabels; // arr

  // CSR
  uint32_t *offsets_; // arr
  vtype *neighbors_;  // arr
  etype *edgeIDs_;    // arr
  // vtype *src_vtx;     // arr

  gpuGraph();
  ~gpuGraph();
};

#endif