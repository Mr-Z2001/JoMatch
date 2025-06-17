#include "gpuGraph.h"

gpuGraph::gpuGraph()
{
  // elCount = 0;
  degree_ = nullptr;
  vLabels_ = nullptr;
  // eLabels = nullptr;
  offsets_ = nullptr;
  neighbors_ = nullptr;
  edgeIDs_ = nullptr;
  // src_vtx = nullptr;
}

gpuGraph::~gpuGraph()
{
  // if (degree != nullptr)
  //   delete[] degree;
  // if (vLabels != nullptr)
  //   delete[] vLabels;
  // // if (eLabels != nullptr)
  // //   delete[] eLabels;
  // if (vertexIDs_ != nullptr)
  //   delete[] vertexIDs_;
  // if (offsets_ != nullptr)
  //   delete[] offsets_;
  // if (neighbors_ != nullptr)
  //   delete[] neighbors_;
  // if (edgeIDs_ != nullptr)
  //   delete[] edgeIDs_;
}