#ifndef CALLSTACK_H
#define CALLSTACK_H

#include "defs.h"
#include "globals.cuh"

class CallStack
{
public:
  vtype level = 0;
  vtype iter[MAX_VQ];
  // numtype map_res_[MAX_VQ];
  numtype num_candidates_[MAX_VQ];
  vtype *candidates_; // max_size for each u: `NUM_CAN_UB` candidates.
  bool stealed_task = 0;
};

#endif // CALLSTACK_H