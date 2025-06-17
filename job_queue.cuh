#ifndef JOB_QUEUE_H
#define JOB_QUEUE_H

#include "defs.h"
#include "globals.cuh"

class Job
{
public:
  vtype nodes[2];
};

class JobQueue
{
public:
  Job *q;
  int start_level;
  vtype length;
  unsigned long long cur = 0;
  int mutex = 0;
};

class JobQueuePreprocessor
{
  JobQueue q;

  JobQueuePreprocessor();
};

#endif // JOB_QUEUE_H