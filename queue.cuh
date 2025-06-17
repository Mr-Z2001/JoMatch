#ifndef QUEUE_H
#define QUEUE_H

template <typename DataType>
class DeletionMarker
{
public:
  static constexpr void *val{nullptr};
};

template <>
class DeletionMarker<int>
{
public:
  static constexpr uint32_t val{0xFFFFFFFF};
};

template <>
class DeletionMarker<unsigned long long>
{
public:
  static constexpr unsigned long long val{0xFFFFFFFFFFFFFFFF};
};

class Queue
{
public:
  int *queue_;
  // vtype *candidate_queue;
  int count_{0};
  unsigned int front_{0};
  unsigned int back_{0};
  int size_{0};

public:
  __forceinline__ __host__ __device__ void resetQueue()
  {
    count_ = 0;
    front_ = 0;
    back_ = 0;
  }

  __forceinline__ __device__ void init()
  {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = tid + bid * blockDim.x;
    while (idx < size_)
    {
      queue_[idx] = DeletionMarker<int>::val;
      idx += blockDim.x * gridDim.x;
    }
  }

  __forceinline__ __device__ bool enqueue(int *var_arr_, int num)
  {
    int fill = atomicAdd(&count_, num);
    if (fill < size_)
    {
      unsigned int pos = atomicAdd(&back_, num) % size_;
      for (int i = 0; i < num; ++i)
      {
        int val = var_arr_[i];
        while (atomicCAS(&queue_[pos + i], DeletionMarker<int>::val, val) != DeletionMarker<int>::val)
          __nanosleep(10);
      }
      return true;
    }
    else
    {
      atomicSub(&count_, num);
      return false;
    }
  }

  __forceinline__ __device__ bool dequeue(int *var_arr_, int num)
  {
    int readable = atomicSub(&count_, num);
    if (readable > 0)
    {
      unsigned int pos = atomicAdd(&front_, num) % size_;
      for (int i = 0; i < num; ++i)
      {
        while ((var_arr_[i] = atomicExch(&queue_[pos + i], DeletionMarker<int>::val)) == DeletionMarker<int>::val)
          __nanosleep(10);
      }
      return true;
    }
    else
    {
      atomicAdd(&count_, num);
      return false;
    }
  }
};

#endif