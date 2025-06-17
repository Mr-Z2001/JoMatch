#pragma once

#ifndef RES_TABLE_H
#define RES_TABLE_H

#include <cinttypes>
#include <cstring>

#include <vector>

class ResTable
{
public:
  int cur_level;
  int max_res;
  uint32_t *res_table;
  int head, tail, size;

  ResTable()
  {
    cur_level = 0;
    max_res = 0;
    res_table = nullptr;
    head = tail = size = 0;
  }

  ResTable(int max_res)
  {
    cur_level = 0;
    this->max_res = max_res;
    res_table = new uint32_t[max_res];
    head = tail = size = 0;
  }
  ~ResTable()
  {
    if (res_table != nullptr)
      delete[] res_table;
  }

  bool insert(uint32_t order_id, std::vector<uint32_t> &res)
  {
    if (size + res.size() + 1 > max_res)
      return false;
    res_table[tail++] = order_id;
    memcpy(res_table + tail, res.data(), sizeof(uint32_t) * res.size());
    tail += res.size();
    size += res.size() + 1;
    return true;
  }
  bool pop(uint32_t &order_id, std::vector<uint32_t> &res)
  {
    if (size == 0)
      return false;
    order_id = res_table[head++];

    res.resize(cur_level);
    memcpy(res.data(), res_table + head, sizeof(uint32_t) * cur_level);
    head += cur_level;
    size -= cur_level + 1;
    return true;
  }
};

#endif