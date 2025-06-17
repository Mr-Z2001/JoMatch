#include <cstring>
#include <queue>

#include "decycle.h"

void decycle(cpuGraph *q)
{
  bool *vis_v = new bool[q->num_v];
  memset(vis_v, false, sizeof(bool) * q->num_v);
  uint32_t dumped = 0;

  vtype root = 0;
  degtype maxdeg = q->outdeg_[root];

  for (vtype u = 1; u < q->num_v; ++u)
  {
    if (q->outdeg_[u] > maxdeg)
    {
      root = u;
      maxdeg = q->outdeg_[u];
    }
  }

  std::queue<vtype> qu;
  qu.push(root);
  vis_v[root] = true;
  while (!qu.empty())
  {
    vtype top = qu.front();
    qu.pop();

    for (offtype u_nbr_off = q->offsets_[top]; u_nbr_off < q->offsets_[top + 1]; ++u_nbr_off)
    {
      vtype u_nbr = q->neighbors_[u_nbr_off];
      if (!vis_v[u_nbr])
      {
        vis_v[u_nbr] = true;
        qu.push(u_nbr);
        etype e = q->vve[{top, u_nbr}];
        q->keep[e] = true;
      }
    }
  }

  for (etype e = 0; e < q->num_e * 2; e += 2)
  {
    if (q->keep[e] || q->keep[e + 1])
    {
      q->keep[e] = q->keep[e + 1] = true;
    }
    else
    {
#ifndef NDEBUG
      std::cout << q->evv[e].first << " " << q->evv[e].second << " dumped." << std::endl;
#endif
      dumped++;
      q->outdeg_[q->evv[e].first]--;
      q->outdeg_[q->evv[e].second]--;
    }
  }

  // for (int e = 0; e < q->num_e * 2; e += 2)
  // {
  //   vtype u = q->evv[e].first, v = q->evv[e].second;

  //   std::cout << "e = " << e << ", u = " << u << ", v = " << v;

  //   if (vis_v[u] && vis_v[v])
  //   {
  //     std::cout << " dumped";
  //     q->keep[e] = false;
  //     q->keep[e + 1] = false;
  //     dumped++;
  //     q->outdeg_[u]--;
  //     q->outdeg_[v]--;
  //   }
  //   std::cout << std::endl;
  //   vis_v[u] = vis_v[v] = true;
  // }

  offtype *temp_off = new offtype[q->num_v + 1];
  vtype *temp_nbrs = new vtype[(q->num_e - dumped) * 2];
  etype *temp_eids = new etype[(q->num_e - dumped) * 2];

  q->evv.clear();
  q->vve.clear();

  temp_off[0] = 0;

  offtype global_offset = 0;
  etype global_eid = 0;

  for (vtype u = 0; u < q->num_v; ++u)
  {
    for (offtype u_nbr_off = q->offsets_[u]; u_nbr_off < q->offsets_[u + 1]; ++u_nbr_off)
    {
      vtype u_nbr = q->neighbors_[u_nbr_off];
      etype e = q->edgeIDs_[u_nbr_off];
      if (q->keep[e])
      {
        temp_nbrs[global_offset] = u_nbr;
        if (q->vve.find(std::make_pair(u, u_nbr)) == q->vve.end())
        {
          q->vve.insert(std::make_pair(std::make_pair(u, u_nbr), global_eid));
          q->evv.insert(std::make_pair(global_eid, std::make_pair(u, u_nbr)));

          // reverse
          q->vve.insert(std::make_pair(std::make_pair(u_nbr, u), global_eid + 1));
          q->evv.insert(std::make_pair(global_eid + 1, std::make_pair(u_nbr, u)));

          global_eid += 2;
        }
        temp_eids[global_offset] = q->vve[std::make_pair(u, u_nbr)];
        global_offset++;
      }
    }
    temp_off[u + 1] = global_offset;
  }

  q->num_e -= dumped;
  for (int i = 0; i < q->num_v; ++i)
    q->maxDegree = std::max(q->maxDegree, q->outdeg_[i]);

  delete[] vis_v;

  delete[] q->offsets_;
  delete[] q->neighbors_;
  delete[] q->edgeIDs_;

  q->offsets_ = temp_off;
  q->neighbors_ = temp_nbrs;
  q->edgeIDs_ = temp_eids;

  return;
}