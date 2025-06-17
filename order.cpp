#include "order.h"
#include "globals.cuh"
#include "structure.cuh"

#include <cstring>
#include <queue>
#include <iostream>
#include <set>

#include "cuda_helpers.h"

using namespace std;

Order::Order()
{
  root_u = UINT32_MAX;
  v_order_ = new vtype[NUM_VQ];
  u2l_ = new int[NUM_VQ];
  e_order_ = new etype[NUM_EQ];
  num_backward_neighbors_ = new numtype[NUM_VQ];
  backward_neighbors_ = new vtype *[NUM_VQ];
  for (int i = 0; i < NUM_VQ; ++i)
    backward_neighbors_[i] = new vtype[NUM_VQ];
  e_is_tree_ = new bool[NUM_EQ];

  memset(v_order_, UINT32_MAX, sizeof(vtype) * NUM_VQ);
  memset(u2l_, 0, sizeof(int) * NUM_VQ);
  memset(e_order_, UINT32_MAX, sizeof(etype) * NUM_EQ);
  memset(num_backward_neighbors_, 0, sizeof(numtype) * NUM_VQ);
  for (int i = 0; i < NUM_VQ; ++i)
    memset(backward_neighbors_[i], UINT32_MAX, sizeof(vtype) * NUM_VQ);
  memset(e_is_tree_, false, sizeof(bool) * NUM_EQ);
}

Order::~Order()
{
  if (v_order_ != nullptr)
    delete[] v_order_;
  if (u2l_ != nullptr)
    delete[] u2l_;
  if (e_order_ != nullptr)
    delete[] e_order_;

  if (num_backward_neighbors_ != nullptr)
    delete[] num_backward_neighbors_;
  for (int i = 0; i < NUM_VQ; ++i)
    if (backward_neighbors_[i] != nullptr)
      delete[] backward_neighbors_[i];
  if (backward_neighbors_ != nullptr)
    delete[] backward_neighbors_;
  if (e_is_tree_ != nullptr)
    delete[] e_is_tree_;
}

void Order::getEdgeOrderBFS(
    cpuGraph *query_graph,
    vtype start_u)
{
  uint32_t num_v = query_graph->num_v;

  vtype start_v_nbr = query_graph->neighbors_[query_graph->offsets_[start_u]];

  // e_order_[0] = query_graph->vve[{start_u, start_v_nbr}];

  bool *vis_v = new bool[query_graph->num_v];
  memset(vis_v, false, sizeof(bool) * query_graph->num_v);
  bool *vis_e = new bool[query_graph->num_e];
  memset(vis_e, false, sizeof(bool) * query_graph->num_e);

  std::set<vtype> visited_v;

  int order_off = 0;

  // go bfs
  std::set<std::pair<vtype, vtype>> visited_edges;
  // visited_edges.insert({start_u, start_v_nbr});

  queue<vtype> q;
  q.push(start_u);
  vis_v[start_u] = true;
  while (!q.empty())
  {
    vtype u = q.front();
    q.pop();

    for (uint32_t off = query_graph->offsets_[u]; off < query_graph->offsets_[u + 1]; ++off)
    {
      vtype u_nbr = query_graph->neighbors_[off];

      if (visited_edges.find(std::make_pair(u_nbr, u)) == visited_edges.end() &&
          visited_edges.find(std::make_pair(u, u_nbr)) == visited_edges.end())
      {
        e_order_[order_off] = query_graph->edgeIDs_[off];
        if (vis_v[u] && vis_v[u_nbr])
          e_is_tree_[order_off++] = false;
        else
          e_is_tree_[order_off++] = true;
        visited_edges.insert(std::make_pair(u, u_nbr));
      }

      if (!vis_v[u_nbr])
      {
        q.push(u_nbr);
        vis_v[u_nbr] = true;
      }

      for (uint32_t off_2 = query_graph->offsets_[u_nbr]; off_2 < query_graph->offsets_[u_nbr + 1]; ++off_2)
      {
        vtype u_nbr_nbr = query_graph->neighbors_[off_2];
        if (!vis_v[u_nbr_nbr] || u_nbr_nbr == u)
          continue;
        if (visited_edges.find(std::make_pair(u_nbr, u_nbr_nbr)) == visited_edges.end() &&
            visited_edges.find(std::make_pair(u_nbr_nbr, u_nbr)) == visited_edges.end())
        {
          e_order_[order_off] = query_graph->vve[{u_nbr, u_nbr_nbr}];
          if (vis_v[u_nbr] && vis_v[u_nbr_nbr])
            e_is_tree_[order_off++] = false;
          else
            e_is_tree_[order_off++] = true;
          visited_edges.insert(std::make_pair(u_nbr, u_nbr_nbr));
        }
      }
    }
  }
}

void Order::getVertexOrderBFS(
    cpuGraph *query_graph,
    vtype start_u)
{
  int cnt = 0;
  std::queue<vtype> q;
  bool vis[MAX_VQ] = {false};
  q.push(start_u);
  vis[start_u] = true;
  while (!q.empty())
  {
    vtype top = q.front();
    q.pop();
    v_order_[cnt] = top;
    u2l_[top] = cnt++; // maintain reverse order.
    for (offtype u_off = query_graph->offsets_[top]; u_off < query_graph->offsets_[top + 1]; ++u_off)
    {
      vtype u_nbr = query_graph->neighbors_[u_off];
      if (!vis[u_nbr])
      {
        vis[u_nbr] = true;
        q.push(u_nbr);
      }
    }
  }
}

// __forceinline__ bool is_neighbor(
//     cpuGraph *query_graph,
//     vtype u, vtype u_other)
// {
//   bool res = false;

//   auto it = lower_bound(query_graph->neighbors_ + query_graph->offsets_[u], query_graph->neighbors_ + query_graph->offsets_[u + 1], u_other);
//   if (it != query_graph->neighbors_ + query_graph->offsets_[u + 1] && *it == u_other)
//     res = true;

//   return res;

//   // for (offtype off = query_graph->offsets_[u]; off < query_graph->offsets_[u + 1]; ++off)
//   // {
//   //   if (u_other == query_graph->neighbors_[off])
//   //     res = true;
//   // }
//   // return res;
// }

void Order::constructBackwardNeighbors(
    cpuGraph *query_graph)
{
  for (int order_off = 1; order_off < NUM_VQ; ++order_off)
  {
    vtype u = v_order_[order_off];
    for (int order_off_bn = 0; order_off_bn < order_off; ++order_off_bn)
    {
      vtype u_back = v_order_[order_off_bn];
      if (is_neighbor(query_graph, u, u_back))
      {
        backward_neighbors_[u][num_backward_neighbors_[u]++] = u_back;
      }
    }
  }
}

OrderGPU::OrderGPU(int v_num_orders)
{
  // std::cout << "In constructing OrderGPU, num_orders = " << v_num_orders << std::endl;
  // std::cout << "NUM_VQ = " << NUM_VQ << std::endl;
  cuchk(cudaMalloc((void **)&num_orders, sizeof(int)));
  cuchk(cudaMalloc((void **)&roots_, sizeof(vtype) * v_num_orders));
  cuchk(cudaMalloc((void **)&v_orders_, sizeof(vtype) * v_num_orders * NUM_VQ));
  cuchk(cudaMalloc((void **)&u2ls_, sizeof(int) * v_num_orders * NUM_VQ));

  cuchk(cudaMalloc((void **)&num_backward_neighbors_, sizeof(numtype) * v_num_orders * NUM_VQ));
  cuchk(cudaMalloc((void **)&backward_neighbors_, sizeof(vtype) * v_num_orders * NUM_VQ * NUM_VQ));

  cuchk(cudaMemcpy(this->num_orders, &v_num_orders, sizeof(int), cudaMemcpyHostToDevice));

  // cudaMalloc((void **)&root_u, sizeof(vtype));
  // cudaMalloc((void **)&v_order_, sizeof(vtype) * NUM_VQ);
  // cudaMalloc((void **)&u2l_, sizeof(int) * NUM_VQ);
  // cudaMalloc((void **)&e_order_, sizeof(etype) * NUM_EQ);
  // cudaMalloc((void **)&shared_neighbors_with_, sizeof(vtype) * NUM_VQ);
  // cudaMalloc((void **)&num_backward_neighbors_, sizeof(numtype) * NUM_VQ);
  // cudaMalloc((void **)&backward_neighbors_, sizeof(vtype) * NUM_VQ * NUM_VQ);
}

OrderGPU::OrderGPU()
{
  std::cout << "Constructing OrderGPU default" << std::endl;
  // // cudaMalloc((void **)&root_u, sizeof(vtype));
  // // cudaMalloc((void **)&v_order_, sizeof(vtype) * NUM_VQ);
  // // cudaMalloc((void **)&u2l_, sizeof(int) * NUM_VQ);
  // // cudaMalloc((void **)&e_order_, sizeof(etype) * NUM_EQ);
  // // cudaMalloc((void **)&shared_neighbors_with_, sizeof(vtype) * NUM_VQ);
  // cudaMalloc((void **)&num_backward_neighbors_, sizeof(numtype) * NUM_VQ);
  // cudaMalloc((void **)&backward_neighbors_, sizeof(vtype) * NUM_VQ * NUM_VQ);

  // // cudaMemcpy(root_u, &order_obj->root_u, sizeof(vtype), cudaMemcpyHostToDevice);
  // // cudaMemcpy(v_order_, order_obj->v_order_, sizeof(vtype) * NUM_VQ, cudaMemcpyHostToDevice);
  // // cudaMemcpy(u2l_, order_obj->u2l_, sizeof(int) * NUM_VQ, cudaMemcpyHostToDevice);
  // // cudaMemcpy(e_order_, order_obj->e_order_, sizeof(etype) * NUM_EQ, cudaMemcpyHostToDevice);
  // // cudaMemcpy(shared_neighbors_with_, order_obj->shared_neighbors_with_, sizeof(vtype) * NUM_VQ, cudaMemcpyHostToDevice);
  // cudaMemcpy(num_backward_neighbors_, order_obj->num_backward_neighbors_, sizeof(numtype) * NUM_VQ, cudaMemcpyHostToDevice);
  // offtype off = 0;
  // for (int i = 0; i < NUM_VQ; ++i)
  // {
  //   cudaMemcpy(backward_neighbors_ + off, order_obj->backward_neighbors_[i], sizeof(vtype) * NUM_VQ, cudaMemcpyHostToDevice);
  //   off += NUM_VQ;
  // }
}

OrderGPU::~OrderGPU()
{
  cudaFree(num_orders);
  cudaFree(roots_);
  cudaFree(v_orders_);
  cudaFree(u2ls_);
  // cudaFree(e_order_);
  // cudaFree(shared_neighbors_with_);
  cudaFree(num_backward_neighbors_);
  cudaFree(backward_neighbors_);
}

OrderCPU::OrderCPU()
{
  num_orders = 0;
}

OrderCPU::OrderCPU(numtype v_num_orders)
{
  num_orders = v_num_orders;
}

OrderCPU::~OrderCPU()
{
}

void OrderCPU::init_roots(
    std::vector<vtype> &roots)
{
  this->roots = roots;
}

void OrderCPU::getEdgeOrderBFS(
    cpuGraph *query_graph)
{
  if (num_orders == 0)
  {
    std::cout << "Error, `num_orders` is 0." << std::endl;
    exit(1);
  }

  uint32_t num_v = query_graph->num_v;
  bool *vis_v = new bool[query_graph->num_v];
  bool *vis_e = new bool[query_graph->num_e];

  e_orders.resize(num_orders);
  e_is_trees.resize(num_orders);

  for (int order_id = 0; order_id < num_orders; ++order_id)
  {
    vtype start_u = roots[order_id];
    vtype start_u_nbr = sub_roots[order_id];
    e_orders[order_id].push_back(query_graph->vve[{start_u, start_u_nbr}]);

    // e_order_[0] = query_graph->vve[{start_u, start_v_nbr}];

    memset(vis_v, false, sizeof(bool) * query_graph->num_v);
    memset(vis_e, false, sizeof(bool) * query_graph->num_e);

    std::set<vtype> visited_v;

    // go bfs
    std::set<std::pair<vtype, vtype>> visited_edges;
    // visited_edges.insert({start_u, start_v_nbr});

    queue<vtype> q;
    q.push(start_u);
    vis_v[start_u] = true;
    q.push(start_u_nbr);
    vis_v[start_u_nbr] = true;
    visited_edges.insert(std::make_pair(start_u, start_u_nbr));
    e_is_trees[order_id].push_back(true);
    while (!q.empty())
    {
      vtype u = q.front();
      q.pop();

      for (uint32_t off = query_graph->offsets_[u]; off < query_graph->offsets_[u + 1]; ++off)
      {
        vtype u_nbr = query_graph->neighbors_[off];

        if (visited_edges.find(std::make_pair(u_nbr, u)) == visited_edges.end() &&
            visited_edges.find(std::make_pair(u, u_nbr)) == visited_edges.end())
        {
          e_orders[order_id].push_back(query_graph->edgeIDs_[off]);
          if (vis_v[u] && vis_v[u_nbr])
            e_is_trees[order_id].push_back(false);
          else
            e_is_trees[order_id].push_back(true);
          visited_edges.insert(std::make_pair(u, u_nbr));
        }

        if (!vis_v[u_nbr])
        {
          q.push(u_nbr);
          vis_v[u_nbr] = true;
        }

        for (uint32_t off_2 = query_graph->offsets_[u_nbr]; off_2 < query_graph->offsets_[u_nbr + 1]; ++off_2)
        {
          vtype u_nbr_nbr = query_graph->neighbors_[off_2];
          if (!vis_v[u_nbr_nbr] || u_nbr_nbr == u)
            continue;
          if (visited_edges.find(std::make_pair(u_nbr, u_nbr_nbr)) == visited_edges.end() &&
              visited_edges.find(std::make_pair(u_nbr_nbr, u_nbr)) == visited_edges.end())
          {
            e_orders[order_id].push_back(query_graph->vve[{u_nbr, u_nbr_nbr}]);
            if (vis_v[u_nbr] && vis_v[u_nbr_nbr])
              e_is_trees[order_id].push_back(false);
            else
              e_is_trees[order_id].push_back(true);
            visited_edges.insert(std::make_pair(u_nbr, u_nbr_nbr));
          }
        }
      }
    }
  }

  delete[] vis_v;
  delete[] vis_e;
}
void OrderCPU::getVertexOrderBFS(
    cpuGraph *query_graph)
{
  std::queue<vtype> q;

  v_orders.resize(num_orders);
  u2ls.resize(num_orders);
  for (int order_id = 0; order_id < num_orders; ++order_id)
  {
    u2ls[order_id].resize(NUM_VQ);
    bool vis[MAX_VQ] = {false};
    vtype start_u = roots[order_id];
    q.push(start_u);
    vis[start_u] = true;
    q.push(sub_roots[order_id]);
    vis[sub_roots[order_id]] = true;
    while (!q.empty())
    {
      vtype top = q.front();
      q.pop();
      v_orders[order_id].push_back(top);
      // v_order_[cnt] = top;
      u2ls[order_id][top] = v_orders[order_id].size() - 1; // maintain reverse order.
      // u2l_[top] = cnt++; // maintain reverse order.
      for (offtype u_off = query_graph->offsets_[top]; u_off < query_graph->offsets_[top + 1]; ++u_off)
      {
        vtype u_nbr = query_graph->neighbors_[u_off];
        if (!vis[u_nbr])
        {
          vis[u_nbr] = true;
          q.push(u_nbr);
        }
      }
    }
  }
}

void OrderCPU::constructBackwardNeighbors(
    cpuGraph *query_graph)
{
  num_backward_neighbors.resize(num_orders);
  backward_neighbors.resize(num_orders);

  for (int order_id = 0; order_id < num_orders; ++order_id)
  {
    num_backward_neighbors[order_id].resize(NUM_VQ);
    backward_neighbors[order_id].resize(NUM_VQ);
    for (int order_off = 1; order_off < NUM_VQ; ++order_off)
    {
      vtype u = v_orders[order_id][order_off];
      // backward_neighbors[order_id][u].resize(NUM_VQ);
      // vtype u = v_order_[order_off];
      for (int order_off_bn = 0; order_off_bn < order_off; ++order_off_bn)
      {
        vtype u_back = v_orders[order_id][order_off_bn];
        // vtype u_back = v_order_[order_off_bn];
        if (is_neighbor(query_graph, u, u_back))
        {
          backward_neighbors[order_id][u].push_back(u_back);
          // backward_neighbors[order_id][u][num_backward_neighbors[order_id][u]++] = u_back;
          // backward_neighbors_[u][num_backward_neighbors_[u]++] = u_back;
        }
      }
      num_backward_neighbors[order_id][u] = backward_neighbors[order_id][u].size();
    }
  }
}

// void getBFSorder(
//     cpuGraph *g,
//     vtype *order,
//     vtype start_v)
// {
// #ifndef NDEBUG
//   std::cout << "offsets_:" << std::endl;
//   for (int i = 0; i < g->num_v; ++i)
//     std::cout << g->vertexIDs_[i] << " ";
//   std::cout << std::endl;
//   for (int i = 0; i < NUM_VQ + 1; ++i)
//     std::cout << g->offsets_[i] << " ";
//   std::cout << std::endl;
// #endif

//   uint32_t num_v = g->num_v;

//   bool vis[MAX_VQ];
//   memset(vis, false, sizeof(bool) * MAX_VQ);
//   int order_off = 0;

//   // go bfs
//   queue<vtype> q;
//   vtype start_u = start_v;
//   q.push(start_u);
//   while (!q.empty())
//   {
//     vtype u = q.front();
//     q.pop();
//     vis[u] = true;

//     order[order_off++] = u;

// #ifndef NDEBUG
//     std::cout << u << std::endl;
// #endif

//     vtype u_nxt = NUM_VQ;
//     uint32_t u_off = g->get_u_off(u);
//     if (u_off != num_v - 1)
//       u_nxt = g->vertexIDs_[u_off + 1];

// #ifndef NDEBUG
//     std::cout << "u=" << u << ", u_nxt=" << u_nxt << std::endl;
//     std::cout << "off_st=" << g->offsets_[u] << ", off_end=" << g->offsets_[u_nxt] << std::endl;
// #endif
//     for (uint32_t off = g->offsets_[u]; off < g->offsets_[u_nxt]; ++off)
//     {
// #ifndef NDEBUG
//       std::cout << "off=" << off << std::endl;
// #endif
//       vtype u_nbr = g->neighbors_[off];
// #ifndef NDEBUG
//       std::cout << "u_nbr = " << u_nbr << std::endl;
// #endif
//       if (!vis[u_nbr] && u_nbr != UINT32_MAX)
//         q.push(u_nbr);
//     }
//   }

// #ifndef NDEBUG
//   std::cout << "In getBFSorder(): " << std::endl;
//   for (int i = 0; i < num_v; ++i)
//     std::cout << order[i] << " ";
//   std::cout << std::endl;
// #endif
// }

// struct cmp
// {
//   bool operator()(std::pair<degtype, vtype> a, std::pair<degtype, vtype> b)
//   {
//     if (a.first == b.first)
//       return a.second < b.second;
//     return a.first > b.first;
//   }
// };

// // designed for undirected graph
// // if directed, need to modify some details. --> change of degree.
// void getCFLorder(
//     cpuGraph *q,
//     vtype *order)
// {
//   bool *vis = new bool[NUM_VQ];
//   memset(vis, false, sizeof(vis));
//   degtype *out_deg_copy = new degtype[NUM_VQ];
//   memcpy(out_deg_copy, q->outdeg_, sizeof(degtype) * NUM_VQ);

//   offtype order_off = 0;

//   bool is_leaf_left = true;
//   while (is_leaf_left)
//   {
//     is_leaf_left = false;
//     for (vtype u = 0; u < NUM_VQ; ++u)
//     {
//       if (vis[u])
//         continue;
//       if (out_deg_copy[u] == 1)
//       {
//         order[order_off++] = u;
//         vis[u] = true;
//         is_leaf_left = true;
//         // decrease the out degree of the neighbor
//         for (offtype off = q->offsets_[u]; off < q->offsets_[u + 1]; ++off)
//         {
//           vtype u_nbr = q->neighbors_[off];
//           if (!vis[u_nbr])
//             --out_deg_copy[u_nbr];
//         }
//       }
//     }
//   }

//   std::priority_queue<std::pair<degtype, vtype>, std::vector<std::pair<degtype, vtype>>, cmp> pq;
//   for (vtype u = 0; u < NUM_VQ; ++u)
//   {
//     if (!vis[u])
//       pq.push(std::make_pair(out_deg_copy[u], u));
//   }

//   while (!pq.empty())
//   {
//     vtype u = pq.top().second;
//     pq.pop();
//     order[order_off++] = u;
//   }

//   reverse(order, order + NUM_VQ);

//   delete[] vis;
//   delete[] out_deg_copy;
// }

// // one direction of an undirected edge is enough.
// void getBFSEdgeOrder(
//     cpuGraph *g,
//     etype *order,
//     // cpuRelation *cpu_relations_,
//     vtype start_v)
// {
//   uint32_t num_v = g->num_v;

//   vtype start_v_nbr = g->neighbors_[g->offsets_[start_v]];

//   order[0] = g->vve[{start_v, start_v_nbr}];

//   bool *vis_v = new bool[g->num_v];
//   memset(vis_v, false, sizeof(bool) * g->num_v);
//   bool *vis_e = new bool[g->num_e];
//   memset(vis_e, false, sizeof(bool) * g->num_e);

//   std::set<vtype> visited_v;

//   int order_off = 1;

//   // go bfs
//   std::set<std::pair<vtype, vtype>> visited_edges;
//   visited_edges.insert({start_v, start_v_nbr});

//   queue<vtype> q;
//   q.push(start_v);
//   vis_v[start_v] = true;
//   while (!q.empty())
//   {
//     vtype u = q.front();
//     q.pop();

//     for (uint32_t off = g->offsets_[u]; off < g->offsets_[u + 1]; ++off)
//     {
//       vtype u_nbr = g->neighbors_[off];

//       if (!vis_v[u_nbr])
//       {
//         q.push(u_nbr);
//         vis_v[u_nbr] = true;
//       }
//       if (visited_edges.find(std::make_pair(u_nbr, u)) == visited_edges.end() &&
//           visited_edges.find(std::make_pair(u, u_nbr)) == visited_edges.end())
//       {
//         order[order_off++] = g->edgeIDs_[off];
//         visited_edges.insert(std::make_pair(u, u_nbr));
//       }

//       for (uint32_t off_2 = g->offsets_[u_nbr]; off_2 < g->offsets_[u_nbr + 1]; ++off_2)
//       {
//         vtype u_nbr_nbr = g->neighbors_[off_2];
//         if (!vis_v[u_nbr_nbr] || u_nbr_nbr == u)
//           continue;
//         if (visited_edges.find(std::make_pair(u_nbr, u_nbr_nbr)) == visited_edges.end() &&
//             visited_edges.find(std::make_pair(u_nbr_nbr, u_nbr)) == visited_edges.end())
//         {
//           order[order_off++] = g->vve[{u_nbr, u_nbr_nbr}];
//           visited_edges.insert(std::make_pair(u_nbr, u_nbr_nbr));
//         }
//       }
//     }
//   }
// }
