#ifndef ORDER_H
#define ORDER_H

#include "cpuGraph.h"
#include "structure.cuh"

#include <algorithm>

__forceinline__ bool is_neighbor(
		cpuGraph *query_graph,
		vtype u, vtype u_other)
{
	bool res = false;

	auto it = std::lower_bound(query_graph->neighbors_ + query_graph->offsets_[u], query_graph->neighbors_ + query_graph->offsets_[u + 1], u_other);
	if (it != query_graph->neighbors_ + query_graph->offsets_[u + 1] && *it == u_other)
		res = true;

	return res;

	// for (offtype off = query_graph->offsets_[u]; off < query_graph->offsets_[u + 1]; ++off)
	// {
	//   if (u_other == query_graph->neighbors_[off])
	//     res = true;
	// }
	// return res;
}

class Order
{
public:
	vtype root_u;
	vtype *v_order_;
	int *u2l_; // given query u, return the order(0 ~ NUM_VQ-1)
	etype *e_order_;
	bool *e_is_tree_;

	// vtype *shared_neighbors_with_;
	// vtype **backward_neighbors_sh_;
	// numtype *num_backward_neighbors_sh_;

	numtype *num_backward_neighbors_;
	vtype **backward_neighbors_;

	Order();
	~Order();

	void getEdgeOrderBFS(
			cpuGraph *query_graph,
			vtype start_u = 0); // get e_order_

	void getVertexOrderBFS(
			cpuGraph *query_graph,
			vtype start_u = 0); // get v_order_

	void constructBackwardNeighbors(
			cpuGraph *query_graph);
};

// class OrderGPU
// {
// public:
// 	vtype *root_u;
// 	vtype *v_order_;
// 	int *u2l_;
// 	// etype *e_order_;

// 	numtype *num_backward_neighbors_;
// 	vtype *backward_neighbors_;

// 	OrderGPU();
// 	OrderGPU(Order *order_obj);
// 	~OrderGPU();
// };

class OrderCPU
{
public:
	numtype num_orders;
	std::vector<vtype> roots;
	std::vector<vtype> sub_roots;
	std::vector<std::vector<vtype>> v_orders;
	std::vector<std::vector<numtype>> u2ls;
	std::vector<std::vector<etype>> e_orders;
	std::vector<std::vector<bool>> e_is_trees;

	std::vector<std::vector<numtype>> num_backward_neighbors;
	std::vector<std::vector<std::vector<vtype>>> backward_neighbors; // backward_neighbors[order_id][v] is a vector.

	OrderCPU();
	OrderCPU(numtype v_num_orders);
	~OrderCPU();

	void init_roots(
			std::vector<vtype> &roots);

	void getEdgeOrderBFS(
			cpuGraph *query_graph);

	void getVertexOrderBFS(
			cpuGraph *query_graph);

	void constructBackwardNeighbors(
			cpuGraph *query_graph);
};

class OrderGPU
{
public:
	int *num_orders;
	vtype *roots_;		// size: num_orders
	vtype *v_orders_; // size: num_orders * NUM_VQ
	int *u2ls_;				// size: num_orders * NUM_VQ

	numtype *num_backward_neighbors_; // size: num_orders * NUM_VQ
	vtype *backward_neighbors_;				// size: num_orders * NUM_VQ * NUM_VQ
	// backward_neighbors[order_id * NUM_VQ * NUM_VQ + v * NUM_VQ + u] = v's u-th backward neighbor.

	OrderGPU();
	OrderGPU(int num_orders);
	// OrderGPU(Order *order_obj);
	~OrderGPU();
};

// void getBFSorder(
// 		cpuGraph *g,
// 		vtype *order,
// 		vtype start_v = 0);

// void getCFLorder(
// 		cpuGraph *g,
// 		vtype *order);

// void getBFSEdgeOrder(
// 		cpuGraph *g,
// 		etype *order,
// 		// cpuRelation *cpu_relations_,
// 		vtype start_v = 0);

#endif // ORDER_H