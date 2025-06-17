#include "cpuGraph.h"
#include "globals.cuh"

#include <iostream>
#include <cstring>
#include <set>

cpuGraph::cpuGraph() : vve()
{
	num_v = 0;
	num_e = 0;
	largest_l = 0;
	// elCount = 0;
	maxDegree = 0;

	outdeg_ = nullptr;
	vLabels_ = nullptr;
	maxLabelFreq = 0;
	// eLabels = nullptr;

	vertexIDs_ = nullptr;
	offsets_ = nullptr;
	neighbors_ = nullptr;
	edgeIDs_ = nullptr;
	isQuery = true;
	keep = nullptr;
}

cpuGraph::~cpuGraph()
{
	if (outdeg_ != nullptr)
		delete[] outdeg_;
	if (vLabels_ != nullptr)
		delete[] vLabels_;
	if (vertexIDs_ != nullptr)
		delete[] vertexIDs_;
	if (offsets_ != nullptr)
		delete[] offsets_;
	if (neighbors_ != nullptr)
		delete[] neighbors_;
	if (edgeIDs_ != nullptr)
		delete[] edgeIDs_;
	if (keep != nullptr)
		delete[] keep;
}

uint32_t cpuGraph::get_u_off(vtype u)
{
	uint32_t u_off = UINT32_MAX;
	for (uint32_t i = 0; i < num_v; i++)
	{
		if (vertexIDs_[i] == u)
		{
			u_off = i;
			break;
		}
	}

	if (u_off == UINT32_MAX)
	{
		std::cerr << "ERROR! in get_u_off(): u_off = uint32max" << std::endl;
		exit(EXIT_FAILURE);
	}

	return u_off;
}

void cpuGraph::Print()
{
	std::cout << "============================\n";
	std::cout << "num_v: " << num_v << std::endl;
	std::cout << "num_e: " << num_e << std::endl;
	std::cout << "largest_l: " << largest_l << std::endl;
	std::cout << "maxDegree: " << maxDegree << std::endl;

	std::cout << "outdeg_: \n";
	for (int i = 0; i < num_v; ++i)
		std::cout << outdeg_[i] << " \n"[i == num_v - 1];
	std::cout << "vLabels: \n";
	for (int i = 0; i < num_v; ++i)
		std::cout << vLabels_[i] << " \n"[i == num_v - 1];
	std::cout << "maxLabelFreq: " << maxLabelFreq << std::endl;

	std::cout << "vertexIDs_: \n";
	for (int i = 0; i < num_v; ++i)
		std::cout << vertexIDs_[i] << " \n"[i == num_v - 1];
	std::cout << "offsets_: \n";
	for (int i = 0; i < num_v + 1; ++i)
		std::cout << offsets_[i] << " \n"[i == num_v];
	std::cout << "neighbors_: \n";
	for (int i = 0; i < num_e * 2; ++i)
		std::cout << neighbors_[i] << " \n"[i == num_e * 2 - 1];
	std::cout << "edgeIDs_: \n";
	for (int i = 0; i < num_e * 2; ++i)
		std::cout << edgeIDs_[i] << " \n"[i == num_e * 2 - 1];
	std::cout << "============================" << std::endl;
}

// void updateHostGraph(cpuGraph &graph, std::vector<change> &batch)
// {

// 	int sz = batch.size();
// 	for (int i = 0; i < sz; ++i)
// 	{
// 		change c = batch[i];
// 		std::swap(c.src, c.dst);
// 		batch.push_back(c);
// 	}

// 	std::sort(batch.begin(), batch.end(),
// 						[](change &a, change &b)
// 						{
// 		if (a.src != b.src)
// 			return a.src < b.src;
// 		return a.dst < b.dst; });

// 	std::vector<vtype> new_nbrs;
// 	new_nbrs.assign(graph.neighbors_, graph.neighbors_ + graph.num_e * 2);

// 	bool a_flip = true;
// 	bool d_flip = true;
// 	for (int i = 0; i < batch.size(); ++i)
// 	{
// 		vtype &src = batch[i].src;
// 		vtype &dst = batch[i].dst;
// 		if (batch[i].add)
// 		{
// 			int start = graph.offsets_[src];
// 			int end = graph.offsets_[src + 1];
// 			auto it = std::lower_bound(new_nbrs.begin() + start, new_nbrs.begin() + end, dst);

// 			if (it == new_nbrs.begin() + end || *it != dst)
// 			{
// 				int pos = it - new_nbrs.begin();
// 				new_nbrs.insert(it, dst);
// 				if (a_flip)
// 				{
// 					graph.num_e++;
// 					a_flip = false;
// 				}
// 				else
// 				{
// 					a_flip = true;
// 				}
// 				graph.outdeg_[src]++;
// 				for (int j = src + 1; j <= graph.num_v; ++j)
// 					graph.offsets_[j]++;
// 			}
// 		}
// 		else
// 		{
// 			int start = graph.offsets_[src];
// 			int end = graph.offsets_[src + 1];
// 			auto it = std::lower_bound(new_nbrs.begin() + start, new_nbrs.begin() + end, dst);

// 			if (it != new_nbrs.begin() + end && *it == dst)
// 			{
// 				int pos = it - new_nbrs.begin();
// 				new_nbrs.erase(it);
// 				if (d_flip)
// 				{
// 					graph.num_e--;
// 					d_flip = false;
// 				}
// 				else
// 				{
// 					d_flip = true;
// 				}
// 				graph.outdeg_[src]--;
// 				for (int j = src + 1; j <= graph.num_v; ++j)
// 					graph.offsets_[j]--;
// 			}
// 		}
// 	}

// 	for (int i = 0; i < graph.num_v; ++i)
// 		graph.maxDegree = std::max(graph.maxDegree, graph.outdeg_[i]);

// 	delete[] graph.neighbors_;
// 	graph.neighbors_ = new vtype[graph.num_e * 2];
// 	memcpy(graph.neighbors_, new_nbrs.data(), sizeof(vtype) * graph.num_e * 2);
// }

void updateHostGraph(cpuGraph &graph, std::vector<change> &batch)
{
	// 复制原始 batch 以保留原始请求
	std::vector<change> temp_batch = batch;
	temp_batch.reserve(batch.size() * 2);

	// 生成反向边
	for (const auto c : batch)
	{
		change rev_c = c;
		std::swap(rev_c.src, rev_c.dst);
		temp_batch.push_back(rev_c);
	}

	// 按 src 和 dst 排序
	std::sort(temp_batch.begin(), temp_batch.end(),
						[](const change &a, const change &b)
						{
							if (a.src != b.src)
								return a.src < b.src;
							return a.dst < b.dst;
						});

	// 初始化临时邻接表
	std::vector<std::set<int>> temp_adj(graph.num_v);
	for (int i = 0; i < graph.num_v; ++i)
	{
		int start = graph.offsets_[i];
		int end = graph.offsets_[i + 1];

		temp_adj[i].insert(graph.neighbors_ + start, graph.neighbors_ + end);

		// for (int j = start; j < end; ++j)
		// {
		// 	temp_adj[i].insert(graph.neighbors_[j]);
		// }
	}

	// 处理每条边
	for (const auto &c : temp_batch)
	{
		if (c.add)
		{
			temp_adj[c.src].insert(c.dst);
			// temp_adj[c.dst].insert(c.src);
		}
		else
		{
			temp_adj[c.src].erase(c.dst);
			// temp_adj[c.dst].erase(c.src);
		}
	}

	// 生成新的 neighbors 和 offsets
	std::vector<int> new_neighbors;
	std::vector<int> new_offsets(graph.num_v + 1, 0);
	int current_offset = 0;

	for (int i = 0; i < graph.num_v; ++i)
	{
		new_offsets[i] = current_offset;
		for (auto neighbor : temp_adj[i])
		{
			new_neighbors.push_back(neighbor);
		}
		current_offset += temp_adj[i].size();
	}
	new_offsets[graph.num_v] = current_offset;

	// 更新图的属性
	graph.num_e = new_neighbors.size() / 2; // 假设图是无向的
	graph.neighbors_ = new vtype[new_neighbors.size()];
	memcpy(graph.neighbors_, new_neighbors.data(), sizeof(uint32_t) * new_neighbors.size());

	// delete[] graph.offsets_;
	// graph.offsets_ = new vtype[graph.num_v + 1];
	memcpy(graph.offsets_, new_offsets.data(), sizeof(int) * (graph.num_v + 1));

	// graph.outdeg_.resize(graph.num_v);
	graph.maxDegree = 0;
	for (int i = 0; i < graph.num_v; ++i)
	{
		graph.outdeg_[i] = temp_adj[i].size();
		graph.maxDegree = std::max(graph.maxDegree, graph.outdeg_[i]);
	}
}