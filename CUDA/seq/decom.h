#pragma once
#ifndef DECOM_H
#define DECOM_H

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <unordered_set>
#include <fstream>

// typedef unsigned int vid_t;
typedef int vid_t;
typedef int num_t;

struct lrval_index_block {
	std::vector<vid_t> nodeset;
	lrval_index_block* next = NULL;
};

struct lrval_index_block_dual_pointer {
	std::unordered_set<vid_t> nodeset;
	lrval_index_block_dual_pointer* horizontal_pointer = NULL;
	lrval_index_block_dual_pointer* vertical_pointer = NULL;
};

class Edge
{
public:
	Edge(int u_, int v_) { u = u_; v = v_; }
	bool operator<(const Edge &other) const
	{
		if (u == other.u)
			return v < other.v;
		return u < other.u;
	}

	int u;
	int v;
};

class DegreeNode
{
public:
	int id;
	int degree;
};

class BiGraph
{

public:

	BiGraph(std::string dir);
	BiGraph();
	~BiGraph() {}

	void addEdge(vid_t u, vid_t v);
	num_t getV1Num() { return num_v1; }
	num_t getV2Num() { return num_v2; }
	num_t getV1Degree(vid_t u) { return degree_v1[u]; }
	num_t getV2Degree(vid_t u) { return degree_v2[u]; }
	std::vector<vid_t> & getV2Neighbors(vid_t u) { return neighbor_v2[u]; }
	std::vector<vid_t> & getV1Neighbors(vid_t u) { return neighbor_v1[u]; }

public:

	void init(unsigned int num_v1, unsigned int num_v2);
	void loadGraph(std::string dir);

	std::string dir;
	num_t num_v1;
	num_t num_v2;
	num_t num_edges;

	std::vector<std::vector<vid_t>> neighbor_v1;
	std::vector<std::vector<vid_t>> neighbor_v2;

	std::vector<int> degree_v1;
	std::vector<int> degree_v2;

	std::vector<std::vector<int>> left_index;
	std::vector<std::vector<int>> right_index;
	int v1_max_degree;
	int v2_max_degree;
	std::vector<bool> left_delete;
	std::vector<bool> right_delete;

};

extern void build_lrval_index(BiGraph&g, std::vector<std::vector<lrval_index_block*>>& lrval_index_u, std::vector<std::vector<lrval_index_block*>>& lrval_index_v);

extern void retrieve_via_lrval_index(BiGraph& g, std::vector<std::vector<lrval_index_block*>>& lrval_index_u, std::vector<std::vector<lrval_index_block*>>& lrval_index_v,
	std::vector<bool>& left_node, std::vector<bool>& right_node, int l_val, int r_val);

extern void retrieve_via_lrval_index_check(BiGraph& g, std::vector<std::vector<lrval_index_block*>>& lrval_index_u, std::vector<std::vector<lrval_index_block*>>& lrval_index_v,
	std::vector<bool>& left_node, std::vector<bool>& right_node, int l_val, int r_val, int & flag);

extern void loadQuery(std::string dir, std::vector<std::vector<int>>& queryStream, int& n_query);

extern void inv(BiGraph& g);

extern void lrIndexBasic(BiGraph& g);

extern void __leftCopyPeel(int left, BiGraph& g);


#endif // !GEPHI_H