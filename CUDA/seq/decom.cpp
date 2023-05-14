#include "decom.h"
#include <algorithm>
#include <ctime>
#include <iostream>
#include <set>
#include <cmath>
#include <list>

using namespace std;

void crossUpdate(BiGraph& g, int left_k, int k_x, vid_t v) {
	for (int right_k = 1; right_k < g.right_index[v].size(); right_k++) {
		if (right_k <= k_x) {
			if (g.right_index[v][right_k] < left_k) {
				g.right_index[v][right_k] = left_k;
			}
		}
		else {
			break;
		}
	}
}

bool compare_deg_array(const pair<int, int>& p1, const pair<int, int>& p2) {
	return p1.second < p2.second;
}

BiGraph::BiGraph(string dir)
{
	num_v1 = 0;
	num_v2 = 0;
	num_edges = 0;

	neighbor_v1.clear();
	neighbor_v2.clear();

	degree_v1.clear();
	degree_v2.clear();

	// index left (x,*) right (*,x)
	left_index.clear();
	right_index.clear();
	v1_max_degree = 0;
	v2_max_degree = 0;

	this->dir = dir;
	loadGraph(dir);
}

BiGraph::BiGraph() {
	dir = "";
	num_v1 = 0;
	num_v2 = 0;
	num_edges = 0;

	neighbor_v1.clear();
	neighbor_v2.clear();

	degree_v1.clear();
	degree_v2.clear();

	left_index.clear();
	right_index.clear();
	v1_max_degree = 0;
	v2_max_degree = 0;
}

void BiGraph::loadGraph(string dir)
{
	unsigned int n1, n2;
	unsigned int edges = 0;
	int u, v;
	int r;

	string metaFile = dir + "graph.meta";
	string edgeFile = dir + "graph.e";

	FILE * metaGraph = fopen(metaFile.c_str(), "r");
	FILE * edgeGraph = fopen(edgeFile.c_str(), "r");

	if (fscanf(metaGraph, "%d\n%d", &n1, &n2) != 2)
	{
		fprintf(stderr, "Bad file format: n1 n2 incorrect\n");
		exit(1);
	}

	// fprintf(stdout, "n1: %d, n2: %d\n", n1, n2);

	init(n1, n2);

	while ((r = fscanf(edgeGraph, "%d %d", &u, &v)) != EOF)
	{
		//fprintf(stderr, "%d, %d\n", u, v);
		if (r != 2)
		{
			fprintf(stderr, "Bad file format: u v incorrect\n");
			exit(1);
		}

		addEdge(u, v);
		//num_edges++;
	}

	fclose(metaGraph);
	fclose(edgeGraph);

	for (int i = 0; i < num_v1; ++i)
	{
		neighbor_v1[i].shrink_to_fit();
		sort(neighbor_v1[i].begin(), neighbor_v1[i].end());

	}
	for (int i = 0; i < num_v2; ++i)
	{
		neighbor_v2[i].shrink_to_fit();
		sort(neighbor_v2[i].begin(), neighbor_v2[i].end());
	}

}

void loadQuery(string dir, vector<vector<int>>& queryStream, int &line)
{
	int r, lval, rval;
	string queryFile = dir + "querystream.txt";
	FILE * queryVec = fopen(queryFile.c_str(), "r");
	line = 0;
	while ((r = fscanf(queryVec, "%d %d", &lval, &rval)) != EOF)
	{
		if (r != 2)
		{
			fprintf(stderr, "Bad file format: u v incorrect\n");
			exit(1);
		}
		queryStream[line].resize(2);
		queryStream[line][0] = lval;
		queryStream[line][1] = rval;
		line++;
	}
	// cout<<"line: " << line;

	fclose(queryVec);
}

void BiGraph::init(unsigned int num1, unsigned int num2)
{
	num_v1 = num1;
	num_v2 = num2;
	num_edges = 0;

	neighbor_v1.resize(num_v1);
	neighbor_v2.resize(num_v2);

	degree_v1.resize(num_v1);
	degree_v2.resize(num_v2);

	fill_n(degree_v1.begin(), num_v1, 0);
	fill_n(degree_v2.begin(), num_v2, 0);

	left_delete.resize(num_v1);
	right_delete.resize(num_v2);
}

void BiGraph::addEdge(vid_t u, vid_t v)
{

	neighbor_v1[u].push_back(v);
	++degree_v1[u];
	if (degree_v1[u] > v1_max_degree) v1_max_degree = degree_v1[u];
	neighbor_v2[v].push_back(u);
	++degree_v2[v];
	if (degree_v2[v] > v2_max_degree) v2_max_degree = degree_v2[v];
	num_edges++;
}

void inv(BiGraph& g) {
	swap(g.degree_v1, g.degree_v2);
	swap(g.left_index, g.right_index);
	swap(g.num_v1, g.num_v2);
	swap(g.neighbor_v1, g.neighbor_v2);
	swap(g.v1_max_degree, g.v2_max_degree);
	swap(g.left_delete, g.right_delete);
}

void __leftCopyPeel(int left_k, BiGraph& g) {
	vector<bool> left_deletion_next_round;
	vector<bool> right_deletion_next_round;
	vector<int> left_degree_next_round;
	vector<int> right_degree_next_round;
	vector<vid_t> left_vertices_to_be_peeled;
	vector<vid_t> right_vertices_to_be_peeled;
	// definition of dyn array
	vector<int> right_k_x_index; right_k_x_index.resize(g.v2_max_degree + 2);
	fill_n(right_k_x_index.begin(), right_k_x_index.size(), -1); right_k_x_index[0] = 0; right_k_x_index[g.v2_max_degree + 1] = g.num_v2;
	vector<int> right_vertices_index; right_vertices_index.resize(g.num_v2);
	vector<pair<int, int>> degree_array; degree_array.resize(g.num_v2);
	// end of definition
	for (vid_t u = 0; u < g.getV1Num(); u++) {
		if (g.degree_v1[u] < left_k && !g.left_delete[u]) {
			left_vertices_to_be_peeled.push_back(u);
		}
	}
	// initialize dyn array
	for (vid_t v = 0; v < g.num_v2; v++) {
		degree_array[v] = make_pair(v, g.degree_v2[v]);
	}
	sort(degree_array.begin(), degree_array.end(), compare_deg_array);
	for (int i = 0; i < degree_array.size(); i++) {
		if (right_k_x_index[degree_array[i].second] == -1) {
			right_k_x_index[degree_array[i].second] = i;
		}
		right_vertices_index[degree_array[i].first] = i;
	}
	for (int i = right_k_x_index.size() - 1; i >= 0; i--) {
		if (right_k_x_index[i] == -1) {
			right_k_x_index[i] = right_k_x_index[i + 1];
		}
	}
	// end of initialization
	for (int right_k = 1; right_k <= g.v2_max_degree + 1; right_k++) {
		if (right_k_x_index[right_k - 1] == g.num_v2) break;
		for (int i = right_k_x_index[right_k - 1]; i < right_k_x_index[right_k]; i++) {
			right_vertices_to_be_peeled.push_back(degree_array[i].first);
		}
		while (!left_vertices_to_be_peeled.empty() || !right_vertices_to_be_peeled.empty()) {
			// peel left
			for (auto j = left_vertices_to_be_peeled.begin(); j != left_vertices_to_be_peeled.end(); j++) {
				vid_t u = *j;
				if (g.left_delete[u]) continue;
				for (int k = 0; k < g.neighbor_v1[u].size(); k++) {
					vid_t v = g.neighbor_v1[u][k];
					if (g.right_delete[v]) continue;
					g.degree_v2[v]--;
					g.degree_v1[u]--;
					if (g.getV2Degree(v) == 0 && right_k - 1 > 0) {
						crossUpdate(g, left_k, right_k - 1, v);
						g.right_delete[v] = true;
					}
					if (g.degree_v2[v] < right_k && !g.right_delete[v]) {
						right_vertices_to_be_peeled.push_back(v);
					}
					if (g.degree_v2[v] >= right_k - 1) {
						int olddegree = g.degree_v2[v] + 1;
						vid_t stack = degree_array[right_k_x_index[olddegree]].first;
						swap(degree_array[right_k_x_index[olddegree]], degree_array[right_vertices_index[v]]);
						swap(right_vertices_index[stack], right_vertices_index[v]);
						right_k_x_index[olddegree]++;
					}
				}
				g.left_delete[u] = true;
				if (right_k - 1 > 0) {
					g.left_index[u][left_k] = right_k - 1;
				}
			}
			left_vertices_to_be_peeled.clear();
			// peel right
			for (auto j = right_vertices_to_be_peeled.begin(); j != right_vertices_to_be_peeled.end(); j++) {
				vid_t v = *j;
				if (g.right_delete[v]) continue;
				for (int k = 0; k < g.neighbor_v2[v].size(); k++) {
					vid_t u = g.neighbor_v2[v][k];
					if (g.left_delete[u]) continue;
					g.degree_v2[v]--;
					g.degree_v1[u]--;
					if (g.getV1Degree(u) == 0 && right_k - 1 > 0) {
						g.left_index[u][left_k] = right_k - 1;
						g.left_delete[u] = true;
					}
					if (g.degree_v1[u] < left_k && !g.left_delete[u]) {
						left_vertices_to_be_peeled.push_back(u);
					}
				}
				g.right_delete[v] = true;
				if (right_k - 1 > 0) {
					crossUpdate(g, left_k, right_k - 1, v);
				}
			}
			right_vertices_to_be_peeled.clear();
		}
		if (right_k == 1) {
			left_degree_next_round = g.degree_v1;
			right_degree_next_round = g.degree_v2;
			left_deletion_next_round = g.left_delete;
			right_deletion_next_round = g.right_delete;
		}
	}
	g.degree_v1 = left_degree_next_round;
	g.degree_v2 = right_degree_next_round;
	g.left_delete = left_deletion_next_round;
	g.right_delete = right_deletion_next_round;
	g.v1_max_degree = 0;
	g.v2_max_degree = 0;
	for (vid_t u = 0; u < g.degree_v1.size(); u++) {
		if (g.v1_max_degree < g.degree_v1[u]) g.v1_max_degree = g.degree_v1[u];
	}
	for (vid_t v = 0; v < g.degree_v2.size(); v++) {
		if (g.v2_max_degree < g.degree_v2[v]) g.v2_max_degree = g.degree_v2[v];
	}
}

void lrIndexBasic(BiGraph& g) {
	// initialize_bigraph(g);

	int left_degree_max = 0;
	int m = 0;
	for (int i = 0; i < g.getV1Num(); i++) {
		if (left_degree_max < g.getV1Degree(i)) left_degree_max = g.getV1Degree(i);
		m += g.getV1Degree(i);
	}
	int right_degree_max = 0;
	for (int i = 0; i < g.getV2Num(); i++) {
		if (right_degree_max < g.getV2Degree(i)) right_degree_max = g.getV2Degree(i);
	}
	// init g's max degree and index
	g.v1_max_degree = left_degree_max;
	g.v2_max_degree = right_degree_max;
	g.left_index.resize(g.getV1Num());
	g.right_index.resize(g.getV2Num());
	g.left_delete.resize(g.getV1Num());
	g.right_delete.resize(g.getV2Num());
	fill_n(g.left_delete.begin(), g.left_delete.size(), false);
	fill_n(g.right_delete.begin(), g.right_delete.size(), false);
	for (int i = 0; i < g.getV1Num(); i++) {
		g.left_index[i].resize(g.getV1Degree(i) + 1);
		fill_n(g.left_index[i].begin(), g.left_index[i].size(), 0);
	}
	for (int i = 0; i < g.getV2Num(); i++) {
		g.right_index[i].resize(g.getV2Degree(i) + 1);
		fill_n(g.right_index[i].begin(), g.right_index[i].size(), 0);
	}


	for (int left_k = 1; left_k <= g.v1_max_degree; left_k++) {
		__leftCopyPeel(left_k, g);
	}

	// restore g
	fill_n(g.left_delete.begin(), g.left_delete.size(), false);
	g.v1_max_degree = left_degree_max;
	for (vid_t u = 0; u < g.num_v1; u++) {
		g.degree_v1[u] = g.neighbor_v1[u].size();
	}
	fill_n(g.right_delete.begin(), g.right_delete.size(), false);
	g.v2_max_degree = right_degree_max;
	for (vid_t v = 0; v < g.num_v2; v++) {
		g.degree_v2[v] = g.neighbor_v2[v].size();
	}

	inv(g);
	for (int left_k = 1; left_k <= g.v1_max_degree; left_k++) {
		__leftCopyPeel(left_k, g);
	}
	inv(g);

	// restore g
	fill_n(g.left_delete.begin(), g.left_delete.size(), false);
	g.v1_max_degree = left_degree_max;
	for (vid_t u = 0; u < g.num_v1; u++) {
		g.degree_v1[u] = g.neighbor_v1[u].size();
	}
	fill_n(g.right_delete.begin(), g.right_delete.size(), false);
	g.v2_max_degree = right_degree_max;
	for (vid_t v = 0; v < g.num_v2; v++) {
		g.degree_v2[v] = g.neighbor_v2[v].size();
	}
}

void build_lrval_index(BiGraph&g, vector<vector<lrval_index_block*>>& lrval_index_u, vector<vector<lrval_index_block*>>& lrval_index_v) {
	lrval_index_u.clear(); lrval_index_u.resize(g.v1_max_degree + 1); lrval_index_v.clear(); lrval_index_v.resize(g.v2_max_degree + 1);
	// build left
	vector<int> r_val_m; r_val_m.resize(g.v1_max_degree + 1);
	for (vid_t u = 0; u < g.num_v1; u++) {
		for (int l_val = 1; l_val < g.left_index[u].size(); l_val++) {
			int r_val = g.left_index[u][l_val];
			if (r_val_m[l_val] < r_val) r_val_m[l_val] = r_val;
		}
	}
	for (int l_val = 1; l_val < r_val_m.size(); l_val++) {
		lrval_index_u[l_val].resize(r_val_m[l_val] + 1);
		for (int kkk = 1; kkk < lrval_index_u[l_val].size(); kkk++) {
			lrval_index_u[l_val][kkk] = new lrval_index_block;
		}
	}
	for (vid_t u = 0; u < g.num_v1; u++) {
		for (int l_val = 1; l_val < g.left_index[u].size(); l_val++) {
			int r_val = g.left_index[u][l_val];
			lrval_index_u[l_val][r_val]->nodeset.push_back(u);
		}
	}
	for (int l_val = 1; l_val < lrval_index_u.size(); l_val++) {
		lrval_index_block* pre = NULL;
		for (int r_val = lrval_index_u[l_val].size() - 1; r_val > 0; r_val--) {
			lrval_index_u[l_val][r_val]->next = pre;
			if (lrval_index_u[l_val][r_val]->nodeset.size() > 0) pre = lrval_index_u[l_val][r_val];
		}
	}
	// build right
	vector<int> l_val_m; l_val_m.resize(g.v2_max_degree + 1);
	for (vid_t v = 0; v < g.num_v2; v++) {
		for (int r_val = 1; r_val < g.right_index[v].size(); r_val++) {
			int l_val = g.right_index[v][r_val];
			if (l_val_m[r_val] < l_val) l_val_m[r_val] = l_val;
		}
	}
	for (int r_val = 1; r_val < l_val_m.size(); r_val++) {
		lrval_index_v[r_val].resize(l_val_m[r_val] + 1);
		for (int kkk = 1; kkk < lrval_index_v[r_val].size(); kkk++) {
			lrval_index_v[r_val][kkk] = new lrval_index_block;
		}
	}
	for (vid_t v = 0; v < g.num_v2; v++) {
		for (int r_val = 1; r_val < g.right_index[v].size(); r_val++) {
			int l_val = g.right_index[v][r_val];
			lrval_index_v[r_val][l_val]->nodeset.push_back(v);
		}
	}
	for (int r_val = 1; r_val < lrval_index_v.size(); r_val++) {
		lrval_index_block* pre = NULL;
		for (int l_val = lrval_index_v[r_val].size() - 1; l_val > 0; l_val--) {
			lrval_index_v[r_val][l_val]->next = pre;
			if (lrval_index_v[r_val][l_val]->nodeset.size() > 0) pre = lrval_index_v[r_val][l_val];
		}
	}
}

void retrieve_via_lrval_index(BiGraph& g, vector<vector<lrval_index_block*>>& lrval_index_u, vector<vector<lrval_index_block*>>& lrval_index_v,
	vector<bool>& left_node, vector<bool>& right_node, int l_val, int r_val) {
	//left_node.clear(); right_node.clear();
	//left_node.resize(g.num_v1); right_node.resize(g.num_v2);
	fill_n(left_node.begin(), left_node.size(), false);
	fill_n(right_node.begin(), right_node.size(), false);
	if (lrval_index_u.size() <= l_val) return;
	if (lrval_index_u[l_val].size() <= r_val) return;
	lrval_index_block* block = lrval_index_u[l_val][r_val];
	while (block != NULL) {
		for (auto i = block->nodeset.begin(); i != block->nodeset.end(); i++) {
			left_node[*i] = true;
		}
		block = block->next;
	}
	block = lrval_index_v[r_val][l_val];
	while (block != NULL) {
		for (auto i = block->nodeset.begin(); i != block->nodeset.end(); i++) {
			right_node[*i] = true;
		}
		block = block->next;
	}
}

void retrieve_via_lrval_index_check(BiGraph& g, vector<vector<lrval_index_block*>>& lrval_index_u, vector<vector<lrval_index_block*>>& lrval_index_v,
	vector<bool>& left_node, vector<bool>& right_node, int l_val, int r_val, int & flag) {

	if (lrval_index_u.size() <= l_val || lrval_index_u[l_val].size() <= r_val){
		flag = 0;
	} else {
		flag = 1;
	}
	
}

