// MSBD5009 SPRING2023 HW3

// Sequentail ver.:
	// Compile:
	// g++ -std=c++11 -g decom.h decom.cpp main.cpp -o lrds
	// Run:
	// ./lrds -BasicDecom ../data/<DATASET>/
	// ./lrds -Query ../data/<DATASET>/


#include <ctime>
#include <cmath>
#include <iostream>
#include <chrono>
#include "decom.h"


using namespace std;

const int Q_MAX = 10000000;

int main(int argc, char **argv) {
	if (argc == 1) {
		cout << "error in number of arguments" << endl;
	}
	string exec_type = argv[1];
	if (exec_type == "-BasicDecom") {
		cout << "start BasicDecom for " << argv[2] << endl;
		auto start = chrono::system_clock::now();
		BiGraph g(argv[2]);
		lrIndexBasic(g);
		auto end = chrono::system_clock::now();
		chrono::duration<double> time = end - start;
		cout << "run time: " << time.count() << endl;
	}
	else if (exec_type == "-Query") {
		cout << "start sequential query for " << argv[2] << endl;
		auto start = chrono::system_clock::now();

		BiGraph g(argv[2]);
		lrIndexBasic(g);
		vector<vector<lrval_index_block*>> lrval_index_u; vector<vector<lrval_index_block*>> lrval_index_v;
		build_lrval_index(g, lrval_index_u, lrval_index_v);
		vector<bool> left; vector<bool> right;
		// all the vertices in query result are set as true
		vector<vector<int>> queryStream;
		queryStream.resize(Q_MAX);
		int n_query = 0;
		loadQuery(argv[2], queryStream, n_query);
		queryStream.resize(n_query);
		int queryAns[n_query*3];
		
		for(int i = 0; i<n_query; i++){
			int flag = 0;
			left.resize(g.num_v1, false); right.resize(g.num_v2, false);
			int lval = queryStream[i][0];
			int rval = queryStream[i][1];
			retrieve_via_lrval_index_check(g, lrval_index_u, lrval_index_v, left, right, lval, rval, flag);
			
			queryAns[i*3] = lval;
			queryAns[i*3+1] = rval;
			queryAns[i*3+2] = flag;

			
		}
		
		cout << "queryAns: ";
		for(int i = 0; i< n_query*3; i++){
			cout << queryAns[i] << " ";
		}
		cout << endl;
		
		auto end = chrono::system_clock::now();
		chrono::duration<double> time = end - start;
		
		cout << "Sequential query time: " << time.count() << endl;
	}
	
	else {
		cout << "illegal arguments" << endl;
	}
	return 0;
}