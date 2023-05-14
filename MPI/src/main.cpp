// PAN Han, student id:20881280
// email:hpanan@connect.ust.hk

// MSBD5009/DSAA5015 SPRING2023 HW1

// Sequentail ver.:
	// Compile:
	// mpic++ -std=c++11 -g decom.h decom.cpp main.cpp -o lrds
	// Run:
	// ./lrds -BasicDecom ../data/<DATASET>/
	// ./lrds -Query ../data/<DATASET>/

// MPI ver.:
	// Compile:
	// mpic++ -std=c++11 -g decom.h decom.cpp main.cpp -o lrds
	// Run:
	// mpiexec -n num_threads --oversubscribe ./lrds -ParallelQuery ../data/<DATASET>/


#include <ctime>
#include <cmath>
#include <iostream>
#include <chrono>
#include "decom.h"
#include "mpi.h"
#include "string.h"

using namespace std;

const int Q_MAX = 100000;

int main(int argc, char **argv) {
	if (argc == 1) {
		cout << "error in number of arguments" << endl;
	}
	string exec_type = argv[1];
	if (exec_type == "-BasicDecom") {
		cout << "start BasicDecom for " << argv[2] << endl;
		BiGraph g(argv[2]);
		auto start = chrono::system_clock::now();
		lrIndexBasic(g);
		auto end = chrono::system_clock::now();
		chrono::duration<double> time = end - start;
		cout << "run time: " << time.count() << endl;
	}
	else if (exec_type == "-Query") {
		cout << "start sequential query for " << argv[2] << endl;
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
		int queryAns[n_query*4];
		auto start = chrono::system_clock::now();
		for(int i = 0; i<n_query; i++){
			left.resize(g.num_v1, false); right.resize(g.num_v2, false);
			int lval = queryStream[i][0];
			int rval = queryStream[i][1];
			retrieve_via_lrval_index(g, lrval_index_u, lrval_index_v, left, right, lval, rval);
			int nl = 0, nr = 0;
			for(auto e: left){
				if(e) nl++;
			}
			for (auto e: right){
				if(e) nr++;
			}
			// cout << "number of left in the answer: " << nl << ", " << "number of right in the answer: " << nr << endl;
			queryAns[i*4] = lval;
			queryAns[i*4+1] = rval;
			queryAns[i*4+2] = nl;
			queryAns[i*4+3] = nr;
			
		}
		
		cout << "queryAns: ";
		for(int i = 0; i< n_query*4; i++){
			cout << queryAns[i] << " ";
		}
		cout << endl;
		
		auto end = chrono::system_clock::now();
		chrono::duration<double> time = end - start;
		
		cout << "Sequential query time: " << time.count() << endl;
	}
	else if (exec_type == "-ParallelQuery") {
		
		MPI_Init(&argc, &argv);
		MPI_Comm comm;
		int num_process; // number of processors
		int my_rank;     // my global rank

		comm = MPI_COMM_WORLD;

		MPI_Comm_size(comm, &num_process);
		MPI_Comm_rank(comm, &my_rank);

		cout << "start ParallelQuery for " << argv[2] << "at the process " << my_rank << endl;

		if (argc != 3) {
			std::cerr << "usage: mpiexec -n num_threads --oversubscribe ./lrds -ParallelQuery ../data/<DATASET>/"
					<< std::endl;
			return -1;
		}

		int final_num_v1 = 0;
		int final_num_v2 = 0;
		int final_num_edges = 0;
		int final_v1_maxdeg = 0;
		int final_v2_maxdeg = 0;
		int *deg_v1=nullptr;
		int *deg_v2=nullptr;
		int *left_index_long=nullptr;
		int *right_index_long=nullptr;

		vector<vector<int>> final_left_index;
		vector<vector<int>> final_right_index;

		if(my_rank==0){
			BiGraph g(argv[2]);
			final_num_v1 = g.num_v1;
			final_num_v2 = g.num_v2;
			final_num_edges = g.num_edges;
		}

		MPI_Bcast(&final_num_v1, 1, MPI_INT, 0 , MPI_COMM_WORLD);
		MPI_Bcast(&final_num_v2, 1, MPI_INT, 0 , MPI_COMM_WORLD);
		MPI_Bcast(&final_num_edges, 1, MPI_INT, 0 , MPI_COMM_WORLD);

		deg_v1=(int *)calloc(final_num_v1, sizeof(int));
		deg_v2=(int *)calloc(final_num_v2, sizeof(int));

		left_index_long=(int *)calloc(final_num_v1 + final_num_edges, sizeof(int));
		right_index_long=(int *)calloc(final_num_v2 + final_num_edges, sizeof(int));

		

		if(my_rank==0){
			BiGraph g(argv[2]);
			lrIndexBasic(g);
			final_v1_maxdeg = g.v1_max_degree;
			final_v2_maxdeg = g.v2_max_degree;

			// change g.degree_v1 (std::vector) to c style array for MPI
			for (int i = 0; i < g.num_v1; i++){
				deg_v1[i] = g.degree_v1[i];
			}
			for (int i = 0; i < g.num_v2; i++){
				deg_v2[i] = g.degree_v2[i];
			}

			// change g.left_index (vector<vector<int>>) to a one dimen long array (left_index_long) for MPI
			final_left_index.resize(final_num_v1);
			for (int i = 0; i < g.getV1Num(); i++) {
				final_left_index[i].resize(g.getV1Degree(i) + 1);
				fill_n(final_left_index[i].begin(), final_left_index[i].size(), 0);
			}
			final_right_index.resize(final_num_v2);
			for (int i = 0; i < g.getV2Num(); i++) {
				final_right_index[i].resize(g.getV2Degree(i) + 1);
				fill_n(final_right_index[i].begin(), final_right_index[i].size(), 0);
			}

			final_left_index = g.left_index;
			final_right_index = g.right_index;
	
			int cur_i = 0;
			for (int i = 0; i < final_num_v1; i++){
				for (int j = 0; j < final_left_index[i].size(); j++){
					left_index_long[cur_i + j] = final_left_index[i][j];
				}
				cur_i += final_left_index[i].size();
			}

			cur_i = 0;
			for (int i = 0; i < final_num_v2; i++){
				for (int j = 0; j < final_right_index[i].size(); j++){
					right_index_long[cur_i + j] = final_right_index[i][j];
				}
				cur_i += final_right_index[i].size();
			}
		}

		
		BiGraph local_g;

		// BLANK1: make sure the local_g on each process is ready for query. Note that you are not allowed to load graph again (i.e. BiGraph g(argv[2]); lrIndexBasic(g);) on each process.
		// Hint: For answering the query, only the following members are used:
		// num_v1, num_v2, v1_max_degree, v2_max_degree, left_index, right_index; you need to send these members to local_g.

		// BLANK1 - ADD YOUR CODE HERE

		// num_v1, num_v2
		local_g.num_v1 = final_num_v1;
		local_g.num_v2 = final_num_v2;

		// final_v1_maxdeg
		MPI_Bcast(&final_v1_maxdeg, 1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(&final_v2_maxdeg, 1, MPI_INT, 0, MPI_COMM_WORLD);

		local_g.v1_max_degree = final_v1_maxdeg;
		local_g.v2_max_degree = final_v2_maxdeg;

		// degree_v1[i],degree_v2[i]

		MPI_Bcast(deg_v1, final_num_v1, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(deg_v2, final_num_v2, MPI_INT, 0, MPI_COMM_WORLD);

		// deg_v1->g.degree_v1 for getV1Degree(i)
		local_g.degree_v1.resize(local_g.num_v1);
		local_g.degree_v2.resize(local_g.num_v2);

		for (int i = 0; i < local_g.num_v1; i++)
		{
			local_g.degree_v1[i] = deg_v1[i];
			// cout<<my_rank<<" "<<i<<" "<<local_g.degree_v1[i]<<" "<<deg_v1[i]<<endl;
		}
		for (int i = 0; i < local_g.num_v2; i++)
		{
			local_g.degree_v2[i] = deg_v2[i];
		}

		// left_index, right_index

		MPI_Bcast(left_index_long, final_num_v1 + final_num_edges, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Bcast(right_index_long, final_num_v2 + final_num_edges, MPI_INT, 0, MPI_COMM_WORLD);

		// left_index_long -> left_index

		local_g.left_index.resize(final_num_v1);
		int cur_i = 0;
		for (int i = 0; i < local_g.getV1Num(); i++)
		{
			local_g.left_index[i].resize(local_g.getV1Degree(i) + 1);
			for (int j = 0; j < local_g.getV1Degree(i) + 1; j++)
			{
				local_g.left_index[i][j] = left_index_long[cur_i++];
				// cout<<local_g.left_index[i][j]<<" ";
			}
		}

		local_g.right_index.resize(final_num_v2);
		cur_i = 0;
		for (int i = 0; i < local_g.getV2Num(); i++)
		{
			local_g.right_index[i].resize(local_g.getV2Degree(i) + 1);
			for (int j = 0; j < local_g.getV2Degree(i) + 1; j++)
			{
				local_g.right_index[i][j] = right_index_long[cur_i++];
			}
		}

		// BLANK1 - END HERE
		

		// load querystream on each process, you don't need to scatter them.
		vector<vector<int>> queryStream;
		queryStream.resize(Q_MAX);
		int n_query = 0;
		loadQuery(argv[2], queryStream, n_query);
		queryStream.resize(n_query);
		int queryAns[n_query*4];
		
		// build_lrval_index based on local_g
		vector<vector<lrval_index_block*>> lrval_index_u; vector<vector<lrval_index_block*>> lrval_index_v;
		build_lrval_index(local_g, lrval_index_u, lrval_index_v);
		vector<bool> left; vector<bool> right;

		auto start = chrono::system_clock::now();

		// BLANK2: you need to decide which part of queries are answered on each process, and gather all results in the same format as the sequential version (i.e. the c-style array queryAns).  
		// Hint1: As stated above, we load querystream on each process, you don't need to scatter them.
		// Hint2: You may use retrieve_via_lrval_index(local_g, lrval_index_u, lrval_index_v, left, right, lval, rval).
		// Hint3: You may find MPI_Gatherv() is useful.
		
		// BLANK2 - ADD YOUR CODE HERE

		int cur_temp = 0;
		int query_size = n_query / num_process;
		int remainder = n_query % num_process;

		int *displs = nullptr, *recvCount = nullptr;
		displs = (int *)calloc(num_process, sizeof(int));	 // displs
		recvCount = (int *)calloc(num_process, sizeof(int)); // send num
		int begin = 0, curr_begin = 0, cur_query_size = query_size;
		int dis=0;
		for (int i = 0; i < num_process; i++)
		{
			displs[i] = dis;
			if (my_rank == i)
			{
				curr_begin = begin;
			}
			if (remainder != 0 && i < remainder)
			{
				recvCount[i] = (query_size + 1)*2;
				begin += query_size + 1;
			}
			else
			{
				recvCount[i] = query_size*2;
				begin += query_size;
			}
			dis=dis+recvCount[i];
			// cout<<i<<" "<<displs[i]<<" "<<recvCount[i]<<endl;
		}
		if (remainder != 0 && my_rank < remainder)
		{
			cur_query_size = query_size + 1;
		}

		// change to c style array for MPI
		int *query_temp = nullptr;
		query_temp = (int *)calloc(cur_query_size * 2, sizeof(int));
		// cout << my_rank << " " << curr_begin << " " << curr_begin + cur_query_size - 1 << endl;
		for (int i = curr_begin; i <= curr_begin + cur_query_size - 1; i++)
		{
			left.resize(local_g.num_v1, false);
			right.resize(local_g.num_v2, false);
			int lval = queryStream[i][0]; // 读取
			int rval = queryStream[i][1]; // 读取
			// cout<<lval<<" "<<rval<<endl;
			retrieve_via_lrval_index(local_g, lrval_index_u, lrval_index_v, left, right, lval, rval);
			int nl = 0, nr = 0;
			for (auto e : left)
			{
				if (e)
					nl++;
			}
			for (auto e : right)
			{
				if (e)
					nr++;
			}
			// cout << "number of left in the answer: " << nl << ", " << "number of right in the answer: " << nr << endl;

			query_temp[cur_temp++] = nl;
			query_temp[cur_temp++] = nr;
		}

		int *query_all = nullptr;
		query_all = (int *)calloc(n_query * 2, sizeof(int));
		MPI_Gatherv(query_temp, cur_temp, MPI_INT, query_all, recvCount, displs, MPI_INT, 0, MPI_COMM_WORLD);

		if (my_rank == 0)
		{
			cur_temp = 0;
			for (int i = 0; i < n_query; i++)
			{
				int lval = queryStream[i][0]; 
				int rval = queryStream[i][1]; 

				queryAns[i * 4] = lval;
				queryAns[i * 4 + 1] = rval;
				queryAns[i * 4 + 2] = query_all[cur_temp++];
				queryAns[i * 4 + 3] = query_all[cur_temp++];
			}
		}
		
		// BLANK2 - END HERE


		// You may see that whether your results are correctly gathered.
		if(my_rank == 0){
			cout << "queryAns: ";
			for(int i = 0; i< n_query*4; i++){
				cout << queryAns[i] << " ";
			}
			cout << endl;
		}

		MPI_Barrier(comm);
		auto end = chrono::system_clock::now();
		chrono::duration<double> time = end - start;

		if(my_rank == 0){
			cout << "MPI Query time on the process " << my_rank << ": " << time.count() << endl;
		}
		
		MPI_Finalize();	
		
	}
	
	else {
		cout << "illegal arguments" << endl;
	}
	return 0;
}