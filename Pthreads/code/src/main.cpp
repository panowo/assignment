// PAN Han, student id:20881280
// email:hpanan@connect.ust.hk

// MSBD5009 SPRING2023 HW2

// Sequentail ver.:
// Compile:
// g++ -std=c++11 -pthread decom.h decom.cpp main.cpp -o lrds
// Run:
// ./lrds -BasicDecom ../data/<DATASET>/
// ./lrds -Query ../data/<DATASET>/

// Pthread ver.:
// Compile:
// g++ -std=c++11 -pthread decom.h decom.cpp main.cpp -o lrds
// Run:
// ./lrds -ParallelQuery num_threads ../data/<DATASET>/

#include <ctime>
#include <cmath>
#include <iostream>
#include <chrono>
#include "decom.h"
#include <pthread.h>
#include "string.h"
#include <cstring>

using namespace std;

const int Q_MAX = 100000;
int *query_ans;

struct AllThings
{
	/*
		BLANK1: AllThings contains arguments that are passed to the thread function.
	*/
	/*========== Add elements and construct AllThings below this line ==========*/
	int num_threads;
	int my_rank;
	int n_query;
	int num_v1;
	int num_v2;
	vector<vector<int>> queryStream;
	vector<vector<lrval_index_block *>> lrval_index_u;
	vector<vector<lrval_index_block *>> lrval_index_v;
	AllThings(){};
	AllThings(int inum_v1, int inum_v2, int inum_threads, int imy_rank, int in_query, vector<vector<int>> iqueryStream,
			  vector<vector<lrval_index_block *>> ilrval_index_u,
			  vector<vector<lrval_index_block *>> ilrval_index_v)
	{
		num_v1 = inum_v1;
		num_v2 = inum_v2;
		num_threads = inum_threads;
		my_rank = imy_rank;
		n_query = in_query;
		for (int i = 0; i < iqueryStream.size(); i++)
		{
			// queryStream.push_back(iqueryStream[i]);
			vector<int> temp;
			for (int j = 0; j < iqueryStream[i].size(); j++)
			{
				temp.push_back(iqueryStream[i][j]);
			}
			queryStream.push_back(temp);
			// cout<<queryStream[i].size()<<endl;
		}
		lrval_index_u = ilrval_index_u;
		lrval_index_v = ilrval_index_v;
	};

	/*==========Add elements and construct AllThings above this line    ==========*/
};

void *parallel(void *arg)
{
	/*
		BLANK2: Thread function - the function that threads are to run.
		You need to implement the thread function parallel to query.
		You can call the function retrieve_via_lrval_index which has been implemented.
	*/
	/*========== Fill the body of your thread function below this line ==========*/

	struct AllThings *pstru;
	pstru = (struct AllThings *)arg;

	int num_threads = pstru->num_threads;
	int my_rank = pstru->my_rank;
	int n_query = pstru->n_query;
	// pstru->queryStream
	// pstru->lrval_index_u;
	// pstru->lrval_index_v;

	// for (int i = 0; i < pstru->queryStream.size(); i++)
	// 		{
	// 			cout<<i<<endl;
	// 			cout<<pstru->queryStream[i][0]<<" "<<pstru->queryStream[i][1]<<endl;
	// 			cout<<pstru->queryStream[i].size()<<endl;
	// 		}

	int cur_temp = 0;
	int query_size = n_query / num_threads;
	int remainder = n_query % num_threads;

	int begin = 0, curr_begin = 0, cur_query_size = query_size;
	int dis = 0;
	for (int i = 0; i < num_threads; i++)
	{
		if (my_rank == i)
		{
			curr_begin = begin;
		}
		if (remainder != 0 && i < remainder)
		{
			begin += query_size + 1;
		}
		else
		{
			begin += query_size;
		}
	}
	if (remainder != 0 && my_rank < remainder)
	{
		cur_query_size = query_size + 1;
	}

	vector<bool> left;
	vector<bool> right;

	// cout << my_rank << " " << curr_begin << " " << curr_begin + cur_query_size - 1 << endl;
	for (int i = curr_begin; i <= curr_begin + cur_query_size - 1; i++)
	{
		// cout << i << endl;
		left.resize(pstru->num_v1, false);
		right.resize(pstru->num_v2, false);
		int lval = pstru->queryStream[i][0]; // 读取
		int rval = pstru->queryStream[i][1]; // 读取
		// cout << my_rank << " " << i << " " << lval << " " << rval << endl;
		retrieve_via_lrval_index(pstru->lrval_index_u, pstru->lrval_index_v, left, right, lval, rval);
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

		query_ans[i * 4] = lval;
		query_ans[i * 4 + 1] = rval;
		query_ans[i * 4 + 2] = nl;
		query_ans[i * 4 + 3] = nr;
	}

	/*========== Fill the body of your thread function above this line ==========*/

	return 0;
}

int main(int argc, char **argv)
{
	if (argc == 1)
	{
		cout << "error in number of arguments" << endl;
	}
	string exec_type = argv[1];
	if (exec_type == "-BasicDecom")
	{
		cout << "start BasicDecom for " << argv[2] << endl;
		BiGraph g(argv[2]);
		auto start = chrono::system_clock::now();
		lrIndexBasic(g);
		auto end = chrono::system_clock::now();
		chrono::duration<double> time = end - start;
		cout << "run time: " << time.count() << endl;
	}
	else if (exec_type == "-Query")
	{
		cout << "start sequential query for " << argv[2] << endl;
		BiGraph g(argv[2]);
		lrIndexBasic(g);
		vector<vector<lrval_index_block *>> lrval_index_u;
		vector<vector<lrval_index_block *>> lrval_index_v;
		build_lrval_index(g, lrval_index_u, lrval_index_v);
		vector<bool> left;
		vector<bool> right;
		// all the vertices in query result are set as true
		vector<vector<int>> queryStream;
		queryStream.resize(Q_MAX);
		int n_query = 0;
		loadQuery(argv[2], queryStream, n_query);
		queryStream.resize(n_query);
		int queryAns[n_query * 4];
		auto start = chrono::system_clock::now();
		for (int i = 0; i < n_query; i++)
		{
			left.resize(g.num_v1, false);
			right.resize(g.num_v2, false);
			int lval = queryStream[i][0];
			int rval = queryStream[i][1];
			retrieve_via_lrval_index(lrval_index_u, lrval_index_v, left, right, lval, rval);
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
			queryAns[i * 4] = lval;
			queryAns[i * 4 + 1] = rval;
			queryAns[i * 4 + 2] = nl;
			queryAns[i * 4 + 3] = nr;
		}

		cout << "query_ans: ";
		for (int i = 0; i < n_query * 4; i++)
		{
			cout << queryAns[i] << " ";
		}
		cout << endl;

		auto end = chrono::system_clock::now();
		chrono::duration<double> time = end - start;

		cout << "Sequential query time: " << time.count() << endl;
	}
	else if (exec_type == "-ParallelQuery")
	{

		cout << "start parallel query for " << argv[3] << endl;

		int num_of_threads = 0;
		num_of_threads = atoi(argv[2]);
		BiGraph g(argv[3]);
		lrIndexBasic(g);

		vector<vector<lrval_index_block *>> lrval_index_u;
		vector<vector<lrval_index_block *>> lrval_index_v;
		build_lrval_index(g, lrval_index_u, lrval_index_v);

		vector<vector<int>> query_stream;

		query_stream.resize(Q_MAX);
		int n_query = 0;
		loadQuery(argv[3], query_stream, n_query);
		query_stream.resize(n_query);

		query_ans = new int[n_query * 4];

		auto start = chrono::system_clock::now();

		/*
			BLANK3: Initiate threads, call the thread function.
		*/
		/*========== Fill in your code below this line ==========*/

		long thread;
		pthread_t *thread_handles;
		thread_handles = (pthread_t *)malloc(num_of_threads * sizeof(pthread_t));

		// struct AllThings args[9];
		// for (thread = 0; thread < num_of_threads; thread++)
		// {
		// 	struct AllThings temp(g.num_v1, g.num_v2, num_of_threads, thread, n_query, query_stream, lrval_index_u, lrval_index_v);
		// 	args[thread] = temp;
		// }

		for (thread = 0; thread < num_of_threads; thread++)
		{
			AllThings *all_t = new AllThings(g.num_v1, g.num_v2, num_of_threads, thread, n_query, query_stream, lrval_index_u, lrval_index_v);
			pthread_create(&thread_handles[thread], (pthread_attr_t *)NULL, parallel, (void *)all_t);
			// pthread_create(&thread_handles[thread], (pthread_attr_t *)NULL, parallel, (void *)&(args[thread]));
		}

		for (thread = 0; thread < num_of_threads; thread++)
		{
			pthread_join(thread_handles[thread], NULL);
		}

		free(thread_handles);

		/*========== Fill in your code above this line ==========*/

		auto end = chrono::system_clock::now();
		chrono::duration<double> time = end - start;

		cout << "query_ans: ";
		for (int i = 0; i < n_query * 4; i++)
		{
			cout << query_ans[i] << " ";
		}
		cout << endl;
		cout << "Parrallel query time: " << time.count() << endl;

		return 0;
	}
}
