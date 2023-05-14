// MSBD5009 SPRING2023 HW3

// CUDA ver.:
	// Compile:
	// nvcc -std=c++11 cuda_skeleton.cu decom.cpp main.cpp -o lrds
	// Run:
	// ./lrds -ParallelQuery ../data/<DATASET>/ <num_blocks_per_grid> <num_threads_per_block>


#include <ctime>
#include <cmath>
#include <iostream>
#include <chrono>
#include "decom.h"

using namespace std;

int main(int argc, char **argv) {
	if (argc == 1) {
		cout << "error in number of arguments" << endl;
	}
	string exec_type = argv[1];
	if (exec_type == "-ParallelQuery") {
		
		int num_blocks_per_grid = stoi(argv[3]);
		int num_threads_per_block = stoi(argv[4]);

		int n_query = 0;
		get_n_query(argv[2], n_query);
		int *queryAns = nullptr;
		queryAns=(int *)calloc(n_query*3, sizeof(int));

		cudaDeviceReset();
		cudaEvent_t cuda_start, cuda_end;

		float kernel_time;
		auto start_clock = chrono::high_resolution_clock::now();
		
		cudaEventCreate(&cuda_start);
		cudaEventCreate(&cuda_end);

		cudaEventRecord(cuda_start);

		cuda_query(argv[2], num_blocks_per_grid, num_threads_per_block, queryAns);
		
		cudaEventRecord(cuda_end);

		cudaEventSynchronize(cuda_start);
		cudaEventSynchronize(cuda_end);

		cout << "queryAns: ";
		for(int i = 0; i< n_query*3; i++){
			cout << queryAns[i] << " ";
		}
		cout << endl;

		cudaEventElapsedTime(&kernel_time, cuda_start, cuda_end);
		GPUErrChk(cudaDeviceSynchronize());

		auto end_clock = chrono::high_resolution_clock::now();

		chrono::duration<double> diff = end_clock - start_clock;
		
		printf("Elapsed Time: %.9lf s\n", diff.count());
		fprintf(stderr, "Driver Time: %.9lf s\n", kernel_time / pow(10, 3));
		
	}
	
	else {
		cout << "illegal arguments" << endl;
	}
	return 0;
}