// PAN Han, student id:20881280
// email:hpanan@connect.ust.hk

#include "decom.h"

using namespace std;

__global__ void retrieve_via_lrval_index_check_kernel(int *d_lrval_index_u_size, int d_length_lrval_index_u, int *queryStream, int n_query, int *d_queryAns)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int nthread = blockDim.x * gridDim.x;

    for (int i = tid; i < n_query; i += nthread)
    {

        int flag = 0;
        int l_val = queryStream[i * 2];
        int r_val = queryStream[i * 2 + 1];

        if (d_length_lrval_index_u <= l_val || d_lrval_index_u_size[l_val] <= r_val)
        {
            flag = 0;
        }
        else
        {
            flag = 1;
        }

        d_queryAns[i * 3] = l_val;
        d_queryAns[i * 3 + 1] = r_val;
        d_queryAns[i * 3 + 2] = flag;
    }
}

// void retrieve_via_lrval_index_check(vector<vector<lrval_index_block *>> &lrval_index_u,
//                                     int l_val, int r_val, int &flag)
// {

//     if (lrval_index_u.size() <= l_val || lrval_index_u[l_val].size() <= r_val)
//     {
//         flag = 0;
//     }
//     else
//     {
//         flag = 1;
//     }
// }

void cuda_query(string dir, int num_blocks_per_grid, int num_threads_per_block, int *queryAns)
{
    int Q_MAX = 100000;
    BiGraph g(dir);
    lrIndexBasic(g);
    vector<vector<lrval_index_block *>> lrval_index_u;
    vector<vector<lrval_index_block *>> lrval_index_v;
    build_lrval_index(g, lrval_index_u, lrval_index_v);
    // all the vertices in query result are set as true
    vector<vector<int>> queryStream;
    queryStream.resize(Q_MAX);
    int n_query = 0;
    get_n_query(dir, n_query);
    loadQuery(dir, queryStream);
    queryStream.resize(n_query);

    // prepare data
    int *d_queryStream;
    int *h_queryStream;
    size_t size_query = sizeof(queryStream[0][0]) * n_query * 2;
    h_queryStream = (int *)malloc(sizeof(queryStream[0][0]) * n_query * 2);
    for (int i = 0; i < n_query; i++)
    {
        h_queryStream[i * 2] = queryStream[i][0];
        h_queryStream[i * 2 + 1] = queryStream[i][1];
    }

    int d_length_lrval_index_u;
    int h_length_lrval_index_u;
    h_length_lrval_index_u=lrval_index_u.size();
    d_length_lrval_index_u=h_length_lrval_index_u;

    int *d_lrval_index_u_size;
    int *h_lrval_index_u_size;
    size_t lrval_index_u_size = sizeof(lrval_index_u.size()) * lrval_index_u.size();
    h_lrval_index_u_size = (int *)malloc(lrval_index_u_size);
    for (int i = 0; i < lrval_index_u.size(); i++)
    {
        h_lrval_index_u_size[i] = lrval_index_u[i].size();
    }

    int *d_queryAns;

    cudaMalloc((void **)&d_queryStream, sizeof(queryStream[0][0]) * n_query * 2);
    cudaMemcpy(d_queryStream, h_queryStream, size_query, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_lrval_index_u_size, lrval_index_u_size);
    cudaMemcpy(d_lrval_index_u_size, h_lrval_index_u_size, lrval_index_u_size, cudaMemcpyHostToDevice);

    cudaMalloc((void **)&d_queryAns, sizeof(int) * n_query * 3);

    retrieve_via_lrval_index_check_kernel<<<num_blocks_per_grid, num_threads_per_block>>>(
        d_lrval_index_u_size, d_length_lrval_index_u, d_queryStream, n_query, d_queryAns);

    cudaDeviceSynchronize();

    cudaMemcpy(queryAns,d_queryAns,sizeof(int) * n_query * 3,cudaMemcpyDeviceToHost);

    cudaFree(d_queryStream);
    cudaFree(d_lrval_index_u_size);
    cudaFree(d_queryAns);

    // for (int i = 0; i < n_query; i++)
    // {
    //     int flag = 0;
    //     int lval = queryStream[i][0];
    //     int rval = queryStream[i][1];

    //     retrieve_via_lrval_index_check(lrval_index_u, lval, rval, flag);

    //     queryAns[i * 3] = lval;
    //     queryAns[i * 3 + 1] = rval;
    //     queryAns[i * 3 + 2] = flag;
    // }
}
