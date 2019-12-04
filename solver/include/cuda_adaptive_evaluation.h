#ifndef CUDA_ADAPTIVE_EVALUATION_H
#define CUDA_ADAPTIVE_EVALUATION_H

#include <stdio.h>
#include <cuda_runtime.h>

/*This software contains source code provided by NVIDIA Corporation.

Read more at: http://docs.nvidia.com/cuda/eula/index.html#ixzz3h4aD9UbD 
Follow us: @GPUComputing on Twitter | NVIDIA on Facebook
*/
/***checkCudaErrors from NVIDIA CUDA Samples***/

#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)
inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
	if (err != cudaSuccess)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString(err));
		throw cudaGetErrorString(err);
    }
}
/**********************************************/

template<class T> struct TreeNode;

struct CSC
{
	float* cscValA;
	int* cscRowIndA;
	int* cscColPtrA;
	int nnz;
};

struct CSR
{
	float* csrValA;
	int* csrRowPtrA;
	int* csrColIndA;
	int nnz;
};

struct TwoArray
{
	int* data;
	int* row_width;
	int* row_ptr;
	int n_rows;
};

struct CudaVariable
{
	int width;
	int height;
	int* index;
	float3* coefs;
	float2* points;
	TreeNode<float>* regions;
	CSC laplacian_matrix;
	TwoArray neighbors;
};

void cuda_evaluation_func(const CudaVariable& var, float3* image);
void jacobian_func(float3** image, int* region_index, int2* critical_points, int critical_points_size, int width, int height, int round, int radius);

#endif
