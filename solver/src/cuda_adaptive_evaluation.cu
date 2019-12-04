#include "cuda_adaptive_evaluation.h"
#include "structure.h"
#include <device_functions.h>
#include <math_constants.h>

__forceinline__ __device__ float atanf_Pade(float x)
{
	const float CORRECTED_VALUE = ((1.0f + 7.0f / 9.0f + 64.0f / 945.0f) / (1.0f + 10.0f / 9.0f + 5.0f / 21.0f) - CUDART_PIO4_F);
	float x2 = x*x;
	float x3 = x2*x;
	float x4 = x3*x;
	float x5 = x4*x;
	return __fdividef(x + 7.0f / 9.0f*x3 + 64.0f / 945.0f*x5, 1.0f + 10.0f / 9.0f*x2 + 5.0f / 21.0f*x4) - CORRECTED_VALUE*x;
}

__forceinline__ __device__ float fast_atanf(float y, float x)
{
	bool b = fabsf(y) > fabsf(x);
	if (b) {
		float t = y;
		y = x;
		x = t;
	}

	float div = __fdividef(y, x);

	float v = atanf_Pade(div);

	if (div > 0 && b) return CUDART_PIO2_F - v;
	else if (div < 0 && b) return -CUDART_PIO2_F - v;
	else return v;
}

__forceinline__ __device__ float source(const float2* func, float pt_x, float pt_y)
{
	float x = pt_x - func->x;
	float y = pt_y - func->y;

	//float v1 = atanf(__fdividef(y, x));
	float v1 = fast_atanf(y, x);

	float x2 = x*x;
	float y2 = y*y;
	float v = x*y*(__logf(x2 + y2) - 3.0f) + (x2 - y2)*v1;
	y2 *= CUDART_PIO2_F;

	if (x == 0.0f || y == 0.0f)  v = y2 = 0.0f;

	return v1 > 0 ? v + y2 : v - y2;
}

__forceinline__ __device__ float green_integral(const float2* func, float x1, float x2, float y1, float y2)
{
	return source(func, x1, y1) + source(func, x2, y2) - source(func, x1, y2) - source(func, x2, y1);
}

__device__ float3 basis_evaluator(const float2* pt, int basis_id, const CudaVariable* var_constant)
{
	float val = 0.0f;
	int id = var_constant->laplacian_matrix.cscColPtrA[basis_id + 1];
	for (int i = var_constant->laplacian_matrix.cscColPtrA[basis_id]; i < id; ++i)
	{
		const TreeNode<float>& r = var_constant->regions[var_constant->laplacian_matrix.cscRowIndA[i]];
		val += __fdividef(var_constant->laplacian_matrix.cscValA[i] * green_integral(pt, r.row, r.row + r.width, r.col, r.col + r.width), r.width*r.width);
	}
	val *= 0.25f / CUDART_PI_F;
	float3 value;
	value.x = val*var_constant->coefs[basis_id].x;
	value.y = val*var_constant->coefs[basis_id].y;
	value.z = val*var_constant->coefs[basis_id].z;
	return value;
}

__global__ void pixel_evaluator(float3* image, const CudaVariable var_constant, const int2 base)
{
	int row = base.x + blockIdx.x*blockDim.x + threadIdx.x;
	int col = base.y + blockIdx.y*blockDim.y + threadIdx.y;

	float2* pt = var_constant.points + row*var_constant.width + col;

	if (row < var_constant.height && col < var_constant.width)
	{
		int region_id = var_constant.index[row*var_constant.width + col];
		if (region_id != -1)
		{
			float3 color = { 0.0f, 0.0f, 0.0f };

			int row_width = var_constant.neighbors.row_width[region_id];
			int row_ptr = var_constant.neighbors.row_ptr[region_id];
			const float radius = 27.0f;
			const float inner = radius - 8.0f;

			for (int i = 0; i < row_width; ++i)
			{
				int basis_id = var_constant.neighbors.data[row_ptr + i];

				const TreeNode<float>& n = var_constant.regions[basis_id];
				float d1 = n.row + 0.5f*n.width - pt->x;
				float d2 = n.col + 0.5f*n.width - pt->y;
				float d = sqrt(d1*d1 + d2*d2);

				if (d <= radius)
				{
					float w;
					if (d <= inner) w = 1.0f;
					else if (d < inner + 0.126f) w = sqrt(1.0f - (d - inner)*(d - inner));
					else if (d <= inner + 7.874f) w = 0.992f - 0.127f*(d - (inner + 0.126f));
					else w = 1.0f - sqrt(1.0f - (d - radius)*(d - radius));

					const float3& c = basis_evaluator(pt, basis_id, &var_constant);
					color.x += w*c.x;
					color.y += w*c.y;
					color.z += w*c.z;
				}
			}

			image[row*var_constant.width + col] = color;
		}
	}
}

__global__ void jacobian_iteration(float3* image, float3* image_cache, int* region_index, int2* critical_points, int critical_points_size, int width, int height,  int radius)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index < critical_points_size)
	{
		int count = 0;
		int2 c = critical_points[index];
		float3 val = { 0, 0, 0 };
		if (c.x >= radius)
		{
			bool flag = true;
			int n = 0;
			for (int i = 0; i <= radius; ++i)
			{
				n += region_index[(c.x - i)*width + c.y];
				if (n == 2){
					flag = false;
					break;
				}
			}
			
			if (flag)
			{
				float3 v = image_cache[(c.x - radius)*width + c.y];
				if (v.x != -10000 && v.y != -10000 && v.z != -10000)
				{
					val.x += v.x;
					val.y += v.y;
					val.z += v.z;
					++count;
				}
			}
		}
		if (c.x + radius < height)
		{
			bool flag = true;
			int n = 0;
			for (int i = 0; i <= radius; ++i)
			{
				n += region_index[(c.x + i)*width + c.y];
				if (n == 2){
					flag = false;
					break;
				}
			}

			if (flag)
			{
				float3 v = image_cache[(c.x + radius)*width + c.y];
				if (v.x != -10000 && v.y != -10000 && v.z != -10000)
				{
					val.x += v.x;
					val.y += v.y;
					val.z += v.z;
					++count;
				}
			}
		}
		if (c.y >= radius)
		{
			bool flag = true;
			int n = 0;
			for (int i = 0; i <= radius; ++i)
			{
				n += region_index[c.x*width + c.y - i];
				if (n == 2){
					flag = false;
					break;
				}
			}

			if (flag)
			{
				float3 v = image_cache[c.x *width + c.y - radius];
				if (v.x != -10000 && v.y != -10000 && v.z != -10000)
				{
					val.x += v.x;
					val.y += v.y;
					val.z += v.z;
					++count;
				}
			}
		}
		if (c.y + radius < width)
		{
			bool flag = true;
			int n = 0;
			for (int i = 0; i <= radius; ++i)
			{
				n += region_index[c.x*width + c.y + i];
				if (n == 2){
					flag = false;
					break;
				}
			}

			if (flag)
			{
				float3 v = image_cache[c.x*width + c.y + radius];
				if (v.x != -10000 && v.y != -10000 && v.z != -10000)
				{
					val.x += v.x;
					val.y += v.y;
					val.z += v.z;
					++count;
				}
			}
		}
		if (count != 0)
		{
			float div = 1.0f / count;
			val.x *= div;
			val.y *= div;
			val.z *= div;
			image[c.x*width + c.y] = val;
		}
		else image[c.x*width + c.y] = image_cache[c.x*width + c.y];
	}
}

void cuda_evaluation_func(const CudaVariable& var, float3* image)
{
#ifdef _DEBUG
	int grid = 64;
#else
	int grid = 1024;
#endif
	const dim3 threads_per_block(16, 16);
	int grid_row = (var.height + grid - 1) / grid;
	int grid_col = (var.width + grid - 1) / grid;
	for (int r = 0; r < grid_row; ++r)
	{
		for (int c = 0; c < grid_col; ++c)
		{
			int2 base = { r*grid, c*grid };
			int thread_row = std::min(var.height - base.x, grid);
			int thread_col = std::min(var.width - base.y, grid);
			dim3 blocks_per_grid((thread_row + threads_per_block.x - 1) / threads_per_block.x, (thread_col + threads_per_block.y - 1) / threads_per_block.y);
			pixel_evaluator << <blocks_per_grid, threads_per_block >> >(image, var, base);
			checkCudaErrors(cudaDeviceSynchronize());
		}
	}
}

void jacobian_func(float3** image, int* region_index, int2* critical_points, int critical_points_size, int width, int height, int round, int radius)
{
	dim3 threads_per_block(128);
	dim3 blocks_per_grid((critical_points_size + threads_per_block.x - 1) / threads_per_block.x);
	float3* image_tmp;
	checkCudaErrors(cudaMalloc((void**)&image_tmp, sizeof(float3)*width*height));
	checkCudaErrors(cudaMemcpy(image_tmp, *image, sizeof(float3)*width*height, cudaMemcpyDeviceToDevice));

	for (int i = 0; i < round; ++i)
	{
		std::swap(*image, image_tmp);
		jacobian_iteration << <blocks_per_grid, threads_per_block >> >(*image, image_tmp, region_index, critical_points, critical_points_size, width, height, radius);
		checkCudaErrors(cudaDeviceSynchronize());
	}

	if (radius > 1)
	{
		for (int i = 0; i < 50; ++i)
		{
			std::swap(*image, image_tmp);
			jacobian_iteration << <blocks_per_grid, threads_per_block >> >(*image, image_tmp, region_index, critical_points, critical_points_size, width, height, 1);
			checkCudaErrors(cudaDeviceSynchronize());
		}
	}

	checkCudaErrors(cudaFree(image_tmp));
}

////////////////////BLURRING FOR DEBUG////////////////////////////////

__global__ void blur_image(float3* blured_image, const float3* image, const float* kernel_size, int rows, int cols, int2 base, int count)
{
	int r = base.x + blockIdx.x*blockDim.x + threadIdx.x;
	int c = base.y + blockIdx.y*blockDim.y + threadIdx.y;
	const int2 trans[] = { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };
	if (r < rows && c < cols)
	{
		blured_image[r*cols + c] = image[r*cols + c];
		if (count >= kernel_size[r*cols + c]) return;
		float3 val = { 0, 0, 0 };
		int num = 0;
		for(int i = 0; i < 4; ++i)
		{
			int2 p = { r + trans[i].x, c + trans[i].y };
			if (p.x >= 0 && p.x < rows && p.y >= 0 && p.y < cols)
			{
				val.x += image[p.x*cols + p.y].x;
				val.y += image[p.x*cols + p.y].y;
				val.z += image[p.x*cols + p.y].z;
				++num;
			}
		}
		val.x /= num;
		val.y /= num;
		val.z /= num;
		blured_image[r*cols + c] = val;
	}
}

void blur(float3** image, const float* kernel_size, int rows, int cols, int round)
{
	float3* blured_image;
	checkCudaErrors(cudaMalloc((void**)&blured_image, sizeof(float3)*rows*cols));
	checkCudaErrors(cudaMemcpy(blured_image,*image,sizeof(float3)*rows*cols,cudaMemcpyDeviceToDevice));
#ifdef _DEBUG
	int grid = 64;
#else
	int grid = 1024;
#endif
	const dim3 threads_per_block(16, 16);
	int grid_row = (rows + grid - 1) / grid;
	int grid_col = (cols + grid - 1) / grid;
	for (int n = 0; n < round; ++n)
	{
		for (int r = 0; r < grid_row; ++r)
		{
			for (int c = 0; c < grid_col; ++c)
			{
				int2 base = { r*grid, c*grid };
				int thread_row = std::min(rows - base.x, grid);
				int thread_col = std::min(cols - base.y, grid);
				dim3 blocks_per_grid((thread_row + threads_per_block.x - 1) / threads_per_block.x, (thread_col + threads_per_block.y - 1) / threads_per_block.y);
				blur_image << <blocks_per_grid, threads_per_block >> >(blured_image, *image, kernel_size, rows, cols, base, n);
				checkCudaErrors(cudaDeviceSynchronize());
				std::swap(*image, blured_image);
			}
		}
	}
	checkCudaErrors(cudaFree(blured_image));
}

__global__ void jacobian_iteration(float* image, float* image_cache, int2* critical_points, int critical_points_size, int rows, int cols)
{
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	const int2 trans[] = { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };
	if (index < critical_points_size)
	{
		int2* pt = &critical_points[index];
		float val = 0;
		int num = 0;
		for (int i = 0; i < 4; ++i)
		{
			int2 p = { pt->x + trans[i].x, pt->y + trans[i].y };
			if (p.x >= 0 && p.x < rows && p.y >= 0 && p.y < cols)
			{
				if (image_cache[p.x*cols + p.y] >= 0)
				{
					val += image_cache[p.x*cols + p.y];
					++num;
				}
			}
		}
		if (num != 0) image[pt->x*cols + pt->y] = val / num;
	}
}

void jacobian_func(float** image, int2* critical_points, int critical_points_size, int width, int height, int round)
{
	dim3 threads_per_block(128);
	dim3 blocks_per_grid((critical_points_size + threads_per_block.x - 1) / threads_per_block.x);
	float* image_tmp;
	checkCudaErrors(cudaMalloc((void**)&image_tmp, sizeof(float)*width*height));
	checkCudaErrors(cudaMemcpy(image_tmp, *image, sizeof(float)*width*height, cudaMemcpyDeviceToDevice));

	for (int i = 0; i < round; ++i)
	{
		std::swap(*image, image_tmp);
		jacobian_iteration << <blocks_per_grid, threads_per_block >> >(*image, image_tmp, critical_points, critical_points_size, width, height);
		checkCudaErrors(cudaDeviceSynchronize());
	}

	checkCudaErrors(cudaFree(image_tmp));
}


//////////////////////////////////////////////////////////////////////
