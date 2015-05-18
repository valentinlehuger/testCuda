#include <iostream>
#include <stdio.h>
#include <cuda.h>

#define BLOCK_SIZE 1024


#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

__global__ void max_min_cuda(float *d_in1, float *d_in2, float *d_max, float *d_min, size_t nb)
{
	int ft_id = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;
	int size = (blockIdx.x == gridDim.x - 1) ? (nb % blockDim.x) : blockDim.x;
 
	for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (ft_id + s < nb && tid < s)
		{
			d_in1[ft_id] = (d_in1[ft_id] > d_in1[ft_id + s]) ? d_in1[ft_id] : d_in1[ft_id + s];
			if (size % 2 == 1 && ft_id + s + s == size - 1)
				d_in1[ft_id] = (d_in1[ft_id] > d_in1[ft_id + s + s]) ? d_in1[ft_id] : d_in1[ft_id + s + s];
			d_in2[ft_id] = (d_in2[ft_id] < d_in2[ft_id + s]) ? d_in2[ft_id] : d_in2[ft_id + s];
			if (size % 2 == 1 && ft_id + s + s == size - 1)
				d_in2[ft_id] = (d_in2[ft_id] < d_in2[ft_id + s + s]) ? d_in2[ft_id] : d_in2[ft_id + s + s];
		}
		__syncthreads();
		size /= 2;
	}
	if (tid == 0)
	{
		d_max[blockIdx.x] = d_in1[ft_id];
		d_min[blockIdx.x] = d_in2[ft_id];
	}
	// __syncthreads();
	// for (int i = 0; i < GRID_SIZE; i++)
	// 	printf("d_out[%d] = %f\n", i, d_out[i]);
}

void				max_min(float *h_values, size_t size, float &h_min, float &h_max)
{
	size_t 		grid_size = size / BLOCK_SIZE + 1;

	float		*d_values;
	float		**d_values_ = &d_values;
	float		*d_values2;
	float		**d_values2_ = &d_values2;

	float		*maxs = (float *)malloc(sizeof(float) * grid_size);
	float		*d_max;
	float		**d_max_ = &d_max;

	float		*mins = (float *)malloc(sizeof(float) * grid_size);
	float		*d_min;
	float		**d_min_ = &d_min;


	// malloc values and max
	checkCudaErrors(cudaMalloc(d_values_, sizeof(float) * size));
	checkCudaErrors(cudaMalloc(d_values2_, sizeof(float) * size));
	checkCudaErrors(cudaMalloc(d_max_, sizeof(float) * grid_size));
	checkCudaErrors(cudaMalloc(d_min_, sizeof(float) * grid_size));

	// memcopy values
	checkCudaErrors(cudaMemcpy(d_values, h_values, sizeof(float) * size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_values2, h_values, sizeof(float) * size, cudaMemcpyHostToDevice));

	// kernel
	max_min_cuda<<<grid_size, BLOCK_SIZE>>>(d_values, d_values2, d_max, d_min, size);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	
	// printf("GRID SIZE %d\n", GRID_SIZE);

	// memcpy results
	checkCudaErrors(cudaMemcpy(maxs, d_max, sizeof(float) * grid_size, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(mins, d_min, sizeof(float) * grid_size, cudaMemcpyDeviceToHost));
	h_min = mins[0];
	h_max = maxs[0];
	for (int i = 0; i < grid_size; i++) {
	  if (h_max < maxs[i])
	    h_max = maxs[i];
	  if (h_min > mins[i])
	    h_min = mins[i];
	}

	// free the three
	cudaFree(d_max_);
	cudaFree(d_min_);
	cudaFree(d_values_);
	cudaFree(d_values2_);

}
