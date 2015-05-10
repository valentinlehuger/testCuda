#include <iostream>
#include <stdio.h>
#include <cuda.h>

#define NB 10
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

__global__ void max(float *d_in, float *d_out)
{
	int ft_id = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;
	int size = (blockIdx.x == gridDim.x - 1) ? (NB % blockDim.x) : blockDim.x;

	for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (ft_id + s < NB && tid < s)
		{
			// printf("Compare d_in[%d] = %f with d_in[%d] = %f\n", (int)ft_id, d_in[ft_id], (int)(ft_id + s), d_in[ft_id + s]);
			d_in[ft_id] = (d_in[ft_id] > d_in[ft_id + s]) ? d_in[ft_id] : d_in[ft_id + s];

			if (size % 2 == 1 && ft_id + s + s == size - 1)
				d_in[ft_id] = (d_in[ft_id] > d_in[ft_id + s + s]) ? d_in[ft_id] : d_in[ft_id + s + s];
		}
		__syncthreads();
		size /= 2;
	}
	if (tid == 0)
	{
		d_out[blockIdx.x] = d_in[ft_id];
	}
	// __syncthreads();
	// for (int i = 0; i < GRID_SIZE; i++)
	// 	printf("d_out[%d] = %f\n", i, d_out[i]);
}

float			*init_values(size_t size)
{
	float		*values;

	if ((values = (float *)malloc(sizeof(float) * size)) != NULL)
	{
		for (size_t i = 0; i < size; i++)
			values[i] = float(i + 2);
		values[size / 2] = 100.f;
		return (values);
	}
	return (NULL);
}

int				main(void)
{
	size_t 		grid_size = NB / BLOCK_SIZE + 1;

	float		*h_values;

	float		*d_values;
	float		**d_values_ = &d_values;

	float		*h_max = (float *)malloc(sizeof(float) * 1);
	float		*d_max;
	float		**d_max_ = &d_max;

	h_values = init_values(NB);

	printf ("Initial values :\n");
	for (int i = 0; i < NB; i++)
		printf("%f\n", h_values[i]);
	printf("\n");
	

	// malloc values and max
	checkCudaErrors(cudaMalloc(d_values_, sizeof(float) * NB));
	checkCudaErrors(cudaMalloc(d_max_, sizeof(float) * 1));

	// memcopy values
	checkCudaErrors(cudaMemcpy(d_values, h_values, sizeof(float) * NB, cudaMemcpyHostToDevice));

	// kernel
	max<<<grid_size, BLOCK_SIZE>>>(d_values, d_max);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	
	// printf("GRID SIZE %d\n", GRID_SIZE);

	// memcpy max result
	checkCudaErrors(cudaMemcpy(h_max, d_max, sizeof(float) * 1, cudaMemcpyDeviceToHost));



	for (int i = 0; i < grid_size; i++)
		printf("h_max[%d] = %f\n", i, h_max[i]);

	// free the two
	cudaFree(d_max_);
	cudaFree(d_values_);

	return(0);
}
