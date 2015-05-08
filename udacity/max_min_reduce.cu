#include <iostream>
#include <stdio.h>
#include <cuda.h>

#define NB 1000
#define BLOCK_SIZE 512
#define GRID_SIZE NB / BLOCK_SIZE + 1


#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

__global__ void max(float *d_in, float *d_out)
{
	int ft_id = threadIdx.x + blockDim.x * blockIdx.x;
	int tid = threadIdx.x;
	int size = (blockIdx.x == gridDim.x - 1) ? (NB % blockDim.x) : blockDim.x;
	printf("%d\n", size);
	// for (int i = 0; i < blockDim.x; i ++)
	// {
	// 	printf("%f\n", d_in[i]);
	// }

	for (size_t s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (ft_id + s < NB && tid < s)
		{
			d_in[ft_id] = (d_in[ft_id] > d_in[ft_id + s]) ? d_in[ft_id] : d_in[ft_id + s];

			if (size % 2 == 1 && ft_id + s + s == size - 1)
				d_in[ft_id] = (d_in[ft_id] > d_in[ft_id + s + s]) ? d_in[ft_id] : d_in[ft_id + s + s];
		}
		__syncthreads();
		size /= 2;
	}
	if (tid == 0)
		d_out[blockIdx.x] = d_in[ft_id];
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
	float		*h_values;

	float		*d_values;
	float		**d_values_ = &d_values;

	float		*h_max = (float *)malloc(sizeof(float) * GRID_SIZE);
	float		*d_max;
	float		**d_max_ = &d_max;

	h_values = init_values(NB);

	// printf ("Initial values :\n");
	// for (int i = 0; i < NB; i++)
	// 	printf("%f\n", h_values[i]);
	// printf("\n");
	

	// malloc values and max
	checkCudaErrors(cudaMalloc(d_values_, sizeof(float) * NB));
	checkCudaErrors(cudaMalloc(d_max_, sizeof(float) * GRID_SIZE));

	// memcopy values
	checkCudaErrors(cudaMemcpy(d_values, h_values, sizeof(float) * NB, cudaMemcpyHostToDevice));

	// kernel
	max<<<GRID_SIZE, BLOCK_SIZE>>>(d_values, d_max);

	// memcpy max result
	checkCudaErrors(cudaMemcpy(h_max, d_max, sizeof(float) * GRID_SIZE, cudaMemcpyDeviceToHost));

	// free the two
	cudaFree(d_max_);
	cudaFree(d_values_);

	for (int i = 0; i < GRID_SIZE; i++)
		printf("h_max = %f\n", h_max[i]);

	return(0);
}
