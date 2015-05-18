#include <iostream>
#include <stdio.h>
#include <cuda.h>

#define THREADS 1024

#define BINS 3
#define NB 9000
#define ITEMS_PER_THREAD 100

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

float			*init_values(size_t size)
{
	float		*values;

	if ((values = (float *)malloc(sizeof(float) * size)) != NULL)
	{
		for (size_t i = 0; i < size; i++)
			values[i] = float(i + 2);
		values[size / 2] = 8.f;
		return (values);
	}
	return (NULL);
}

__global__ void	histogram(int *histogram, float *values, int min, int max, int bins, int nb_thread)
{
	int			id = (blockIdx.x * blockDim.x + threadIdx.x) * ITEMS_PER_THREAD;
	int			thread_id = threadIdx.x;
	int			*local_hist = (int *)malloc(sizeof(int) * bins);

	float		bin_size = (float)(max - min) / (float)bins;

	// printf("Bin size : %f\n", bin_size);

	// Init local histogram
	for (int i = 0; i < bins; i++)
		local_hist[i] = 0;

	// One shared array per bin
	extern __shared__ int s_bins[];

	// Compute serially local bin
	for (int i = 0; i < ITEMS_PER_THREAD; i++)
	{
		for (int j = 0; j < BINS; j += 1)
		{
			// if (id + i < NB)
			// 	printf("values[%d] = %f <= %f\n", id + i, values[id + i], (float)min + (float)(j + 1) * bin_size);

			if (id + i < NB && values[id + i] <= ((float)min + (float)(j + 1) * bin_size))
			{
				local_hist[j] += 1;
				// printf("BlockIdx : %d - Thread %d : values[%d] = %f -> local_hist[%d] = %d\n", blockIdx.x, thread_id, id + i, values[id + i], j, local_hist[j]);
				break ;
			}
		}
	}
	__syncthreads();
	// Store local bins into shared bins
	for (int i = 0; i < bins; i++)
	{
		s_bins[THREADS * i + thread_id] = local_hist[i];
		printf("Block %d - Thread %d : s_bins[%d] = local_hist[%d] = %d\n", blockIdx.x, thread_id, THREADS * i + thread_id, i, local_hist[i]);
	}

	__syncthreads();

	// if (thread_id == 0)
	// {
	// 	for (int i = 0; i < nb_thread * 3; i++)
	// 	{
	// 		printf("s_bins[%d] = %d\n", i, s_bins[i]);
	// 	}
	// }

	// Reduce each shared bin
	// int size = (blockIdx.x == gridDim.x - 1) ? (NB % blockDim.x) : blockDim.x;

	int size = THREADS;

	for (size_t s = THREADS / 2; s > 0; s >>= 1)
	{
		if (thread_id + s < THREADS && thread_id < s)
		{
			for (size_t j = 0; j < bins; j++)
			{
				s_bins[j * THREADS + thread_id] = s_bins[j * THREADS + thread_id] + s_bins[j * THREADS + thread_id + s];

				if (size % 2 == 1 && thread_id + s + s == size - 1)
					s_bins[j * THREADS + thread_id] = s_bins[j * THREADS + thread_id] + s_bins[j * THREADS + thread_id + s + s];
			}
		}
		__syncthreads();
		size = s;
	}

	// Store the result into histogram
	if (thread_id == 0)
	{
		histogram[0 + blockIdx.x * bins] = s_bins[0];
		histogram[1 + blockIdx.x * bins] = s_bins[THREADS];
		histogram[2 + blockIdx.x * bins] = s_bins[THREADS * 2];
		printf("histogram[%d] = %d\n", 0 + blockIdx.x * bins, s_bins[0]);
		printf("histogram[%d] = %d\n", 1 + blockIdx.x * bins, s_bins[THREADS]);
		printf("histogram[%d] = %d\n", 2 + blockIdx.x * bins, s_bins[THREADS * 2]);
	}
}


int				main(void)
{

	float		*h_values = init_values(NB);
	float		*d_values;
	float		**d_values_ = &d_values;

	int			*h_histogram = (int *)malloc(sizeof(int) * BINS);
	int			*d_histogram;
	int			**d_histogram_ = &d_histogram;

	int			min = 2;
	int			max = NB + 2;

	int			nb_thread = NB / ITEMS_PER_THREAD + 1;
	int			grid_dim = nb_thread / THREADS + 1;

	printf("NB = %d\n", NB);
	printf("ITEMS_PER_THREAD = %d\n", ITEMS_PER_THREAD);
	printf("nb_thread = %d\n", nb_thread);
	printf("grid dim = %d\n", grid_dim);

	// cudaMalloc
	checkCudaErrors(cudaMalloc(d_values_, sizeof(float) * NB));
	checkCudaErrors(cudaMalloc(d_histogram_, sizeof(int) * BINS * grid_dim));

	// cudaMemcpy HostToDevice
	checkCudaErrors(cudaMemcpy(d_values, h_values, sizeof(float) * NB, cudaMemcpyHostToDevice));

	// cudaMemset
	checkCudaErrors(cudaMemset(d_histogram, 0, sizeof(int) * BINS));

	printf("size of shared mem = %d\n", THREADS * BINS);
	printf("\n\n");
	// // kernel HISTOGRAM
	histogram<<<grid_dim, THREADS, THREADS * BINS * sizeof(int) >>>(d_histogram, d_values, min, max, BINS, nb_thread);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


	// cudaMemcpy DeviceToHost
	checkCudaErrors(cudaMemcpy(h_histogram, d_histogram, sizeof(int) * BINS * grid_dim, cudaMemcpyDeviceToHost));

	printf("\n\n");
	for (int i = 0; i < BINS * grid_dim; i++)
	{
		printf("Histogram[%d] = %d\n", i, h_histogram[i]);
	}

	// cudaFree
	cudaFree(d_values_);
	cudaFree(d_histogram_);


	return (0);
}
