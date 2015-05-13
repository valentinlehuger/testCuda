#include <iostream>
#include <stdio.h>
#include <cuda.h>

#define THREADS 19

#define BINS 3
#define NB 100
#define ITEMS_PER_THREAD 10

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

			if (id < NB && values[id + i] <= ((float)min + (float)(j + 1) * bin_size))
			{
				local_hist[j] += 1;
				// printf("Thread %d : values[%d] = %f -> local_hist[%d] = %d\n", thread_id, id + i, values[id + i], j, local_hist[j]);
				break ;
			}
		}
	}
	__syncthreads();
	// Store local bins into shared bins
	for (int i = 0; i < bins; i++)
	{
		s_bins[nb_thread + bins + i] = local_hist[i];
		if (local_hist[i] > 0)
			printf("s_bins[%d] = %d\n", thread_id * bins + i, local_hist[i]);
	}

	__syncthreads();

	// Reduce each shared bin
	//
	//

	// Store the result into histogram
	//
	//
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

	printf("nb_thread = %d\n", nb_thread);

	// cudaMalloc
	checkCudaErrors(cudaMalloc(d_values_, sizeof(float) * NB));
	checkCudaErrors(cudaMalloc(d_histogram_, sizeof(int) * BINS));

	// cudaMemcpy HostToDevice
	checkCudaErrors(cudaMemcpy(d_values, h_values, sizeof(float) * NB, cudaMemcpyHostToDevice));

	// cudaMemset
	checkCudaErrors(cudaMemset(d_histogram, 0, sizeof(int) * BINS));


	// // kernel HISTOGRAM
	histogram<<<grid_dim, THREADS, THREADS * BINS * sizeof(int) >>>(d_histogram, d_values, min, max, BINS, nb_thread);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


	// cudaMemcpy DeviceToHost
	checkCudaErrors(cudaMemcpy(h_histogram, d_histogram, sizeof(int) * BINS, cudaMemcpyDeviceToHost));


	// cudaFree
	cudaFree(d_values_);
	cudaFree(d_histogram_);


	return (0);
}