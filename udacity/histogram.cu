#include <iostream>
#include <stdio.h>
#include <cuda.h>

#define THREADS 1024

#define BINS 3
#define NB 101
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
		values[size / 2] = 1000.f;
		return (values);
	}
	return (NULL);
}

__global__ void	histogram(int *histogram, float *values, int min, int max, int bins)
{
	int			id = blockIdx.x * blockDim.x + threadIdx.x;
	int			thread_id = threadIdx.x;
	int			*local_hist = (int *)malloc(sizeof(int) * bins);

	float		bin_size = (float)(max - min) / (float)bins;

	// Init local histogram
	for (int i = 0; i < bins; i++)
		local_hist[i] = 0;

	// One shared array per bin
	extern __shared__ int s_bins[];

	// Compute serially local bin
	for (int i = 0; i < ITEMS_PER_THREAD; i++)
	{
		for (int j = 0; j < bins; j += 1)
		{
			if (values[id] <= ((float)min + j * bin_size))
				local_hist[j] += 1;
		}
	}	
	// Store local bins into shared bins
	for (int i = 0; i < bins; i++)
	{
		s_bins[(NB / ITEMS_PER_THREAD + 1) * i + thread_id] = local_hist[i];
		// printf("Thread %d local[%d] = %d\n", thread_id, i, local_hist[i]);
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
	int			max = 1000;

	int			nb_thread = NB / ITEMS_PER_THREAD + 1;
	int			grid_dim = nb_thread / THREADS + 1;


	// cudaMalloc
	checkCudaErrors(cudaMalloc(d_values_, sizeof(float) * NB));
	checkCudaErrors(cudaMalloc(d_histogram_, sizeof(int) * BINS));

	// cudaMemcpy HostToDevice
	checkCudaErrors(cudaMemcpy(d_values, h_values, sizeof(float) * NB, cudaMemcpyHostToDevice));

	// cudaMemset
	checkCudaErrors(cudaMemset(d_histogram, 0, sizeof(int) * BINS));


	// // kernel HISTOGRAM
	histogram<<<grid_dim, THREADS, THREADS * BINS * sizeof(int) >>>(d_histogram, d_values, min, max, BINS);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());


	// cudaMemcpy DeviceToHost
	checkCudaErrors(cudaMemcpy(h_histogram, d_histogram, sizeof(int) * BINS, cudaMemcpyDeviceToHost));


	// cudaFree
	cudaFree(d_values_);
	cudaFree(d_histogram_);


	return (0);
}