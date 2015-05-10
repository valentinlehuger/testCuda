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



int				main(void)
{

	float		*h_values = init_values(NB);
	float		*d_values;
	float		**d_values_ = &d_values;

	float		*h_histogram = (float *)malloc(sizeof(float) * BINS);
	float		**d_histograms;
	float		***d_histograms_ = &d_histograms;
	float		*d_histogram;
	float		**d_histogram_ = &d_histogram;

	float		min = 2;
	float		max = 1000;

	int			nb_thread = NB / ITEMS_PER_THREAD + 1;
	int			grid_dim = nb_thread / THREADS + 1;


	// cudaMalloc
	checkCudaErrors(cudaMalloc(d_values_, sizeof(float) * NB));
	checkCudaErrors(cudaMalloc(d_histograms_, sizeof(float*) * nb_thread));
	for (int i = 0; i < nb_thread; i++)
	{
		checkCudaErrors(cudaMalloc(d_histograms_[i], sizeof(float) * BINS));
		checkCudaErrors(cudaMemset(d_histograms_[i], 0, sizeof(float) * BINS));
	}
	checkCudaErrors(cudaMalloc(d_histogram_, sizeof(float) * BINS));


	// cudaMemcpy HostToDevice
	checkCudaErrors(cudaMemcpy(d_values, h_values, sizeof(float) * NB, cudaMemcpyHostToDevice));

	// cudaMemset
	checkCudaErrors(cudaMemset(d_histogram_, 0, sizeof(float) * BINS));


	// // kernel HISTOGRAM
	// histogram<<<grid_dim, THREADS>>>(d_histograms, d_values, min, max, BINS);
	// cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	// // kernel reduce
	// reduce_histogram<<< grid_dim, THREADS>>>(d_histogram, d_histograms, min, max, BINS);
	// cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	// cudaMemcpy DeviceToHost
	checkCudaErrors(cudaMemcpy(h_histogram, d_histogram, sizeof(float) * BINS, cudaMemcpyDeviceToHost));


	// cudaFree
	cudaFree(d_values_);
	for (int i = 0; i < nb_thread; i++)
	{
		cudaFree(d_histograms_[i]);
	}
	cudaFree(d_histograms_);
	cudaFree(d_histogram_);


	return (0);
}