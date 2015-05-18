#include <iostream>
#include <stdio.h>
#include <cuda.h>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}


#define NB 8


int			*init_values(size_t size)
{
	int		*values;

	if ((values = (int *)malloc(sizeof(int) * size)) != NULL)
	{
		for (size_t i = 0; i < size; i++)
			values[i] = i + 1;
		// values[size / 2] = 8.f;
		return (values);
	}
	return (NULL);
}

__global__ void	b_scan_reduce_cuda(int *values, int *cumulative)
{

	int		id = blockDim.x * blockIdx.x + threadIdx.x;
	int		tid = threadIdx.x;
	
	int		nb_here = (NB - (blockDim.x * blockIdx.x) < blockDim.x) ? NB - (blockDim.x * blockIdx.x) : blockDim.x;

	if (tid == 0)
	  printf("Block %d nb_here = %d\n", blockIdx.x, nb_here);

////////// REDUCE
//
//
	int		next_th = 2;
	// printf("size = %u\n", size);
	for (int threshold = 1; threshold < blockDim.x / 2; threshold = threshold << 1)
	{
		// printf("Thread %d \t id = %d \t threshold = %d \t NB = 8 \t threadIdx.x = %d \t next_th = %d\n", tid, id, threshold, tid, next_th);
		// printf("NB %d \t tid = %d \t next_th = %d\n", (NB - 1), tid, next_th);
		if (tid < nb_here && tid - threshold >= 0 && ((nb_here - 1) - tid) % next_th == 0)
		{
		  //	printf("HERE : values[%d] = %d\n", id, values[id] + values[id - threshold]);
			values[id] = values[id] + values[id - threshold];
		}
		next_th = next_th << 1;
		__syncthreads();
	}

////////// DOWNSWEEP
//
//

	values[nb_here - 1] = 0;
	next_th = nb_here >> 1;
	int	tmp;
	for (int threshold = nb_here; threshold > 1; threshold >>= 1)
	{
	  if (tid == 0)
	    printf("threshold = %d && next_th = %d\n", threshold, next_th);
	  // printf("threshold = %d\n", threshold);
	  if (tid < nb_here && tid - next_th >= 0 && (nb_here - 1 - tid) % threshold == 0) {
		  tmp = values[id];
		  printf("Thread %d : values[%d] = %d\n", tid, id, tmp + values[id - threshold]);
		  printf("Thread %d : values[%d] = %d\n", tid, id - threshold, tmp);
		  values[id] += values[id - next_th];
		  values[id - next_th] = tmp;
		}
	  next_th = next_th >> 1;
	  if (tid == 0)
	    printf("\n");
	  __syncthreads();
	}


// Store into cumulative
        cumulative[id] = values[id];
}

void		blelloch_scan_reduce(int *h_values, int *h_cumulative)
{
	int		*d_values;
	int		**d_values_ = &d_values;

	int		*d_cumulative;
	int		**d_cumulative_ = &d_cumulative;

	// mallocs
	checkCudaErrors(cudaMalloc(d_values_, sizeof(int) * NB));
	checkCudaErrors(cudaMalloc(d_cumulative_, sizeof(int) * NB));

	// memcpy & memset
	checkCudaErrors(cudaMemcpy(d_values, h_values, sizeof(int) * NB, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_cumulative, h_values, sizeof(int) * NB, cudaMemcpyHostToDevice));
	// checkCudaErrors(cudaMemset(d_cumulative, 0, sizeof(int) * NB));

	b_scan_reduce_cuda<<<1, NB>>>(d_values, d_cumulative);

	// memcpy
	checkCudaErrors(cudaMemcpy(h_cumulative, d_cumulative, sizeof(int) * NB, cudaMemcpyDeviceToHost));
	// checkCudaErrors(cudaMemcpy(h_values, d_values, sizeof(int) * NB, cudaMemcpyDeviceToHost));

	// free
	cudaFree(d_values_);
	cudaFree(d_cumulative_);
}

int			main(void)
{
	int		*values;
	int		*cumulative;

	values = init_values(NB);
	cumulative = (int *)malloc(sizeof(int) * NB);
	blelloch_scan_reduce(values, cumulative);

	for (size_t i = 0; i < NB; i++)
	{
		printf("%d\t%d\n", values[i], cumulative[i]);
	}

	return (0);
}
