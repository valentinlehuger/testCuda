#include "utils.h"
#include <cuda.h>



__global__
void incrementOne(const int* tab,
                       int* newTab)
{
    int x = threadIdx.x;
    
    newTab[x] = tab[x] + 1;
}


void			incrementOne_cu(const int *h_tab,
								int ** d_tab,
								int ** d_newTab,
								int size)
{

	checkCudaErrors(cudaMalloc(d_tab, sizeof(int) * size));
	checkCudaErrors(cudaMalloc(d_newTab, sizeof(int) * size));
	checkCudaErrors(cudaMemset(*d_newTab, 0, size * sizeof(int)));
	checkCudaErrors(cudaMemcpy(*d_tab, h_tab, sizeof(int) * size, cudaMemcpyHostToDevice));

	const dim3 blockSize(size, 1, 1);
	const dim3 gridSize(1, 1, 1);
	incrementOne<<<gridSize, blockSize>>>(*d_tab, *d_newTab);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
