#include "utils.h"
#include <cuda.h>



__global__
void incrementOne(const int* tab,
                       int* newTab)
{
    int x = blockIdx.x;
    newTab[x] = tab[x] + 1;
}


void			incrementOne_cu(const int *h_tab,
								int ** d_tab,
								int ** d_newTab,
								int size)
{

	checkCudaErrors(cudaMalloc(d_tab, sizeof(int) * size));
	checkCudaErrors(cudaMalloc(d_newTab, sizeof(int) * size));
	checkCudaErrors(cudaMemset(*d_newTab, 0, sizeof(int) * size));
	checkCudaErrors(cudaMemcpy(*d_tab, h_tab, sizeof(int) * size, cudaMemcpyHostToDevice));

	const dim3 blockSize(1, 1, 1);
	const dim3 gridSize(size, 1, 1);
	incrementOne<<<gridSize, blockSize>>>(*d_tab, *d_newTab);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

int main()
{
	return (0);
}