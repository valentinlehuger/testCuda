#include <cmath>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.h"

void			incrementOne_cu(const int *h_tab,
								int ** d_tab,
								int ** d_newTab,
								int size);


int				main(int ac, char **av)
{
	int size = 0;
	int	*tab;
	int *d_tab = NULL;
	int *d_newTab = NULL;
	int *h_newTab;

	h_newTab = static_cast<int *>(malloc(sizeof(int) * size));
	d_newTab = static_cast<int *>(malloc(sizeof(int) * size));
	d_tab = static_cast<int *>(malloc(sizeof(int) * size));

	if (ac != 2)
	{
		std::cout << "Usage : ./test [size]" << std::endl;
		return (-1);
	}

	size = std::atoi(av[1]);
	tab = static_cast<int *>(malloc(sizeof(int) * size));
	for (int i = 0; i < size; i++)
		tab[i] = i;

	incrementOne_cu(tab, &d_tab, &d_newTab, size);
  	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaMemcpy(h_newTab, d_newTab, sizeof(int) * size, cudaMemcpyDeviceToHost));

	cudaFree(d_tab);
	cudaFree(d_newTab);



	for (int i = 0; i < size; i++)
		std::cout << tab[i] << "->" << h_newTab[i] << std::endl;

	return (0);
}
