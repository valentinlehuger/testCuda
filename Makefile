NVCC=nvcc

CUDA_INCLUDEPATH=/Developer/NVIDIA/CUDA-7.0/include/
CUDA_LIBPATH=/Developer/NVIDIA/CUDA-7.0/lib/

NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64

GCC_OPTS=-O3 -Wall -Wextra -m64

student: main.o incrementOne.o Makefile
	$(NVCC) -o test main.o incrementOne.o $(NVCC_OPTS)

main.o: main.cpp
	g++ -c main.cpp $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

incrementOne.o: incrementOne.cu
	nvcc -c incrementOne.cu $(NVCC_OPTS)

clean:
	rm -f *.o

fclean: clean
	rm -f test