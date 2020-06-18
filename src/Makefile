CC = nvcc
FLAGS = -O3 -arch=sm_70 -rdc=true
LIBS = -lm

all: dmsim_omp


dmsim_omp: dmsim_omp.cu gate.cuh configuration.h circuit.cuh util.cuh
	$(CC) $(FLAGS) $(LIBS) -Xcompiler -fopenmp dmsim_omp.cu -o $@


clean:
	rm -rf *.o dmsim_omp
