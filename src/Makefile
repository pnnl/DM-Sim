# ---------------------------------------------------------------------------
# DM-Sim: Density-Matrix quantum circuit simulator based on GPU clusters
# Version 2.0
# ---------------------------------------------------------------------------
# File: Makefile
# Please update -arch with respect to the compute capability of your GPU.
# Pascal P100 GPU: sm_60
# Volta  V100 GPU: sm_70
# Turing T4   GPU: sm_75
# Ampere A100 GPU: sm_80
# ---------------------------------------------------------------------------
# Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
# Homepage: http://www.angliphd.com
# GitHub repo: http://www.github.com/pnnl/DM-Sim
# PNNL-IPID: 31919-E, ECCN: EAR99, IR: PNNL-SA-143160
# BSD Lincese.
# ---------------------------------------------------------------------------

CC = nvcc
FLAGS = -O3 -arch=sm_70 -std=c++11 -rdc=true 
LIBS = -lm

#all: adder_n10_omp dmsim_py_omp_wrapper dmsim_py_mpi_wrapper adder_n10_mpi 
all: adder_n10_omp 

adder_n10_omp: adder_n10_omp.cu gate_omp.cuh config.hpp util.cuh
	$(CC) $(FLAGS) $(LIBS) -Xcompiler -fopenmp adder_n10_omp.cu -o $@

adder_n10_mpi: adder_n10_mpi.cu gate_mpi.cuh config.hpp util.cuh
	$(CC) $(FLAGS) $(LIBS) -ccbin mpicxx adder_n10_mpi.cu -o $@

dmsim_py_omp_wrapper: py_omp_wrapper.cu gate_omp.cuh config.hpp util.cuh
	$(CC) $(FLAGS) $(LIBS) --compiler-options=" -fopenmp -Wall -shared -std=c++11 -fPIC" -I/usr/include/python2.7 -lpython2.7 `python -m pybind11 --includes` py_omp_wrapper.cu -o dmsim_py_omp_wrapper.so

dmsim_py_mpi_wrapper: py_mpi_wrapper.cu gate_mpi.cuh config.hpp util.cuh
	$(CC) $(FLAGS) $(LIBS) --compiler-options=" -Wall -shared -std=c++11 -fPIC" -I/usr/include/python2.7 -lpython2.7 `python -m pybind11 --includes` -I/opt/openmpi/include/ py_mpi_wrapper.cu -o dmsim_py_mpi_wrapper.so

clean:
	rm -rf *.o adder_n10_omp adder_n10_mpi dmsim_py_omp_wrapper.so dmsim_py_mpi_wrapper.so
