# ---------------------------------------------------------------------------
# DM-Sim: Density-Matrix quantum circuit simulator based on GPU clusters
# Version 2.1
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
FLAGS = -O3 -arch=sm_70 -m64 -std=c++11 -rdc=true --compiler-options -fPIC
LIBS = -lm

# For QIR:
CXX = /autofs/nccs-svm1_sw/summit/llvm/10.0.1-rc1/10.0.1-rc1-0/bin/clang++
QIR_BRIDGE_PUBLIC = /ccs/home/angli/work/dmsim_qir/qir-pnnl/src/Bridge/public
QIR_BRIDGE_BUILD = /ccs/home/angli/work/dmsim_qir/qir-pnnl/src/Bridge/build/Linux/Release

CXX_FLAGS = -std=c++11 -m64 -O3 -I. -I$(QIR_BRIDGE_PUBLIC) -fPIC
QIR_FLAGS = -I. -I$(QIR_BRIDGE_PUBLIC) -L$(QIR_BRIDGE_BUILD)/lib/QIR -L$(QIR_BRIDGE_BUILD)/lib/Simulators -L$(QIR_BRIDGE_BUILD)/lib/GateDecompositions -L$(QIR_BRIDGE_BUILD)/test/QIR/ -ldefault_gates  -lsimulators -lqir_bridge_qis -lqir_qis -lqir_bridge -lqir_rt --compiler-options -fPIC


all: adder_n10_omp dmsim_py_omp_wrapper dmsim_py_mpi_wrapper adder_n10_mpi qir_vqe_mpi

adder_n10_omp: adder_n10_omp.cu gate_omp.cuh config.hpp util.cuh
	$(CC) $(FLAGS) $(LIBS) -Xcompiler -fopenmp adder_n10_omp.cu -o $@

adder_n10_mpi: adder_n10_mpi.cu gate_mpi.cuh config.hpp util.cuh
	$(CC) $(FLAGS) $(LIBS) -ccbin mpicxx adder_n10_mpi.cu -o $@

dmsim_py_omp_wrapper: py_omp_wrapper.cu gate_omp.cuh config.hpp util.cuh
	$(CC) $(FLAGS) $(LIBS) --compiler-options=" -fopenmp -Wall -shared -std=c++11 -fPIC" -I/usr/include/python2.7 -lpython2.7 `python -m pybind11 --includes` py_omp_wrapper.cu -o dmsim_py_omp_wrapper.so

dmsim_py_mpi_wrapper: py_mpi_wrapper.cu gate_mpi.cuh config.hpp util.cuh
	$(CC) $(FLAGS) $(LIBS) --compiler-options=" -Wall -shared -std=c++11 -fPIC" -I/usr/include/python2.7 -lpython2.7 `python -m pybind11 --includes` -I/opt/openmpi/include/ py_mpi_wrapper.cu -o dmsim_py_mpi_wrapper.so

qir_vqe_mpi: vqe.ll vqe_mpi_driver.cc qir_mpi_wrapper.cu gate_mpi.cuh
	$(CXX) $(CXX_FLAGS) -o vqe.o -c vqe.ll
	$(CXX) $(CXX_FLAGS) -o vqe_mpi_driver.o -c vqe_mpi_driver.cc
	$(CC) $(FLAGS) $(QIR_FLAGS) $(LIBS) -ccbin mpicxx qir_mpi_wrapper.cu vqe.o vqe_mpi_driver.o -o $@


clean:
	rm -rf *.o adder_n10_omp adder_n10_mpi dmsim_py_omp_wrapper.so dmsim_py_mpi_wrapper.so qir_vqe_mpi
