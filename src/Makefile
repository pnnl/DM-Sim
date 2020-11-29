# ---------------------------------------------------------------------------
# DM-Sim: Density-Matrix Quantum Circuit Simulation Environement 
# Version 2.2
# Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
# Homepage: http://www.angliphd.com
# GitHub repo: http://www.github.com/pnnl/DM-Sim
# PNNL-IPID: 31919-E, ECCN: EAR99, IR: PNNL-SA-143160
# BSD Lincese.
# ---------------------------------------------------------------------------
# File: Makefile
#
# ----------------- For CPU: --------------- 
# For AVX support:
# AVX256: -mavx2
# AVX512: -mavx512
#
# ----------------- For NVIDIA GPU: --------------- 
# Update -arch with corresponding compute capability:
# Pascal P100 GPU: sm_60
# Volta  V100 GPU: sm_70
# Turing T4   GPU: sm_75
# Ampere A100 GPU: sm_80
# ---------------------------------------------------------------------------

# GPU Compiler
NVCC = nvcc
NVCC_FLAGS = -O3 -arch=sm_70 -m64 -std=c++11 -rdc=true --compiler-options -fPIC

# CPU Compiler
CC = g++
CC_FLAGS = -O3 -m64 -std=c++11 -fPIC

# MPI Compiler
MPICC = mpicxx
MPICC_FLAGS = -O3 -m64 -std=c++11 -fPIC
MPI_INC = /share/apps/openmpi/3.1.3/gcc/7.3.0/include/ 

# Python
PY_NVGPU_OMP_FLAGS = --compiler-options=" -fopenmp -Wall -shared -std=c++11 -fPIC" -I/usr/include/python2.7 -lpython2.7 `python -m pybind11 --includes`
PY_NVGPU_MPI_FLAGS = --compiler-options=" -Wall -shared -std=c++11 -fPIC" -I/usr/include/python2.7 -lpython2.7 `python -m pybind11 --includes` -I$(MPI_INC)
PY_CPU_OMP_FLAGS = -fopenmp -Wall -shared -std=c++11 -fPIC -I/usr/include/python2.7 -lpython2.7 `python -m pybind11 --includes`
PY_CPU_MPI_FLAGS = -Wall -shared -std=c++11 -fPIC -I/usr/include/python2.7 -lpython2.7 `python -m pybind11 --includes` -I$(MPI_INC)

# QIR requires Clang, QIR and QIR-Bridge
QIRCC = /autofs/nccs-svm1_sw/summit/llvm/10.0.1-rc1/10.0.1-rc1-0/bin/clang++
CIRCC_FLAGS = -std=c++11 -m64 -O3 -I. -I$(QIR_BRIDGE_PUBLIC) -fPIC

QIR_BRIDGE_PUBLIC = /ccs/home/angli/work/dmsim_qir/qir-pnnl/src/Bridge/public
QIR_BRIDGE_BUILD = /ccs/home/angli/work/dmsim_qir/qir-pnnl/src/Bridge/build/Linux/Release
QIR_BRIDGE_FLAGS = -I. -I$(QIR_BRIDGE_PUBLIC) -L$(QIR_BRIDGE_BUILD)/lib/QIR -L$(QIR_BRIDGE_BUILD)/lib/Simulators -L$(QIR_BRIDGE_BUILD)/lib/GateDecompositions -L$(QIR_BRIDGE_BUILD)/test/QIR/ -ldefault_gates  -lsimulators -lqir_bridge_qis -lqir_qis -lqir_bridge -lqir_rt --compiler-options -fPIC

# Linking
LIBS = -lm

#OBJ = adder_n10_cpu_omp adder_n10_nvgpu_omp adder_n10_cpu_mpi adder_n10_nvgpu_mpi 
#OBJ = dmsim_py_nvgpu_omp.so dmsim_py_cpu_omp.so 
#OBJ = dmsim_py_nvgpu_mpi.so dmsim_py_cpu_mpi.so 

OBJ = vqe_qir_omp

#vqe_qir_mpi 

all: $(OBJ) 

#================================== CPU =======================================
adder_n10_cpu_omp: adder_n10_cpu_omp.cpp dmsim_cpu_omp.hpp config.hpp util_cpu.h
	$(CC) $(CC_FLAGS) $(LIBS) -fopenmp adder_n10_cpu_omp.cpp -o $@

adder_n10_cpu_mpi: adder_n10_cpu_mpi.cpp dmsim_cpu_mpi.hpp config.hpp util_cpu.h 
	$(MPICC) $(CC_FLAGS) $(LIBS) adder_n10_cpu_mpi.cpp -o $@

dmsim_py_cpu_omp.so: py_cpu_omp_wrapper.cpp dmsim_cpu_omp.hpp config.hpp util_cpu.h
	$(CC) $(CC_FLAGS) $(LIBS) $(PY_CPU_OMP_FLAGS) py_cpu_omp_wrapper.cpp -o $@

dmsim_py_cpu_mpi.so: py_cpu_mpi_wrapper.cpp dmsim_cpu_mpi.hpp config.hpp util_cpu.h
	$(MPICC) $(CC_FLAGS) $(LIBS) $(PY_CPU_MPI_FLAGS) py_cpu_mpi_wrapper.cpp -o $@

#============================== NVIDIA GPU ====================================
adder_n10_nvgpu_omp: adder_n10_nvgpu_omp.cu dmsim_nvgpu_omp.cuh config.hpp util.cuh
	$(NVCC) $(NVCC_FLAGS) $(LIBS) -Xcompiler -fopenmp adder_n10_nvgpu_omp.cu -o $@

adder_n10_nvgpu_mpi: adder_n10_nvgpu_mpi.cu dmsim_nvgpu_mpi.cuh config.hpp util.cuh
	$(NVCC) $(NVCC_FLAGS) $(LIBS) -ccbin mpicxx adder_n10_nvgpu_mpi.cu -o $@

dmsim_py_nvgpu_omp.so: py_nvgpu_omp_wrapper.cu dmsim_nvgpu_omp.cuh config.hpp util.cuh
	$(NVCC) $(NVCC_FLAGS) $(LIBS) $(PY_NVGPU_OMP_FLAGS) py_nvgpu_omp_wrapper.cu -o $@

dmsim_py_nvgpu_mpi.so: py_nvgpu_mpi_wrapper.cu dmsim_nvgpu_mpi.cuh config.hpp util.cuh
	$(NVCC) $(NVCC_FLAGS) $(LIBS) $(PY_NVGPU_MPI_FLAGS) py_nvgpu_mpi_wrapper.cu -o $@


#============================== Microsoft QIR ====================================
# Update config.hpp to change a backend (e.g., CPU, NVGPU)
vqe_qir_mpi: vqe.ll vqe_mpi_driver.cc qir_mpi_wrapper.cpp dmsim_nvgpu_mpi.cuh dmsim_cpu_mpi.hpp config.hpp
	$(QIRCC) $(QIRCC_FLAGS) -o vqe.o -c vqe.ll
	$(QIRCC) $(QIRCC_FLAGS) -o vqe_mpi_driver.o -c vqe_mpi_driver.cc
	$(NVCC) $(NVCC_FLAGS) $(QIR_BRIDGE_FLAGS) $(LIBS) -ccbin mpicxx qir_mpi_wrapper.cpp vqe.o vqe_mpi_driver.o -o $@

vqe_qir_omp: vqe.ll vqe_omp_driver.cc qir_omp_wrapper.cpp dmsim_nvgpu_omp.cuh dmsim_cpu_omp.hpp config.hpp
	$(QIRCC) $(QIRCC_FLAGS) -o vqe.o -c vqe.ll
	$(QIRCC) $(QIRCC_FLAGS) -o vqe_mpi_driver.o -c vqe_mpi_driver.cc
	$(CC) $(CC_FLAGS) $(QIR_BRIDGE_FLAGS) $(LIBS) -fopenmp qir_omp_wrapper.cpp vqe.o vqe_mpi_driver.o -o $@


clean:
	rm -rf *.o $(OBJ)

# Have Fun!
