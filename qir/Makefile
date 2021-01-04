# ---------------------------------------------------------------------------
# DM-Sim: Density-Matrix Quantum Circuit Simulation Environement 
# ---------------------------------------------------------------------------
# Ang Li, Senior Computer Scientist
# Pacific Northwest National Laboratory(PNNL), U.S.
# Homepage: http://www.angliphd.com
# GitHub repo: http://www.github.com/pnnl/DM-Sim
# PNNL-IPID: 31919-E, ECCN: EAR99, IR: PNNL-SA-143160
# BSD Lincese.
# ---------------------------------------------------------------------------
# File: Makefile
# Please use CMake for more automatic configuration!!
# ---------------------------------------------------------------------------

include ../src/config.mk

# Please update config.hpp to change a backend (CPU, NVGPU, AMDGPU)

OBJ = vqe_qir_omp vqe_qir_mpi
all: $(OBJ) 

vqe_qir_omp: vqe.ll vqe_omp_driver.cc qir_omp_wrapper.cu ../src/dmsim_nvgpu_omp.cuh ../src/dmsim_cpu_omp.hpp ../src/config.hpp
	$(QIRCC) $(QIRCC_FLAGS) -I$(QIR_BRIDGE_PUBLIC) -o vqe.o -c vqe.ll
	$(QIRCC) $(QIRCC_FLAGS) -I$(QIR_BRIDGE_PUBLIC) -o vqe_omp_driver.o -c vqe_omp_driver.cc
	$(NVCC) $(NVCC_FLAGS) $(QIR_BRIDGE_FLAGS) $(LIBS) -Xcompiler -fopenmp qir_omp_wrapper.cu vqe.o vqe_omp_driver.o -o $@

vqe_qir_mpi: vqe.ll vqe_mpi_driver.cc qir_mpi_wrapper.cu ../src/dmsim_nvgpu_mpi.cuh ../src/dmsim_cpu_mpi.hpp ../src/config.hpp
	$(QIRCC) $(QIRCC_FLAGS) -I$(QIR_BRIDGE_PUBLIC) -o vqe.o -c vqe.ll
	$(QIRCC) $(QIRCC_FLAGS) -I$(QIR_BRIDGE_PUBLIC) -o vqe_mpi_driver.o -c vqe_mpi_driver.cc
	$(NVCC) $(NVCC_FLAGS) $(QIR_BRIDGE_FLAGS) $(LIBS) -ccbin mpicxx qir_mpi_wrapper.cu vqe.o vqe_mpi_driver.o -o $@


