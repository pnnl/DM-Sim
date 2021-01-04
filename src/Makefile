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

include config.mk

# CPU object (OpenMP + MPI)
CPU_PY_OBJ = libdmsim_py_cpu_omp.so libdmsim_py_cpu_mpi.so 
# NVGPU object (OpenMP + MPI)
NVGPU_PY_OBJ = libdmsim_py_nvgpu_omp.so libdmsim_py_nvgpu_mpi.so
# AMDGPU object (OpenMP + MPI)
AMDGPU_PY_OBJ = libdmsim_py_amdgpu_omp.so libdmsim_py_amdgpu_mpi.so

# Objects to be built
OBJ = $(CPU_PY_OBJ) $(NVGPU_PY_OBJ) $(AMDGPU_PY_OBJ)
all: $(OBJ) 

#================================== CPU =======================================
dmsim_py_cpu_omp.so: py_cpu_omp_wrapper.cpp dmsim_cpu_omp.hpp config.hpp util_cpu.h
	$(CC) $(CC_FLAGS) $(LIBS) $(PY_CPU_OMP_FLAGS) py_cpu_omp_wrapper.cpp -o $@

dmsim_py_cpu_mpi.so: py_cpu_mpi_wrapper.cpp dmsim_cpu_mpi.hpp config.hpp util_cpu.h
	$(MPICC) $(CC_FLAGS) $(LIBS) $(PY_CPU_MPI_FLAGS) py_cpu_mpi_wrapper.cpp -o $@

#============================== NVIDIA GPU ====================================
dmsim_py_nvgpu_omp.so: py_nvgpu_omp_wrapper.cu dmsim_nvgpu_omp.cuh config.hpp util_nvgpu.cuh
	$(NVCC) $(NVCC_FLAGS) $(LIBS) $(PY_NVGPU_OMP_FLAGS) py_nvgpu_omp_wrapper.cu -o $@

dmsim_py_nvgpu_mpi.so: py_nvgpu_mpi_wrapper.cu dmsim_nvgpu_mpi.cuh config.hpp util_nvgpu.cuh
	$(NVCC) $(NVCC_FLAGS) $(LIBS) $(PY_NVGPU_MPI_FLAGS) py_nvgpu_mpi_wrapper.cu -o $@

#============================== AMD GPU ====================================
libdmsim_py_amdgpu_omp.so: py_amdgpu_omp_wrapper.cpp dmsim_amdgpu_omp.hpp config.hpp util_amdgpu.hpp
	$(HIPCC) $(HIPCC_FLAGS) $(LIBS) $(PY_AMDGPU_OMP_FLAGS) py_amdgpu_omp_wrapper.cpp -o $@

libdmsim_py_amdgpu_mpi.so: py_amdgpu_mpi_wrapper.cpp dmsim_amdgpu_mpi.hpp config.hpp util_amdgpu.hpp
	$(HIPCC) $(HIPCC_FLAGS) $(LIBS) $(PY_AMDGPU_MPI_FLAGS) py_amdgpu_mpi_wrapper.cpp -o $@

#============================== Microsoft QIR ====================================
# Update config.hpp to change a backend (e.g., CPU, NVGPU, AMDGPU)
vqe_qir_mpi: vqe.ll vqe_mpi_driver.cc qir_mpi_wrapper.cu dmsim_nvgpu_mpi.cuh dmsim_cpu_mpi.hpp config.hpp
	$(QIRCC) $(QIRCC_FLAGS) -I$(QIR_BRIDGE_PUBLIC) -o vqe.o -c vqe.ll
	$(QIRCC) $(QIRCC_FLAGS) -I$(QIR_BRIDGE_PUBLIC) -o vqe_mpi_driver.o -c vqe_mpi_driver.cc
	$(NVCC) $(NVCC_FLAGS) $(QIR_BRIDGE_FLAGS) $(LIBS) -ccbin mpicxx qir_mpi_wrapper.cu vqe.o vqe_mpi_driver.o -o $@

vqe_qir_omp: vqe.ll vqe_omp_driver.cc qir_omp_wrapper.cu dmsim_nvgpu_omp.cuh dmsim_cpu_omp.hpp config.hpp
	$(QIRCC) $(QIRCC_FLAGS) -I$(QIR_BRIDGE_PUBLIC) -o vqe.o -c vqe.ll
	$(QIRCC) $(QIRCC_FLAGS) -I$(QIR_BRIDGE_PUBLIC) -o vqe_omp_driver.o -c vqe_omp_driver.cc
	$(NVCC) $(NVCC_FLAGS) $(QIR_BRIDGE_FLAGS) $(LIBS) -Xcompiler -fopenmp qir_omp_wrapper.cu vqe.o vqe_omp_driver.o -o $@

clean:
	rm -rf *.o $(OBJ)

