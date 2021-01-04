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

# CPU adder (OpenMP + MPI)
CPU_OBJ = adder_n10_cpu_omp adder_n10_cpu_mpi
# NVGPU object (OpenMP + MPI)
NVGPU_OBJ = adder_n10_nvgpu_omp adder_n10_nvgpu_mpi
# AMDGPU object (OpenMP + MPI)
AMDGPU_OBJ = adder_n10_amdgpu_omp adder_n10_amdgpu_mpi

# Objects to be built
OBJ = $(CPU_OBJ) $(NVGPU_OBJ) $(AMDGPU_OBJ)
all: $(OBJ) 

#================================== CPU =======================================
adder_n10_cpu_omp: adder_n10_cpu_omp.cpp ../src/dmsim_cpu_omp.hpp ../src/config.hpp ../src/util_cpu.h
	$(CC) $(CC_FLAGS) $(LIBS) -fopenmp adder_n10_cpu_omp.cpp -o $@

adder_n10_cpu_mpi: adder_n10_cpu_mpi.cpp ../src/dmsim_cpu_mpi.hpp ../src/config.hpp ../src/util_cpu.h 
	$(MPICC) $(CC_FLAGS) $(LIBS) adder_n10_cpu_mpi.cpp -o $@

#============================== NVIDIA GPU ====================================
adder_n10_nvgpu_omp: adder_n10_nvgpu_omp.cu ../src/dmsim_nvgpu_omp.cuh ../src/config.hpp ../src/util_nvgpu.cuh
	$(NVCC) $(NVCC_FLAGS) $(LIBS) -Xcompiler -fopenmp adder_n10_nvgpu_omp.cu -o $@

adder_n10_nvgpu_mpi: adder_n10_nvgpu_mpi.cu ../src/dmsim_nvgpu_mpi.cuh ../src/config.hpp ../src/util_nvgpu.cuh
	$(NVCC) $(NVCC_FLAGS) $(LIBS) -ccbin mpicxx adder_n10_nvgpu_mpi.cu -o $@

#============================== AMD GPU ====================================
adder_n10_amdgpu_omp: adder_n10_amdgpu_omp.cpp ../src/dmsim_amdgpu_omp.hpp ../src/config.hpp ../src/util_amdgpu.hpp
	$(HIPCC) $(HIPCC_FLAGS) $(LIBS) -fopenmp adder_n10_amdgpu_omp.cpp -o $@

adder_n10_amdgpu_mpi: adder_n10_amdgpu_mpi.cpp ../src/dmsim_amdgpu_mpi.hpp ../src/config.hpp ../src/util_amdgpu.hpp
	$(HIPCC) $(HIPCC_FLAGS) $(LIBS) -lmpi -I/usr/mpi/gcc/openmpi-4.0.3rc4/include/ -L/usr/mpi/gcc/openmpi-4.0.3rc4/lib64/ adder_n10_amdgpu_mpi.cpp -o $@



clean:
	rm -rf *.o $(OBJ)


