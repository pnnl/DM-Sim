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
# File: config.mk
# This is the file to configure Makefile.
# Please use CMake for more automatic configuration!!
# ----------------- For CPU: --------------- 
# For AVX512 acceleration support:
# AVX512: -mavx512f
# ----------------- For NVIDIA GPU: --------------- 
# Update -arch with corresponding compute capability:
# Pascal P100 GPU: sm_60
# Volta  V100 GPU: sm_70
# Turing T4   GPU: sm_75
# Ampere A100 GPU: sm_80
# ---------------------------------------------------------------------------

#======================= Compiler Configuration =============================
# CPU Compiler
CC = g++
# Should add -mavx512f for AVX512 acceleration
CC_FLAGS = -O3 -std=c++11 -fPIC 

# GPU Compiler
NVCC = nvcc
# Should adjust -arch=sm_XX based on NVIDIA GPU compute capability
NVCC_FLAGS = -O3 -arch=sm_70 -m64 -std=c++11 -rdc=true --compiler-options -fPIC

# AMD GPU Compiler
HIPCC = hipcc
HIPCC_FLAGS = -O3 -std=c++11 -fPIC

# MPI Compiler
MPICC = mpicxx
# Should add -mavx512f for AVX512 acceleration
MPICC_FLAGS = -O3 -std=c++11 -fPIC
# Should specify MPI include path
MPI_INC = /usr/include/mpi/

# Linking
LIBS = -lm

#========================= Python Configuration ==============================
# CPU
PY_CPU_OMP_FLAGS = -fopenmp -Wall -shared -std=c++11 -fPIC `python -m pybind11 --includes`
PY_CPU_MPI_FLAGS = -Wall -shared -std=c++11 -fPIC `python -m pybind11 --includes` -I$(MPI_INC)

# NVIDIA GPU
PY_NVGPU_OMP_FLAGS = --compiler-options=" -fopenmp -Wall -shared -std=c++11 -fPIC" `python -m pybind11 --includes`
PY_NVGPU_MPI_FLAGS = --compiler-options=" -Wall -shared -std=c++11 -fPIC" `python -m pybind11 --includes` -I$(MPI_INC)

# AMD GPU
PY_AMDGPU_OMP_FLAGS = -fopenmp -Wall -shared -std=c++11 -fPIC `python -m pybind11 --includes`
PY_AMDGPU_MPI_FLAGS = -Wall -shared -std=c++11 -fPIC `python -m pybind11 --includes` -I$(MPI_INC)


#==================== Microsoft Q#/QIR Configuration =========================

# QIR requires Clang, QIR and QIR-Bridge
# On OLCF Summit, it should be /autofs/nccs-svm1_sw/summit/llvm/10.0.1-rc1/10.0.1-rc1-0/bin/clang++
QIRCC = /home/lian599/raid/qir/llvm-project/build/bin/clang++
QIRCC_FLAGS = -std=c++11 -m64 -O3 -I. -I$(QIR_BRIDGE_PUBLIC) -I$(QIR_BRIDGE_TEST) -I$(MPI_INC) -fPIC

# QIR Bridge path
QIR_BRIDGE_PUBLIC = /home/angli/raid/qir/irina/public_repo/qsharp-runtime/src/QirRuntime/public/
QIR_BRIDGE_TEST = /home/angli/raid/qir/irina/public_repo/qsharp-runtime/src/QirRuntime/test/
QIR_BRIDGE_BUILD =  /home/angli/raid/qir/irina/public_repo/qsharp-runtime/src/QirRuntime/build/Linux/Release

# QIR Bridge linking flags
QIR_BRIDGE_FLAGS = -I. -I$(QIR_BRIDGE_PUBLIC) -I$(QIR_BRIDGE_TEST) -L$(QIR_BRIDGE_BUILD)/lib/QIR -L$(QIR_BRIDGE_BUILD)/lib/Simulators -lqir-bridge-qis-u -lqir-bridge-u -lqir-qis -lqir-qis-support -lqir-rt -lqir-rt-support -lsimulators


# Have Fun!
