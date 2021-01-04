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
# File: CMakeLists.txt
# ---------------------------------------------------------------------------

CMAKE_MINIMUM_REQUIRED(VERSION 3.9)

#================================== Configuration =======================================
SET(CMAKE_C_COMPILER "/usr/bin/gcc")
SET(CMAKE_CXX_COMPILER "/usr/bin/g++")

MESSAGE(STATUS "CMake version: ${CMAKE_VERSION}")
PROJECT(DMSim VERSION 2.4 LANGUAGES CXX CUDA)

# Define frontend
OPTION(USE_CXX "Use C++ frontend" ON)
OPTION(USE_PYTHON "Use Python frontend" OFF)
OPTION(USE_QIR "Use QIR (Q#/QDK) frontend" OFF)

# Define backend
OPTION(USE_CPU "Use CPU backend" ON)
OPTION(USE_NVGPU "Use NVIDIA GPU backend" OFF)
OPTION(USE_AMD "Use AMD GPU backend" OFF)

# Define OMP/MPI
OPTION(USE_OPENMP "Use OpenMP for scale-up" ON)
OPTION(USE_MPI "Use MPI for scale-out" OFF)

# AVX512 for CPU backend
OPTION(USE_AVX512 "Accelerated by AVX512 on X86-CPU platforms" OFF)

# Define NVIDIA GPU Arch
SET(CUDA_ARCH_BIN 70)
IF (NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
    SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_BIN})
ENDIF()

# Define Clang++ for Q#/QIR
IF (USE_QIR)
    SET(CMAKE_C_COMPILER "/home/lian599/raid/qir/llvm-project/build/bin/clang")
    SET(CMAKE_CXX_COMPILER "/home/lian599/raid/qir/llvm-project/build/bin/clang++")
    SET(QIR_INC_PATH "/home/angli/raid/qir/irina/public_repo/qsharp-runtime/src/QirRuntime/")
    SET(QIR_LIB_PATH "/home/angli/raid/qir/irina/public_repo/qsharp-runtime/src/QirRuntime/build/Linux/Release/")
ENDIF()

# Debug or Release
IF (DEFINED DEBUG)
    MESSAGE("Building DM-Sim in DEBUG mode.")
    ADD_DEFINITIONS(-DDMSIM_DEBUG=${DEBUG})
    SET(CMAKE_BUILD_TYPE Debug)
ELSE()
    MESSAGE("Building DM-Sim in RELEASE mode.")
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

# Common flags
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c++11 -fPIC")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")

# Debug or Release
IF(DEFINED DEBUG)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
ELSE()
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
ENDIF()

# MPI
IF(USE_MPI)
    FIND_PACKAGE(MPI REQUIRED)
    INCLUDE_DIRECTORIES(SYSTEM ${MPI_INCLUDE_PATH})
ENDIF()

# OpenMP
IF(USE_OPENMP)
    FIND_PACKAGE(OpenMP REQUIRED)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")
ENDIF()

# X86 CPU with AVX512
IF(USE_CPU)
    IF(USE_AVX512)
        ADD_DEFINITIONS(-DUSE_AVX512)
        SET(CMAKE_C_FLAGS "-mavx512f ${CMAKE_C_FLAGS}")
        SET(CMAKE_CXX_FLAGS "-mavx512f ${CMAKE_CXX_FLAGS}")
    ENDIF(USE_AVX512)
ENDIF()

# NVIDIA GPU Backend
IF(USE_NVGPU)
    FIND_PACKAGE(CUDA REQUIRED)
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${CUDA_ARCH_BIN},code=sm_${CUDA_ARCH_BIN} -rdc=true --compiler-options -fPIC")
    IF(USE_OPENMP)
        SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
    ENDIF()
    IF(USE_MPI)
        SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin mpicxx")
    ENDIF()
ENDIF()

# AMD GPU Backend
IF(USE_AMDGPU)
    FIND_PACKAGE(hcc PATHS /opt/rocm/hcc)
    SET(CMAKE_C_COMPILER "/opt/rocm/hcc/bin/hipcc")
    SET(CMAKE_CXX_COMPILER "/opt/rocm/hcc/bin/hipcc")
ENDIF()

# Python Frontend
IF(USE_PYTHON)
    FIND_PACKAGE(pybind11)
    INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_DIRS})
    IF(USE_NVGPU)
        IF(USE_OPENMP)
            SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options=' -fopenmp -Wall -shared -std=c++11 -fPIC' ")
        ENDIF()
        IF(USE_MPI)
            SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options=' -Wall -shared -std=c++11 -fPIC' ")
        ENDIF()
    ENDIF()
    IF(USE_CPU)
        IF(USE_OPENMP)
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Wall -shared -std=c++11 -fPIC ")
        ENDIF()
        IF(USE_MPI)
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -shared -std=c++11 -fPIC ")
        ENDIF()
    ENDIF()
ENDIF()

# Q#/QDK QIR Frontend
IF(USE_QIR)
    INCLUDE_DIRECTORIES("${QIR_INC_PATH}/public/")
    INCLUDE_DIRECTORIES("${QIR_INC_PATH}/test/")
    INCLUDE_DIRECTORIES("src")
    #ADD_SUBDIRECTORY("qir")
ENDIF()



IF(USE_PYTHON)
    IF(USE_NVGPU)
        IF(USE_OPENMP)
            ADD_LIBRARY("dmsim_py_nvgpu_omp" SHARED "src/py_nvgpu_omp_wrapper.cu")
            TARGET_LINK_LIBRARIES("dmsim_py_nvgpu_omp" ${PYTHON_LIBRARIES})
            TARGET_LINK_LIBRARIES("dmsim_py_nvgpu_omp" ${OpenMP_CXX_FLAGS})
            SET_TARGET_PROPERTIES("dmsim_py_nvgpu_omp" PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
            SET_TARGET_PROPERTIES("dmsim_py_nvgpu_omp" PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

        ENDIF()
        IF(USE_MPI)
            ADD_LIBRARY("dmsim_py_nvgpu_mpi" SHARED "src/py_nvgpu_mpi_wrapper.cu")
            TARGET_LINK_LIBRARIES("dmsim_py_nvgpu_mpi" ${MPI_LIBRARIES})
            TARGET_LINK_LIBRARIES("dmsim_py_nvgpu_mpi" ${PYTHON_LIBRARIES})
            SET_TARGET_PROPERTIES("dmsim_py_nvgpu_mpi" PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
            SET_TARGET_PROPERTIES("dmsim_py_nvgpu_mpi" PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        ENDIF()

    ENDIF()
    IF(USE_CPU)
        IF(USE_OPENMP)
            ADD_LIBRARY("dmsim_py_cpu_omp" SHARED "src/py_cpu_omp_wrapper.cpp")
        ENDIF()
        IF(USE_MPI)
            ADD_LIBRARY("dmsim_py_cpu_mpi" SHARED "src/py_cpu_mpi_wrapper.cpp")
            TARGET_LINK_LIBRARIES("dmsim_py_cpu_mpi" ${MPI_LIBRARIES})
        ENDIF()
    ENDIF()
ENDIF()

INCLUDE_DIRECTORIES("src")
#INCLUDE_DIRECTORIES("example")
#INCLUDE_DIRECTORIES("qasmbench")
#INCLUDE_DIRECTORIES("qir")

