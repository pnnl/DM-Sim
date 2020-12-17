CMAKE_MINIMUM_REQUIRED(VERSION 3.9)

SET(CMAKE_C_COMPILER "/usr/bin/gcc")
SET(CMAKE_CXX_COMPILER "/usr/bin/g++")

MESSAGE(STATUS "CMake version: ${CMAKE_VERSION}")
PROJECT(DMSim VERSION 2.4 LANGUAGES CXX CUDA)

# Define frontend
OPTION(USE_CXX "Use C++ frontend" ON)
OPTION(USE_PYTHON "Use Python frontend" OFF)

# Define backend
OPTION(USE_NVGPU "Use NVIDIA GPU backend" ON)
OPTION(USE_CPU "Use CPU backend" OFF)

OPTION(USE_OPENMP "Use OpenMP for scale-up" ON)
OPTION(USE_MPI "Use MPI for scale-out" OFF)
OPTION(USE_AVX512 "Accelerated by AVX512 on X86-CPU platforms" OFF)

# Define GPU Arch
SET(CUDA_ARCH_BIN 70)

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
    SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11 -gencode arch=compute_${CUDA_ARCH_BIN},code=sm_${CUDA_ARCH_BIN} -rdc=true --compiler-options -fPIC")
    IF(USE_OPENMP)
        SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
    ENDIF()
    IF(USE_MPI)
        SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin mpicxx")
    ENDIF()
ENDIF()


# Python Frontend
IF(USE_PYTHON)
    FIND_PACKAGE(PythonInterp 3.4 REQUIRED)
    FIND_PACKAGE(PythonLibs 3.4 REQUIRED)
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



IF(USE_PYTHON)
    IF(USE_NVGPU)
        IF(USE_OPENMP)
            ADD_EXECUTABLE("dmsim_py_nvgpu_omp.so" "src/py_nvgpu_omp_wrapper.cu")
            #ADD_LIBRARY("dmsim_py_nvgpu_omp" SHARED "src/py_nvgpu_omp_wrapper.cu")
            TARGET_LINK_LIBRARIES("dmsim_py_nvgpu_omp.so" ${PYTHON_LIBRARIES})
        ENDIF()
        IF(USE_MPI)
            ADD_EXECUTABLE("dmsim_py_nvgpu_mpi.so" "src/py_nvgpu_mpi_wrapper.cu")
            TARGET_LINK_LIBRARIES("dmsim_py_nvgpu_mpi.so" ${MPI_LIBRARIES})
            TARGET_LINK_LIBRARIES("dmsim_py_nvgpu_mpi.so" ${PYTHON_LIBRARIES})
        ENDIF()
    ENDIF()
    IF(USE_CPU)
        IF(USE_OPENMP)
            ADD_EXECUTABLE("dmsim_py_cpu_omp.so" "src/py_cpu_omp_wrapper.cpp")
        ENDIF()
        IF(USE_MPI)
            ADD_EXECUTABLE("dmsim_py_cpu_mpi.so" "src/py_cpu_mpi_wrapper.cpp")
            TARGET_LINK_LIBRARIES("dmsim_py_cpu_mpi.so" ${MPI_LIBRARIES})
        ENDIF()
    ENDIF()
ENDIF()

INCLUDE_DIRECTORIES("src")
ADD_SUBDIRECTORY("example")
#ADD_SUBDIRECTORY("qasmbench")


    
#include_directories(${MPI_INCLUDE_PATH})

#find_package(CUDA REQUIRED)
##set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" "-O3 -arch=sm_70 -m64 -std=c++11 -rdc=true --compiler-options -fPIC -Xcompiler -fopenmp -lcudadevrt")

#set(CMAKE_CXX_FLAGS "-O3 -arch=sm_70 -m64 -std=c++11 -rdc=true --compiler-options -fPIC -Xcompiler -fopenmp -lcudadevrt")
#set(CMAKE_C_FLAGS "-O3 -arch=sm_70 -m64 -std=c++11 -rdc=true --compiler-options -fPIC -Xcompiler -fopenmp -lcudadevrt")

#set(CMAKE_INCLUDE_CURRENT_DIR ON)

#include_directories(${CUDA_INCLUDE_DIRS})
    
#find_package(OpenMP)

#set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)

#if(NOT CMAKE_BUILD_TYPE)
    #set(CMAKE_BUILD_TYPE Release)
#endif()

#add_executable(adder_n10_cpu_omp
#src/adder_n10_cpu_omp.cpp
#src/dmsim_cpu_omp.hpp
#src/config.hpp
#src/util_cpu.h
#)
#
#if(OpenMP_CXX_FOUND)
#target_link_libraries(adder_n10_cpu_omp PUBLIC OpenMP::OpenMP_CXX)
#endif()

#if(OpenMP_CXX_FOUND)
#target_link_libraries(adder_n10_cpu_omp PUBLIC OpenMP::OpenMP_CXX)
#endif()

#add_executable(adder_n10_nvgpu_omp
#src/adder_n10_nvgpu_omp.cu
#src/dmsim_nvgpu_omp.cuh
#src/config.hpp
#src/util.cuh
#)

#if(OpenMP_CXX_FOUND)
#target_link_libraries(adder_n10_nvgpu_omp PUBLIC OpenMP::OpenMP_CXX)
#endif()




#add_executable(adder_n10_cpu_mpi
    #src/adder_n10_cpu_mpi.cpp
    #src/dmsim_cpu_mpi.hpp
    #src/config.hpp
    #src/util_cpu.h
    #)
#target_link_libraries(adder_n10_cpu_mpi ${MPI_LIBRARIES})

#add_executable(adder_n10_nvgpu_mpi
#src/adder_n10_nvgpu_mpi.cu
#src/dmsim_nvgpu_mpi.cuh
#src/config.hpp
#src/util.cuh
#)
#target_link_libraries(adder_n10_nvgpu_mpi ${MPI_LIBRARIES})






#if(MPI_COMPILE_FLAGS)
    #set_target_properties(adder_n10_cpu_mpi PROPERTIES
        #COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
#endif()




#set_target_properties(adder
#PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
