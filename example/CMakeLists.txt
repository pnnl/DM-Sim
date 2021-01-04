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
PROJECT(DMSim)

IF(USE_NVGPU)
    IF (USE_OPENMP)
        FILE(GLOB_RECURSE EXAMPLE_SRCS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*_nvgpu_omp.cu")
    ENDIF()
    IF (USE_MPI)
        FILE(GLOB_RECURSE EXAMPLE_SRCS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*_nvgpu_mpi.cu")
    ENDIF()
ELSE()
    IF (USE_OPENMP)
        FILE(GLOB_RECURSE EXAMPLE_SRCS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*_cpu_omp.cpp")
    ENDIF()
    IF (USE_MPI)
        FILE(GLOB_RECURSE EXAMPLE_SRCS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*_cpu_mpi.cpp")
    ENDIF()
ENDIF()

FOREACH(EXAMPLE_SRC IN LISTS EXAMPLE_SRCS)
    GET_FILENAME_COMPONENT(EXAMPLE_EXE "${EXAMPLE_SRC}" NAME_WE)
    IF(USE_NVGPU)
        #CUDA_ADD_EXECUTABLE("${EXAMPLE_EXE}" "${EXAMPLE_SRC}")
        ADD_EXECUTABLE("${EXAMPLE_EXE}" "${EXAMPLE_SRC}")
        TARGET_LINK_LIBRARIES("${EXAMPLE_EXE}" "${CUDA_LIBRARIES}")
        SET_TARGET_PROPERTIES("${EXAMPLE_EXE}" PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    ELSE()
        ADD_EXECUTABLE("${EXAMPLE_EXE}" "${EXAMPLE_SRC}")
    ENDIF()
    
    #IF(USE_OPENMP)
    #TARGET_LINK_LIBRARIES("${EXAMPLE_EXE}" PUBLIC OpenMP::OpenMP_CXX)
    #ENDIF()

    IF(USE_MPI)
        TARGET_LINK_LIBRARIES("${EXAMPLE_EXE}" ${MPI_LIBRARIES})
    ENDIF()
ENDFOREACH(EXAMPLE_SRC)

