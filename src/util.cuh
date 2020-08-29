// ---------------------------------------------------------------------------
// DM-Sim: Density-Matrix quantum circuit simulator based on GPU clusters
// Version 2.0
// ---------------------------------------------------------------------------
// File: util.cuh
// Header file in which we defined some utility functions.
// ---------------------------------------------------------------------------
// Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
// Homepage: http://www.angliphd.com
// GitHub repo: http://www.github.com/pnnl/DM-Sim
// PNNL-IPID: 31919-E, ECCN: EAR99, IR: PNNL-SA-143160
// BSD Lincese.
// ---------------------------------------------------------------------------

#ifndef UTIL_CUH
#define UTIL_CUH

#include <stdio.h>
#include <sys/time.h>
#include <assert.h>

#include "config.hpp"

namespace DMSim
{

//==================================== Error Checking =======================================
// Error checking for CUDA API call
#define cudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString(err));
        exit(-1);
    }
#endif
    return;
}

// Error checking for CUDA API call
#define cudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
    // Expensive checking
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif
    return;
}

// Checking null pointer
#define CHECK_NULL_POINTER(X) __checkNullPointer( __FILE__, __LINE__, (void**)&(X))
inline void __checkNullPointer( const char *file, const int line, void** ptr)
{
    if ((*ptr) == NULL)
    {
        fprintf( stderr, "Error: NULL pointer at %s:%i.\n", file, line);
        exit(-1);
    }
}

//================================= Allocation and Free ====================================
//CPU host allocation
#define SAFE_ALOC_HOST(X,Y) cudaSafeCall(cudaMallocHost((void**)&(X),(Y)));
//GPU device allocation
#define SAFE_ALOC_GPU(X,Y) cudaSafeCall(cudaMalloc((void**)&(X),(Y)));
//CPU host free
#define SAFE_FREE_HOST(X) if ((X) != NULL) { \
               cudaSafeCall( cudaFreeHost((X))); \
               (X) = NULL;}
//GPU device free
#define SAFE_FREE_GPU(X) if ((X) != NULL) { \
               cudaSafeCall( cudaFree((X))); \
               (X) = NULL;}

//======================================== Timer ==========================================
double get_cpu_timer()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    //get current timestamp in milliseconds
    return (double)tp.tv_sec * 1e3 + (double)tp.tv_usec * 1e-3;
}

// CPU Timer object definition
typedef struct CPU_Timer
{
    CPU_Timer() { start = stop = 0.0; }
    void start_timer() { start = get_cpu_timer(); }
    void stop_timer() { stop = get_cpu_timer(); }
    double measure() { double millisconds = stop - start; return millisconds; }
    double start;
    double stop;
} cpu_timer;

// GPU Timer object definition
typedef struct GPU_Timer
{
    GPU_Timer()
    {
        cudaSafeCall( cudaEventCreate(&this->start) );
        cudaSafeCall( cudaEventCreate(&this->stop) );
    }
    void start_timer() { cudaSafeCall( cudaEventRecord(this->start) ); }
    void stop_timer() { cudaSafeCall( cudaEventRecord(this->stop) ); }
    double measure()
    {
        cudaSafeCall( cudaEventSynchronize(this->stop) );
        float millisconds = 0;
        cudaSafeCall(cudaEventElapsedTime(&millisconds, this->start, this->stop) ); 
        return (double)millisconds;
    }
    cudaEvent_t start;
    cudaEvent_t stop;
} gpu_timer;


//======================================== Print ==========================================
void print_binary(IdxType v, int width)
{
    for (int i=width-1; i>=0; i--) putchar('0' + ((v>>i)&1));
}


//======================================== Other ==========================================
//Swap two pointers
inline void swap_pointers(ValType** pa, ValType** pb)
{
    ValType* tmp = (*pa); (*pa) = (*pb); (*pb) = tmp;
}
//Verify whether a number is power of 2
inline bool is_power_of_2(int x)
{
    return x > 0 && !(x & (x-1));
}


}; //namespace DMSim

#endif
