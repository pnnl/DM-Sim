// ---------------------------------------------------------------------------
// DM-Sim: Density-Matrix Quantum Circuit Simulation Environement.
// ---------------------------------------------------------------------------
// Ang Li, Senior Computer Scientist
// Pacific Northwest National Laboratory(PNNL), U.S.
// Homepage: http://www.angliphd.com
// GitHub repo: http://www.github.com/pnnl/DM-Sim
// PNNL-IPID: 31919-E, ECCN: EAR99, IR: PNNL-SA-143160
// BSD Lincese.
// ---------------------------------------------------------------------------
// File: util_amdgpu.hpp
// Header file in which we defined some utility functions.
// ---------------------------------------------------------------------------

#ifndef UTIL_AMDGPU_CUH
#define UTIL_AMDGPU_CUH

#include <stdio.h>
#include <sys/time.h>
#include <assert.h>
#include <hip/hip_runtime.h>

#include "config.hpp"

namespace DMSim
{

//==================================== Error Checking =======================================
// Error checking for HIP API call
#define hipSafeCall( err ) __hipSafeCall( err, __FILE__, __LINE__ )
inline void __hipSafeCall( hipError_t err, const char *file, const int line )
{
#ifdef HIP_ERROR_CHECK
    if ( hipSuccess != err )
    {
        fprintf(stderr, "hipSafeCall() failed at %s:%i : %s\n",
                 file, line, hipGetErrorString(err));
        exit(-1);
    }
#endif
    return;
}

// Error checking for HIP API call
#define hipCheckError()    __hipCheckError( __FILE__, __LINE__ )
inline void __hipCheckError( const char *file, const int line )
{
#ifdef HIP_ERROR_CHECK
    hipError_t err = hipGetLastError();
    if ( hipSuccess != err )
    {
        fprintf( stderr, "hipCheckError() failed at %s:%i : %s\n",
                 file, line, hipGetErrorString( err ) );
        exit( -1 );
    }
    // Expensive checking
    err = hipDeviceSynchronize();
    if( hipSuccess != err )
    {
        fprintf( stderr, "hipCheckError() with sync failed at %s:%i : %s\n",
                 file, line, hipGetErrorString( err ) );
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
#define SAFE_ALOC_HOST(X,Y) hipSafeCall(hipHostMalloc((void**)&(X),(Y)));
//GPU device allocation
#define SAFE_ALOC_GPU(X,Y) hipSafeCall(hipMalloc((void**)&(X),(Y)));
//CPU host free
#define SAFE_FREE_HOST(X) if ((X) != NULL) { \
               hipSafeCall( hipHostFree((X))); \
               (X) = NULL;}
//GPU device free
#define SAFE_FREE_GPU(X) if ((X) != NULL) { \
               hipSafeCall( hipFree((X))); \
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
        hipSafeCall( hipEventCreate(&this->start) );
        hipSafeCall( hipEventCreate(&this->stop) );
    }
    ~GPU_Timer()
    {
        hipSafeCall( hipEventDestroy(this->start));
        hipSafeCall( hipEventDestroy(this->stop));
    }
    void start_timer() { hipSafeCall( hipEventRecord(this->start) ); }
    void stop_timer() { hipSafeCall( hipEventRecord(this->stop) ); }
    double measure()
    {
        hipSafeCall( hipEventSynchronize(this->stop) );
        float millisconds = 0;
        hipSafeCall(hipEventElapsedTime(&millisconds, this->start, this->stop) ); 
        return (double)millisconds;
    }
    hipEvent_t start;
    hipEvent_t stop;
} gpu_timer;


//======================================== Print ==========================================
void print_binary(IdxType v, int width)
{
    for (int i=width-1; i>=0; i--) putchar('0' + ((v>>i)&1));
}

void print_measurement(IdxType* res_state, IdxType n_qubits, int repetition)
{
    assert(res_state != NULL);
    printf("\n===============  Measurement (tests=%d) ================\n", repetition);
    for (int i=0; i<repetition; i++)
    {
        printf("Test-%d: ",i);
        print_binary(res_state[i], n_qubits);
        printf("\n");
    }
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
    return (x > 0 && !(x & (x-1)));
}


}; //namespace DMSim

#endif
