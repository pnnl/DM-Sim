// ---------------------------------------------------------------------------
// File: util.cuh
// Header file in which we defined some utility functions.
// ---------------------------------------------------------------------------
// See our SC-20 paper for detail.
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

#include "configuration.h"

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
        fprintf( stderr, "Error: NULL pointer at %s:%i.", file, line);
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


//======================================== Validation ==========================================
// print output density matrix
void print_dm(const double* dm_real_cpu, const double* dm_imag_cpu)
{
    for (int i=0; i<DIM; i++)
    {
        for (int j=0; j<DIM; j++)
        {
            printf("(%.1lf,%.1lf) ",dm_real_cpu[i*DIM+j],dm_imag_cpu[i*DIM+j]);
        }
        printf("\n");
    }
}

// verify output density matrix with cpu
bool valid_dm(const double* dm_real_cpu, const double* dm_imag_cpu, 
        const double* dm_real_res, const double* dm_imag_res)
{
    bool valid = true;
    for (int i=0; i<DIM; i++)
        for (int j=0; j<DIM; j++)
            if ((dm_real_cpu[i*DIM+j] - dm_real_res[i*DIM+j] > ERROR_BAR)
             || (dm_imag_cpu[i*DIM+j] - dm_imag_res[i*DIM+j] > ERROR_BAR))
            {
                valid = false;
                break;
                //printf("%lf,%lf ",dm_real_cpu[i*DIM+j] - dm_real_res[i*DIM+j], dm_imag_cpu[i*DIM+j] - dm_imag_res[i*DIM+j]);
            }
    return valid;
}

// verify output density matrix with cpu under adjoint operation
bool valid_dm_adjoint(const double* dm_real_cpu, const double* dm_imag_cpu, 
        const double* dm_real_res, const double* dm_imag_res)
{
    bool valid = true;
    for (int i=0; i<DIM; i++)
        for (int j=0; j<DIM; j++)
            if ((abs(dm_real_cpu[i*DIM+j] - dm_real_res[j*DIM+i]) > ERROR_BAR)
             || (abs(dm_imag_cpu[i*DIM+j] + dm_imag_res[j*DIM+i]) > ERROR_BAR))
            {
                //printf("%d,%d,%lf,%lf ",i,j,dm_real_cpu[i*DIM+j] - dm_real_res[j*DIM+i], dm_imag_cpu[i*DIM+j] + dm_imag_res[j*DIM+i]);
                
                valid = false;
                break;
            }
    return valid;
}

//======================================== Other ==========================================
inline void swap_pointers(double** pa, double** pb)
{
    double* tmp = (*pa); (*pa) = (*pb); (*pb) = tmp;
}

#endif
