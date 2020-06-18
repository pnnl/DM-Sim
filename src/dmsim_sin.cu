// ---------------------------------------------------------------------------
// File: dmsim_sin.cuh
// Single-GPU implementation of DM-Sim. 
// No inter-GPU communication is required.
// ---------------------------------------------------------------------------
// See our SC-20 paper for detail.
// Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
// Homepage: http://www.angliphd.com
// GitHub repo: http://www.github.com/pnnl/DM-Sim
// PNNL-IPID: 31919-E, ECCN: EAR99, IR: PNNL-SA-143160
// BSD Lincese.
// ---------------------------------------------------------------------------

#include <stdio.h>
#include "util.cuh"
#include "gate.cuh"
#include "circuit.cuh"

//================ Single GPU version ===================
//Forward: In(dm_real, dm_imag) => Out(dm_real, dm_imag)
//Adjoint: In(dm_real, dm_imag) => Out(dm_real_buf, dm_imag_buf)
//Bakward: In(dm_real_buf, dm_imag_buf) => Out(dm_real_buf, dm_imag_buf)
__global__ void simulation(double* dm_real, double* dm_imag, int dev,
        double* dm_real_buf, double* dm_imag_buf, bool isforward)
{
    grid_group grid = this_grid(); 
    if (isforward)
    {
        circuit(dm_real, dm_imag);
    }
    else
    {
        block_transpose(dm_real_buf, dm_imag_buf, dm_real, dm_imag); //(out,in)
        grid.sync();
        circuit(dm_real_buf, dm_imag_buf);
    }
}

int main()
{
//=================================== Initialization =====================================
    int dev = 0; //Use GPU-0 by default
    cudaSafeCall(cudaSetDevice(dev));
    srand(RAND_SEED);
    
    const idxtype dm_num = DIM*DIM;
    const idxtype dm_size = dm_num*(idxtype)sizeof(double);
    const idxtype dm_num_per_GPU = dm_num; 
    const idxtype dm_size_per_GPU = dm_size;

    double* dm_real_cpu = NULL;
    double* dm_imag_cpu = NULL;
    double* dm_real_res = NULL;
    double* dm_imag_res = NULL;

    double* dm_real = NULL;
    double* dm_imag = NULL;
    double* dm_real_buf = NULL;
    double* dm_imag_buf = NULL;

//=================================== Settings =====================================
    SAFE_ALOC_HOST(dm_real_cpu, dm_size_per_GPU);
    SAFE_ALOC_HOST(dm_imag_cpu, dm_size_per_GPU);
    SAFE_ALOC_HOST(dm_real_res, dm_size_per_GPU);
    SAFE_ALOC_HOST(dm_imag_res, dm_size_per_GPU);

    memset(dm_real_cpu, 0, dm_size_per_GPU);
    memset(dm_imag_cpu, 0, dm_size_per_GPU);
    memset(dm_real_res, 0, dm_size_per_GPU);
    memset(dm_imag_res, 0, dm_size_per_GPU);

#ifdef RAND_INIT_DM
    for (idxtype i=0; i<dm_num_per_GPU; i++)
    {
        dm_real_cpu[i] = (double)rand() / (double)RAND_MAX - 0.5;
        dm_imag_cpu[i] = (double)rand() / (double)RAND_MAX - 0.5;
    }
#endif

    gpu_timer sim;
    double gpu_mem = 0;
    SAFE_ALOC_GPU(dm_real,dm_size_per_GPU);
    SAFE_ALOC_GPU(dm_imag,dm_size_per_GPU);
    SAFE_ALOC_GPU(dm_real_buf, dm_size_per_GPU);
    SAFE_ALOC_GPU(dm_imag_buf, dm_size_per_GPU);
    gpu_mem += dm_size_per_GPU*4;

    cudaSafeCall(cudaMemcpy(dm_real, dm_real_cpu, 
                dm_size_per_GPU, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(dm_imag, dm_imag_cpu, 
                dm_size_per_GPU, cudaMemcpyHostToDevice));

    //printf("======= DM_Sim using 1 GPUs with %.1lf MB GPU memory =======\n", gpu_mem/1024/1024);

    cudaSafeCall(cudaMemset(dm_real_buf, 0, dm_size_per_GPU));
    cudaSafeCall(cudaMemset(dm_imag_buf, 0, dm_size_per_GPU));

//=================================== Kernel =====================================
    dim3 gridDim(1,1,1);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    int numBlocksPerSm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, 
            simulation, THREADS_PER_BLOCK, 0);
    gridDim.x = numBlocksPerSm * deviceProp.multiProcessorCount;

    bool isforward = true;
    void* args[] = {&dm_real, &dm_imag, &dev, &dm_real_buf, 
        &dm_imag_buf, &isforward};

    cudaSafeCall(cudaDeviceSynchronize());

    sim.start_timer();
    //Empty function for shallow circuit
    DEEP_SIM(dm_real, dm_imag, gridDim); 
    //Empty function for deep circuit
    cudaLaunchCooperativeKernel((void*)simulation,gridDim,THREADS_PER_BLOCK,args,0);
    isforward = false;
    //Only perform adjoint operation for deep circuit
    cudaLaunchCooperativeKernel((void*)simulation,gridDim,THREADS_PER_BLOCK,args,0);
    //Empty function for shallow circuit
    DEEP_SIM(dm_real_buf, dm_imag_buf, gridDim); 
    sim.stop_timer();

    cudaSafeCall(cudaDeviceSynchronize());
    swap_pointers(&dm_real, &dm_real_buf);//Actual results stored in buf after adjoint
    swap_pointers(&dm_imag, &dm_imag_buf);//Actual results stored in buf after adjoint

//=================================== Copy Back =====================================
    cudaSafeCall(cudaMemcpy(dm_real_res, dm_real, 
                dm_size_per_GPU, cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(dm_imag_res, dm_imag, 
                dm_size_per_GPU, cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaDeviceSynchronize());

    SAFE_FREE_GPU(dm_real);
    SAFE_FREE_GPU(dm_imag);
    SAFE_FREE_GPU(dm_real_buf);
    SAFE_FREE_GPU(dm_imag_buf);

#ifdef PROFILE
    //printf("\n==== After Simulation ======\n");
    //print_dm(dm_real_res, dm_imag_res);
    //bool valid = valid_dm(dm_real_cpu, dm_imag_cpu, dm_real_res, dm_imag_res);
    bool valid = valid_dm_adjoint(dm_real_cpu, dm_imag_cpu, dm_real_res, dm_imag_res);
    printf("\n\tSingle GPU Version Validation: %s \n\n", valid ? "True" : "False"); 
#endif
    
    printf("\nnqubits:%d, ngpus:%d, comp:%.3lf, comm:%.3lf, sim:%.3lf, mem:%.3lf\n",
            N_QUBITS, 1, sim.measure(), 0.0, 
            sim.measure(), gpu_mem/1024/1024);

//=================================== Finalize =====================================
    SAFE_FREE_HOST(dm_real_cpu);
    SAFE_FREE_HOST(dm_imag_cpu);
    SAFE_FREE_HOST(dm_real_res);
    SAFE_FREE_HOST(dm_imag_res);
    
    return 0;
}

