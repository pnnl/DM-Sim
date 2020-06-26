// ---------------------------------------------------------------------------
// File: dmsim_omp.cuh
// Single-node-multi-GPU implementation of DM-Sim for scaling up. 
// Since we need all-to-all communication, if using NVLink, all 
// GPUs are required to be directly connected.
// ---------------------------------------------------------------------------
// See our SC-20 paper for detail.
// Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
// Homepage: http://www.angliphd.com
// GitHub repo: http://www.github.com/pnnl/DM-Sim
// PNNL-IPID: 31919-E, ECCN: EAR99, IR: PNNL-SA-143160
// BSD Lincese.
// ---------------------------------------------------------------------------

#include <stdio.h>
#include <omp.h>

#include "util.cuh"
#include "gate.cuh"
#include "circuit.cuh"

//================ Scale-up OpenMP version ===================
//Forward: In(dm_real, dm_imag) => Out(dm_real, dm_imag)
//Packing: In(dm_real, dm_imag) => Out(dm_real_buf, dm_imag_buf)
//All2All: In(dm_real_buf, dm_imag_buf) => Out(dm_real, dm_imag)
//Unpack:  In(dm_real, dm_imag) => Out(dm_real_buf, dm_imag_buf)
//BlkTran: In(dm_real_buf, dm_imag_buf) => Out(dm_real, dm_imag)
//Bakward: In(dm_real, dm_imag) => Out(dm_real, dm_imag)
__global__ void simulation(double* dm_real, double* dm_imag, int dev,
        double* dm_real_buf, double* dm_imag_buf,
        bool isforward)
{
    grid_group grid = this_grid(); 
    if (isforward) //Forward pass, see our paper
    {
        circuit(dm_real, dm_imag);
        packing(dm_real, dm_imag, dm_real_buf, dm_imag_buf); //(in,out)
    }
    else //Adjoint and Backward pass
    {
        unpacking(dm_real_buf, dm_imag_buf, dm_real, dm_imag); //(out,in)
        grid.sync();
        block_transpose(dm_real, dm_imag, dm_real_buf, dm_imag_buf); //(out,in)
        grid.sync();
        circuit(dm_real, dm_imag);
    }
}

int main()
{
//=================================== Initialization =====================================
    srand(RAND_SEED);
    int n_devs = N_GPUS;
    const idxtype slices_per_dev = M_GPU;
    const idxtype dm_num = DIM*DIM;
    const idxtype dm_size = dm_num*(idxtype)sizeof(double);

    double* dm_real_cpu = NULL;
    double* dm_imag_cpu = NULL;
    double* dm_real_res = NULL;
    double* dm_imag_res = NULL;


//=================================== Settings =====================================

    SAFE_ALOC_HOST(dm_real_cpu, dm_size);
    SAFE_ALOC_HOST(dm_imag_cpu, dm_size);
    SAFE_ALOC_HOST(dm_real_res, dm_size);
    SAFE_ALOC_HOST(dm_imag_res, dm_size);

    memset(dm_real_res, 0, dm_size);
    memset(dm_imag_res, 0, dm_size);
    memset(dm_real_cpu, 0, dm_size);
    memset(dm_imag_cpu, 0, dm_size);
    dm_real_cpu[0] = 1; //Initial State: all 0s.

#ifdef RAND_INIT_DM
    for (int i=0; i<DIM*DIM; i++)
    {
        dm_real_cpu[i] = (double)rand() / (double)RAND_MAX - 0.5;
        dm_imag_cpu[i] = (double)rand() / (double)RAND_MAX - 0.5;
    }
#endif
    assert(DIM % n_devs == 0);
    const idxtype dm_size_per_GPU = dm_size / n_devs;
    double* dm_real_buf[N_GPUS] = {NULL};
    double* dm_imag_buf[N_GPUS] = {NULL};
    double* dm_real[N_GPUS] = {NULL};
    double* dm_imag[N_GPUS] = {NULL};

    float comp_times[N_GPUS] = {0};
    float comm_times[N_GPUS] = {0};
    float sim_times[N_GPUS] = {0};
    double total_gpu_mem = 0;

#pragma omp parallel num_threads (n_devs)
    {
        int dev = omp_get_thread_num();
        cudaSetDevice(dev);
        gpu_timer comp, comm, sim;
        float gpu_mem = 0;

        for (int g=0; g<N_GPUS; g++)
        {
            if (g != dev)
            {
                cudaSafeCall( cudaDeviceEnablePeerAccess(g,0));
            }
        }
        cudaStream_t streams[n_devs];
        for (int i=0; i<n_devs; i++) 
        {
            cudaSafeCall(cudaStreamCreate(&streams[i]));
        }

        SAFE_ALOC_GPU(dm_real[dev],dm_size_per_GPU);
        SAFE_ALOC_GPU(dm_imag[dev],dm_size_per_GPU);
        gpu_mem += dm_size_per_GPU*2;

        cudaSafeCall(cudaMemcpy(dm_real[dev], &dm_real_cpu[dev*DIM*M_GPU], 
                    dm_size_per_GPU, cudaMemcpyHostToDevice));
        cudaSafeCall(cudaMemcpy(dm_imag[dev], &dm_imag_cpu[dev*DIM*M_GPU], 
                    dm_size_per_GPU, cudaMemcpyHostToDevice));

        SAFE_ALOC_GPU(dm_real_buf[dev], dm_size_per_GPU);
        SAFE_ALOC_GPU(dm_imag_buf[dev], dm_size_per_GPU);
        gpu_mem += dm_size_per_GPU*2;

        cudaSafeCall(cudaMemset(dm_real_buf[dev], 0, dm_size_per_GPU));
        cudaSafeCall(cudaMemset(dm_imag_buf[dev], 0, dm_size_per_GPU));

#pragma omp barrier
//=================================== Kernel =====================================
        
        dim3 gridDim(1,1,1);
        cudaDeviceProp deviceProp;
        cudaSafeCall(cudaGetDeviceProperties(&deviceProp, dev));
        int numBlocksPerSm;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, 
                simulation, THREADS_PER_BLOCK, 0);
        gridDim.x = numBlocksPerSm * deviceProp.multiProcessorCount;

        bool isforward = true;

        void* args[] = {&dm_real[dev], &dm_imag[dev], &dev, &dm_real_buf[dev], 
            &dm_imag_buf[dev], &isforward};
        cudaSafeCall(cudaDeviceSynchronize());

#pragma omp barrier

        sim.start_timer();
        comp.start_timer();

        //Empty function for shallow circuit
        DEEP_SIM(dm_real[dev], dm_imag[dev], gridDim); 
        //Only perform packing function for deep circuit
        cudaLaunchCooperativeKernel((void*)simulation,gridDim,THREADS_PER_BLOCK,args,0);
        
        cudaSafeCall(cudaDeviceSynchronize());
        comp.stop_timer();
        comm.start_timer();

#pragma unroll
        //All2All: In(dm_real_buf, dm_imag_buf) => Out(dm_real, dm_imag)
        for (int g = 0; g<N_GPUS; g++)
        {
            unsigned dst = (dev + g) % (N_GPUS); 
            cudaSafeCall( cudaMemcpyAsync(&dm_real[dst][M_GPU*M_GPU*dev], 
                        &dm_real_buf[dev][dst*M_GPU*M_GPU],
                        M_GPU*M_GPU*sizeof(double),
                        cudaMemcpyDefault, streams[dst]) );
            cudaSafeCall( cudaMemcpyAsync(&dm_imag[dst][M_GPU*M_GPU*dev], 
                        &dm_imag_buf[dev][dst*M_GPU*M_GPU],
                        M_GPU*M_GPU*sizeof(double),
                        cudaMemcpyDefault, streams[dst]) );
        }
        //for (int g = 0; g<N_GPUS; g++)
        //cudaSafeCall(cudaStreamSynchronize(streams[g]));
        cudaSafeCall(cudaStreamSynchronize(0));

        comm.stop_timer();

#pragma omp barrier

        isforward = false;
        //Only perform unpacking function and adjoint for deep circuit
        cudaLaunchCooperativeKernel((void*)simulation,gridDim,THREADS_PER_BLOCK,args,0);
        //Empty function for shallow circuit
        DEEP_SIM(dm_real[dev], dm_imag[dev], gridDim);
        sim.stop_timer();

//=================================== Copy Back =====================================
        cudaSafeCall(cudaDeviceSynchronize());
#pragma omp barrier
        cudaSafeCall(cudaMemcpy(&dm_real_res[dev*DIM*slices_per_dev], dm_real[dev], 
                    dm_size_per_GPU, cudaMemcpyDeviceToHost));
        cudaSafeCall(cudaMemcpy(&dm_imag_res[dev*DIM*slices_per_dev], dm_imag[dev], 
                    dm_size_per_GPU, cudaMemcpyDeviceToHost));

        comp_times[dev] = comp.measure();
        comm_times[dev] = comm.measure();
        sim_times[dev] = sim.measure();
        
        //printf("\nGPU-%d: Comp: %.3fms, Comm: %.3fms, Sim: %.3fms, GMem:%.1f MB",dev,comp_times[dev], comm_times[dev], sim_times[dev], gpu_mem/1024/1204);

#ifdef PROFILE
        printf("\nGPU-%d: Comp: %.3fms, Comm: %.3fms, Sim: %.3fms, GMem:%.1f MB",dev,comp_times[dev], comm_times[dev], sim_times[dev], gpu_mem/1024/1204);
#endif

//=================================== Finalize =====================================
        
        SAFE_FREE_GPU(dm_real[dev]);
        SAFE_FREE_GPU(dm_imag[dev]);
        SAFE_FREE_GPU(dm_real_buf[dev]);
        SAFE_FREE_GPU(dm_imag_buf[dev]);

        for (int g=0; g<N_GPUS; g++)
        {
            if (g != dev)
            {
                cudaSafeCall(cudaDeviceDisablePeerAccess(g));
            }
        }
        if (dev == 0) total_gpu_mem = gpu_mem * n_devs;
    }

#ifdef PROFILE
    //printf("\n==== After Simulation ======\n");
    //print_dm(dm_real_res, dm_imag_res);

    //bool valid = valid_dm(dm_real_cpu, dm_imag_cpu, dm_real_res, dm_imag_res);
    bool valid = valid_dm_adjoint(dm_real_cpu, dm_imag_cpu, dm_real_res, dm_imag_res);
    printf("\n\tOpenMP Version Validation: %s \n\n", valid ? "True" : "False"); 
#endif

    double avg_comp_time, avg_comm_time, avg_sim_time;
    for (int k=0; k<N_GPUS; k++)
    {
        avg_comm_time += comm_times[k];
        avg_comp_time += comp_times[k];
        avg_sim_time += sim_times[k];
    }
    avg_comp_time /= (double)N_GPUS;
    avg_comm_time /= (double)N_GPUS;
    avg_sim_time /= (double)N_GPUS;

    //printf("\n======= DM_Sim using %d GPUs with %.1lf MB GPU memory all-together =======\n", 
        //n_devs, total_gpu_mem/1024/1024);
    printf("\nnqubits:%d, ngpus:%d, comp:%.3lf, comm:%.3lf, sim:%.3lf, mem:%.3lf\n",
            N_QUBITS, N_GPUS, (avg_sim_time-avg_comm_time), avg_comm_time, 
            avg_sim_time, total_gpu_mem/1024/1024);

//=================================== Measure =====================================
    //print_sv(dm_real_res, dm_imag_res); //print state-vector (i.e., diag of resulting dm)
    //print_dm(dm_real_res, dm_imag_res); //print resulting density-matrix
    measurement(dm_real_res);

//=================================== Finalize =====================================
    SAFE_FREE_HOST(dm_real_cpu);
    SAFE_FREE_HOST(dm_imag_cpu);
    SAFE_FREE_HOST(dm_real_res);
    SAFE_FREE_HOST(dm_imag_res);

}

