// ---------------------------------------------------------------------------
// File: dmsim_mpi.cuh
// Multi-node-multi-GPU implementation of DM-Sim for scaling out. 
// We use MPI_All_to_All for Adjoint and GPUDirect-RDMA for 
// direct memory access.
// ---------------------------------------------------------------------------
// See our SC-20 paper for detail.
// Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
// Homepage: http://www.angliphd.com
// GitHub repo: http://www.github.com/pnnl/DM-Sim
// PNNL-IPID: 31919-E, ECCN: EAR99, IR: PNNL-SA-143160
// BSD Lincese.
// ---------------------------------------------------------------------------


#include <stdio.h>
#include <mpi.h>

#include "util.cuh"
#include "gate.cuh"
#include "circuit.cuh"

//================ Scale-out MPI version ===================
//Forward: In(dm_real, dm_imag) => Out(dm_real, dm_imag)
//Packing: In(dm_real, dm_imag) => Out(dm_real_buf, dm_imag_buf)
//All2All: In(dm_real_buf, dm_imag_buf) => Out(dm_real, dm_imag)
//Unpack:  In(dm_real, dm_imag) => Out(dm_real_buf, dm_imag_buf)
//BlkTran: In(dm_real_buf, dm_imag_buf) => Out(dm_real, dm_imag)
//Bakward: In(dm_real, dm_imag) => Out(dm_real, dm_imag)
__global__ void simulation(double* dm_real, double* dm_imag, int dev,
        double* dm_real_buf, double* dm_imag_buf, bool isforward)
{
    grid_group grid = this_grid(); 
    if (isforward)
    {
        circuit(dm_real, dm_imag);
        packing(dm_real, dm_imag, dm_real_buf, dm_imag_buf); //(in,out)
    }
    else
    {
        unpacking(dm_real_buf, dm_imag_buf, dm_real, dm_imag); //(out,in)
        grid.sync();
        block_transpose(dm_real, dm_imag, dm_real_buf, dm_imag_buf); //(out,in)
        grid.sync();
        circuit(dm_real, dm_imag);
    }
}


int main(int argc, char *argv[])
{
//=================================== Initialization =====================================
    int dev;
    int n_devs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &dev);
    MPI_Comm_size(MPI_COMM_WORLD, &n_devs);

    cudaSetDevice(0); 
    srand(RAND_SEED);

    //assert(DIM % n_devs == 0);
    //assert(n_devs == N_GPUS);

    const idxtype dm_num = DIM*DIM;
    const idxtype dm_size = dm_num*sizeof(double);

    const idxtype dm_num_per_GPU = dm_num / n_devs; 
    const idxtype dm_size_per_GPU = dm_size / n_devs;

    double* dm_real_cpu = NULL;
    double* dm_imag_cpu = NULL;
    double* dm_real_res = NULL;
    double* dm_imag_res = NULL;

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
    double* dm_real_buf = NULL;
    double* dm_imag_buf = NULL;
    float* comp_times = NULL;
    float* comm_times = NULL;
    float* sim_times = NULL;

//================================== Settings =====================================
    if (dev == 0)
    {
        SAFE_ALOC_HOST(comp_times, N_GPUS*sizeof(float));
        memset(comp_times, 0, N_GPUS*sizeof(float));
        SAFE_ALOC_HOST(comm_times, N_GPUS*sizeof(float));
        memset(comm_times, 0, N_GPUS*sizeof(float));
        SAFE_ALOC_HOST(sim_times, N_GPUS*sizeof(float));
        memset(sim_times, 0, N_GPUS*sizeof(float));
    }
    double total_gpu_mem = 0;
    gpu_timer comp, comm, sim;
    float gpu_mem = 0;
    double* dm_real = NULL;
    double* dm_imag = NULL;

    SAFE_ALOC_GPU(dm_real,dm_size_per_GPU);
    SAFE_ALOC_GPU(dm_imag,dm_size_per_GPU);
    gpu_mem += dm_size_per_GPU*2;

    cudaSafeCall(cudaMemcpy(dm_real, dm_real_cpu, 
                dm_size_per_GPU, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(dm_imag, dm_imag_cpu, 
                dm_size_per_GPU, cudaMemcpyHostToDevice));

    SAFE_ALOC_GPU(dm_real_buf, dm_size_per_GPU);
    SAFE_ALOC_GPU(dm_imag_buf, dm_size_per_GPU);
    gpu_mem += dm_size_per_GPU*2;
    
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
    void* args[] = {&dm_real, &dm_imag, &dev, &dm_real_buf, &dm_imag_buf, &isforward};

    cudaSafeCall(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    sim.start_timer();
    comp.start_timer();

    //Empty function for shallow circuit
    DEEP_SIM(dm_real, dm_imag, gridDim); 
    //Only perform packing function for deep circuit
    cudaLaunchCooperativeKernel((void*)simulation,gridDim,THREADS_PER_BLOCK,args,0);

    cudaSafeCall(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    comp.stop_timer();
    comm.start_timer();

    //All2All: In(dm_real_buf, dm_imag_buf) => Out(dm_real, dm_imag)
    MPI_Alltoall(dm_real_buf, M_GPU*M_GPU, MPI_DOUBLE,
            dm_real, M_GPU*M_GPU, MPI_DOUBLE,
            MPI_COMM_WORLD);
    MPI_Alltoall(dm_imag_buf, M_GPU*M_GPU, MPI_DOUBLE,
            dm_imag, M_GPU*M_GPU, MPI_DOUBLE,
            MPI_COMM_WORLD);

    comm.stop_timer();
    MPI_Barrier(MPI_COMM_WORLD);
    isforward = false;

    //Only perform unpacking function and adjoint for deep circuit
    cudaLaunchCooperativeKernel((void*)simulation,gridDim,THREADS_PER_BLOCK,args,0);
    //Empty function for shallow circuit
    DEEP_SIM(dm_real, dm_imag, gridDim);

    cudaSafeCall(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    sim.stop_timer();

//=================================== Copy Back =====================================
    cudaSafeCall(cudaMemcpy(dm_real_res, dm_real, 
                dm_size_per_GPU, cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(dm_imag_res, dm_imag, 
                dm_size_per_GPU, cudaMemcpyDeviceToHost));

    cudaSafeCall(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    float comp_time = comp.measure();
    float comm_time = comm.measure();
    float sim_time = sim.measure();

    MPI_Gather(&comp_time, 1, MPI_FLOAT,
            &comp_times[dev], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(&comm_time, 1, MPI_FLOAT,
            &comm_times[dev], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gather(&sim_time, 1, MPI_FLOAT,
            &sim_times[dev], 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

#ifdef PROFILE
    printf("\nGPU-%d: Comp: %.3fms, Comm: %.3fms, Sim: %.3fms, GMem:%.1f MB",dev,comp_time, comm_time, sim_time, gpu_mem/1024/1204);
#endif

    SAFE_FREE_GPU(dm_real);
    SAFE_FREE_GPU(dm_imag);
    SAFE_FREE_GPU(dm_real_buf);
    SAFE_FREE_GPU(dm_imag_buf);

#ifdef PROFILE
    //printf("\n==== After Simulation ======\n");
    //print_dm(dm_real_res, dm_imag_res);

    //bool valid = valid_dm(dm_real_cpu, dm_imag_cpu, dm_real_res, dm_imag_res);
    bool valid = valid_dm_adjoint(dm_real_cpu, dm_imag_cpu, dm_real_res, dm_imag_res);
    printf("\n\tMPI Version Validation: %s \n\n", valid ? "True" : "False"); 
#endif
    
    if (dev == 0) 
    {
        printf("\n======= DM_Sim using %d GPUs with %d MPI ranks,\
                %.1lf MB total GPU memory =======\n", 
                N_GPUS, n_devs, N_GPUS*gpu_mem/1024/1024);

        total_gpu_mem = gpu_mem * N_GPUS;
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

        printf("\nnqubits:%d, ngpus:%d, comp:%.3lf, comm:%.3lf, sim:%.3lf, mem:%.3lf\n",
                N_QUBITS, N_GPUS, (avg_sim_time-avg_comm_time), 
                avg_comm_time, avg_sim_time, total_gpu_mem/1024/1024);

        SAFE_FREE_HOST(comp_times);
        SAFE_FREE_HOST(comm_times);
        SAFE_FREE_HOST(sim_times);
    }

    SAFE_FREE_HOST(dm_real_cpu);
    SAFE_FREE_HOST(dm_imag_cpu);
    SAFE_FREE_HOST(dm_real_res);
    SAFE_FREE_HOST(dm_imag_res);

    MPI_Finalize();

    return 0;
}
