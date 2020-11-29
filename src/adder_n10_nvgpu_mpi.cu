// ---------------------------------------------------------------------------
// DM-Sim: Density-Matrix Quantum Circuit Simulation Environment
// Version 2.2
// Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
// Homepage: http://www.angliphd.com
// GitHub repo: http://www.github.com/pnnl/DM-Sim
// PNNL-IPID: 31919-E, ECCN: EAR99, IR: PNNL-SA-143160
// BSD Lincese.
// ---------------------------------------------------------------------------
// File: adder_n10_nvgpu_mpi.cu
// A 10-qubit adder example based on MPI using NVIDIA GPU backend.
// !!!! This design requires GPUDirect-RDMA support !!!!
// ---------------------------------------------------------------------------

#include <stdio.h>
#include <mpi.h>
#include "util.cuh"
#include "dmsim_nvgpu_mpi.cuh"

//Use the DMSim namespace to enable C++/CUDA APIs
using namespace DMSim;

//You can define circuit module functions as below.
void majority(Simulation &sim, const IdxType a, const IdxType b, const IdxType c)
{
    sim.append(Simulation::CX(c, b));
    sim.append(Simulation::CX(c, a));
    sim.append(Simulation::CCX(a, b, c));
}
void unmaj(Simulation &sim, const IdxType a, const IdxType b, const IdxType c)
{
    sim.append(Simulation::CCX(a, b, c));
    sim.append(Simulation::CX(c, a));
    sim.append(Simulation::CX(a, b));
}
//argc and argv are required for MPI.
int main(int argc, char *argv[])
{
//=================================== Initialization =====================================
    //Initialize
    int n_gpus = 0;
    int i_gpu = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &i_gpu);
    MPI_Comm_size(MPI_COMM_WORLD, &n_gpus);
    //printf("Rank-%d of %d processes.\n", i_gpu, n_gpus);
    int n_qubits = 10;
    srand(RAND_SEED);

    //Obtain a simulator object
    Simulation sim(n_qubits);

    //Add the gates to the circuit
    sim.append(Simulation::X(1));
    sim.append(Simulation::X(5));
    sim.append(Simulation::X(6));
    sim.append(Simulation::X(7));
    sim.append(Simulation::X(8));

    //Call user-defined module functions 
    majority(sim, 0, 5, 1);
    majority(sim, 1, 6, 2);
    majority(sim, 2, 7, 3);
    majority(sim, 3, 8, 4);
    sim.append(Simulation::CX(4, 9));
    unmaj(sim, 3, 8, 4);
    unmaj(sim, 2, 7, 3);
    unmaj(sim, 1, 6, 2);
    unmaj(sim, 0, 5, 1);

    //Upload to GPU, ready for execution
    sim.upload();

    //Run the simulation
    sim.sim();

    //Measure
    auto res = sim.measure(5);
    if (i_gpu == 0) print_measurement(res, 10, 5);

    //Finalize 
    MPI_Finalize();
    return 0;
}

