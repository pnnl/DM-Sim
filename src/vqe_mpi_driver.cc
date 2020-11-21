// ---------------------------------------------------------------------------
// DM-Sim: Density-Matrix quantum circuit simulator based on GPU clusters
// Version 2.1
// ---------------------------------------------------------------------------
// File: vqe_mpi_driver.cc
// The Vartional-Quantum-Eigensolver (VQE) driver for running Q# vqe on 
// scale-out DM-Sim. Tested on ORNL Summit HPC.
// ---------------------------------------------------------------------------
// Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
// Homepage: http://www.angliphd.com
// GitHub repo: http://www.github.com/pnnl/DM-Sim
// PNNL-IPID: 31919-E, ECCN: EAR99, IR: PNNL-SA-143160
// BSD Lincese.
// ---------------------------------------------------------------------------

#include <cassert>
#include <iostream>
#include <mpi.h>
//#include <pybind11/pybind11.h>
//#include <pybind11/numpy.h>

#include "CoreTypes.hpp"
#include "IQuantumApi.hpp"
#include "ITranslator.hpp"
#include "QPFactory.hpp"

extern "C" Result ResultOne = nullptr;
extern "C" Result ResultZero = nullptr;

//Calling the entry function defined in vqe.ll
extern "C" double Microsoft__Quantum__Samples__Chemistry__SimpleVQE__GetEnergyHydrogenVQE__body(double, double, double);

extern "C" Result* measure(QUBIT* measureops, QUBIT* registers)
{
    printf("Here has some issues in qis__mesure()\n");
    fflush(stdout);
    //assert(0);
    Result* res = new Result[1];
    //Result res[4] = {Result(0), Result(0), Result(0), Result(0)};
    return res;
}

//These functions are for debugging code in vqe.ll
extern "C" void printval_i8(int8_t* x)
{
    std::cout << "== Value is " << int(*((char*)x)) << std::endl;
}

extern "C" void printval_i64(int64_t x)
{
    std::cout << "== Value is " << x << std::endl;
}

//argc and argv are required for MPI.
int main(int argc, char *argv[])
{
    //Initialize
    int n_gpus = 0;
    int i_gpu = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &i_gpu);
    MPI_Comm_size(MPI_COMM_WORLD, &n_gpus);

    double theta1 = atof(argv[1]);
    double theta2 = atof(argv[2]);
    double theta3 = atof(argv[3]);

    //if (i_gpu == 0) std::cout << "*** Testing QIR VQE example with DM-Sim ***" << std::endl;
    //if (i_gpu == 0) std::cout << theta1 << "," << theta2 << "," << theta3;

    double jwTermEnergy = 0;
    jwTermEnergy = 
        Microsoft__Quantum__Samples__Chemistry__SimpleVQE__GetEnergyHydrogenVQE__body(theta1, theta2, theta3);
    
    if (i_gpu == 0) 
    {
        //std::cout << "\n===============================\n";
        //std::cout << "VQE_jwTermEnergy is " << jwTermEnergy << std::endl;
        //std::cout << "===============================\n";
        std::cout << jwTermEnergy << std::endl;
    }
    //Finalize 
    MPI_Finalize();

    return 0;
}
