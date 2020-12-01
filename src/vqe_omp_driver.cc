// ---------------------------------------------------------------------------
// DM-Sim: Density-Matrix Quantum Circuit Simulation Environement
// Version 2.2
// Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
// Homepage: http://www.angliphd.com
// GitHub repo: http://www.github.com/pnnl/DM-Sim
// PNNL-IPID: 31919-E, ECCN: EAR99, IR: PNNL-SA-143160
// BSD Lincese.
// ---------------------------------------------------------------------------
// File: vqe_omp_driver.cc
// The Vartional-Quantum-Eigensolver (VQE) driver for running Q# vqe on 
// scale-up DM-Sim. Tested on DGX-1V and ORNL Summit HPC.
// ---------------------------------------------------------------------------

#include <cassert>
#include <iostream>
#include "QuantumApi_I.hpp"
#include "BitStates.hpp"
#include "SimFactory.hpp"

//Calling the entry function defined in vqe.ll
extern "C" double Microsoft__Quantum__Samples__Chemistry__SimpleVQE__GetEnergyHydrogenVQE__body(double, double, double);
//Calling to get the simulator instance
extern "C" Microsoft::Quantum::ISimulator* GetDMSim(); 

int main(int argc, char *argv[])
{
    Microsoft::Quantum::ISimulator* dmsim = GetDMSim();
    Microsoft::Quantum::SetSimulatorForQIR(dmsim);

    double theta1 = atof(argv[1]);
    double theta2 = atof(argv[2]);
    double theta3 = atof(argv[3]);

    double jwTermEnergy = 0;
    jwTermEnergy = Microsoft__Quantum__Samples__Chemistry__SimpleVQE__GetEnergyHydrogenVQE__body(theta1, theta2, theta3);
    
    //std::cout << "*** Testing QIR VQE example with DM-Sim ***" << std::endl;
    //std::cout << "\n===============================\n";
    //std::cout << "VQE_jwTermEnergy is " << jwTermEnergy << std::endl;
    //std::cout << "===============================\n";
    
    std::cout << jwTermEnergy << std::endl;
    return 0;
}
