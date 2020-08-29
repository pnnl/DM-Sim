// ---------------------------------------------------------------------------
// DM-Sim: Density-Matrix quantum circuit simulator based on GPU clusters
// Version 2.0
// ---------------------------------------------------------------------------
// File: config.hpp
// Configuration file in which we define the gate and runtime settings.
// ---------------------------------------------------------------------------
// Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
// Homepage: http://www.angliphd.com
// GitHub repo: http://www.github.com/pnnl/DM-Sim
// PNNL-IPID: 31919-E, ECCN: EAR99, IR: PNNL-SA-143160
// BSD Lincese.
// ---------------------------------------------------------------------------
#ifndef CONFIG_H
#define CONFIG_H

//Error check for all CUDA Runtim-API calls and Kernel check
#define CUDA_ERROR_CHECK

// ================================= Configurations =====================================
namespace DMSim 
{
//Basic Type for indices, adjust to uint64_t when qubits > 15
using IdxType = unsigned;
//Basic Type for value, expect to support half, float and double
using ValType = double;
//Random seed
#define RAND_SEED 5
//Tile for transposition in the adjoint operation
#define TILE 16
//Threads per GPU Thread BLock (Fixed)
#define THREADS_PER_BLOCK 256
//Error bar for validation
#define ERROR_BAR (1e-3)
// constant value of PI
#define PI 3.14159265358979323846
// constant value of 1/sqrt(2)
#define S2I 0.70710678118654752440 

}; //namespace DMSim

#endif
