// ---------------------------------------------------------------------------
// File: configuration_sample.h
// Configuration file in which we define the gate and runtime settings.
// configuration.h will be generated based on this configuration_sample.h file.
// ---------------------------------------------------------------------------
// See our SC-20 paper for detail.
// Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
// Homepage: http://www.angliphd.com
// GitHub repo: http://www.github.com/pnnl/DM-Sim
// PNNL-IPID: 31919-E, ECCN: EAR99, IR: PNNL-SA-143160
// BSD Lincese.
// ---------------------------------------------------------------------------

#ifndef CONFIGURATION_H
#define CONFIGURATION_H

// ================================= Configurations =====================================
// N_QUBITS is the number of qubits
// GPU_SCALE is 2^x of the number of GPUs, e.g., with 8 GPUs the GPU_SCALE is 3 (2^3=8)

#define N_QUBITS 10
#define GPU_SCALE 0 

// Index type: for N_QUBITS>16, we should use unsigned long long int, 
// otherwise we use unsigned int.
typedef unsigned idxtype;
// typedef unsigned long long idxtype;

// whether perform error check (possibly with overhead)
#define CUDA_ERROR_CHECK

// whether perform detailed profiling
//#define PROFILE

// random seed
#define RAND_SEED 5

// randomly initialize density-matrix
// #define RAND_INIT_DM 1

// Size of the density matrix
#define DIM ((idxtype)1<<(N_QUBITS))
#define HALF_DIM ((idxtype)1<<(N_QUBITS-1))

//number of GPUs
#define N_GPUS (1u<<GPU_SCALE)
//log2 of M_GPU
#define LG2_M_GPU (N_QUBITS-GPU_SCALE)
//share of columns per GPU with partition along columns (see our SC-20 paper)
#define M_GPU (1u<<LG2_M_GPU) 
//Tile for transposition in the adjoint operation
#define TILE 16
//Number of tiles along column per GPU
#define N_TILE ((M_GPU+TILE-1)/TILE)
//Threads per GPU Thread BLock (Fixed)
#define THREADS_PER_BLOCK 256
//Constant value of 1/sqrt(2)
#define S2I 0.70710678118654752440 
//Constant value of PI
#define PI  3.14159265358979323846
//Error bar for validation
#define ERROR_BAR (1e-3)

// =============================== Standard Gates ===================================
//3-parameter 2-pulse single qubit gate
#define U3(A,B,C,D) U3_GATE(dm_real, dm_imag, A, B, C, D)
//2-parameter 1-pulse single qubit gate
#define U2(A,B,C) U2_GATE(dm_real, dm_imag, A, B, C)
//1-parameter 0-pulse single qubit gate
#define U1(A,B) U1_GATE(dm_real, dm_imag, A, B)
//controlled-NOT
#define CX(M,N) CX_GATE(dm_real, dm_imag, M, N)
//idle gate(identity)
#define ID(M) ID_GATE(dm_real, dm_imag, M)
//Pauli gate: bit-flip
#define X(M) X_GATE(dm_real, dm_imag, M)
//Pauli gate: bit and phase flip
#define Y(M) Y_GATE(dm_real, dm_imag, M)
//Pauli gate: phase flip
#define Z(M) Z_GATE(dm_real, dm_imag, M)
//Clifford gate: Hadamard
#define H(M) H_GATE(dm_real, dm_imag, M)
//Clifford gate: sqrt(Z) phase gate
#define S(M) S_GATE(dm_real, dm_imag, M)
//Clifford gate: conjugate of sqrt(Z)
#define SDG(M) SDG_GATE(dm_real, dm_imag, M)
//C3 gate: sqrt(S) phase gate
#define T(M) T_GATE(dm_real, dm_imag, M)
//C3 gate: conjugate of sqrt(S)
#define TDG(M) TDG_GATE(dm_real, dm_imag, M)
//Rotation around X-axis
#define RX(A,B) RX_GATE(dm_real, dm_imag, A, B)
//Rotation around Y-axis
#define RY(A,B) RY_GATE(dm_real, dm_imag, A, B)
//Rotation around Z-axis
#define RZ(A,B) RZ_GATE(dm_real, dm_imag, A, B)

// =============================== Composition Gates ===================================
//Controlled-Phase
#define CZ(A,B) CZ_GATE(dm_real, dm_imag, A, B)
//Controlled-Y
#define CY(A,B) CY_GATE(dm_real, dm_imag, A, B)
//Swap
#define SWAP(A,B) SWAP_GATE(dm_real, dm_imag, A, B)
//Controlled-H
#define CH(A,B) CH_GATE(dm_real, dm_imag, A, B)
//C3 gate: Toffoli
#define CCX(A,B,C) CCX_GATE(dm_real, dm_imag, A, B, C)
//Fredkin gate
#define CSWAP(A,B,C) CSWAP_GATE(dm_real, dm_imag, A, B, C)
//Controlled RX rotation
#define CRX(L,A,B) CRX_GATE(dm_real, dm_imag, L, A, B)
//Controlled RY rotation
#define CRY(L,A,B) CRY_GATE(dm_real, dm_imag, L, A, B)
//Controlled RZ rotation
#define CRZ(L,A,B) CRZ_GATE(dm_real, dm_imag, L, A, B)
//Controlled phase rotation
#define CU1(L,A,B) CU1_GATE(dm_real, dm_imag, L, A, B)
//Controlled-U
#define CU3(T,P,L,A,B) CU3_GATE(dm_real, dm_imag, T, P, L, A, B)
//2-qubit XX rotation
#define RXX(T,A,B) RXX_GATE(dm_real, dm_imag, T, A, B)
//2-qubit ZZ rotation
#define RZZ(T,A,B) RZZ_GATE(dm_real, dm_imag, T, A, B)
//Relative-phase CCX
#define RCCX(A,B,C) RCCX_GATE(dm_real, dm_imag, A, B, C)
//Relative-phase 3-controlled X gate
#define RC3X(A,B,C,D) RC3X_GATE(dm_real, dm_imag, A, B, C, D)
//3-controlled X gate
#define C3X(A,B,C,D) C3X_GATE(dm_real, dm_imag, A, B, C, D)
//3-controlled sqrt(X) gate
#define C3SQRTX(A,B,C,D) C3SQRTX_GATE(dm_real, dm_imag, A, B, C, D)
//4-controlled X gate
#define C4X(A,B,C,D,E) C4X_GATE(dm_real, dm_imag, A, B, C, D, E)


// =============================== DM_Sim Native Gates ===================================
#define R(M,N) R_GATE(dm_real, dm_imag, M, N)
#define N(M) SRN_GATE(dm_real, dm_imag, M)
#define D(A,B,C,E,I) D_GATE(dm_real, dm_imag, A, B, C, E, I)

//Arbitrary 1-qubit gate
#define C1(E0R,E0I,E1R,E1I,E2R,E2I,E3R,E3I,I) C1_GATE(dm_real, dm_imag, E0R, \
        E0I, E1R, E1I, E2R, E2I, E3R, E3I, I) 
//Arbitrary 2-qubit gate
#define C2(E00R,E00I,E01R,E01I,E02R,E02I,E03R,E03I,\
              E10R,E10I,E11R,E11I,E12R,E12I,E13R,E13I,\
              E20R,E20I,E21R,E21I,E22R,E22I,E23R,E23I,\
              E30R,E30I,E31R,E31I,E32R,E32I,E33R,E33I,\
              M,N) C2_GATE(dm_real, dm_imag, \
                       E00R, E00I, E01R, E01I, E02R, E02I, E03R, E03I,\
                       E10R, E10I, E11R, E11I, E12R, E12I, E13R, E13I,\
                       E20R, E20I, E21R, E21I, E22R, E22I, E23R, E23I,\
                       E30R, E30I, E31R, E31I, E32R, E32I, E33R, E33I,\
                       M, N)

#endif
