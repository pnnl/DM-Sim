// ---------------------------------------------------------------------------
// File: gate.cuh
// Implementation of the gates.
// ---------------------------------------------------------------------------
// See our SC-20 paper for detail.
// Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
// Homepage: http://www.angliphd.com
// GitHub repo: http://www.github.com/pnnl/DM-Sim
// PNNL-IPID: 31919-E, ECCN: EAR99, IR: PNNL-SA-143160
// BSD Lincese.
// ---------------------------------------------------------------------------

#ifndef GATE_CUH
#define GATE_CUH

#include <assert.h>
#include <cooperative_groups.h>

#include "configuration.h"

using namespace cooperative_groups;

//Define MG-BSP machine operation header (Original version with semantics)
#define OP_HEAD_ORIGIN grid_group grid = this_grid(); \
    const int tid = blockDim.x * blockIdx.x + threadIdx.x; \
    const idxtype outer_bound = (1 << ( N_QUBITS - qubit - 1)); \
    const idxtype inner_bound = (1 << qubit); \
        for (idxtype i = tid; i < outer_bound* inner_bound * M_GPU; i+=blockDim.x*gridDim.x){ \
            idxtype col = i / (inner_bound * outer_bound); \
        idxtype outer = (i % (inner_bound * outer_bound)) / inner_bound; \
        idxtype inner =  i % inner_bound; \
        idxtype offset = (2 * outer) * inner_bound; \
        idxtype pos0 = col * DIM + offset + inner; \
        idxtype pos1 = pos0 + inner_bound; 

//Define MG-BSP machine operation header (Optimized version)
#define OP_HEAD grid_group grid = this_grid(); \
    const int tid = blockDim.x * blockIdx.x + threadIdx.x; \
        for (idxtype i = tid; i < (HALF_DIM<<LG2_M_GPU); i+=blockDim.x*gridDim.x){ \
            idxtype col = (i >> (N_QUBITS-1)); \
            idxtype outer = ((i & (HALF_DIM-1)) >> qubit); \
            idxtype inner =  (i & ((1<<qubit)-1)); \
            idxtype offset = (outer << (qubit+1)); \
            idxtype pos0 = (col << (N_QUBITS)) + offset + inner; \
            idxtype pos1 = pos0 + (1<<qubit); 

//Define MG-BSP machine operation footer
#define OP_TAIL  } grid.sync(); 

//For subscription in Deep Simulation
#define DEEP_SIM(X,Y,Z)  deep_simulation((X),(Y),(Z)); 

//=================================== Gate Definition ==========================================

//============== Unified 1-qubit Gate ================
__device__ __inline__ void C1_GATE(double* dm_real, double* dm_imag, 
        const double e0_real, const double e0_imag,
        const double e1_real, const double e1_imag,
        const double e2_real, const double e2_imag,
        const double e3_real, const double e3_imag,
        const idxtype qubit)
{
    OP_HEAD;
    const double el0_real = dm_real[pos0]; 
    const double el0_imag = dm_imag[pos0];
    const double el1_real = dm_real[pos1]; 
    const double el1_imag = dm_imag[pos1];
    dm_real[pos0] = (e0_real * el0_real) - (e0_imag * el0_imag)
                   +(e1_real * el1_real) - (e1_imag * el1_imag);
    dm_imag[pos0] = (e0_real * el0_imag) + (e0_imag * el0_real)
                   +(e1_real * el1_imag) + (e1_imag * el1_real);
    dm_real[pos1] = (e2_real * el0_real) - (e2_imag * el0_imag)
                   +(e3_real * el1_real) - (e3_imag * el1_imag);
    dm_imag[pos1] = (e2_real * el0_imag) + (e2_imag * el0_real)
                   +(e3_real * el1_imag) + (e3_imag * el1_real);
    OP_TAIL;
}

//============== Unified 2-qubit Gate ================
__device__ __inline__ void C2_GATE(double* dm_real, double* dm_imag, 
        const double e00_real, const double e00_imag,
        const double e01_real, const double e01_imag,
        const double e02_real, const double e02_imag,
        const double e03_real, const double e03_imag,
        const double e10_real, const double e10_imag,
        const double e11_real, const double e11_imag,
        const double e12_real, const double e12_imag,
        const double e13_real, const double e13_imag,
        const double e20_real, const double e20_imag,
        const double e21_real, const double e21_imag,
        const double e22_real, const double e22_imag,
        const double e23_real, const double e23_imag,
        const double e30_real, const double e30_imag,
        const double e31_real, const double e31_imag,
        const double e32_real, const double e32_imag,
        const double e33_real, const double e33_imag,
        const idxtype qubit1, const idxtype qubit2)
{
    grid_group grid = this_grid(); 
    const int tid = blockDim.x * blockIdx.x + threadIdx.x; 
    const idxtype outer_bound1 = (1 << ( N_QUBITS - qubit1 - 1)); 
    const idxtype inner_bound1 = (1 << qubit1); 
    const idxtype outer_bound2 = (1 << ( N_QUBITS - qubit2 - 1)); 
    const idxtype inner_bound2 = (1 << qubit2); 
    for (idxtype i = tid; i < outer_bound1* inner_bound2 * M_GPU; i++)
    { 
        idxtype col = i / (inner_bound1 * outer_bound1); 
        idxtype outer1 = (i % (inner_bound1 * outer_bound1)) / inner_bound1; 
        idxtype inner1 =  i % inner_bound1; 
        idxtype offset1 = (2 * outer1) * inner_bound1; 
        idxtype pos0 = col * DIM + offset1 + inner1; 
        idxtype pos1 = pos0 + inner_bound1; 
        const double el0_real = dm_real[pos0]; 
        const double el0_imag = dm_imag[pos0];
        const double el1_real = dm_real[pos1]; 
        const double el1_imag = dm_imag[pos1];

        for (idxtype i = tid; i < outer_bound2* inner_bound2; i+=blockDim.x*gridDim.x)
        { 
            idxtype outer2 = i / inner_bound2; 
            idxtype inner2 = i % inner_bound2;
            idxtype offset2 = (2 * outer2) * inner_bound2; 
            idxtype pos2 = col * DIM + offset2 + inner2; 
            idxtype pos3 = pos2 + inner_bound2; 
            const double el2_real = dm_real[pos2]; 
            const double el2_imag = dm_imag[pos2];
            const double el3_real = dm_real[pos3]; 
            const double el3_imag = dm_imag[pos3];
            dm_real[pos0] = (e00_real * el0_real) - (e00_imag * el0_imag)
                           +(e01_real * el1_real) - (e01_imag * el1_imag)
                           +(e02_real * el2_real) - (e02_imag * el2_imag)
                           +(e03_real * el3_real) - (e03_imag * el3_imag);
            dm_real[pos1] = (e10_real * el0_real) - (e10_imag * el0_imag)
                           +(e11_real * el1_real) - (e11_imag * el1_imag)
                           +(e12_real * el2_real) - (e12_imag * el2_imag)
                           +(e13_real * el3_real) - (e13_imag * el3_imag);
            dm_real[pos1] = (e20_real * el0_real) - (e20_imag * el0_imag)
                           +(e21_real * el1_real) - (e21_imag * el1_imag)
                           +(e22_real * el2_real) - (e22_imag * el2_imag)
                           +(e23_real * el3_real) - (e23_imag * el3_imag);
            dm_real[pos1] = (e30_real * el0_real) - (e30_imag * el0_imag)
                           +(e31_real * el1_real) - (e31_imag * el1_imag)
                           +(e32_real * el2_real) - (e32_imag * el2_imag)
                           +(e33_real * el3_real) - (e33_imag * el3_imag);
        }
    }
}

//============== CX Gate ================
//Controlled-NOT or CNOT
/** CX   = [1 0 0 0]
           [0 1 0 0]
           [0 0 0 1]
           [0 0 1 0]
*/
__device__ __inline__ void CX_GATE(double* dm_real, double* dm_imag, const idxtype qubit,
        const idxtype ctrl)
{
    grid_group grid = this_grid(); 
    const int tid = blockDim.x * blockIdx.x + threadIdx.x; 
    const idxtype q0dim = (1 << max(ctrl, qubit) );
    const idxtype q1dim = (1 << min(ctrl, qubit) );
    assert (ctrl != qubit); //Non-cloning
    const idxtype outer_factor = (DIM + q0dim + q0dim - 1) >> (max(ctrl,qubit)+1);
    const idxtype mider_factor = (q0dim + q1dim + q1dim - 1) >> (min(ctrl,qubit)+1);
    const idxtype inner_factor = q1dim;

    for (idxtype i = tid; i < outer_factor * mider_factor * inner_factor * M_GPU; 
            i+=blockDim.x*gridDim.x)
    {
        idxtype col = i / (outer_factor * mider_factor * inner_factor);
        idxtype row = i % (outer_factor * mider_factor * inner_factor);
        idxtype outer = ((row/inner_factor) / (mider_factor)) * (q0dim+q0dim);
        idxtype mider = ((row/inner_factor) % (mider_factor)) * (q1dim+q1dim);
        idxtype inner = row % inner_factor;
        idxtype pos2 = col * DIM + outer + mider + inner;
        idxtype pos3 = col * DIM + outer + mider + inner + q1dim;
        const double el2_real = dm_real[pos2]; 
        const double el2_imag = dm_imag[pos2];
        const double el3_real = dm_real[pos3]; 
        const double el3_imag = dm_imag[pos3];

        if ((el2_real == 1.0) && (el2_imag == 0.0) && (el3_real == 0.0) && (el3_imag == 0.0))
        {
            idxtype pos0 = col * DIM + outer + mider + inner + q0dim;
            idxtype pos1 = col * DIM + outer + mider + inner + q0dim + q1dim;
            assert (pos0 < DIM*M_GPU); //ensure not out of bound
            assert (pos1 < DIM*M_GPU); //ensure not out of bound
            const double el0_real = dm_real[pos0]; 
            const double el0_imag = dm_imag[pos0];
            const double el1_real = dm_real[pos1]; 
            const double el1_imag = dm_imag[pos1];
            dm_real[pos0] = el1_real; 
            dm_imag[pos0] = el1_imag;
            dm_real[pos1] = el0_real; 
            dm_imag[pos1] = el0_imag;
        }
    }
    grid.sync();
}

//============== X Gate ================
//Pauli gate: bit flip
/** X = [0 1]
        [1 0]
*/
__device__ __inline__ void X_GATE(double* dm_real, double* dm_imag, const idxtype qubit)
{
    OP_HEAD;
    const double el0_real = dm_real[pos0]; 
    const double el0_imag = dm_imag[pos0];
    const double el1_real = dm_real[pos1]; 
    const double el1_imag = dm_imag[pos1];
    dm_real[pos0] = el1_real; 
    dm_imag[pos0] = el1_imag;
    dm_real[pos1] = el0_real; 
    dm_imag[pos1] = el0_imag;
    OP_TAIL;
}

//============== Y Gate ================
//Pauli gate: bit and phase flip
/** Y = [0 -i]
        [i  0]
*/
__device__ __inline__ void Y_GATE(double* dm_real, double* dm_imag, const idxtype qubit)
{
    OP_HEAD;
    const double el0_real = dm_real[pos0]; 
    const double el0_imag = dm_imag[pos0];
    const double el1_real = dm_real[pos1]; 
    const double el1_imag = dm_imag[pos1];
    dm_real[pos0] = el1_imag; 
    dm_imag[pos0] = -el1_real;
    dm_real[pos1] = -el0_imag;
    dm_imag[pos1] = el0_real;
    OP_TAIL;
}

//============== Z Gate ================
//Pauli gate: phase flip
/** Z = [1  0]
        [0 -1]
*/
__device__ __inline__ void Z_GATE(double* dm_real, double* dm_imag, const idxtype qubit)
{
    OP_HEAD;
    const double el1_real = dm_real[pos1]; 
    const double el1_imag = dm_imag[pos1];
    dm_real[pos1] = -el1_real;
    dm_imag[pos1] = -el1_imag;
    OP_TAIL;
}

//============== H Gate ================
//Clifford gate: Hadamard
/** H = 1/sqrt(2) * [1  1]
                    [1 -1]
*/
__device__ __inline__ void H_GATE(double* dm_real, double* dm_imag,  const idxtype qubit)
{
    OP_HEAD;
    const double el0_real = dm_real[pos0]; 
    const double el0_imag = dm_imag[pos0];
    const double el1_real = dm_real[pos1]; 
    const double el1_imag = dm_imag[pos1];
    dm_real[pos0] = S2I*(el0_real + el1_real); 
    dm_imag[pos0] = S2I*(el0_imag + el1_imag);
    dm_real[pos1] = S2I*(el0_real - el1_real);
    dm_imag[pos1] = S2I*(el0_imag - el1_imag);
    OP_TAIL;
}

//============== SRN Gate ================
//Square Root of X gate, it maps |0> to ((1+i)|0>+(1-i)|1>)/2,
//and |1> to ((1-i)|0>+(1+i)|1>)/2
/** SRN = 1/2 * [1+i 1-i]
                [1-i 1+1]
*/
__device__ __inline__ void SRN_GATE(double* dm_real, double* dm_imag,  const idxtype qubit)
{
    OP_HEAD;
    const double el0_real = dm_real[pos0]; 
    const double el0_imag = dm_imag[pos0];
    const double el1_real = dm_real[pos1]; 
    const double el1_imag = dm_imag[pos1];
    dm_real[pos0] = 0.5*( el0_real + el1_real); 
    dm_imag[pos0] = 0.5*( el0_imag - el1_imag);
    dm_real[pos1] = 0.5*( el0_real + el1_real);
    dm_imag[pos1] = 0.5*(-el0_imag + el1_imag);
    OP_TAIL;
}

//============== ID Gate ================
/** ID = [1 0]
         [0 1]
*/
__device__ __inline__ void ID_GATE(double* dm_real, double* dm_imag, const idxtype qubit)
{
}

//============== R Gate ================
//Phase-shift gate, it leaves |0> unchanged
//and maps |1> to e^{i\psi}|1>
/** R = [1 0]
        [0 0+p*i]
*/
__device__ __inline__ void R_GATE(double* dm_real, double* dm_imag, 
        const double phase, const idxtype qubit)
{
    OP_HEAD;
    const double el1_real = dm_real[pos1]; 
    const double el1_imag = dm_imag[pos1];
    dm_real[pos1] = -(el1_imag*phase);
    dm_imag[pos1] = el1_real*phase;
    OP_TAIL;
}

//============== S Gate ================
//Clifford gate: sqrt(Z) phase gate
/** S = [1 0]
        [0 i]
*/
__device__ __inline__ void S_GATE(double* dm_real, double* dm_imag,  const idxtype qubit)
{
    OP_HEAD;
    const double el1_real = dm_real[pos1]; 
    const double el1_imag = dm_imag[pos1];
    dm_real[pos1] = -el1_imag;
    dm_imag[pos1] = el1_real;
    OP_TAIL;
}

//============== SDG Gate ================
//Clifford gate: conjugate of sqrt(Z) phase gate
/** SDG = [1  0]
          [0 -i]
*/
__device__ __inline__ void SDG_GATE(double* dm_real, double* dm_imag,  const idxtype qubit)
{
    OP_HEAD;
    const double el1_real = dm_real[pos1]; 
    const double el1_imag = dm_imag[pos1];
    dm_real[pos1] = el1_imag;
    dm_imag[pos1] = -el1_real;
    OP_TAIL;
}

//============== T Gate ================
//C3 gate: sqrt(S) phase gate
/** T = [1 0]
        [0 s2i+s2i*i]
*/
__device__ __inline__ void T_GATE(double* dm_real, double* dm_imag, const idxtype qubit)
{
    OP_HEAD;
    const double el1_real = dm_real[pos1]; 
    const double el1_imag = dm_imag[pos1];
    dm_real[pos1] = S2I*(el1_real-el1_imag);
    dm_imag[pos1] = S2I*(el1_real+el1_imag);
    OP_TAIL;
}

//============== TDG Gate ================
//C3 gate: conjugate of sqrt(S) phase gate
/** TDG = [1 0]
          [0 s2i-s2i*i]
*/
__device__ __inline__ void TDG_GATE(double* dm_real, double* dm_imag, const idxtype qubit)
{
    OP_HEAD;
    const double el1_real = dm_real[pos1]; 
    const double el1_imag = dm_imag[pos1];
    dm_real[pos1] = S2I*( el1_real+el1_imag);
    dm_imag[pos1] = S2I*(-el1_real+el1_imag);
    OP_TAIL;
}


//============== D Gate ================
/** D = [e0_real+i*e0_imag 0]
        [0 e3_real+i*e3_imag]
*/
__device__ __inline__ void D_GATE(double* dm_real, double* dm_imag, 
        const double e0_real, const double e0_imag,
        const double e3_real, const double e3_imag,
        const idxtype qubit)
{
    OP_HEAD;
    const double el0_real = dm_real[pos0]; 
    const double el0_imag = dm_imag[pos0];
    const double el1_real = dm_real[pos1]; 
    const double el1_imag = dm_imag[pos1];
    dm_real[pos0] = (e0_real * el0_real) - (e0_imag * el0_imag);
    dm_imag[pos0] = (e0_real * el0_imag) + (e0_imag * el0_real);
    dm_real[pos1] = (e3_real * el1_real) - (e3_imag * el1_imag);
    dm_imag[pos1] = (e3_real * el1_imag) + (e3_imag * el1_real);
    OP_TAIL;
}

//============== U1 Gate ================
//1-parameter 0-pulse single qubit gate
__device__ __inline__ void U1_GATE(double* dm_real, double* dm_imag,
        const double lambda, const idxtype qubit)
{
    double e0_real = cos(-lambda/2.0);
    double e0_imag = sin(-lambda/2.0);
    double e3_real = cos(lambda/2.0);
    double e3_imag = sin(lambda/2.0);
    D(e0_real, e0_imag, e3_real, e3_imag, qubit);
}

//============== U2 Gate ================
//2-parameter 1-pulse single qubit gate
__device__ __inline__ void U2_GATE(double* dm_real, double* dm_imag,
        const double phi, const double lambda, const idxtype qubit)
{
    double e0_real = S2I * cos((-phi-lambda)/2.0);
    double e0_imag = S2I * sin((-phi-lambda)/2.0);
    double e1_real = -S2I * cos((-phi+lambda)/2.0);
    double e1_imag = -S2I * sin((-phi+lambda)/2.0);
    double e2_real = S2I * cos((phi-lambda)/2.0);
    double e2_imag = S2I * sin((phi-lambda)/2.0);
    double e3_real = S2I * cos((phi+lambda)/2.0);
    double e3_imag = S2I * sin((phi+lambda)/2.0);
    C1(e0_real, e0_imag, e1_real, e1_imag,
            e2_real, e2_imag, e3_real, e3_imag, qubit);
}

//============== U3 Gate ================
//3-parameter 2-pulse single qubit gate
__device__ __inline__ void U3_GATE(double* dm_real, double* dm_imag,
         const double theta, const double phi, 
         const double lambda, const idxtype qubit)
{
    double e0_real = cos(theta/2.0) * cos((-phi-lambda)/2.0);
    double e0_imag = cos(theta/2.0) * sin((-phi-lambda)/2.0);
    double e1_real = -sin(theta/2.0) * cos((-phi+lambda)/2.0);
    double e1_imag = -sin(theta/2.0) * sin((-phi+lambda)/2.0);
    double e2_real = sin(theta/2.0) * cos((phi-lambda)/2.0);
    double e2_imag = sin(theta/2.0) * sin((phi-lambda)/2.0);
    double e3_real = cos(theta/2.0) * cos((phi+lambda)/2.0);
    double e3_imag = cos(theta/2.0) * sin((phi+lambda)/2.0);
    C1(e0_real, e0_imag, e1_real, e1_imag,
            e2_real, e2_imag, e3_real, e3_imag, qubit);
}

//============== RX Gate ================
//Rotation around X-axis
__device__ __inline__ void RX_GATE(double* dm_real, double* dm_imag,
       const double theta, const idxtype qubit)
{
    double rx_real = cos(theta/2.0);
    double rx_imag = -sin(theta/2.0);
    OP_HEAD;
    const double el0_real = dm_real[pos0]; 
    const double el0_imag = dm_imag[pos0];
    const double el1_real = dm_real[pos1]; 
    const double el1_imag = dm_imag[pos1];
    dm_real[pos0] = (rx_real * el0_real) - (rx_imag * el1_imag);
    dm_imag[pos0] = (rx_real * el0_imag) + (rx_imag * el1_real);
    dm_real[pos1] =  - (rx_imag * el0_imag) +(rx_real * el1_real);
    dm_imag[pos1] =  + (rx_imag * el0_real) +(rx_real * el1_imag);
    OP_TAIL;
}

//============== RY Gate ================
//Rotation around Y-axis
__device__ __inline__ void RY_GATE(double* dm_real, double* dm_imag,
        const double theta, const idxtype qubit)
{
    double e0_real = cos(theta/2.0);
    double e1_real = -sin(theta/2.0);
    double e2_real = sin(theta/2.0);
    double e3_real = cos(theta/2.0);

    OP_HEAD;
    const double el0_real = dm_real[pos0]; 
    const double el0_imag = dm_imag[pos0];
    const double el1_real = dm_real[pos1]; 
    const double el1_imag = dm_imag[pos1];
    dm_real[pos0] = (e0_real * el0_real) +(e1_real * el1_real);
    dm_imag[pos0] = (e0_real * el0_imag) +(e1_real * el1_imag);
    dm_real[pos1] = (e2_real * el0_real) +(e3_real * el1_real);
    dm_imag[pos1] = (e2_real * el0_imag) +(e3_real * el1_imag);
    OP_TAIL;
}

//============== RZ Gate ================
//Rotation around Z-axis
__device__ __inline__ void RZ_GATE(double* dm_real, double* dm_imag,
     const double phi, const idxtype qubit)
{
    U1(phi, qubit);
}

//============== CZ Gate ================
//Controlled-Phase
__device__ __inline__ void CZ_GATE(double* dm_real, double* dm_imag,
        const idxtype a, const idxtype b)
{
    H(b);
    CX(a,b);
    H(b);
}

//============== CY Gate ================
//Controlled-Y
__device__ __inline__ void CY_GATE(double* dm_real, double* dm_imag,
        const idxtype a, const idxtype b)
{
    SDG(b);
    CX(a,b);
    S(b);
}

//============== CH Gate ================
//Controlled-H
__device__ __inline__ void CH_GATE(double* dm_real, double* dm_imag,
        const idxtype a, const idxtype b)
{
    H(b);
    SDG(b);
    CX(a,b);
    H(b); T(b);
    CX(a,b);
    T(b); H(b); S(b); X(b); S(a);
}

//============== CRZ Gate ================
//Controlled RZ rotation
__device__ __inline__ void CRZ_GATE(double* dm_real, double* dm_imag,
        const double lambda, const idxtype a, const idxtype b)
{
    U1(lambda/2, b);
    CX(a,b);
    U1(-lambda/2, b);
    CX(a,b);
}

//============== CU1 Gate ================
//Controlled phase rotation 
__device__ __inline__ void CU1_GATE(double* dm_real, double* dm_imag,
        const double lambda, const idxtype a, const idxtype b)
{
    U1(lambda/2, b);
    CX(a,b);
    U1(-lambda/2, b);
    CX(a,b);
    U1(lambda/2, b);
}

//============== CU1 Gate ================
//Controlled U
__device__ __inline__ void CU3_GATE(double* dm_real, double* dm_imag,
        const double theta, const double phi, const double lambda, 
        const idxtype c, const idxtype t)
{
    double temp1 = (lambda-phi)/2;
    double temp2 = theta/2;
    double temp3 = -(phi+lambda)/2;
    U1(temp1,t);
    CX(c,t);
    U3(-temp2,0,temp3,t);
    CX(c,t);
    U3(temp2,phi,0,t);
}

//========= Toffoli Gate ==========
__device__ __inline__ void CCX_GATE(double* dm_real, double* dm_imag,
        const idxtype a, const idxtype b, const idxtype c)
{
    H(c);
    CX(b,c); TDG(c);
    CX(a,c); T(c);
    CX(b,c); TDG(c);
    CX(a,c); T(b); T(c); H(c);
    CX(a,b); T(a); TDG(b);
    CX(a,b);
}

//========= SWAP Gate ==========
__device__ __inline__ void SWAP_GATE(double* dm_real, double* dm_imag,
        const idxtype a, const idxtype b)
{
    CX(a,b);
    CX(b,a);
    CX(a,b);
}

//========= Fredkin Gate ==========
__device__ __inline__ void CSWAP_GATE(double* dm_real, double* dm_imag,
        const idxtype a, const idxtype b, const idxtype c)
{
    CX(c,b);
    CCX(a,b,c);
    CX(c,b);
}

//============== CRX Gate ================
//Controlled RX rotation
__device__ __inline__ void CRX_GATE(double* dm_real, double* dm_imag,
       const double lambda, const idxtype a, const idxtype b)
{
    U1(PI/2, b);
    CX(a,b);
    U3(-lambda/2,0,0,b);
    CX(a,b);
    U3(lambda/2,-PI/2,0,b);
}
 
//============== CRY Gate ================
//Controlled RY rotation
__device__ __inline__ void CRY_GATE(double* dm_real, double* dm_imag,
       const double lambda, const idxtype a, const idxtype b)
{
    U3(lambda/2,0,0,b);
    CX(a,b);
    U3(-lambda/2,0,0,b);
    CX(a,b);
}
 
//============== RXX Gate ================
//2-qubit XX rotation
__device__ __inline__ void RXX_GATE(double* dm_real, double* dm_imag,
       const double theta, const idxtype a, const idxtype b)
{
    U3(PI/2,theta,0,a);
    H(b);
    CX(a,b);
    U1(-theta,b);
    CX(a,b);
    H(b);
    U2(-PI,PI-theta,a);
}
 
//============== RZZ Gate ================
//2-qubit ZZ rotation
__device__ __inline__ void RZZ_GATE(double* dm_real, double* dm_imag,
       const double theta, const idxtype a, const idxtype b)
{
    CX(a,b);
    U1(theta,b);
    CX(a,b);
}
 
//============== RCCX Gate ================
//Relative-phase CCX
__device__ __inline__ void RCCX_GATE(double* dm_real, double* dm_imag,
       const idxtype a, const idxtype b, const idxtype c)
{
    U2(0,PI,c);
    U1(PI/4,c);
    CX(b,c);
    U1(-PI/4,c);
    CX(a,c);
    U1(PI/4,c);
    CX(b,c);
    U1(-PI/4,c);
    U2(0,PI,c);
}
 
//============== RC3X Gate ================
//Relative-phase 3-controlled X gate
__device__ __inline__ void RC3X_GATE(double* dm_real, double* dm_imag,
       const idxtype a, const idxtype b, const idxtype c, const idxtype d)
{
    U2(0,PI,d);
    U1(PI/4,d);
    CX(c,d);
    U1(-PI/4,d);
    U2(0,PI,d);
    CX(a,d);
    U1(PI/4,d);
    CX(b,d);
    U1(-PI/4,d);
    CX(a,d);
    U1(PI/4,d);
    CX(b,d);
    U1(-PI/4,d);
    U2(0,PI,d);
    U1(PI/4,d);
    CX(c,d);
    U1(-PI/4,d);
    U2(0,PI,d);
}
 
//============== C3X Gate ================
//3-controlled X gate
__device__ __inline__ void C3X_GATE(double* dm_real, double* dm_imag,
       const idxtype a, const idxtype b, const idxtype c, const idxtype d)
{
    H(d); CU1(-PI/4,a,d); H(d);
    CX(a,b);
    H(d); CU1(PI/4,b,d); H(d);
    CX(a,b);
    H(d); CU1(-PI/4,b,d); H(d);
    CX(b,c);
    H(d); CU1(PI/4,c,d); H(d);
    CX(a,c);
    H(d); CU1(-PI/4,c,d); H(d);
    CX(b,c);
    H(d); CU1(PI/4,c,d); H(d);
    CX(a,c);
    H(d); CU1(-PI/4,c,d); H(d);
}
 
//============== C3SQRTX Gate ================
//3-controlled sqrt(X) gate, this equals the C3X gate where the CU1
//rotations are -PI/8 not -PI/4
__device__ __inline__ void C3SQRTX_GATE(double* dm_real, double* dm_imag,
       const idxtype a, const idxtype b, const idxtype c, const idxtype d)
{
    H(d); CU1(-PI/8,a,d); H(d);
    CX(a,b);
    H(d); CU1(PI/8,b,d); H(d);
    CX(a,b);
    H(d); CU1(-PI/8,b,d); H(d);
    CX(b,c);
    H(d); CU1(PI/8,c,d); H(d);
    CX(a,c);
    H(d); CU1(-PI/8,c,d); H(d);
    CX(b,c);
    H(d); CU1(PI/8,c,d); H(d);
    CX(a,c);
    H(d); CU1(-PI/8,c,d); H(d);
}
 
//============== C4X Gate ================
//4-controlled X gate
__device__ __inline__ void C4X_GATE(double* dm_real, double* dm_imag,
       const idxtype a, const idxtype b, const idxtype c, 
       const idxtype d, const idxtype e)
{
    H(e); CU1(-PI/2,d,e); H(e);
    C3X(a,b,c,d);
    H(d); CU1(PI/4,d,e); H(d);
    C3X(a,b,c,d);
    C3SQRTX(a,b,c,e);
}


//================================= Adjoint Related  ========================================

//Packing portions for all-to-all communication, see our paper for detail.
__device__ __inline__ void packing(const double* dm_real, const double* dm_imag,
        double* real_buf, double* imag_buf)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x; 
    for (idxtype i = tid; i < DIM * M_GPU; i+=blockDim.x*gridDim.x)
    {
        ////Original version with sementics
        //idxtype w_in_block = i / DIM;
        //idxtype block_id = (i % DIM) / M_GPU;
        //idxtype h_in_block = (i % DIM) % M_GPU;
        //idxtype id_in_dm = w_in_block*DIM+(i%DIM);
        //idxtype id_in_buf = block_id * M_GPU * M_GPU + w_in_block * M_GPU + h_in_block;

        //Optimized version
        idxtype w_in_block = (i >> N_QUBITS);
        idxtype block_id = (i & (DIM-1)) >> LG2_M_GPU;
        idxtype h_in_block = (i & (DIM-1)) & (M_GPU-1);
        idxtype id_in_dm = (w_in_block << N_QUBITS)+(i & (DIM-1));
        idxtype id_in_buf = (block_id << (LG2_M_GPU+LG2_M_GPU)) 
            + (w_in_block << LG2_M_GPU) + h_in_block;

        real_buf[id_in_buf] = dm_real[id_in_dm];
        imag_buf[id_in_buf] = dm_imag[id_in_dm];
    }
}

//Unpacking portions after all-to-all communication, see our paper for detail.
__device__ __inline__ void unpacking(double* send_real, double* send_imag,
        const double* recv_real, const double* recv_imag)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x; 
    for (idxtype i = tid; i < DIM * M_GPU; i+=blockDim.x*gridDim.x)
    {
        ////Original version with sementics
        //idxtype j = i / DIM; 
        //idxtype id_in_buf = j * DIM + (i % DIM);
        //idxtype block_id = id_in_buf / (M_GPU*M_GPU);
        //idxtype in_block_id = id_in_buf % (M_GPU*M_GPU);
        //idxtype w_in_block = in_block_id / M_GPU;
        //idxtype h_in_block = in_block_id % M_GPU;
        //idxtype dm_w = w_in_block;
        //idxtype dm_h = h_in_block + M_GPU*block_id;
        //idxtype id_in_dim = dm_w * DIM + dm_h;

        //Optimized version
        idxtype j = (i >> N_QUBITS); 
        idxtype id_in_buf = (j << N_QUBITS) + (i & (DIM-0x1));
        idxtype block_id = (id_in_buf >> (LG2_M_GPU+LG2_M_GPU));
        idxtype in_block_id = (id_in_buf & (M_GPU*M_GPU-0x1));
        idxtype w_in_block = (in_block_id >> LG2_M_GPU);
        idxtype h_in_block = (in_block_id & (M_GPU-1));
        idxtype dm_w = w_in_block;
        idxtype dm_h = h_in_block + (block_id<<LG2_M_GPU);
        idxtype id_in_dim = (dm_w << N_QUBITS) + dm_h;

        send_real[id_in_dim] = recv_real[id_in_buf]; 
        send_imag[id_in_dim] = recv_imag[id_in_buf]; 
    }
}

//Blockwise transpose via shared memory
__device__ __inline__ void block_transpose(double* dm_real, double* dm_imag, 
        const double* real_buf, const double* imag_buf)
{
    __shared__ double smem_real[TILE][TILE+1];
    __shared__ double smem_imag[TILE][TILE+1];
    idxtype tlx = threadIdx.x % TILE;
    idxtype tly = threadIdx.x / TILE;
    for (idxtype bid = blockIdx.x; bid < N_TILE*N_TILE*N_GPUS; bid += gridDim.x)
    {
        idxtype bz = bid / (N_TILE * N_TILE); 
        idxtype by = (bid % (N_TILE*N_TILE)) / N_TILE;
        idxtype bx = bid % N_TILE;
        idxtype tx = bx * TILE + tlx;
        idxtype ty = by * TILE + tly;

        if ((tlx < M_GPU) && (tly < M_GPU))
        {
            idxtype in_idx = ty*DIM+bz*M_GPU+tx;
            smem_real[tly][tlx] = real_buf[in_idx];
            smem_imag[tly][tlx] = -imag_buf[in_idx];
        }
        __syncthreads(); //The two ifs cannot merge, looks like a cuda bug on Volta GPU
        if ((tlx < M_GPU) && (tly < M_GPU))
        {
            idxtype out_idx = (bx*TILE+tly)*DIM+ bz*M_GPU + by*TILE+tlx;
            dm_real[out_idx] = smem_real[tlx][tly];
            dm_imag[out_idx] = smem_imag[tlx][tly];
        }
    } 
}







#endif
