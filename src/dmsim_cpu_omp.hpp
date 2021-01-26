// ---------------------------------------------------------------------------
// DM-Sim: Density-Matrix Quantum Circuit Simulation Environment
// ---------------------------------------------------------------------------
// Ang Li, Senior Computer Scientist
// Pacific Northwest National Laboratory(PNNL), U.S.
// Homepage: http://www.angliphd.com
// GitHub repo: http://www.github.com/pnnl/DM-Sim
// PNNL-IPID: 31919-E, ECCN: EAR99, IR: PNNL-SA-143160
// BSD Lincese.
// ---------------------------------------------------------------------------
// File: dmsim_cpu_omp.cuh
// OpenMP based CPU implementation of DM-Sim.
// ---------------------------------------------------------------------------

#ifndef DMSIM_CPU_OMP_CUH
#define DMSIM_CPU_OMP_CUH

#include <assert.h>
#include <vector>
#include <omp.h>
#include <sstream>
#include <string>
#include <math.h>
#include <string.h>
#include <cstdlib>
#include <immintrin.h>
#include <iostream>

#include "config.hpp"

namespace DMSim
{

using namespace std;
class Gate;
class Simulation;
using func_t =void (*)(const Gate*, const Simulation*, ValType*, ValType*);

//Simulation runtime, is_forward?
void simulation_kernel(Simulation*, bool);

//Current DMSim supported gates: 38
enum OP 
{
    U3, U2, U1, CX, ID, X, Y, Z, H, S, 
    SDG, T, TDG, RX, RY, RZ, CZ, CY, SWAP, CH, 
    CCX, CSWAP, CRX, CRY, CRZ, CU1, CU3, RXX, RZZ, RCCX, 
    RC3X, C3X, C3SQRTX, C4X, R, SRN, W, RYY
};

//Name of the gate for tracing purpose
const char *OP_NAMES[] = {
    "U3", "U2", "U1", "CX", "ID", "X", "Y", "Z", "H", "S", 
    "SDG", "T", "TDG", "RX", "RY", "RZ", "CZ", "CY", "SWAP", "CH", 
    "CCX", "CSWAP", "CRX", "CRY", "CRZ", "CU1", "CU3", "RXX", "RZZ", "RCCX", 
    "RC3X", "C3X", "C3SQRTX", "C4X", "R", "SRN", "W", "RYY"
};

//Define gate function pointers
extern  func_t pU3_OP;
extern  func_t pU2_OP;
extern  func_t pU1_OP;
extern  func_t pCX_OP;
extern  func_t pID_OP;
extern  func_t pX_OP;
extern  func_t pY_OP;
extern  func_t pZ_OP;
extern  func_t pH_OP;
extern  func_t pS_OP;
extern  func_t pSDG_OP;
extern  func_t pT_OP;
extern  func_t pTDG_OP;
extern  func_t pRX_OP;
extern  func_t pRY_OP;
extern  func_t pRZ_OP;
extern  func_t pCZ_OP;
extern  func_t pCY_OP;
extern  func_t pSWAP_OP;
extern  func_t pCH_OP;
extern  func_t pCCX_OP;
extern  func_t pCSWAP_OP;
extern  func_t pCRX_OP;
extern  func_t pCRY_OP;
extern  func_t pCRZ_OP;
extern  func_t pCU1_OP;
extern  func_t pCU3_OP;
extern  func_t pRXX_OP;
extern  func_t pRZZ_OP;
extern  func_t pRCCX_OP;
extern  func_t pRC3X_OP;
extern  func_t pC3X_OP;
extern  func_t pC3SQRTX_OP;
extern  func_t pC4X_OP;
extern  func_t pR_OP;
extern  func_t pSRN_OP;
extern  func_t pW_OP;
extern  func_t pRYY_OP;

//Gate definition, currently support up to 5 qubit indices and 3 rotation params
class Gate
{
public:
    Gate(enum OP _op_name, 
         IdxType _qb0, IdxType _qb1, IdxType _qb2, IdxType _qb3, IdxType _qb4, 
         ValType _theta, ValType _phi, ValType _lambda) : 
        op_name(_op_name),
        qb0(_qb0), qb1(_qb1), qb2(_qb2), qb3(_qb3), qb4(_qb4),
        theta(_theta), phi(_phi), lambda(_lambda) {}

    ~Gate() {}

    Gate* upload(int dev) 
    {
        Gate* local_copy;
        SAFE_ALOC_HOST(local_copy, sizeof(Gate)); 

#define GATE_BRANCH(GATE) case GATE: \
    this->op = p ## GATE ## _OP; break;
        switch (op_name)
        {
            GATE_BRANCH(U3);
            GATE_BRANCH(U2);
            GATE_BRANCH(U1);
            GATE_BRANCH(CX);
            GATE_BRANCH(ID);
            GATE_BRANCH(X);
            GATE_BRANCH(Y);
            GATE_BRANCH(Z);
            GATE_BRANCH(H);
            GATE_BRANCH(S);
            GATE_BRANCH(SDG);
            GATE_BRANCH(T);
            GATE_BRANCH(TDG);
            GATE_BRANCH(RX);
            GATE_BRANCH(RY);
            GATE_BRANCH(RZ);
            GATE_BRANCH(CZ);
            GATE_BRANCH(CY);
            GATE_BRANCH(SWAP);
            GATE_BRANCH(CH);
            GATE_BRANCH(CCX);
            GATE_BRANCH(CSWAP);
            GATE_BRANCH(CRX);
            GATE_BRANCH(CRY);
            GATE_BRANCH(CRZ);
            GATE_BRANCH(CU1);
            GATE_BRANCH(CU3);
            GATE_BRANCH(RXX);
            GATE_BRANCH(RZZ);
            GATE_BRANCH(RCCX);
            GATE_BRANCH(RC3X);
            GATE_BRANCH(C3X);
            GATE_BRANCH(C3SQRTX);
            GATE_BRANCH(C4X);
            GATE_BRANCH(R);
            GATE_BRANCH(SRN);
            GATE_BRANCH(W);
            GATE_BRANCH(RYY);
        }
        memcpy(local_copy, this, sizeof(Gate));
        return local_copy;
    }

    //applying the embedded gate operation on cpu side
   void exe_op(Simulation* sim, ValType* dm_real, ValType* dm_imag)
    {
        (*(this->op))(this, sim, dm_real, dm_imag);
    }
    //dump the current circuit
   void dump(std::stringstream& ss)
    {
        ss << OP_NAMES[op_name] << "(" << qb0 << "," << qb1 << "," 
            << qb2 << "," << qb3 << ","
            << qb4 << "," << theta << "," 
            << phi << "," << lambda << ");" << std::endl;
    }
    //Gate operation
    func_t op;
    //Gate name
    enum OP op_name;
    //Qubit position parameters
    IdxType qb0;
    IdxType qb1;
    IdxType qb2;
    IdxType qb3;
    IdxType qb4;
    //Qubit rotation parameters
    ValType theta;
    ValType phi;
    ValType lambda;
}; //end of Gate definition

class Simulation
{
public:
    Simulation(IdxType _n_qubits, IdxType _n_cpus) 
        : n_qubits(_n_qubits), 
        n_cpus(_n_cpus),
        dim((IdxType)1<<(n_qubits)), 
        half_dim((IdxType)1<<(n_qubits-1)),
        cpu_mem(0), 
        n_gates(0), 
        cpu_scale(floor(log((double)_n_cpus+0.5)/log(2.0))),
        lg2_m_cpu(n_qubits-cpu_scale),
        m_cpu((IdxType)1<<(lg2_m_cpu)),
        n_tile((m_cpu+TILE-1)/TILE),
        dm_num(dim*dim), 
        dm_size(dm_num*(IdxType)sizeof(ValType)),
        dm_size_per_cpu(dm_size/n_cpus),
        circuit_cpu(NULL),
        sim_cpu(NULL),
        dm_real(NULL),
        dm_imag(NULL),
        dm_real_buf(NULL),
        dm_imag_buf(NULL)
    {
        //CPU side initialization
        assert(is_power_of_2(n_cpus));
        assert(dim % n_cpus == 0);
        if (!is_power_of_2(n_cpus))
        {
            std::cerr << "Error: Number of CPU threads should be an exponential of 2." << std::endl;
            exit(1);
        }
        if (dim % n_cpus != 0)
        {
            std::cerr << "Error: Number of CPU threads is too large or too small." << std::endl;
            exit(1);
        }
        SAFE_ALOC_HOST(dm_real_cpu, dm_size);
        SAFE_ALOC_HOST(dm_imag_cpu, dm_size);
        SAFE_ALOC_HOST(dm_real_res, dm_size);
        SAFE_ALOC_HOST(dm_imag_res, dm_size);
        cpu_mem += dm_size*4;

        memset(dm_real_cpu, 0, dm_size);
        memset(dm_imag_cpu, 0, dm_size);
        memset(dm_real_res, 0, dm_size);
        memset(dm_imag_res, 0, dm_size);
        //Density matrix initial state [0..0] = 1
        dm_real_cpu[0] = 1.0;

        SAFE_ALOC_HOST(dm_real_buf_ptr, sizeof(ValType*)*n_cpus);
        SAFE_ALOC_HOST(dm_imag_buf_ptr, sizeof(ValType*)*n_cpus);
        SAFE_ALOC_HOST(dm_real_ptr, sizeof(ValType*)*n_cpus);
        SAFE_ALOC_HOST(dm_imag_ptr, sizeof(ValType*)*n_cpus);

        //cpu side initialization
        for (unsigned d=0; d<n_cpus; d++)
        {
            //cpu memory allocation
            SAFE_ALOC_HOST(dm_real_ptr[d], dm_size_per_cpu);
            SAFE_ALOC_HOST(dm_imag_ptr[d], dm_size_per_cpu);
            SAFE_ALOC_HOST(dm_real_buf_ptr[d], dm_size_per_cpu);
            SAFE_ALOC_HOST(dm_imag_buf_ptr[d], dm_size_per_cpu);
            cpu_mem += dm_size_per_cpu*4;
            //cpu memory initilization
            memcpy(dm_real_ptr[d], &dm_real_cpu[d*dim*m_cpu], dm_size_per_cpu);
            memcpy(dm_imag_ptr[d], &dm_imag_cpu[d*dim*m_cpu], dm_size_per_cpu);
            memset(dm_real_buf_ptr[d], 0, dm_size_per_cpu);
            memset(dm_imag_buf_ptr[d], 0, dm_size_per_cpu);
        }
    }

    ~Simulation()
    {
        //Release circuit
        clear_circuit();
        //Release for cpu side
        for (unsigned d=0; d<n_cpus; d++)
        {
            SAFE_FREE_HOST(dm_real_ptr[d]);
            SAFE_FREE_HOST(dm_imag_ptr[d]);
            SAFE_FREE_HOST(dm_real_buf_ptr[d]);
            SAFE_FREE_HOST(dm_imag_buf_ptr[d]);
        }
        //Release for CPU side
        SAFE_FREE_HOST(dm_real_cpu);
        SAFE_FREE_HOST(dm_imag_cpu);
        SAFE_FREE_HOST(dm_real_res);
        SAFE_FREE_HOST(dm_imag_res);

        SAFE_FREE_HOST(dm_real_ptr);
        SAFE_FREE_HOST(dm_imag_ptr);
        SAFE_FREE_HOST(dm_real_buf_ptr);
        SAFE_FREE_HOST(dm_imag_buf_ptr);
    }
   void reset()
    {
        clear_circuit();
        reset_dm();
    }
   void reset_dm()
    {
        memset(dm_real_cpu, 0, dm_size);
        memset(dm_imag_cpu, 0, dm_size);
        memset(dm_real_res, 0, dm_size);
        memset(dm_imag_res, 0, dm_size);
        //Density matrix initial state [0..0] = 1
        dm_real_cpu[0] = 1;
        //cpu side initialization
        for (unsigned d=0; d<n_cpus; d++)
        {
            //cpu memory initilization
            memcpy(dm_real_ptr[d], &dm_real_cpu[d*dim*m_cpu], dm_size_per_cpu);
            memcpy(dm_imag_ptr[d], &dm_imag_cpu[d*dim*m_cpu], dm_size_per_cpu);
            memset(dm_real_buf_ptr[d], 0, dm_size_per_cpu);
            memset(dm_imag_buf_ptr[d], 0, dm_size_per_cpu);
        }

    }
    //add a gate to the current circuit
   void append(Gate* g)
    {
        CHECK_NULL_POINTER(g); 
        assert((g->qb0<n_qubits));
        assert((g->qb1<n_qubits));
        assert((g->qb2<n_qubits));
        assert((g->qb3<n_qubits));
        assert((g->qb4<n_qubits));
 
        //Be careful! PyBind11 will auto-release the object pointed by g, 
        //so we need to creat a new Gate object inside the code
        circuit.push_back(new Gate(*g));
        n_gates++;
    }
    Simulation* upload()
    {
        assert(n_gates == circuit.size());
        //Should be null after calling clear_circuit()
        assert(circuit_cpu == NULL);
        assert(sim_cpu == NULL);

        SAFE_ALOC_HOST(sim_cpu, sizeof(Simulation*)*n_cpus);
        for (unsigned d=0; d<n_cpus; d++)
        {
            for (IdxType t=0; t<n_gates; t++)
            {
                //circuit[t]->dump();
                Gate* g_cpu = circuit[t]->upload(d);
                circuit_copy.push_back(g_cpu);
            }
            SAFE_ALOC_HOST(circuit_cpu, n_gates*sizeof(Gate*));
            memcpy(circuit_cpu, circuit_copy.data(), n_gates*sizeof(Gate*));
            dm_real = dm_real_ptr[d];
            dm_imag = dm_imag_ptr[d];
            dm_real_buf = dm_real_buf_ptr[d];
            dm_imag_buf = dm_imag_buf_ptr[d];

            SAFE_ALOC_HOST(sim_cpu[d], sizeof(Simulation));
            memcpy(sim_cpu[d], this, sizeof(Simulation));
        }
        return this;
    }
    //dump the circuit
    std::string dump()
    {
        stringstream ss;
        for (IdxType t=0; t<n_gates; t++)
        {
            circuit[t]->dump(ss);
        }
        return ss.str();
    }

    //start dm simulation
   void sim()
    {
        double* sim_times;
        double* comm_times;
        SAFE_ALOC_HOST(sim_times, sizeof(double)*n_cpus);
        SAFE_ALOC_HOST(comm_times, sizeof(double)*n_cpus);

#pragma omp parallel num_threads (n_cpus) 
        {
            int d = omp_get_thread_num();
            cpu_timer sim_timer;
            cpu_timer comm_timer;
            bool isforward = true;
            #pragma omp barrier
            //Forward Pass
            sim_timer.start_timer();
            simulation_kernel(sim_cpu[d], isforward);
            #pragma omp barrier

            comm_timer.start_timer();
            if (n_cpus > 1) //Need cpu-to-cpu communication only when there are multi-cpus
            {
                //Transepose All2All Communication: In(dm_real_buf, dm_imag_buf) => Out(dm_real, dm_imag)
                for (unsigned g = 0; g<n_cpus; g++)
                {
                    unsigned dst = (d + g) % (n_cpus); 
                    memcpy(&(dm_real_ptr[dst][m_cpu*m_cpu*d]), 
                            &(dm_real_buf_ptr[d][dst*m_cpu*m_cpu]),
                            m_cpu*m_cpu*sizeof(ValType));
                    memcpy(&(dm_imag_ptr[dst][m_cpu*m_cpu*d]), 
                            &(dm_imag_buf_ptr[d][dst*m_cpu*m_cpu]),
                            m_cpu*m_cpu*sizeof(ValType));
                }
            }
            comm_timer.stop_timer();
            #pragma omp barrier
            //Backward Pass
            isforward = false;
            simulation_kernel(sim_cpu[d], isforward);
            sim_timer.stop_timer();
            sim_times[d] = sim_timer.measure();
            comm_times[d] = comm_timer.measure();
            #pragma omp barrier
            //Copy back
            if (n_cpus == 1)
            {
                swap_pointers(&dm_real_ptr[d], &dm_real_buf_ptr[d]);
                swap_pointers(&dm_imag_ptr[d], &dm_imag_buf_ptr[d]);
                memcpy(dm_real_res, dm_real_ptr[d], dm_size);
                memcpy(dm_imag_res, dm_imag_ptr[d], dm_size);
            }
            else
            {
                memcpy(&dm_real_res[d*dim*m_cpu], dm_real_ptr[d], 
                            dm_size_per_cpu);
                memcpy(&dm_imag_res[d*dim*m_cpu], dm_imag_ptr[d], 
                            dm_size_per_cpu);
            }
        } //end of OpenMP parallel

        double avg_comm_time = 0;
        double avg_sim_time = 0;
        for (unsigned d=0; d<n_cpus; d++)
        {
            avg_comm_time += comm_times[d];
            avg_sim_time += sim_times[d];
        }
        avg_comm_time /= (double)n_cpus;
        avg_sim_time /= (double)n_cpus;

#ifdef PRINT_MEA_PER_CIRCUIT
        printf("\n============== DM-Sim ===============\n");
        printf("nqubits:%d, ngates:%d, ncores:%d, comp:%.3lf ms, comm:%.3lf ms, sim:%.3lf ms, mem:%.3lf MB, mem_per_cpu:%.3lf MB\n",
                n_qubits, n_gates, n_cpus, avg_sim_time-avg_comm_time, avg_comm_time, 
                avg_sim_time, cpu_mem/1024/1024, cpu_mem/1024/1024/n_cpus);
        printf("=====================================\n");
#endif

        SAFE_FREE_HOST(comm_times);
        SAFE_FREE_HOST(sim_times);
    }

   void clear_circuit()
    {
        if (sim_cpu != NULL)
        {
            for (unsigned d=0; d<n_cpus; d++)
            {
                SAFE_FREE_HOST(sim_cpu[d]);
            }
            for (unsigned i=0; i<n_gates; i++)
                SAFE_FREE_HOST(circuit_copy[i]);
            circuit_copy.clear();
        }
        for (unsigned i=0; i<n_gates; i++)
        {
            delete circuit[i];
        }
        SAFE_FREE_HOST(sim_cpu);
        SAFE_FREE_HOST(circuit_cpu);
        circuit.clear();
        n_gates = 0;
        dm_real = NULL;
        dm_imag = NULL;
        dm_real_buf = NULL;
        dm_imag_buf = NULL;
    }
    IdxType* measure(unsigned repetition=10)
    {
        IdxType sv_num = dim;
        IdxType sv_size = sv_num * sizeof(ValType);
        ValType* sv_diag = NULL;
        SAFE_ALOC_HOST(sv_diag, sv_size);

        //for (int i=0; i<sv_num; i++)
        //{
        //for (int j=0; j<sv_num; j++)
        //{
        //cout << dm_real_res[i*dim+j] << " ";
        //}
        //cout << endl;
        //}
        
        for (IdxType i=0; i<sv_num; i++)
        {
            sv_diag[i] = fabs(dm_real_res[i*dim+i]); //sv_diag[i] = dm_real_res[i*dim+i];
        }
        ValType* sv_diag_scan = NULL;
        SAFE_ALOC_HOST(sv_diag_scan, (sv_num+1)*sizeof(ValType));
        sv_diag_scan[0] = 0;
        for (IdxType i=1; i<sv_num+1; i++)
            sv_diag_scan[i] = sv_diag_scan[i-1]+sv_diag[i-1];
        srand(RAND_SEED);
        IdxType* res_state = new IdxType[repetition];
        memset(res_state, 0, (repetition*sizeof(IdxType)));
        for (unsigned i=0; i<repetition; i++)
        {
            ValType r = (ValType)rand()/(ValType)RAND_MAX;
            for (IdxType j=0; j<sv_num; j++)
                if (sv_diag_scan[j]<=r && r<sv_diag_scan[j+1])
                    res_state[i] = j;
        }
        if ( fabs(sv_diag_scan[sv_num] - 1.0) > ERROR_BAR )
            printf("Sum of probability along diag is far from 1.0 with %lf\n", sv_diag_scan[sv_num]);
        SAFE_FREE_HOST(sv_diag);
        SAFE_FREE_HOST(sv_diag_scan);

        return res_state;
    }
   void print_res_sv()
    {
        printf("----- Real SV ------\n");
        for (IdxType i=0; i<dim; i++) 
            printf("%lf ", dm_real_res[i*dim+i]);
        printf("\n");
        printf("----- Imag SV ------\n");
        for (IdxType i=0; i<dim; i++) 
            printf("%lf ", dm_imag_res[i*dim+i]);
        printf("\n");
    }
   void print_res_dm()
    {
        printf("----- Real DM------\n");
        for (IdxType i=0; i<dim; i++) 
        {
            for (IdxType j=0; j<dim; j++)
                printf("%lf ", dm_real_res[i*dim+j]);
            printf("\n");
        }
        printf("----- Imag DM------\n");
        for (IdxType i=0; i<dim; i++) 
        {
            for (IdxType j=0; j<dim; j++)
                printf("%lf ", dm_imag_res[i*dim+j]);
            printf("\n");
        }
    }
    // =============================== Standard Gates ===================================
    //3-parameter 2-pulse single qubit gate
    static Gate* U3(IdxType m, ValType theta, ValType phi, ValType lambda)
    {
        return new Gate(OP::U3, m, 0, 0, 0, 0, theta, phi, lambda);
    }
    //2-parameter 1-pulse single qubit gate
    static Gate* U2(IdxType m, ValType phi, ValType lambda)
    {
        return new Gate(OP::U2, m, 0, 0, 0, 0, 0., phi, lambda);
    }
    //1-parameter 0-pulse single qubit gate
    static Gate* U1(IdxType m, ValType lambda)
    {
        return new Gate(OP::U1, m, 0, 0, 0, 0, 0., 0., lambda);
    }
    //controlled-NOT
    static Gate* CX(IdxType m, IdxType n)
    {
        return new Gate(OP::CX, m, n, 0, 0, 0, 0., 0., 0.);
    }
    //idle gate(identity)
    static Gate* ID(IdxType m)
    {
        return new Gate(OP::ID, m, 0, 0, 0, 0, 0., 0., 0.);
    }
    //Pauli gate: bit-flip
    static Gate* X(IdxType m)
    {
        return new Gate(OP::X, m, 0, 0, 0, 0, 0., 0., 0.);
    }
    //Pauli gate: bit and phase flip
    static Gate* Y(IdxType m)
    {
        return new Gate(OP::Y, m, 0, 0, 0, 0, 0., 0., 0.);
    }
    //Pauli gate: phase flip
    static Gate* Z(IdxType m)
    {
        return new Gate(OP::Z, m, 0, 0, 0, 0, 0., 0., 0.);
    }
    //Clifford gate: Hadamard
    static Gate* H(IdxType m)
    {
        return new Gate(OP::H, m, 0, 0, 0, 0, 0., 0., 0.);
    }
    //Clifford gate: sqrt(Z) phase gate
    static Gate* S(IdxType m)
    {
        return new Gate(OP::S, m, 0, 0, 0, 0, 0., 0., 0.);
    }
    //Clifford gate: conjugate of sqrt(Z)
    static Gate* SDG(IdxType m)
    {
        return new Gate(OP::SDG, m, 0, 0, 0, 0, 0., 0., 0.);
    }
    //C3 gate: sqrt(S) phase gate
    static Gate* T(IdxType m)
    {
        return new Gate(OP::T, m, 0, 0, 0, 0, 0., 0., 0.);
    }
    //C3 gate: conjugate of sqrt(S)
    static Gate* TDG(IdxType m)
    {
        return new Gate(OP::TDG, m, 0, 0, 0, 0, 0., 0., 0.);
    }
    //Rotation around X-axis
    static Gate* RX(ValType theta, IdxType m)
    {
        return new Gate(OP::RX, m, 0, 0, 0, 0, theta, 0., 0.);
    }
    //Rotation around Y-axis
    static Gate* RY(ValType theta, IdxType m)
    {
        return new Gate(OP::RY, m, 0, 0, 0, 0, theta, 0., 0.);
    }
    //Rotation around Z-axis
    static Gate* RZ(ValType phi, IdxType m)
    {
        return new Gate(OP::RZ, m, 0, 0, 0, 0, 0., phi, 0.);
    }
    // =============================== Composition Gates ===================================
    //Controlled-Phase
    static Gate* CZ(IdxType m, IdxType n)
    {
        return new Gate(OP::CZ, m, n, 0, 0, 0, 0., 0., 0.);
    }
    //Controlled-Y
    static Gate* CY(IdxType m, IdxType n)
    {
        return new Gate(OP::CY, m, n, 0, 0, 0, 0., 0., 0.);
    }
    //Swap
    static Gate* SWAP(IdxType m, IdxType n)
    {
        return new Gate(OP::SWAP, m, n, 0, 0, 0, 0., 0., 0.);
    }
    //Controlled-H
    static Gate* CH(IdxType m, IdxType n)
    {
        return new Gate(OP::CH, m, n, 0, 0, 0, 0., 0., 0.);
    }
    //C3 gate: Toffoli
    static Gate* CCX(IdxType l, IdxType m, IdxType n)
    {
        return new Gate(OP::CCX, l, m, n, 0, 0, 0., 0., 0.);
    }
    //Fredkin gate
    static Gate* CSWAP(IdxType l, IdxType m, IdxType n)
    {
        return new Gate(OP::CSWAP, l, m, n, 0, 0, 0., 0., 0.);
    }
    //Controlled RX rotation
    static Gate* CRX(ValType lambda, IdxType m, IdxType n)
    {
        return new Gate(OP::CRX, m, n, 0, 0, 0, 0., 0., lambda);
    }
    //Controlled RY rotation
    static Gate* CRY(ValType lambda, IdxType m, IdxType n)
    {
        return new Gate(OP::CRY, m, n, 0, 0, 0, 0., 0., lambda);
    }
    //Controlled RZ rotation
    static Gate* CRZ(ValType lambda, IdxType m, IdxType n)
    {
        return new Gate(OP::CRZ, m, n, 0, 0, 0, 0., 0., lambda);
    }
    //Controlled phase rotation
    static Gate* CU1(ValType lambda, IdxType m, IdxType n)
    {
        return new Gate(OP::CU1, m, n, 0, 0, 0, 0., 0., lambda);
    }
    //Controlled-U
    static Gate* CU3(ValType theta, ValType phi, ValType lambda, IdxType m, IdxType n)
    {
        return new Gate(OP::CU3, m, n, 0, 0, 0, theta, phi, lambda);
    }
    //2-qubit XX rotation
    static Gate* RXX(ValType theta, IdxType m, IdxType n)
    {
        return new Gate(OP::RXX, m, n, 0, 0, 0, theta, 0., 0.);
    }
    //2-qubit ZZ rotation
    static Gate* RZZ(ValType theta, IdxType m, IdxType n)
    {
        return new Gate(OP::RZZ, m, n, 0, 0, 0, theta, 0., 0.);
    }
    //Relative-phase CCX
    static Gate* RCCX(IdxType l, IdxType m, IdxType n)
    {
        return new Gate(OP::RCCX, l, m, n, 0, 0, 0., 0., 0.);
    }
    //Relative-phase 3-controlled X gate
    static Gate* RC3X(IdxType l, IdxType m, IdxType n, IdxType o)
    {
        return new Gate(OP::RC3X, l, m, n, o, 0, 0., 0., 0.);
    }
    //3-controlled X gate
    static Gate* C3X(IdxType l, IdxType m, IdxType n, IdxType o)
    {
        return new Gate(OP::C3X, l, m, n, o, 0, 0., 0., 0.);
    }
    //3-controlled sqrt(X) gate
    static Gate* C3SQRTX(IdxType l, IdxType m, IdxType n, IdxType o)
    {
        return new Gate(OP::C3SQRTX, l, m, n, o, 0, 0., 0., 0.);
    }
    //4-controlled X gate
    static Gate* C4X(IdxType l, IdxType m, IdxType n, IdxType o, IdxType p)
    {
        return new Gate(OP::C4X, l, m, n, o, p, 0., 0., 0.);
    }
    // =============================== DM_Sim Native Gates ===================================
    static Gate* R(ValType theta, IdxType m)
    {
        return new Gate(OP::R, m, 0, 0, 0, 0, theta, 0., 0.);
    }
    static Gate* SRN(IdxType m)
    {
        return new Gate(OP::SRN, m, 0, 0, 0, 0, 0., 0., 0.);
    }
    static Gate* W(IdxType m)
    {
        return new Gate(OP::W, m, 0, 0, 0, 0, 0., 0., 0.);
    }
    //2-qubit YY rotation
    static Gate* RYY(ValType theta, IdxType m, IdxType n)
    {
        return new Gate(OP::RYY, m, n, 0, 0, 0, theta, 0., 0.);
    }
 
 
public:
    // n_qubits is the number of qubits
    const IdxType n_qubits;
    // cpu_scale is 2^x of the number of cpus, e.g., with 8 cpus the cpu_scale is 3 (2^3=8)
    const IdxType cpu_scale;
    const IdxType n_cpus;
    const IdxType dim;
    const IdxType half_dim;
    const IdxType lg2_m_cpu;
    const IdxType m_cpu;
    const IdxType n_tile;

    const IdxType dm_num;
    const IdxType dm_size;
    const IdxType dm_size_per_cpu;

    IdxType n_gates;
    //CPU arrays
    ValType* dm_real_cpu;
    ValType* dm_imag_cpu;
    ValType* dm_real_res;
    ValType* dm_imag_res;

    //cpu pointers on CPU
    ValType** dm_real_ptr;
    ValType** dm_imag_ptr;
    ValType** dm_real_buf_ptr;
    ValType** dm_imag_buf_ptr;
   
    //cpu arrays, they are used as alias of particular pointers
    ValType* dm_real;
    ValType* dm_imag;
    ValType* dm_real_buf;
    ValType* dm_imag_buf;
    
    ValType cpu_mem;
    //hold the CPU-side gates
    vector<Gate*> circuit;
    //for freeing cpu-side gates in clear(), otherwise there can be cpu memory leak
    vector<Gate*> circuit_copy;
    //hold the cpu-side gates
    Gate** circuit_cpu;
    //hold the cpu-side simulator instances
    Simulation** sim_cpu;
};

void circuit(Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    for (IdxType t=0; t<(sim->n_gates); t++)
    {
        ((sim->circuit_cpu)[t])->exe_op(sim, dm_real, dm_imag);
    }
}

//Blockwise transpose via shared memory
void block_transpose(Simulation* sim, ValType* dm_real, ValType* dm_imag, 
        const ValType* real_buf, const ValType* imag_buf)
{
    for (IdxType bid = 0; bid < (sim->n_tile)*(sim->n_tile)*(sim->n_cpus); 
            bid += 1)
    {
        IdxType bz = bid / ((sim->n_tile) * (sim->n_tile)); 
        IdxType by = (bid % ((sim->n_tile)*(sim->n_tile))) / (sim->n_tile);
        IdxType bx = bid % (sim->n_tile);

        ValType smem_real[TILE][TILE] = {0};
        ValType smem_imag[TILE][TILE] = {0};

        for (IdxType tid = 0; tid < THREADS_PER_BLOCK; tid++)
        {
            IdxType tlx = tid % TILE;
            IdxType tly = tid / TILE;

            IdxType tx = bx * TILE + tlx;
            IdxType ty = by * TILE + tly;

            if ((tlx < (sim->m_cpu)) && (tly < (sim->m_cpu)))
            {
                IdxType in_idx = ty*(sim->dim)+bz*(sim->m_cpu)+tx;
                IdxType out_idx = (bx*TILE+tly)*(sim->dim)+ bz*(sim->m_cpu) + by*TILE+tlx;
                smem_real[tly][tlx] = real_buf[in_idx];
                smem_imag[tly][tlx] = -imag_buf[in_idx];

                dm_real[out_idx] = smem_real[tlx][tly];
                dm_imag[out_idx] = smem_imag[tlx][tly];

            }
        } 
    }
}

//Packing portions for all-to-all communication, see our paper for detail.
void packing(Simulation* sim, const ValType* dm_real, const ValType* dm_imag,
        ValType* real_buf, ValType* imag_buf)
{
    const int tid = 0;
    for (IdxType i = tid; i < (sim->dim) * (sim->m_cpu); i+=1)
    {
        ////Original version with sementics
        //IdxType w_in_block = i / dim;
        //IdxType block_id = (i % dim) / m_cpu;
        //IdxType h_in_block = (i % dim) % m_cpu;
        //IdxType id_in_dm = w_in_block*dim+(i%dim);
        //IdxType id_in_buf = block_id * m_cpu * m_cpu + w_in_block * m_cpu + h_in_block;

        //Optimized version
        IdxType w_in_block = (i >> (sim->n_qubits));
        IdxType block_id = (i & (sim->dim-1)) >> (sim->lg2_m_cpu);
        IdxType h_in_block = (i & (sim->dim-1)) & (sim->m_cpu-1);
        IdxType id_in_dm = (w_in_block << (sim->n_qubits))+(i & (sim->dim-1));
        IdxType id_in_buf = (block_id << (sim->lg2_m_cpu+sim->lg2_m_cpu)) 
            + (w_in_block << (sim->lg2_m_cpu)) + h_in_block;

        real_buf[id_in_buf] = dm_real[id_in_dm];
        imag_buf[id_in_buf] = dm_imag[id_in_dm];
    }
}

//Unpacking portions after all-to-all communication, see our paper for detail.
void unpacking(Simulation* sim, ValType* send_real, ValType* send_imag,
        const ValType* recv_real, const ValType* recv_imag)
{
    const int tid = 0;
    for (IdxType i = tid; i < (sim->dim) * (sim->m_cpu); i+=1)
    {
        ////Original version with sementics
        //IdxType j = i / dim; 
        //IdxType id_in_buf = j * dim + (i % dim);
        //IdxType block_id = id_in_buf / (m_cpu*m_cpu);
        //IdxType in_block_id = id_in_buf % (m_cpu*m_cpu);
        //IdxType w_in_block = in_block_id / m_cpu;
        //IdxType h_in_block = in_block_id % m_cpu;
        //IdxType dm_w = w_in_block;
        //IdxType dm_h = h_in_block + m_cpu*block_id;
        //IdxType id_in_dim = dm_w * dim + dm_h;

        //Optimized version
        IdxType j = (i >> (sim->n_qubits)); 
        IdxType id_in_buf = (j << (sim->n_qubits)) + (i & (sim->dim-0x1));
        IdxType block_id = (id_in_buf >> (sim->lg2_m_cpu+sim->lg2_m_cpu));
        IdxType in_block_id = (id_in_buf & ((sim->m_cpu)*(sim->m_cpu)-0x1));
        IdxType w_in_block = (in_block_id >> (sim->lg2_m_cpu));
        IdxType h_in_block = (in_block_id & (sim->m_cpu-1));
        IdxType dm_w = w_in_block;
        IdxType dm_h = h_in_block + (block_id<<(sim->lg2_m_cpu));
        IdxType id_in_dim = (dm_w << (sim->n_qubits)) + dm_h;

        send_real[id_in_dim] = recv_real[id_in_buf]; 
        send_imag[id_in_dim] = recv_imag[id_in_buf]; 
    }
}

void simulation_kernel(Simulation* sim, bool isforward)
{
    if (sim->n_cpus == 1)
    {
        //================ Single cpu version ===================
        //Forward: In(dm_real, dm_imag) => Out(dm_real, dm_imag)
        //Adjoint: In(dm_real, dm_imag) => Out(dm_real_buf, dm_imag_buf)
        //Bakward: In(dm_real_buf, dm_imag_buf) => Out(dm_real_buf, dm_imag_buf)
        if (isforward)
        {
            circuit(sim, sim->dm_real, sim->dm_imag);
        }
        else
        {
            //(out,in)
            block_transpose(sim, sim->dm_real_buf, sim->dm_imag_buf, 
                    sim->dm_real, sim->dm_imag);
            circuit(sim, sim->dm_real_buf, sim->dm_imag_buf);
        }
    }
    else
    {
        //================ Scale-up OpenMP version ===================
        //Forward: In(dm_real, dm_imag) => Out(dm_real, dm_imag)
        //Packing: In(dm_real, dm_imag) => Out(dm_real_buf, dm_imag_buf)
        //All2All: In(dm_real_buf, dm_imag_buf) => Out(dm_real, dm_imag)
        //Unpack:  In(dm_real, dm_imag) => Out(dm_real_buf, dm_imag_buf)
        //BlkTran: In(dm_real_buf, dm_imag_buf) => Out(dm_real, dm_imag)
        //Bakward: In(dm_real, dm_imag) => Out(dm_real, dm_imag)

        //Forward pass
        if (isforward) 
        {
            circuit(sim, sim->dm_real, sim->dm_imag);
            packing(sim, sim->dm_real, sim->dm_imag, 
                    sim->dm_real_buf, sim->dm_imag_buf); //(in,out)
        }
        //Adjoint and Backward pass
        else 
        {
            unpacking(sim, sim->dm_real_buf, sim->dm_imag_buf,
                    sim->dm_real, sim->dm_imag); //(out,in)
            block_transpose(sim, sim->dm_real, sim->dm_imag,
                    sim->dm_real_buf, sim->dm_imag_buf); //(out,in)
            circuit(sim, sim->dm_real, sim->dm_imag);
        }
    }
}

//=================================== Gate Definition ==========================================

//Define MG-BSP machine operation header (Original version with semantics)
// #define OP_HEAD_ORIGIN \
    const int tid = 0;\
    const IdxType outer_bound = (1 << ( (sim->n_qubits) - qubit - 1)); \
    const IdxType inner_bound = (1 << qubit); \
        for (IdxType i = tid;i<outer_bound*inner_bound*(sim->m_cpu);\
                i+=1){ \
            IdxType col = i / (inner_bound * outer_bound); \
            IdxType outer = (i % (inner_bound * outer_bound)) / inner_bound; \
            IdxType inner =  i % inner_bound; \
            IdxType offset = (2 * outer) * inner_bound; \
            IdxType pos0 = col * (sim->dim) + offset + inner; \
            IdxType pos1 = pos0 + inner_bound; 

//Define MG-BSP machine operation footer
#define OP_TAIL  } 

#ifndef USE_AVX512 //Without AVX512 Acceleration

//Define MG-BSP machine operation header (Optimized version)
#define OP_HEAD \
    const int tid = 0; \
        for (IdxType i=tid; i<((sim->half_dim)<<(sim->lg2_m_cpu));\
                i+=1){ \
            IdxType col = (i >> (sim->n_qubits-1)); \
            IdxType outer = ((i & ((sim->half_dim)-1)) >> qubit); \
            IdxType inner =  (i & ((1<<qubit)-1)); \
            IdxType offset = (outer << (qubit+1)); \
            IdxType pos0 = (col << (sim->n_qubits)) + offset + inner; \
            IdxType pos1 = pos0 + (1<<qubit); 

//============== Unified 1-qubit Gate ================
void C1_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag, 
        const ValType e0_real, const ValType e0_imag,
        const ValType e1_real, const ValType e1_imag,
        const ValType e2_real, const ValType e2_imag,
        const ValType e3_real, const ValType e3_imag,
        const IdxType qubit)
{
    OP_HEAD;
    const ValType el0_real = dm_real[pos0]; 
    const ValType el0_imag = dm_imag[pos0];
    const ValType el1_real = dm_real[pos1]; 
    const ValType el1_imag = dm_imag[pos1];
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
void C2_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag, 
        const ValType e00_real, const ValType e00_imag,
        const ValType e01_real, const ValType e01_imag,
        const ValType e02_real, const ValType e02_imag,
        const ValType e03_real, const ValType e03_imag,
        const ValType e10_real, const ValType e10_imag,
        const ValType e11_real, const ValType e11_imag,
        const ValType e12_real, const ValType e12_imag,
        const ValType e13_real, const ValType e13_imag,
        const ValType e20_real, const ValType e20_imag,
        const ValType e21_real, const ValType e21_imag,
        const ValType e22_real, const ValType e22_imag,
        const ValType e23_real, const ValType e23_imag,
        const ValType e30_real, const ValType e30_imag,
        const ValType e31_real, const ValType e31_imag,
        const ValType e32_real, const ValType e32_imag,
        const ValType e33_real, const ValType e33_imag,
        const IdxType qubit1, const IdxType qubit2)
{
    const int tid = 0;
    const IdxType q0dim = (1 << max(qubit1, qubit2) );
    const IdxType q1dim = (1 << min(qubit1, qubit2) );
    assert (qubit1 != qubit2); //Non-cloning
    const IdxType outer_factor = ((sim->dim) + q0dim + q0dim - 1) >> (max(qubit1,qubit2)+1);
    const IdxType mider_factor = (q0dim + q1dim + q1dim - 1) >> (min(qubit1,qubit2)+1);
    const IdxType inner_factor = q1dim;
    const IdxType qubit1_dim = (1 << qubit1);
    const IdxType qubit2_dim = (1 << qubit2);

    for (IdxType i = tid; i < outer_factor * mider_factor * inner_factor * (sim->m_cpu); 
            i+=1)
    {
        IdxType col = i / (outer_factor * mider_factor * inner_factor);
        IdxType row = i % (outer_factor * mider_factor * inner_factor);
        IdxType outer = ((row/inner_factor) / (mider_factor)) * (q0dim+q0dim);
        IdxType mider = ((row/inner_factor) % (mider_factor)) * (q1dim+q1dim);
        IdxType inner = row % inner_factor;

        IdxType pos0 = col * (sim->dim) + outer + mider + inner;
        IdxType pos1 = col * (sim->dim) + outer + mider + inner + qubit2_dim;
        IdxType pos2 = col * (sim->dim) + outer + mider + inner + qubit1_dim;
        IdxType pos3 = col * (sim->dim) + outer + mider + inner + q0dim + q1dim;

        assert (pos0 < dim*m_cpu); //ensure not out of bound
        assert (pos1 < dim*m_cpu); //ensure not out of bound
        assert (pos2 < dim*m_cpu); //ensure not out of bound
        assert (pos3 < dim*m_cpu); //ensure not out of bound

        const ValType el0_real = dm_real[pos0]; 
        const ValType el0_imag = dm_imag[pos0];
        const ValType el1_real = dm_real[pos1]; 
        const ValType el1_imag = dm_imag[pos1];
        const ValType el2_real = dm_real[pos2]; 
        const ValType el2_imag = dm_imag[pos2];
        const ValType el3_real = dm_real[pos3]; 
        const ValType el3_imag = dm_imag[pos3];

        //Real part
        dm_real[pos0] = (e00_real * el0_real) - (e00_imag * el0_imag)
            +(e01_real * el1_real) - (e01_imag * el1_imag)
            +(e02_real * el2_real) - (e02_imag * el2_imag)
            +(e03_real * el3_real) - (e03_imag * el3_imag);
        dm_real[pos1] = (e10_real * el0_real) - (e10_imag * el0_imag)
            +(e11_real * el1_real) - (e11_imag * el1_imag)
            +(e12_real * el2_real) - (e12_imag * el2_imag)
            +(e13_real * el3_real) - (e13_imag * el3_imag);
        dm_real[pos2] = (e20_real * el0_real) - (e20_imag * el0_imag)
            +(e21_real * el1_real) - (e21_imag * el1_imag)
            +(e22_real * el2_real) - (e22_imag * el2_imag)
            +(e23_real * el3_real) - (e23_imag * el3_imag);
        dm_real[pos3] = (e30_real * el0_real) - (e30_imag * el0_imag)
            +(e31_real * el1_real) - (e31_imag * el1_imag)
            +(e32_real * el2_real) - (e32_imag * el2_imag)
            +(e33_real * el3_real) - (e33_imag * el3_imag);
        
        //Imag part
        dm_imag[pos0] = (e00_real * el0_imag) + (e00_imag * el0_real)
            +(e01_real * el1_imag) + (e01_imag * el1_real)
            +(e02_real * el2_imag) + (e02_imag * el2_real)
            +(e03_real * el3_imag) + (e03_imag * el3_real);
        dm_imag[pos1] = (e10_real * el0_imag) + (e10_imag * el0_real)
            +(e11_real * el1_imag) + (e11_imag * el1_real)
            +(e12_real * el2_imag) + (e12_imag * el2_real)
            +(e13_real * el3_imag) + (e13_imag * el3_real);
        dm_imag[pos2] = (e20_real * el0_imag) + (e20_imag * el0_real)
            +(e21_real * el1_imag) + (e21_imag * el1_real)
            +(e22_real * el2_imag) + (e22_imag * el2_real)
            +(e23_real * el3_imag) + (e23_imag * el3_real);
        dm_imag[pos3] = (e30_real * el0_imag) + (e30_imag * el0_real)
            +(e31_real * el1_imag) + (e31_imag * el1_real)
            +(e32_real * el2_imag) + (e32_imag * el2_real)
            +(e33_real * el3_imag) + (e33_imag * el3_real);
    }
}





//============== CX Gate ================
//Controlled-NOT or CNOT
/** CX   = [1 0 0 0]
           [0 1 0 0]
           [0 0 0 1]
           [0 0 1 0]
*/
void CX_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag, 
        const IdxType ctrl, const IdxType qubit)
{
    const int tid = 0;
    const IdxType q0dim = (1 << max(ctrl, qubit) );
    const IdxType q1dim = (1 << min(ctrl, qubit) );
    assert (ctrl != qubit); //Non-cloning
    const IdxType outer_factor = ((sim->dim) + q0dim + q0dim - 1) >> (max(ctrl,qubit)+1);
    const IdxType mider_factor = (q0dim + q1dim + q1dim - 1) >> (min(ctrl,qubit)+1);
    const IdxType inner_factor = q1dim;
    const IdxType ctrldim = (1 << ctrl);

    for (IdxType i = tid; i < outer_factor * mider_factor * inner_factor * (sim->m_cpu); 
            i+=1)
    {
        IdxType col = i / (outer_factor * mider_factor * inner_factor);
        IdxType row = i % (outer_factor * mider_factor * inner_factor);
        IdxType outer = ((row/inner_factor) / (mider_factor)) * (q0dim+q0dim);
        IdxType mider = ((row/inner_factor) % (mider_factor)) * (q1dim+q1dim);
        IdxType inner = row % inner_factor;

        IdxType pos0 = col * (sim->dim) + outer + mider + inner + ctrldim;
        IdxType pos1 = col * (sim->dim) + outer + mider + inner + q0dim + q1dim;
        //assert (pos0 < dim*m_cpu); //ensure not out of bound
        //assert (pos1 < dim*m_cpu); //ensure not out of bound
        const ValType el0_real = dm_real[pos0]; 
        const ValType el0_imag = dm_imag[pos0];
        const ValType el1_real = dm_real[pos1]; 
        const ValType el1_imag = dm_imag[pos1];
        dm_real[pos0] = el1_real; 
        dm_imag[pos0] = el1_imag;
        dm_real[pos1] = el0_real; 
        dm_imag[pos1] = el0_imag;
    }
}

//============== X Gate ================
//Pauli gate: bit flip
/** X = [0 1]
        [1 0]
*/
void X_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag, const IdxType qubit)
{
    OP_HEAD;
    const ValType el0_real = dm_real[pos0]; 
    const ValType el0_imag = dm_imag[pos0];
    const ValType el1_real = dm_real[pos1]; 
    const ValType el1_imag = dm_imag[pos1];
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
void Y_GATE(const Simulation* sim, ValType* dm_real,
        ValType* dm_imag, const IdxType qubit)
{
    OP_HEAD;
    const ValType el0_real = dm_real[pos0]; 
    const ValType el0_imag = dm_imag[pos0];
    const ValType el1_real = dm_real[pos1]; 
    const ValType el1_imag = dm_imag[pos1];
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
void Z_GATE(const Simulation* sim, ValType* dm_real, 
        ValType* dm_imag, const IdxType qubit)
{
    OP_HEAD;
    const ValType el1_real = dm_real[pos1]; 
    const ValType el1_imag = dm_imag[pos1];
    dm_real[pos1] = -el1_real;
    dm_imag[pos1] = -el1_imag;
    OP_TAIL;
}

//============== H Gate ================
//Clifford gate: Hadamard
/** H = 1/sqrt(2) * [1  1]
                    [1 -1]
*/
void H_GATE(const Simulation* sim, ValType* dm_real, 
        ValType* dm_imag,  const IdxType qubit)
{
    OP_HEAD;
    const ValType el0_real = dm_real[pos0]; 
    const ValType el0_imag = dm_imag[pos0];
    const ValType el1_real = dm_real[pos1]; 
    const ValType el1_imag = dm_imag[pos1];
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
void SRN_GATE(const Simulation* sim, ValType* dm_real, 
        ValType* dm_imag, const IdxType qubit)
{
    OP_HEAD;
    const ValType el0_real = dm_real[pos0]; 
    const ValType el0_imag = dm_imag[pos0];
    const ValType el1_real = dm_real[pos1]; 
    const ValType el1_imag = dm_imag[pos1];
    dm_real[pos0] = 0.5*( el0_real + el1_real); 
    dm_imag[pos0] = 0.5*( el0_imag - el1_imag);
    dm_real[pos1] = 0.5*( el0_real + el1_real);
    dm_imag[pos1] = 0.5*(-el0_imag + el1_imag);
    OP_TAIL;
}

//============== R Gate ================
//Phase-shift gate, it leaves |0> unchanged
//and maps |1> to e^{i\psi}|1>
/** R = [1 0]
        [0 0+p*i]
*/
void R_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag, 
        const ValType phase, const IdxType qubit)
{
    OP_HEAD;
    const ValType el1_real = dm_real[pos1]; 
    const ValType el1_imag = dm_imag[pos1];
    dm_real[pos1] = -(el1_imag*phase);
    dm_imag[pos1] = el1_real*phase;
    OP_TAIL;
}



//============== S Gate ================
//Clifford gate: sqrt(Z) phase gate
/** S = [1 0]
        [0 i]
*/
void S_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,  const IdxType qubit)
{
    OP_HEAD;
    const ValType el1_real = dm_real[pos1]; 
    const ValType el1_imag = dm_imag[pos1];
    dm_real[pos1] = -el1_imag;
    dm_imag[pos1] = el1_real;
    OP_TAIL;
}

//============== SDG Gate ================
//Clifford gate: conjugate of sqrt(Z) phase gate
/** SDG = [1  0]
          [0 -i]
*/
void SDG_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,  const IdxType qubit)
{
    OP_HEAD;
    const ValType el1_real = dm_real[pos1]; 
    const ValType el1_imag = dm_imag[pos1];
    dm_real[pos1] = el1_imag;
    dm_imag[pos1] = -el1_real;
    OP_TAIL;
}

//============== T Gate ================
//C3 gate: sqrt(S) phase gate
/** T = [1 0]
        [0 s2i+s2i*i]
*/
void T_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag, const IdxType qubit)
{
    OP_HEAD;
    const ValType el1_real = dm_real[pos1]; 
    const ValType el1_imag = dm_imag[pos1];
    dm_real[pos1] = S2I*(el1_real-el1_imag);
    dm_imag[pos1] = S2I*(el1_real+el1_imag);
    OP_TAIL;
}

//============== TDG Gate ================
//C3 gate: conjugate of sqrt(S) phase gate
/** TDG = [1 0]
          [0 s2i-s2i*i]
*/
void TDG_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag, const IdxType qubit)
{
    OP_HEAD;
    const ValType el1_real = dm_real[pos1]; 
    const ValType el1_imag = dm_imag[pos1];
    dm_real[pos1] = S2I*( el1_real+el1_imag);
    dm_imag[pos1] = S2I*(-el1_real+el1_imag);
    OP_TAIL;
}

//============== D Gate ================
/** D = [e0_real+i*e0_imag 0]
        [0 e3_real+i*e3_imag]
*/
void D_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag, 
        const ValType e0_real, const ValType e0_imag,
        const ValType e3_real, const ValType e3_imag,
        const IdxType qubit)
{
    OP_HEAD;
    const ValType el0_real = dm_real[pos0]; 
    const ValType el0_imag = dm_imag[pos0];
    const ValType el1_real = dm_real[pos1]; 
    const ValType el1_imag = dm_imag[pos1];
    dm_real[pos0] = (e0_real * el0_real) - (e0_imag * el0_imag);
    dm_imag[pos0] = (e0_real * el0_imag) + (e0_imag * el0_real);
    dm_real[pos1] = (e3_real * el1_real) - (e3_imag * el1_imag);
    dm_imag[pos1] = (e3_real * el1_imag) + (e3_imag * el1_real);
    OP_TAIL;
}

//============== RX Gate ================
//Rotation around X-axis
void RX_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
       const ValType theta, const IdxType qubit)
{
    ValType rx_real = cos(theta/2.0);
    ValType rx_imag = -sin(theta/2.0);
    OP_HEAD;
    const ValType el0_real = dm_real[pos0]; 
    const ValType el0_imag = dm_imag[pos0];
    const ValType el1_real = dm_real[pos1]; 
    const ValType el1_imag = dm_imag[pos1];
    dm_real[pos0] = (rx_real * el0_real) - (rx_imag * el1_imag);
    dm_imag[pos0] = (rx_real * el0_imag) + (rx_imag * el1_real);
    dm_real[pos1] =  - (rx_imag * el0_imag) +(rx_real * el1_real);
    dm_imag[pos1] =  + (rx_imag * el0_real) +(rx_real * el1_imag);
    OP_TAIL;
}

//============== RY Gate ================
//Rotation around Y-axis
void RY_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
        const ValType theta, const IdxType qubit)
{
    ValType e0_real = cos(theta/2.0);
    ValType e1_real = -sin(theta/2.0);
    ValType e2_real = sin(theta/2.0);
    ValType e3_real = cos(theta/2.0);

    OP_HEAD;
    const ValType el0_real = dm_real[pos0]; 
    const ValType el0_imag = dm_imag[pos0];
    const ValType el1_real = dm_real[pos1]; 
    const ValType el1_imag = dm_imag[pos1];
    dm_real[pos0] = (e0_real * el0_real) +(e1_real * el1_real);
    dm_imag[pos0] = (e0_real * el0_imag) +(e1_real * el1_imag);
    dm_real[pos1] = (e2_real * el0_real) +(e3_real * el1_real);
    dm_imag[pos1] = (e2_real * el0_imag) +(e3_real * el1_imag);
    OP_TAIL;
}

//============== W Gate ================
//W gate: e^(-i*pi/4*X)
/** W = [s2i    -s2i*i]
        [-s2i*i s2i   ]
*/
void W_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag, const IdxType qubit)
{
    OP_HEAD;
    const ValType el0_real = dm_real[pos0]; 
    const ValType el0_imag = dm_imag[pos0];
    const ValType el1_real = dm_real[pos1]; 
    const ValType el1_imag = dm_imag[pos1];
    dm_real[pos0] = S2I * (el0_real + el1_imag);
    dm_imag[pos0] = S2I * (el0_imag - el1_real);
    dm_real[pos1] = S2I * (el0_imag + el1_real);
    dm_imag[pos1] = S2I * (-el0_real + el1_imag);
    OP_TAIL;
}

#else //With AVX512 Acceleration

//Define MG-BSP machine operation header (AVX512 Optimized version)
#define OP_HEAD \
    __m256i idx=_mm256_set_epi32(0,1,2,3,4,5,6,7); \
    const __m256i inc=_mm256_set1_epi32(8); \
    const __m256i cons0 = _mm256_set1_epi32((sim->half_dim)-1); \
    const __m256i cons1 = _mm256_set1_epi32((1<<qubit)-1); \
    const __m256i cons2 = _mm256_set1_epi32(1<<qubit); \
    for (IdxType i=0; i<((sim->half_dim)<<(sim->lg2_m_cpu)); i+=8, idx=_mm256_add_epi32(idx,inc)) \
    { \
        __m256i col, tmp, outer, inner, offset, pos0, pos1; \
        col = _mm256_srli_epi32(idx,sim->n_qubits-1); /*IdxType col = (i >> (sim->n_qubits-1));*/ \
        tmp = _mm256_and_si256(idx,cons0); \
        outer = _mm256_srli_epi32(tmp,qubit); /*IdxType outer = ((i & ((sim->half_dim)-1)) >> qubit);*/ \
        inner = _mm256_and_si256(idx,cons1); /*IdxType inner =  (i & ((1<<qubit)-1));*/ \
        offset = _mm256_slli_epi32(outer, qubit+1); /*IdxType offset = (outer << (qubit+1));*/ \
        tmp = _mm256_slli_epi32(col, sim->n_qubits);\
        tmp = _mm256_add_epi32(tmp, offset);\
        pos0 = _mm256_add_epi32(tmp, inner); /*IdxType pos0 = (col << (sim->n_qubits)) + offset + inner;*/ \
        pos1 = _mm256_add_epi32(pos0, cons2); /*IdxType pos1 = pos0 + (1<<qubit);*/ 

//============== Unified 1-qubit Gate ================
void C1_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag, 
        const ValType e0_real, const ValType e0_imag,
        const ValType e1_real, const ValType e1_imag,
        const ValType e2_real, const ValType e2_imag,
        const ValType e3_real, const ValType e3_imag,
        const IdxType qubit)
{
    const __m512d e0_real_v = _mm512_set1_pd(e0_real);
    const __m512d e0_imag_v = _mm512_set1_pd(e0_imag);
    const __m512d e1_real_v = _mm512_set1_pd(e1_real);
    const __m512d e1_imag_v = _mm512_set1_pd(e1_imag);
    const __m512d e2_real_v = _mm512_set1_pd(e2_real);
    const __m512d e2_imag_v = _mm512_set1_pd(e2_imag);
    const __m512d e3_real_v = _mm512_set1_pd(e3_real);
    const __m512d e3_imag_v = _mm512_set1_pd(e3_imag);

    OP_HEAD;

    const __m512d el0_real = _mm512_i32gather_pd(pos0, dm_real, 8);
    const __m512d el0_imag = _mm512_i32gather_pd(pos0, dm_imag, 8);
    const __m512d el1_real = _mm512_i32gather_pd(pos1, dm_real, 8);
    const __m512d el1_imag = _mm512_i32gather_pd(pos1, dm_imag, 8);

    __m512d tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6;
    
    //dm_real[pos0] = (e0_real * el0_real) - (e0_imag * el0_imag)
    //               +(e1_real * el1_real) - (e1_imag * el1_imag);
    tmp0 = _mm512_mul_pd(e0_real_v, el0_real);
    tmp1 = _mm512_mul_pd(e0_imag_v, el0_imag);
    tmp2 = _mm512_mul_pd(e1_real_v, el1_real);
    tmp3 = _mm512_mul_pd(e1_imag_v, el1_imag);
    tmp4 = _mm512_sub_pd(tmp0, tmp1);
    tmp5 = _mm512_sub_pd(tmp2, tmp3);
    tmp6 = _mm512_add_pd(tmp4, tmp5);
    _mm512_i32scatter_pd(dm_real, pos0, tmp6, 8);

    //dm_imag[pos0] = (e0_real * el0_imag) + (e0_imag * el0_real)
    //               +(e1_real * el1_imag) + (e1_imag * el1_real);
    tmp0 = _mm512_mul_pd(e0_real_v, el0_imag);
    tmp1 = _mm512_mul_pd(e0_imag_v, el0_real);
    tmp2 = _mm512_mul_pd(e1_real_v, el1_imag);
    tmp3 = _mm512_mul_pd(e1_imag_v, el1_real);
    tmp4 = _mm512_add_pd(tmp0, tmp1);
    tmp5 = _mm512_add_pd(tmp2, tmp3);
    tmp6 = _mm512_add_pd(tmp4, tmp5);
    _mm512_i32scatter_pd(dm_imag, pos0, tmp6, 8);

    //dm_real[pos1] = (e2_real * el0_real) - (e2_imag * el0_imag)
    //               +(e3_real * el1_real) - (e3_imag * el1_imag);
    tmp0 = _mm512_mul_pd(e2_real_v, el0_real);
    tmp1 = _mm512_mul_pd(e2_imag_v, el0_imag);
    tmp2 = _mm512_mul_pd(e3_real_v, el1_real);
    tmp3 = _mm512_mul_pd(e3_imag_v, el1_imag);
    tmp4 = _mm512_sub_pd(tmp0, tmp1);
    tmp5 = _mm512_sub_pd(tmp2, tmp3);
    tmp6 = _mm512_add_pd(tmp4, tmp5);
    _mm512_i32scatter_pd(dm_real, pos1, tmp6, 8);

    //dm_imag[pos1] = (e2_real * el0_imag) + (e2_imag * el0_real)
    //               +(e3_real * el1_imag) + (e3_imag * el1_real);
    tmp0 = _mm512_mul_pd(e2_real_v, el0_imag);
    tmp1 = _mm512_mul_pd(e2_imag_v, el0_real);
    tmp2 = _mm512_mul_pd(e3_real_v, el1_imag);
    tmp3 = _mm512_mul_pd(e3_imag_v, el1_real);
    tmp4 = _mm512_add_pd(tmp0, tmp1);
    tmp5 = _mm512_add_pd(tmp2, tmp3);
    tmp6 = _mm512_add_pd(tmp4, tmp5);
    _mm512_i32scatter_pd(dm_imag, pos1, tmp6, 8);

    OP_TAIL;
}


//============== Unified 2-qubit Gate ================
void C2_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag, 
        const ValType e00_real, const ValType e00_imag,
        const ValType e01_real, const ValType e01_imag,
        const ValType e02_real, const ValType e02_imag,
        const ValType e03_real, const ValType e03_imag,
        const ValType e10_real, const ValType e10_imag,
        const ValType e11_real, const ValType e11_imag,
        const ValType e12_real, const ValType e12_imag,
        const ValType e13_real, const ValType e13_imag,
        const ValType e20_real, const ValType e20_imag,
        const ValType e21_real, const ValType e21_imag,
        const ValType e22_real, const ValType e22_imag,
        const ValType e23_real, const ValType e23_imag,
        const ValType e30_real, const ValType e30_imag,
        const ValType e31_real, const ValType e31_imag,
        const ValType e32_real, const ValType e32_imag,
        const ValType e33_real, const ValType e33_imag,
        const IdxType qubit1, const IdxType qubit2)
{
    const IdxType q0dim = (1 << max(ctrl, qubit) );
    const IdxType q1dim = (1 << min(ctrl, qubit) );
    assert (ctrl != qubit); //Non-cloning
    const IdxType outer_factor = ((sim->dim) + q0dim + q0dim - 1) >> (max(ctrl,qubit)+1);
    const IdxType mider_factor = (q0dim + q1dim + q1dim - 1) >> (min(ctrl,qubit)+1);
    const IdxType inner_factor = q1dim;
    const IdxType qubit1_dim = (1 << qubit1);
    const IdxType qubit2_dim = (1 << qubit2);

    //convert input parameters to vector form
    const __m512d e00_real_v = _mm512_set1_pd(e00_real);
    const __m512d e00_imag_v = _mm512_set1_pd(e00_imag);
    const __m512d e01_real_v = _mm512_set1_pd(e01_real);
    const __m512d e01_imag_v = _mm512_set1_pd(e01_imag);
    const __m512d e02_real_v = _mm512_set1_pd(e02_real);
    const __m512d e02_imag_v = _mm512_set1_pd(e02_imag);
    const __m512d e03_real_v = _mm512_set1_pd(e03_real);
    const __m512d e03_imag_v = _mm512_set1_pd(e03_imag);

    const __m512d e10_real_v = _mm512_set1_pd(e10_real);
    const __m512d e10_imag_v = _mm512_set1_pd(e10_imag);
    const __m512d e11_real_v = _mm512_set1_pd(e11_real);
    const __m512d e11_imag_v = _mm512_set1_pd(e11_imag);
    const __m512d e12_real_v = _mm512_set1_pd(e12_real);
    const __m512d e12_imag_v = _mm512_set1_pd(e12_imag);
    const __m512d e13_real_v = _mm512_set1_pd(e13_real);
    const __m512d e13_imag_v = _mm512_set1_pd(e13_imag);

    const __m512d e20_real_v = _mm512_set1_pd(e20_real);
    const __m512d e20_imag_v = _mm512_set1_pd(e20_imag);
    const __m512d e21_real_v = _mm512_set1_pd(e21_real);
    const __m512d e21_imag_v = _mm512_set1_pd(e21_imag);
    const __m512d e22_real_v = _mm512_set1_pd(e22_real);
    const __m512d e22_imag_v = _mm512_set1_pd(e22_imag);
    const __m512d e23_real_v = _mm512_set1_pd(e23_real);
    const __m512d e23_imag_v = _mm512_set1_pd(e23_imag);

    const __m512d e30_real_v = _mm512_set1_pd(e30_real);
    const __m512d e30_imag_v = _mm512_set1_pd(e30_imag);
    const __m512d e31_real_v = _mm512_set1_pd(e31_real);
    const __m512d e31_imag_v = _mm512_set1_pd(e31_imag);
    const __m512d e32_real_v = _mm512_set1_pd(e32_real);
    const __m512d e32_imag_v = _mm512_set1_pd(e32_imag);
    const __m512d e33_real_v = _mm512_set1_pd(e33_real);
    const __m512d e33_imag_v = _mm512_set1_pd(e33_imag);

    //start
    const __m256i q0dimx2_v = _mm256_set1_epi32(q0dim+q0dim); 
    const __m256i q1dimx2_v = _mm256_set1_epi32(q1dim+q1dim); 
    const __m256i qdimx2_v = _mm256_set1_epi32(q0dim+q1dim); 
    const __m256i mider_factor_v = _mm256_set1_epi32(mider_factor); 
    const __m256i factors_v =  _mm256_set1_epi32(inner_factor*mider_factor*outer_factor); 
    const __m256i qubit1_dim_v = _mm256_set1_epi32(qubit1_dim); 
    const __m256i qubit2_dim_v = _mm256_set1_epi32(qubit2_dim); 

    const __m256i inner_factor_rm_v = _mm256_set1_epi32(inner_factor-1);
    const __m256i dim_v = _mm256_set1_epi32(sim->dim);
    const __m256i inc=_mm256_set1_epi32(8); 
    __m256i idx=_mm256_set_epi32(0,1,2,3,4,5,6,7); 

    assert(outer_factor*mider_factor <= (1u<<20));
    const __m256i div_f0_v = _mm256_set1_epi32( (1u<<20)/(outer_factor*mider_factor));
    const __m256i div_f1_v = _mm256_set1_epi32( (1u<<20)/mider_factor);

    for (IdxType i=0; i<outer_factor*mider_factor*inner_factor*(sim->m_cpu);
            i+=8, idx=_mm256_add_epi32(idx,inc)) 
    {
        __m256i tmp0, tmp1, tmp2, tmp3; 
        tmp0 = _mm256_srli_epi32(idx,min(ctrl,qubit)); //idx/inner_factor
        tmp1 = _mm256_mullo_epi32(tmp0,div_f0_v);
        
        // IdxType col = i / (outer_factor * mider_factor * inner_factor);
        const __m256i col = _mm256_srli_epi32(tmp1,20);
        tmp2 = _mm256_mullo_epi32(col, factors_v);
        // IdxType row = i % (outer_factor * mider_factor * inner_factor);
        const __m256i row = _mm256_sub_epi32(idx, tmp2); 

        // IdxType outer = ((row/inner_factor) / (mider_factor)) * (q0dim+q0dim);
        tmp0 = _mm256_srli_epi32(row,min(ctrl,qubit)); // =>row/inner_factor
        tmp1 = _mm256_mullo_epi32(tmp0,div_f1_v);  
        tmp1 = _mm256_srli_epi32(tmp1,20);// =>(row/inner_factor)/mider_factor

        const __m256i outer = _mm256_mullo_epi32(tmp1,q0dimx2_v);
        // IdxType mider = ((row/inner_factor) % (mider_factor)) * (q1dim+q1dim);
        tmp2 = _mm256_mullo_epi32(tmp1,mider_factor_v);  //(row/inner_factor)/mider_factor * mider_factor
        tmp3 = _mm256_sub_epi32(tmp0,tmp2);//(row/inner_factor) - ((row/inner_factor)/mider_factor * mider_factor)
        const __m256i mider = _mm256_mullo_epi32(tmp3,q1dimx2_v);
        // IdxType inner = row % inner_factor;
        const __m256i inner = _mm256_and_si256(row,inner_factor_rm_v); //row & (inner_factor-1) 

        tmp0 = _mm256_mullo_epi32(col,dim_v);
        tmp1 = _mm256_add_epi32(tmp0,outer);
        tmp2 = _mm256_add_epi32(tmp1,mider);
        tmp3 = _mm256_add_epi32(tmp2,inner);

        const __m256i pos0 = tmp3;
        const __m256i pos1 = _mm256_add_epi32(tmp3,qubit2_dim_v);
        const __m256i pos2 = _mm256_add_epi32(tmp3,qubit1_dim_v);
        const __m256i pos3 = _mm256_add_epi32(tmp3,qdimx2_v);
        
        const __m512d el0_real = _mm512_i32gather_pd(pos0, dm_real, 8);
        const __m512d el0_imag = _mm512_i32gather_pd(pos0, dm_imag, 8);
        const __m512d el1_real = _mm512_i32gather_pd(pos1, dm_real, 8);
        const __m512d el1_imag = _mm512_i32gather_pd(pos1, dm_imag, 8);
        const __m512d el2_real = _mm512_i32gather_pd(pos2, dm_real, 8);
        const __m512d el2_imag = _mm512_i32gather_pd(pos2, dm_imag, 8);
        const __m512d el3_real = _mm512_i32gather_pd(pos3, dm_real, 8);
        const __m512d el3_imag = _mm512_i32gather_pd(pos3, dm_imag, 8);

        __m512d tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6;
        __m512d tmp7, tmp8, tmp9, tmp10, tmp11, tmp12, tmp13, tmp14;
        
        //dm_real[pos0] = (e00_real * el0_real) - (e00_imag * el0_imag)
        //+(e01_real * el1_real) - (e01_imag * el1_imag)
        //+(e02_real * el2_real) - (e02_imag * el2_imag)
        //+(e03_real * el3_real) - (e03_imag * el3_imag);
        tmp0 = _mm512_mul_pd(e00_real_v, el0_real);
        tmp1 = _mm512_mul_pd(e00_imag_v, el0_imag);
        tmp2 = _mm512_mul_pd(e01_real_v, el1_real);
        tmp3 = _mm512_mul_pd(e01_imag_v, el1_imag);
        tmp4 = _mm512_sub_pd(tmp0, tmp1);
        tmp5 = _mm512_sub_pd(tmp2, tmp3);
        tmp6 = _mm512_add_pd(tmp4, tmp5);
        tmp7  = _mm512_mul_pd(e02_real_v, el2_real);
        tmp8  = _mm512_mul_pd(e02_imag_v, el2_imag);
        tmp9  = _mm512_mul_pd(e03_real_v, el3_real);
        tmp10 = _mm512_mul_pd(e03_imag_v, el3_imag);
        tmp11 = _mm512_sub_pd(tmp7,  tmp8);
        tmp12 = _mm512_sub_pd(tmp9,  tmp10);
        tmp13 = _mm512_add_pd(tmp11, tmp12);
        tmp14 = _mm512_add_pd(tmp6, tmp13);
        _mm512_i32scatter_pd(dm_real, pos0, tmp14, 8);

        //dm_real[pos1] = (e10_real * el0_real) - (e10_imag * el0_imag)
        //+(e11_real * el1_real) - (e11_imag * el1_imag)
        //+(e12_real * el2_real) - (e12_imag * el2_imag)
        //+(e13_real * el3_real) - (e13_imag * el3_imag);
        tmp0 = _mm512_mul_pd(e10_real_v, el0_real);
        tmp1 = _mm512_mul_pd(e10_imag_v, el0_imag);
        tmp2 = _mm512_mul_pd(e11_real_v, el1_real);
        tmp3 = _mm512_mul_pd(e11_imag_v, el1_imag);
        tmp4 = _mm512_sub_pd(tmp0, tmp1);
        tmp5 = _mm512_sub_pd(tmp2, tmp3);
        tmp6 = _mm512_add_pd(tmp4, tmp5);
        tmp7  = _mm512_mul_pd(e12_real_v, el2_real);
        tmp8  = _mm512_mul_pd(e12_imag_v, el2_imag);
        tmp9  = _mm512_mul_pd(e13_real_v, el3_real);
        tmp10 = _mm512_mul_pd(e13_imag_v, el3_imag);
        tmp11 = _mm512_sub_pd(tmp7,  tmp8);
        tmp12 = _mm512_sub_pd(tmp9,  tmp10);
        tmp13 = _mm512_add_pd(tmp11, tmp12);
        tmp14 = _mm512_add_pd(tmp6, tmp13);
        _mm512_i32scatter_pd(dm_real, pos1, tmp14, 8);

        //dm_real[pos2] = (e20_real * el0_real) - (e20_imag * el0_imag)
        //+(e21_real * el1_real) - (e21_imag * el1_imag)
        //+(e22_real * el2_real) - (e22_imag * el2_imag)
        //+(e23_real * el3_real) - (e23_imag * el3_imag);
        tmp0 = _mm512_mul_pd(e20_real_v, el0_real);
        tmp1 = _mm512_mul_pd(e20_imag_v, el0_imag);
        tmp2 = _mm512_mul_pd(e21_real_v, el1_real);
        tmp3 = _mm512_mul_pd(e21_imag_v, el1_imag);
        tmp4 = _mm512_sub_pd(tmp0, tmp1);
        tmp5 = _mm512_sub_pd(tmp2, tmp3);
        tmp6 = _mm512_add_pd(tmp4, tmp5);
        tmp7  = _mm512_mul_pd(e22_real_v, el2_real);
        tmp8  = _mm512_mul_pd(e22_imag_v, el2_imag);
        tmp9  = _mm512_mul_pd(e23_real_v, el3_real);
        tmp10 = _mm512_mul_pd(e23_imag_v, el3_imag);
        tmp11 = _mm512_sub_pd(tmp7,  tmp8);
        tmp12 = _mm512_sub_pd(tmp9,  tmp10);
        tmp13 = _mm512_add_pd(tmp11, tmp12);
        tmp14 = _mm512_add_pd(tmp6, tmp13);
        _mm512_i32scatter_pd(dm_real, pos2, tmp14, 8);

        //dm_real[pos3] = (e30_real * el0_real) - (e30_imag * el0_imag)
        //+(e31_real * el1_real) - (e31_imag * el1_imag)
        //+(e32_real * el2_real) - (e32_imag * el2_imag)
        //+(e33_real * el3_real) - (e33_imag * el3_imag);
        tmp0 = _mm512_mul_pd(e30_real_v, el0_real);
        tmp1 = _mm512_mul_pd(e30_imag_v, el0_imag);
        tmp2 = _mm512_mul_pd(e31_real_v, el1_real);
        tmp3 = _mm512_mul_pd(e31_imag_v, el1_imag);
        tmp4 = _mm512_sub_pd(tmp0, tmp1);
        tmp5 = _mm512_sub_pd(tmp2, tmp3);
        tmp6 = _mm512_add_pd(tmp4, tmp5);
        tmp7  = _mm512_mul_pd(e32_real_v, el2_real);
        tmp8  = _mm512_mul_pd(e32_imag_v, el2_imag);
        tmp9  = _mm512_mul_pd(e33_real_v, el3_real);
        tmp10 = _mm512_mul_pd(e33_imag_v, el3_imag);
        tmp11 = _mm512_sub_pd(tmp7,  tmp8);
        tmp12 = _mm512_sub_pd(tmp9,  tmp10);
        tmp13 = _mm512_add_pd(tmp11, tmp12);
        tmp14 = _mm512_add_pd(tmp6, tmp13);
        _mm512_i32scatter_pd(dm_real, pos3, tmp14, 8);

        //dm_imag[pos0] = (e00_real * el0_imag) + (e00_imag * el0_real)
        //+(e01_real * el1_imag) + (e01_imag * el1_real)
        //+(e02_real * el2_imag) + (e02_imag * el2_real)
        //+(e03_real * el3_imag) + (e03_imag * el3_real);
        tmp0 = _mm512_mul_pd(e00_real_v, el0_imag);
        tmp1 = _mm512_mul_pd(e00_imag_v, el0_real);
        tmp2 = _mm512_mul_pd(e01_real_v, el1_imag);
        tmp3 = _mm512_mul_pd(e01_imag_v, el1_real);
        tmp4 = _mm512_add_pd(tmp0, tmp1);
        tmp5 = _mm512_add_pd(tmp2, tmp3);
        tmp6 = _mm512_add_pd(tmp4, tmp5);
        tmp7  = _mm512_mul_pd(e02_real_v, el2_imag);
        tmp8  = _mm512_mul_pd(e02_imag_v, el2_real);
        tmp9  = _mm512_mul_pd(e03_real_v, el3_imag);
        tmp10 = _mm512_mul_pd(e03_imag_v, el3_real);
        tmp11 = _mm512_add_pd(tmp7,  tmp8);
        tmp12 = _mm512_add_pd(tmp9,  tmp10);
        tmp13 = _mm512_add_pd(tmp11, tmp12);
        tmp14 = _mm512_add_pd(tmp6, tmp13);
        _mm512_i32scatter_pd(dm_imag, pos0, tmp14, 8);

        //dm_imag[pos1] = (e10_real * el0_imag) + (e10_imag * el0_real)
        //+(e11_real * el1_imag) + (e11_imag * el1_real)
        //+(e12_real * el2_imag) + (e12_imag * el2_real)
        //+(e13_real * el3_imag) + (e13_imag * el3_real);
        tmp0 = _mm512_mul_pd(e10_real_v, el0_imag);
        tmp1 = _mm512_mul_pd(e10_imag_v, el0_real);
        tmp2 = _mm512_mul_pd(e11_real_v, el1_imag);
        tmp3 = _mm512_mul_pd(e11_imag_v, el1_real);
        tmp4 = _mm512_add_pd(tmp0, tmp1);
        tmp5 = _mm512_add_pd(tmp2, tmp3);
        tmp6 = _mm512_add_pd(tmp4, tmp5);
        tmp7  = _mm512_mul_pd(e12_real_v, el2_imag);
        tmp8  = _mm512_mul_pd(e12_imag_v, el2_real);
        tmp9  = _mm512_mul_pd(e13_real_v, el3_imag);
        tmp10 = _mm512_mul_pd(e13_imag_v, el3_real);
        tmp11 = _mm512_add_pd(tmp7,  tmp8);
        tmp12 = _mm512_add_pd(tmp9,  tmp10);
        tmp13 = _mm512_add_pd(tmp11, tmp12);
        tmp14 = _mm512_add_pd(tmp6, tmp13);
        _mm512_i32scatter_pd(dm_imag, pos1, tmp14, 8);

        //dm_imag[pos2] = (e20_real * el0_imag) + (e20_imag * el0_real)
        //+(e21_real * el1_imag) + (e21_imag * el1_real)
        //+(e22_real * el2_imag) + (e22_imag * el2_real)
        //+(e23_real * el3_imag) + (e23_imag * el3_real);
        tmp0 = _mm512_mul_pd(e20_real_v, el0_imag);
        tmp1 = _mm512_mul_pd(e20_imag_v, el0_real);
        tmp2 = _mm512_mul_pd(e21_real_v, el1_imag);
        tmp3 = _mm512_mul_pd(e21_imag_v, el1_real);
        tmp4 = _mm512_add_pd(tmp0, tmp1);
        tmp5 = _mm512_add_pd(tmp2, tmp3);
        tmp6 = _mm512_add_pd(tmp4, tmp5);
        tmp7  = _mm512_mul_pd(e22_real_v, el2_imag);
        tmp8  = _mm512_mul_pd(e22_imag_v, el2_real);
        tmp9  = _mm512_mul_pd(e23_real_v, el3_imag);
        tmp10 = _mm512_mul_pd(e23_imag_v, el3_real);
        tmp11 = _mm512_add_pd(tmp7,  tmp8);
        tmp12 = _mm512_add_pd(tmp9,  tmp10);
        tmp13 = _mm512_add_pd(tmp11, tmp12);
        tmp14 = _mm512_add_pd(tmp6, tmp13);
        _mm512_i32scatter_pd(dm_imag, pos2, tmp14, 8);

        //dm_imag[pos3] = (e30_real * el0_imag) + (e30_imag * el0_real)
        //+(e31_real * el1_imag) + (e31_imag * el1_real)
        //+(e32_real * el2_imag) + (e32_imag * el2_real)
        //+(e33_real * el3_imag) + (e33_imag * el3_real);
        tmp0 = _mm512_mul_pd(e30_real_v, el0_imag);
        tmp1 = _mm512_mul_pd(e30_imag_v, el0_real);
        tmp2 = _mm512_mul_pd(e31_real_v, el1_imag);
        tmp3 = _mm512_mul_pd(e31_imag_v, el1_real);
        tmp4 = _mm512_add_pd(tmp0, tmp1);
        tmp5 = _mm512_add_pd(tmp2, tmp3);
        tmp6 = _mm512_add_pd(tmp4, tmp5);
        tmp7  = _mm512_mul_pd(e32_real_v, el2_imag);
        tmp8  = _mm512_mul_pd(e32_imag_v, el2_real);
        tmp9  = _mm512_mul_pd(e33_real_v, el3_imag);
        tmp10 = _mm512_mul_pd(e33_imag_v, el3_real);
        tmp11 = _mm512_add_pd(tmp7,  tmp8);
        tmp12 = _mm512_add_pd(tmp9,  tmp10);
        tmp13 = _mm512_add_pd(tmp11, tmp12);
        tmp14 = _mm512_add_pd(tmp6, tmp13);
        _mm512_i32scatter_pd(dm_imag, pos3, tmp14, 8);
    }
}


//============== CX Gate ================
//Controlled-NOT or CNOT
/** CX   = [1 0 0 0]
           [0 1 0 0]
           [0 0 0 1]
           [0 0 1 0]
*/

///*

void CX_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag, 
        const IdxType ctrl, const IdxType qubit)
{
    const IdxType q0dim = (1 << max(ctrl, qubit) );
    const IdxType q1dim = (1 << min(ctrl, qubit) );
    assert (ctrl != qubit); //Non-cloning
    const IdxType outer_factor = ((sim->dim) + q0dim + q0dim - 1) >> (max(ctrl,qubit)+1);
    const IdxType mider_factor = (q0dim + q1dim + q1dim - 1) >> (min(ctrl,qubit)+1);
    const IdxType inner_factor = q1dim;
    const IdxType ctrldim = (1 << ctrl);

    const __m256i q0dimx2_v = _mm256_set1_epi32(q0dim+q0dim); 
    const __m256i q1dimx2_v = _mm256_set1_epi32(q1dim+q1dim); 
    const __m256i qdimx2_v = _mm256_set1_epi32(q0dim+q1dim); 
    const __m256i mider_factor_v = _mm256_set1_epi32(mider_factor); 
    const __m256i factors_v =  _mm256_set1_epi32(inner_factor*mider_factor*outer_factor); 
    const __m256i ctrldim_v = _mm256_set1_epi32(ctrldim); 
    const __m256i inner_factor_rm_v = _mm256_set1_epi32(inner_factor-1);
    const __m256i dim_v = _mm256_set1_epi32(sim->dim);
    const __m256i inc=_mm256_set1_epi32(8); 
    __m256i idx=_mm256_set_epi32(0,1,2,3,4,5,6,7); 

    assert(outer_factor*mider_factor <= (1u<<20));
    const __m256i div_f0_v = _mm256_set1_epi32( (1u<<20)/(outer_factor*mider_factor));
    const __m256i div_f1_v = _mm256_set1_epi32( (1u<<20)/mider_factor);

    for (IdxType i=0; i<outer_factor*mider_factor*inner_factor*(sim->m_cpu);
            i+=8, idx=_mm256_add_epi32(idx,inc)) 
    {
        __m256i tmp0, tmp1, tmp2, tmp3; 
        tmp0 = _mm256_srli_epi32(idx,min(ctrl,qubit)); //idx/inner_factor
        tmp1 = _mm256_mullo_epi32(tmp0,div_f0_v);
        
        // IdxType col = i / (outer_factor * mider_factor * inner_factor);
        const __m256i col = _mm256_srli_epi32(tmp1,20);
        tmp2 = _mm256_mullo_epi32(col, factors_v);
        // IdxType row = i % (outer_factor * mider_factor * inner_factor);
        const __m256i row = _mm256_sub_epi32(idx, tmp2); 

        // IdxType outer = ((row/inner_factor) / (mider_factor)) * (q0dim+q0dim);
        tmp0 = _mm256_srli_epi32(row,min(ctrl,qubit)); // =>row/inner_factor
        tmp1 = _mm256_mullo_epi32(tmp0,div_f1_v);  
        tmp1 = _mm256_srli_epi32(tmp1,20);// =>(row/inner_factor)/mider_factor

        const __m256i outer = _mm256_mullo_epi32(tmp1,q0dimx2_v);
        // IdxType mider = ((row/inner_factor) % (mider_factor)) * (q1dim+q1dim);
        tmp2 = _mm256_mullo_epi32(tmp1,mider_factor_v);  //(row/inner_factor)/mider_factor * mider_factor
        tmp3 = _mm256_sub_epi32(tmp0,tmp2);//(row/inner_factor) - ((row/inner_factor)/mider_factor * mider_factor)
        const __m256i mider = _mm256_mullo_epi32(tmp3,q1dimx2_v);
        // IdxType inner = row % inner_factor;
        const __m256i inner = _mm256_and_si256(row,inner_factor_rm_v); //row & (inner_factor-1) 

        tmp0 = _mm256_mullo_epi32(col,dim_v);
        tmp1 = _mm256_add_epi32(tmp0,outer);
        tmp2 = _mm256_add_epi32(tmp1,mider);
        tmp3 = _mm256_add_epi32(tmp2,inner);

        const __m256i pos0 = _mm256_add_epi32(tmp3,ctrldim_v);
        const __m256i pos1 = _mm256_add_epi32(tmp3,qdimx2_v);
        
        const __m512d el0_real = _mm512_i32gather_pd(pos0, dm_real, 8);
        const __m512d el0_imag = _mm512_i32gather_pd(pos0, dm_imag, 8);
        const __m512d el1_real = _mm512_i32gather_pd(pos1, dm_real, 8);
        const __m512d el1_imag = _mm512_i32gather_pd(pos1, dm_imag, 8);

        _mm512_i32scatter_pd(dm_real, pos0, el1_real, 8);// dm_real[pos0] = el1_real; 
        _mm512_i32scatter_pd(dm_imag, pos0, el1_imag, 8);// dm_imag[pos0] = el1_imag;
        _mm512_i32scatter_pd(dm_real, pos1, el0_real, 8);// dm_real[pos1] = el0_real; 
        _mm512_i32scatter_pd(dm_imag, pos1, el0_imag, 8);// dm_imag[pos1] = el0_imag;
    }
}

//============== X Gate ================
//Pauli gate: bit flip
/** X = [0 1]
        [1 0]
*/
void X_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag, const IdxType qubit)
{
    OP_HEAD;

    const __m512d el0_real = _mm512_i32gather_pd(pos0, dm_real, 8);
    const __m512d el0_imag = _mm512_i32gather_pd(pos0, dm_imag, 8);
    const __m512d el1_real = _mm512_i32gather_pd(pos1, dm_real, 8);
    const __m512d el1_imag = _mm512_i32gather_pd(pos1, dm_imag, 8);

    _mm512_i32scatter_pd(dm_real, pos0, el1_real, 8);
    _mm512_i32scatter_pd(dm_imag, pos0, el1_imag, 8);
    _mm512_i32scatter_pd(dm_real, pos1, el0_real, 8);
    _mm512_i32scatter_pd(dm_imag, pos1, el0_imag, 8);

    OP_TAIL;
}

//============== Y Gate ================
//Pauli gate: bit and phase flip
/** Y = [0 -i]
        [i  0]
*/
void Y_GATE(const Simulation* sim, ValType* dm_real,
        ValType* dm_imag, const IdxType qubit)
{
    OP_HEAD;

    const __m512d el0_real = _mm512_i32gather_pd(pos0, dm_real, 8);
    const __m512d el0_imag = _mm512_i32gather_pd(pos0, dm_imag, 8);
    const __m512d el1_real = _mm512_i32gather_pd(pos1, dm_real, 8);
    const __m512d el1_imag = _mm512_i32gather_pd(pos1, dm_imag, 8);

    _mm512_i32scatter_pd(dm_real, pos0, el1_imag, 8);
    _mm512_i32scatter_pd(dm_imag, pos0, -el1_real, 8);
    _mm512_i32scatter_pd(dm_real, pos1, -el0_imag, 8);
    _mm512_i32scatter_pd(dm_imag, pos1, el0_real, 8);

    OP_TAIL;
}

//============== Z Gate ================
//Pauli gate: phase flip
/** Z = [1  0]
        [0 -1]
*/
void Z_GATE(const Simulation* sim, ValType* dm_real, 
        ValType* dm_imag, const IdxType qubit)
{
    OP_HEAD;

    const __m512d el1_real = _mm512_i32gather_pd(pos1, dm_real, 8);
    const __m512d el1_imag = _mm512_i32gather_pd(pos1, dm_imag, 8);

    _mm512_i32scatter_pd(dm_real, pos1, -el1_real, 8);
    _mm512_i32scatter_pd(dm_imag, pos1, -el1_imag, 8);

    OP_TAIL;
}

//============== H Gate ================
//Clifford gate: Hadamard
/** H = 1/sqrt(2) * [1  1]
                    [1 -1]
*/
void H_GATE(const Simulation* sim, ValType* dm_real, 
        ValType* dm_imag,  const IdxType qubit)
{
    const __m512d s2i_v = _mm512_set1_pd(S2I);
    OP_HEAD;

    const __m512d el0_real = _mm512_i32gather_pd(pos0, dm_real, 8);
    const __m512d el0_imag = _mm512_i32gather_pd(pos0, dm_imag, 8);
    const __m512d el1_real = _mm512_i32gather_pd(pos1, dm_real, 8);
    const __m512d el1_imag = _mm512_i32gather_pd(pos1, dm_imag, 8);
    __m512d tmp0, tmp1;

    // dm_real[pos0] = S2I*(el0_real + el1_real); 
    tmp0 = _mm512_add_pd(el0_real, el1_real);
    tmp1 = _mm512_mul_pd(s2i_v, tmp0);
    _mm512_i32scatter_pd(dm_real, pos0, tmp1, 8);

    // dm_imag[pos0] = S2I*(el0_imag + el1_imag);
    tmp0 = _mm512_add_pd(el0_imag, el1_imag);
    tmp1 = _mm512_mul_pd(s2i_v, tmp0);
    _mm512_i32scatter_pd(dm_imag, pos0, tmp1, 8);

    // dm_real[pos1] = S2I*(el0_real - el1_real);
    tmp0 = _mm512_sub_pd(el0_real, el1_real);
    tmp1 = _mm512_mul_pd(s2i_v, tmp0);
    _mm512_i32scatter_pd(dm_real, pos1, tmp1, 8);

    // dm_imag[pos1] = S2I*(el0_imag - el1_imag);
    tmp0 = _mm512_sub_pd(el0_imag, el1_imag);
    tmp1 = _mm512_mul_pd(s2i_v, tmp0);
    _mm512_i32scatter_pd(dm_imag, pos1, tmp1, 8);

    OP_TAIL;
}

//============== SRN Gate ================
//Square Root of X gate, it maps |0> to ((1+i)|0>+(1-i)|1>)/2,
//and |1> to ((1-i)|0>+(1+i)|1>)/2
/** SRN = 1/2 * [1+i 1-i]
                [1-i 1+1]
*/
void SRN_GATE(const Simulation* sim, ValType* dm_real, 
        ValType* dm_imag, const IdxType qubit)
{
    const __m512d half_v = _mm512_set1_pd(0.5);
    OP_HEAD;

    const __m512d el0_real = _mm512_i32gather_pd(pos0, dm_real, 8);
    const __m512d el0_imag = _mm512_i32gather_pd(pos0, dm_imag, 8);
    const __m512d el1_real = _mm512_i32gather_pd(pos1, dm_real, 8);
    const __m512d el1_imag = _mm512_i32gather_pd(pos1, dm_imag, 8);
    __m512d tmp0, tmp1;

    // dm_real[pos0] = 0.5*( el0_real + el1_real); 
    tmp0 = _mm512_add_pd(el0_real, el1_real);
    tmp1 = _mm512_mul_pd(half_v, tmp0);
    _mm512_i32scatter_pd(dm_real, pos0, tmp1, 8);

    // dm_imag[pos0] = 0.5*( el0_imag - el1_imag);
    tmp0 = _mm512_sub_pd(el0_imag, el1_imag);
    tmp1 = _mm512_mul_pd(half_v, tmp0);
    _mm512_i32scatter_pd(dm_imag, pos0, tmp1, 8);

    // dm_real[pos1] = 0.5*( el0_real + el1_real);
    tmp0 = _mm512_add_pd(el0_real, el1_real);
    tmp1 = _mm512_mul_pd(half_v, tmp0);
    _mm512_i32scatter_pd(dm_real, pos1, tmp1, 8);

    // dm_imag[pos1] = 0.5*(-el0_imag + el1_imag);
    tmp0 = _mm512_add_pd(-el0_imag, el1_imag);
    tmp1 = _mm512_mul_pd(half_v, tmp0);
    _mm512_i32scatter_pd(dm_imag, pos1, tmp1, 8);

    OP_TAIL;
}

//============== R Gate ================
//Phase-shift gate, it leaves |0> unchanged
//and maps |1> to e^{i\psi}|1>
/** R = [1 0]
        [0 0+p*i]
*/
void R_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag, 
        const ValType phase, const IdxType qubit)
{
    const __m512d phase_v = _mm512_set1_pd(phase);
    OP_HEAD;
    const __m512d el1_real = _mm512_i32gather_pd(pos1, dm_real, 8);
    const __m512d el1_imag = _mm512_i32gather_pd(pos1, dm_imag, 8);

    // dm_real[pos1] = -(el1_imag*phase);
    __m512d tmp1 = _mm512_mul_pd(-el1_imag, phase_v);
    _mm512_i32scatter_pd(dm_real, pos1, tmp1, 8);

    // dm_imag[pos1] = el1_real*phase;
    __m512d tmp2 = _mm512_mul_pd(el1_real, phase_v);
    _mm512_i32scatter_pd(dm_imag, pos1, tmp2, 8);

    OP_TAIL;
}

//============== S Gate ================
//Clifford gate: sqrt(Z) phase gate
/** S = [1 0]
        [0 i]
*/
void S_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,  const IdxType qubit)
{
    OP_HEAD;

    const __m512d el1_real = _mm512_i32gather_pd(pos1, dm_real, 8);
    const __m512d el1_imag = _mm512_i32gather_pd(pos1, dm_imag, 8);

    _mm512_i32scatter_pd(dm_real, pos1, -el1_imag, 8); // dm_real[pos1] = -el1_imag;
    _mm512_i32scatter_pd(dm_imag, pos1, el1_real, 8); // dm_imag[pos1] = el1_real;

    OP_TAIL;
}

//============== SDG Gate ================
//Clifford gate: conjugate of sqrt(Z) phase gate
/** SDG = [1  0]
          [0 -i]
*/
void SDG_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,  const IdxType qubit)
{
    OP_HEAD;

    const __m512d el1_real = _mm512_i32gather_pd(pos1, dm_real, 8);
    const __m512d el1_imag = _mm512_i32gather_pd(pos1, dm_imag, 8);

    _mm512_i32scatter_pd(dm_real, pos1, el1_imag, 8); // dm_real[pos1] = el1_imag;
    _mm512_i32scatter_pd(dm_imag, pos1, -el1_real, 8);// dm_imag[pos1] = -el1_real;

    OP_TAIL;
}

//============== T Gate ================
//C3 gate: sqrt(S) phase gate
/** T = [1 0]
        [0 s2i+s2i*i]
*/
void T_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag, const IdxType qubit)
{
    const __m512d s2i_v = _mm512_set1_pd(S2I);
    OP_HEAD;

    const __m512d el1_real = _mm512_i32gather_pd(pos1, dm_real, 8);
    const __m512d el1_imag = _mm512_i32gather_pd(pos1, dm_imag, 8);

    __m512d tmp0, tmp1;

    // dm_real[pos1] = S2I*(el1_real-el1_imag);
    tmp0 = _mm512_sub_pd(el1_real, el1_imag);
    tmp1 = _mm512_mul_pd(s2i_v, tmp0);
    _mm512_i32scatter_pd(dm_real, pos1, tmp1, 8);

    // dm_imag[pos1] = S2I*(el1_real+el1_imag);
    tmp0 = _mm512_add_pd(el1_real, el1_imag);
    tmp1 = _mm512_mul_pd(s2i_v, tmp0);
    _mm512_i32scatter_pd(dm_imag, pos1, tmp1, 8);

    OP_TAIL;
}

//============== TDG Gate ================
//C3 gate: conjugate of sqrt(S) phase gate
/** TDG = [1 0]
          [0 s2i-s2i*i]
*/
void TDG_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag, const IdxType qubit)
{
    const __m512d s2i_v = _mm512_set1_pd(S2I);
    OP_HEAD;

    const __m512d el1_real = _mm512_i32gather_pd(pos1, dm_real, 8);
    const __m512d el1_imag = _mm512_i32gather_pd(pos1, dm_imag, 8);

    __m512d tmp0, tmp1;

    // dm_real[pos1] = S2I*( el1_real+el1_imag);
    tmp0 = _mm512_add_pd(el1_real, el1_imag);
    tmp1 = _mm512_mul_pd(s2i_v, tmp0);
    _mm512_i32scatter_pd(dm_real, pos1, tmp1, 8);

    // dm_imag[pos1] = S2I*(-el1_real+el1_imag);
    tmp0 = _mm512_add_pd(-el1_real, el1_imag);
    tmp1 = _mm512_mul_pd(s2i_v, tmp0);
    _mm512_i32scatter_pd(dm_imag, pos1, tmp1, 8);

    OP_TAIL;
}

//============== D Gate ================
/** D = [e0_real+i*e0_imag 0]
        [0 e3_real+i*e3_imag]
*/
void D_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag, 
        const ValType e0_real, const ValType e0_imag,
        const ValType e3_real, const ValType e3_imag,
        const IdxType qubit)
{
    const __m512d e0_real_v = _mm512_set1_pd(e0_real);
    const __m512d e0_imag_v = _mm512_set1_pd(e0_imag);
    const __m512d e3_real_v = _mm512_set1_pd(e3_real);
    const __m512d e3_imag_v = _mm512_set1_pd(e3_imag);

    OP_HEAD;
    const __m512d el0_real = _mm512_i32gather_pd(pos0, dm_real, 8);
    const __m512d el0_imag = _mm512_i32gather_pd(pos0, dm_imag, 8);
    const __m512d el1_real = _mm512_i32gather_pd(pos1, dm_real, 8);
    const __m512d el1_imag = _mm512_i32gather_pd(pos1, dm_imag, 8);
    __m512d tmp0, tmp1, tmp2;

    // dm_real[pos0] = (e0_real * el0_real) - (e0_imag * el0_imag);
    tmp0 = _mm512_mul_pd(e0_real_v, el0_real);
    tmp1 = _mm512_mul_pd(e0_imag_v, el0_imag);
    tmp2 = _mm512_sub_pd(tmp0, tmp1);
    _mm512_i32scatter_pd(dm_real, pos0, tmp2, 8);

    // dm_imag[pos0] = (e0_real * el0_imag) + (e0_imag * el0_real);
    tmp0 = _mm512_mul_pd(e0_real_v, el0_imag);
    tmp1 = _mm512_mul_pd(e0_imag_v, el0_real);
    tmp2 = _mm512_add_pd(tmp0, tmp1);
    _mm512_i32scatter_pd(dm_imag, pos0, tmp2, 8);

    // dm_real[pos1] = (e3_real * el1_real) - (e3_imag * el1_imag);
    tmp0 = _mm512_mul_pd(e3_real_v, el1_real);
    tmp1 = _mm512_mul_pd(e3_imag_v, el1_imag);
    tmp2 = _mm512_sub_pd(tmp0, tmp1);
    _mm512_i32scatter_pd(dm_real, pos1, tmp2, 8);

    // dm_imag[pos1] = (e3_real * el1_imag) + (e3_imag * el1_real);
    tmp0 = _mm512_mul_pd(e3_real_v, el1_imag);
    tmp1 = _mm512_mul_pd(e3_imag_v, el1_real);
    tmp2 = _mm512_add_pd(tmp0, tmp1);
    _mm512_i32scatter_pd(dm_imag, pos1, tmp2, 8);

    OP_TAIL;
}

//============== RX Gate ================
//Rotation around X-axis
void RX_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
       const ValType theta, const IdxType qubit)
{
    const __m512d rx_real = _mm512_set1_pd(cos(theta/2.0));
    const __m512d rx_imag = _mm512_set1_pd(-sin(theta/2.0));
    
    OP_HEAD;
    const __m512d el0_real = _mm512_i32gather_pd(pos0, dm_real, 8);
    const __m512d el0_imag = _mm512_i32gather_pd(pos0, dm_imag, 8);
    const __m512d el1_real = _mm512_i32gather_pd(pos1, dm_real, 8);
    const __m512d el1_imag = _mm512_i32gather_pd(pos1, dm_imag, 8);
    __m512d tmp0, tmp1, tmp2;

    // dm_real[pos0] = (rx_real * el0_real) - (rx_imag * el1_imag);
    tmp0 = _mm512_mul_pd(rx_real, el0_real);
    tmp1 = _mm512_mul_pd(rx_imag, el1_imag);
    tmp2 = _mm512_sub_pd(tmp0, tmp1);
    _mm512_i32scatter_pd(dm_real, pos0, tmp2, 8);

    // dm_imag[pos0] = (rx_real * el0_imag) + (rx_imag * el1_real);
    tmp0 = _mm512_mul_pd(rx_real, el0_imag);
    tmp1 = _mm512_mul_pd(rx_imag, el1_real);
    tmp2 = _mm512_add_pd(tmp0, tmp1);
    _mm512_i32scatter_pd(dm_imag, pos0, tmp2, 8);

    // dm_real[pos1] =  - (rx_imag * el0_imag) + (rx_real * el1_real);
    tmp0 = _mm512_mul_pd(-rx_imag, el0_imag);
    tmp1 = _mm512_mul_pd(rx_real, el1_real);
    tmp2 = _mm512_add_pd(tmp0, tmp1);
    _mm512_i32scatter_pd(dm_real, pos1, tmp2, 8);

    // dm_imag[pos1] =  + (rx_imag * el0_real) + (rx_real * el1_imag);
    tmp0 = _mm512_mul_pd(rx_imag, el0_real);
    tmp1 = _mm512_mul_pd(rx_real, el1_imag);
    tmp2 = _mm512_add_pd(tmp0, tmp1);
    _mm512_i32scatter_pd(dm_imag, pos1, tmp2, 8);

    OP_TAIL;
}

//============== RY Gate ================
//Rotation around Y-axis
void RY_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
        const ValType theta, const IdxType qubit)
{
    const __m512d e0_real = _mm512_set1_pd(cos(theta/2.0));
    const __m512d e1_real = _mm512_set1_pd(-sin(theta/2.0));
    const __m512d e2_real = _mm512_set1_pd(sin(theta/2.0));
    const __m512d e3_real = _mm512_set1_pd(cos(theta/2.0));
    
    OP_HEAD;
    const __m512d el0_real = _mm512_i32gather_pd(pos0, dm_real, 8);
    const __m512d el0_imag = _mm512_i32gather_pd(pos0, dm_imag, 8);
    const __m512d el1_real = _mm512_i32gather_pd(pos1, dm_real, 8);
    const __m512d el1_imag = _mm512_i32gather_pd(pos1, dm_imag, 8);
    __m512d tmp0, tmp1, tmp2;

    // dm_real[pos0] = (e0_real * el0_real) + (e1_real * el1_real);
    tmp0 = _mm512_mul_pd(e0_real, el0_real);
    tmp1 = _mm512_mul_pd(e1_real, el1_real);
    tmp2 = _mm512_add_pd(tmp0, tmp1);
    _mm512_i32scatter_pd(dm_real, pos0, tmp2, 8);

    // dm_imag[pos0] = (e0_real * el0_imag) + (e1_real * el1_imag);
    tmp0 = _mm512_mul_pd(e0_real, el0_imag);
    tmp1 = _mm512_mul_pd(e1_real, el1_imag);
    tmp2 = _mm512_add_pd(tmp0, tmp1);
    _mm512_i32scatter_pd(dm_imag, pos0, tmp2, 8);

    // dm_real[pos1] = (e2_real * el0_real) + (e3_real * el1_real);
    tmp0 = _mm512_mul_pd(e2_real, el0_real);
    tmp1 = _mm512_mul_pd(e3_real, el1_real);
    tmp2 = _mm512_add_pd(tmp0, tmp1);
    _mm512_i32scatter_pd(dm_real, pos1, tmp2, 8);

    // dm_imag[pos1] = (e2_real * el0_imag) + (e3_real * el1_imag);
    tmp0 = _mm512_mul_pd(e2_real, el0_imag);
    tmp1 = _mm512_mul_pd(e3_real, el1_imag);
    tmp2 = _mm512_add_pd(tmp0, tmp1);
    _mm512_i32scatter_pd(dm_imag, pos1, tmp2, 8);

    OP_TAIL;
}

//============== W Gate ================
//W gate: e^(-i*pi/4*X)
/** W = [s2i    -s2i*i]
        [-s2i*i s2i   ]
*/
void W_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag, const IdxType qubit)
{
    const __m512d s2i = _mm512_set1_pd(S2I);
    
    OP_HEAD;
    const __m512d el0_real = _mm512_i32gather_pd(pos0, dm_real, 8);
    const __m512d el0_imag = _mm512_i32gather_pd(pos0, dm_imag, 8);
    const __m512d el1_real = _mm512_i32gather_pd(pos1, dm_real, 8);
    const __m512d el1_imag = _mm512_i32gather_pd(pos1, dm_imag, 8);
    __m512d tmp0, tmp1;

    // dm_real[pos0] = S2I * (el0_real + el1_imag);
    tmp0 = _mm512_add_pd(el0_real, el1_imag);
    tmp1 = _mm512_mul_pd(s2i, tmp0);
    _mm512_i32scatter_pd(dm_real, pos0, tmp1, 8);

    // dm_imag[pos0] = S2I * (el0_imag - el1_real);
    tmp0 = _mm512_sub_pd(el0_imag, el1_real);
    tmp1 = _mm512_mul_pd(s2i, tmp0);
    _mm512_i32scatter_pd(dm_imag, pos0, tmp1, 8);

    // dm_real[pos1] = S2I * (el0_imag + el1_real);
    tmp0 = _mm512_add_pd(el0_imag, el1_real);
    tmp1 = _mm512_mul_pd(s2i, tmp0);
    _mm512_i32scatter_pd(dm_real, pos1, tmp1, 8);

    // dm_imag[pos1] = S2I * (-el0_real + el1_imag);
    tmp0 = _mm512_add_pd(-el0_real, el1_imag);
    tmp1 = _mm512_mul_pd(s2i, tmp0);
    _mm512_i32scatter_pd(dm_imag, pos1, tmp1, 8);

    OP_TAIL;
}



#endif //end of AVX512


//============== ID Gate ================
/** ID = [1 0]
         [0 1]
*/
void ID_GATE(const Simulation* sim, ValType* dm_real,
        ValType* dm_imag, const IdxType qubit)
{
    return;
}

//============== U1 Gate ================
//1-parameter 0-pulse single qubit gate
void U1_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
        const ValType lambda, const IdxType qubit)
{
    ValType e0_real = cos(-lambda/2.0);
    ValType e0_imag = sin(-lambda/2.0);
    ValType e3_real = cos(lambda/2.0);
    ValType e3_imag = sin(lambda/2.0);
    D_GATE(sim, dm_real, dm_imag, e0_real, e0_imag, e3_real, e3_imag, qubit);
}

//============== U2 Gate ================
//2-parameter 1-pulse single qubit gate
void U2_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
        const ValType phi, const ValType lambda, const IdxType qubit)
{
    ValType e0_real = S2I * cos((-phi-lambda)/2.0);
    ValType e0_imag = S2I * sin((-phi-lambda)/2.0);
    ValType e1_real = -S2I * cos((-phi+lambda)/2.0);
    ValType e1_imag = -S2I * sin((-phi+lambda)/2.0);
    ValType e2_real = S2I * cos((phi-lambda)/2.0);
    ValType e2_imag = S2I * sin((phi-lambda)/2.0);
    ValType e3_real = S2I * cos((phi+lambda)/2.0);
    ValType e3_imag = S2I * sin((phi+lambda)/2.0);
    C1_GATE(sim, dm_real, dm_imag, e0_real, e0_imag, e1_real, e1_imag,
            e2_real, e2_imag, e3_real, e3_imag, qubit);
}

//============== U3 Gate ================
//3-parameter 2-pulse single qubit gate
void U3_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
         const ValType theta, const ValType phi, 
         const ValType lambda, const IdxType qubit)
{
    ValType e0_real = cos(theta/2.0) * cos((-phi-lambda)/2.0);
    ValType e0_imag = cos(theta/2.0) * sin((-phi-lambda)/2.0);
    ValType e1_real = -sin(theta/2.0) * cos((-phi+lambda)/2.0);
    ValType e1_imag = -sin(theta/2.0) * sin((-phi+lambda)/2.0);
    ValType e2_real = sin(theta/2.0) * cos((phi-lambda)/2.0);
    ValType e2_imag = sin(theta/2.0) * sin((phi-lambda)/2.0);
    ValType e3_real = cos(theta/2.0) * cos((phi+lambda)/2.0);
    ValType e3_imag = cos(theta/2.0) * sin((phi+lambda)/2.0);
    C1_GATE(sim, dm_real, dm_imag, e0_real, e0_imag, e1_real, e1_imag,
            e2_real, e2_imag, e3_real, e3_imag, qubit);
}

//============== RZ Gate ================
//Rotation around Z-axis
void RZ_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
     const ValType phi, const IdxType qubit)
{
    U1_GATE(sim, dm_real, dm_imag, phi, qubit);
}

//============== CZ Gate ================
//Controlled-Phase
void CZ_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
        const IdxType a, const IdxType b)
{
    H_GATE(sim, dm_real, dm_imag, b);
    CX_GATE(sim, dm_real, dm_imag, a, b);
    H_GATE(sim, dm_real, dm_imag, b);
}

//============== CY Gate ================
//Controlled-Y
void CY_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
        const IdxType a, const IdxType b)
{
    SDG_GATE(sim, dm_real, dm_imag, b);
    CX_GATE(sim, dm_real, dm_imag, a, b);
    S_GATE(sim, dm_real, dm_imag, b);
}

//============== CH Gate ================
//Controlled-H
void CH_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
        const IdxType a, const IdxType b)
{
    H_GATE(sim, dm_real, dm_imag, b);
    SDG_GATE(sim, dm_real, dm_imag, b);
    CX_GATE(sim, dm_real, dm_imag, a, b);
    H_GATE(sim, dm_real, dm_imag, b);
    T_GATE(sim, dm_real, dm_imag, b);
    CX_GATE(sim, dm_real, dm_imag, a, b);
    T_GATE(sim, dm_real, dm_imag, b);
    H_GATE(sim, dm_real, dm_imag, b);
    S_GATE(sim, dm_real, dm_imag, b);
    X_GATE(sim, dm_real, dm_imag, b);
    S_GATE(sim, dm_real, dm_imag, a);
}

//============== CRZ Gate ================
//Controlled RZ rotation
void CRZ_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
        const ValType lambda, const IdxType a, const IdxType b)
{
    U1_GATE(sim, dm_real, dm_imag, lambda/2, b);
    CX_GATE(sim, dm_real, dm_imag, a, b);
    U1_GATE(sim, dm_real, dm_imag, -lambda/2, b);
    CX_GATE(sim, dm_real, dm_imag, a, b);
}

//============== CU1 Gate ================
//Controlled phase rotation 
void CU1_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
        const ValType lambda, const IdxType a, const IdxType b)
{
    U1_GATE(sim, dm_real, dm_imag, lambda/2, b);
    CX_GATE(sim, dm_real, dm_imag, a, b);
    U1_GATE(sim, dm_real, dm_imag, -lambda/2, b);
    CX_GATE(sim, dm_real, dm_imag, a, b);
    U1_GATE(sim, dm_real, dm_imag, lambda/2, b);
}

//============== CU1 Gate ================
//Controlled U
void CU3_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
        const ValType theta, const ValType phi, const ValType lambda, 
        const IdxType c, const IdxType t)
{
    ValType temp1 = (lambda-phi)/2;
    ValType temp2 = theta/2;
    ValType temp3 = -(phi+lambda)/2;
    U1_GATE(sim, dm_real, dm_imag, temp1, t);
    CX_GATE(sim, dm_real, dm_imag, c, t);
    U3_GATE(sim, dm_real, dm_imag, -temp2, 0, temp3, t);
    CX_GATE(sim, dm_real, dm_imag, c, t);
    U3_GATE(sim, dm_real, dm_imag, temp2, phi, 0, t);
}

//========= Toffoli Gate ==========
void CCX_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
        const IdxType a, const IdxType b, const IdxType c)
{
    H_GATE(sim, dm_real, dm_imag, c);
    CX_GATE(sim, dm_real, dm_imag, b,c); 
    TDG_GATE(sim, dm_real, dm_imag, c);
    CX_GATE(sim, dm_real, dm_imag, a,c); 
    T_GATE(sim, dm_real, dm_imag, c);
    CX_GATE(sim, dm_real, dm_imag, b,c); 
    TDG_GATE(sim, dm_real, dm_imag, c);
    CX_GATE(sim, dm_real, dm_imag, a,c); 
    T_GATE(sim, dm_real, dm_imag, b); 
    T_GATE(sim, dm_real, dm_imag, c); 
    H_GATE(sim, dm_real, dm_imag, c);
    CX_GATE(sim, dm_real, dm_imag, a,b); 
    T_GATE(sim, dm_real, dm_imag, a); 
    TDG_GATE(sim, dm_real, dm_imag, b);
    CX_GATE(sim, dm_real, dm_imag, a,b);
}

//========= SWAP Gate ==========
void SWAP_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
        const IdxType a, const IdxType b)
{
    CX_GATE(sim, dm_real, dm_imag, a,b);
    CX_GATE(sim, dm_real, dm_imag, b,a);
    CX_GATE(sim, dm_real, dm_imag, a,b);
}

//========= Fredkin Gate ==========
void CSWAP_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
        const IdxType a, const IdxType b, const IdxType c)
{
    CX_GATE(sim, dm_real, dm_imag, c,b);
    CCX_GATE(sim, dm_real, dm_imag, a,b,c);
    CX_GATE(sim, dm_real, dm_imag, c,b);
}

//============== CRX Gate ================
//Controlled RX rotation
void CRX_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
       const ValType lambda, const IdxType a, const IdxType b)
{
    U1_GATE(sim, dm_real, dm_imag, PI/2, b);
    CX_GATE(sim, dm_real, dm_imag, a,b);
    U3_GATE(sim, dm_real, dm_imag, -lambda/2,0,0,b);
    CX_GATE(sim, dm_real, dm_imag, a,b);
    U3_GATE(sim, dm_real, dm_imag, lambda/2,-PI/2,0,b);
}
 
//============== CRY Gate ================
//Controlled RY rotation
void CRY_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
       const ValType lambda, const IdxType a, const IdxType b)
{
    U3_GATE(sim, dm_real, dm_imag, lambda/2,0,0,b);
    CX_GATE(sim, dm_real, dm_imag, a,b);
    U3_GATE(sim, dm_real, dm_imag, -lambda/2,0,0,b);
    CX_GATE(sim, dm_real, dm_imag, a,b);
}
 
//============== RXX Gate ================
//2-qubit XX rotation
void RXX_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
       const ValType theta, const IdxType a, const IdxType b)
{
    U3_GATE(sim, dm_real, dm_imag, PI/2,theta,0,a);
    H_GATE(sim, dm_real, dm_imag, b);
    CX_GATE(sim, dm_real, dm_imag, a,b);
    U1_GATE(sim, dm_real, dm_imag, -theta,b);
    CX_GATE(sim, dm_real, dm_imag, a,b);
    H_GATE(sim, dm_real, dm_imag, b);
    U2_GATE(sim, dm_real, dm_imag, -PI,PI-theta,a);
}
 
//============== RZZ Gate ================
//2-qubit ZZ rotation
void RZZ_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
       const ValType theta, const IdxType a, const IdxType b)
{
    CX_GATE(sim, dm_real, dm_imag, a,b);
    U1_GATE(sim, dm_real, dm_imag, theta,b);
    CX_GATE(sim, dm_real, dm_imag, a,b);
}
 
//============== RCCX Gate ================
//Relative-phase CCX
void RCCX_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
       const IdxType a, const IdxType b, const IdxType c)
{
    U2_GATE(sim, dm_real, dm_imag, 0,PI,c);
    U1_GATE(sim, dm_real, dm_imag, PI/4,c);
    CX_GATE(sim, dm_real, dm_imag, b,c);
    U1_GATE(sim, dm_real, dm_imag, -PI/4,c);
    CX_GATE(sim, dm_real, dm_imag, a,c);
    U1_GATE(sim, dm_real, dm_imag, PI/4,c);
    CX_GATE(sim, dm_real, dm_imag, b,c);
    U1_GATE(sim, dm_real, dm_imag, -PI/4,c);
    U2_GATE(sim, dm_real, dm_imag, 0,PI,c);
}
 
//============== RC3X Gate ================
//Relative-phase 3-controlled X gate
void RC3X_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
       const IdxType a, const IdxType b, const IdxType c, const IdxType d)
{
    U2_GATE(sim, dm_real, dm_imag, 0,PI,d);
    U1_GATE(sim, dm_real, dm_imag, PI/4,d);
    CX_GATE(sim, dm_real, dm_imag, c,d);
    U1_GATE(sim, dm_real, dm_imag, -PI/4,d);
    U2_GATE(sim, dm_real, dm_imag, 0,PI,d);
    CX_GATE(sim, dm_real, dm_imag, a,d);
    U1_GATE(sim, dm_real, dm_imag, PI/4,d);
    CX_GATE(sim, dm_real, dm_imag, b,d);
    U1_GATE(sim, dm_real, dm_imag, -PI/4,d);
    CX_GATE(sim, dm_real, dm_imag, a,d);
    U1_GATE(sim, dm_real, dm_imag, PI/4,d);
    CX_GATE(sim, dm_real, dm_imag, b,d);
    U1_GATE(sim, dm_real, dm_imag, -PI/4,d);
    U2_GATE(sim, dm_real, dm_imag, 0,PI,d);
    U1_GATE(sim, dm_real, dm_imag, PI/4,d);
    CX_GATE(sim, dm_real, dm_imag, c,d);
    U1_GATE(sim, dm_real, dm_imag, -PI/4,d);
    U2_GATE(sim, dm_real, dm_imag, 0,PI,d);
}
 
//============== C3X Gate ================
//3-controlled X gate
void C3X_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
       const IdxType a, const IdxType b, const IdxType c, const IdxType d)
{
    H_GATE(sim, dm_real, dm_imag, d); 
    CU1_GATE(sim, dm_real, dm_imag, -PI/4,a,d); 
    H_GATE(sim, dm_real, dm_imag, d);
    CX_GATE(sim, dm_real, dm_imag, a,b);
    H_GATE(sim, dm_real, dm_imag, d); 
    CU1_GATE(sim, dm_real, dm_imag, PI/4,b,d); 
    H_GATE(sim, dm_real, dm_imag, d);
    CX_GATE(sim, dm_real, dm_imag, a,b);
    H_GATE(sim, dm_real, dm_imag, d); 
    CU1_GATE(sim, dm_real, dm_imag, -PI/4,b,d); 
    H_GATE(sim, dm_real, dm_imag, d);
    CX_GATE(sim, dm_real, dm_imag, b,c);
    H_GATE(sim, dm_real, dm_imag, d); 
    CU1_GATE(sim, dm_real, dm_imag, PI/4,c,d); 
    H_GATE(sim, dm_real, dm_imag, d);
    CX_GATE(sim, dm_real, dm_imag, a,c);
    H_GATE(sim, dm_real, dm_imag, d); 
    CU1_GATE(sim, dm_real, dm_imag, -PI/4,c,d); 
    H_GATE(sim, dm_real, dm_imag, d);
    CX_GATE(sim, dm_real, dm_imag, b,c);
    H_GATE(sim, dm_real, dm_imag, d); 
    CU1_GATE(sim, dm_real, dm_imag, PI/4,c,d); 
    H_GATE(sim, dm_real, dm_imag, d);
    CX_GATE(sim, dm_real, dm_imag, a,c);
    H_GATE(sim, dm_real, dm_imag, d); 
    CU1_GATE(sim, dm_real, dm_imag, -PI/4,c,d); 
    H_GATE(sim, dm_real, dm_imag, d);
}
 
//============== C3SQRTX Gate ================
//3-controlled sqrt(X) gate, this equals the C3X gate where the CU1
//rotations are -PI/8 not -PI/4
void C3SQRTX_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
       const IdxType a, const IdxType b, const IdxType c, const IdxType d)
{
    H_GATE(sim, dm_real, dm_imag, d); 
    CU1_GATE(sim, dm_real, dm_imag, -PI/8,a,d); 
    H_GATE(sim, dm_real, dm_imag, d);
    CX_GATE(sim, dm_real, dm_imag, a,b);
    H_GATE(sim, dm_real, dm_imag, d); 
    CU1_GATE(sim, dm_real, dm_imag, PI/8,b,d); 
    H_GATE(sim, dm_real, dm_imag, d);
    CX_GATE(sim, dm_real, dm_imag, a,b);
    H_GATE(sim, dm_real, dm_imag, d); 
    CU1_GATE(sim, dm_real, dm_imag, -PI/8,b,d); 
    H_GATE(sim, dm_real, dm_imag, d);
    CX_GATE(sim, dm_real, dm_imag, b,c);
    H_GATE(sim, dm_real, dm_imag, d); 
    CU1_GATE(sim, dm_real, dm_imag, PI/8,c,d); 
    H_GATE(sim, dm_real, dm_imag, d);
    CX_GATE(sim, dm_real, dm_imag, a,c);
    H_GATE(sim, dm_real, dm_imag, d); 
    CU1_GATE(sim, dm_real, dm_imag, -PI/8,c,d); 
    H_GATE(sim, dm_real, dm_imag, d);
    CX_GATE(sim, dm_real, dm_imag, b,c);
    H_GATE(sim, dm_real, dm_imag, d); 
    CU1_GATE(sim, dm_real, dm_imag, PI/8,c,d); 
    H_GATE(sim, dm_real, dm_imag, d);
    CX_GATE(sim, dm_real, dm_imag, a,c);
    H_GATE(sim, dm_real, dm_imag, d); 
    CU1_GATE(sim, dm_real, dm_imag, -PI/8,c,d); 
    H_GATE(sim, dm_real, dm_imag, d);
}
 
//============== C4X Gate ================
//4-controlled X gate
void C4X_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
       const IdxType a, const IdxType b, const IdxType c, 
       const IdxType d, const IdxType e)
{
    H_GATE(sim, dm_real, dm_imag, e); 
    CU1_GATE(sim, dm_real, dm_imag, -PI/2,d,e); 
    H_GATE(sim, dm_real, dm_imag, e);
    C3X_GATE(sim, dm_real, dm_imag, a,b,c,d);
    H_GATE(sim, dm_real, dm_imag, d); 
    CU1_GATE(sim, dm_real, dm_imag, PI/4,d,e); 
    H_GATE(sim, dm_real, dm_imag, d);
    C3X_GATE(sim, dm_real, dm_imag, a,b,c,d);
    C3SQRTX_GATE(sim, dm_real, dm_imag, a,b,c,e);
}

//============== RYY Gate ================
//2-qubit YY rotation
void RYY_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
       const ValType theta, const IdxType a, const IdxType b)
{
    RX_GATE(sim, dm_real, dm_imag, PI/2, a);
    RX_GATE(sim, dm_real, dm_imag, PI/2, b);
    CX_GATE(sim, dm_real, dm_imag, a, b);
    RZ_GATE(sim, dm_real, dm_imag, theta, b);
    CX_GATE(sim, dm_real, dm_imag, a, b);
    RX_GATE(sim, dm_real, dm_imag, -PI/2, a);
    RX_GATE(sim, dm_real, dm_imag, -PI/2, b);
}
 


//==================================== Gate Ops  ========================================

void U3_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    U3_GATE(sim, dm_real, dm_imag, g->theta, g->phi, g->lambda, g->qb0); 
}

void U2_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    U2_GATE(sim, dm_real, dm_imag, g->phi, g->lambda, g->qb0); 
}

void U1_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    U1_GATE(sim, dm_real, dm_imag, g->lambda, g->qb0); 
}

void CX_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    CX_GATE(sim, dm_real, dm_imag, g->qb0, g->qb1); 
}

void ID_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    ID_GATE(sim, dm_real, dm_imag, g->qb0); 
}

void X_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    X_GATE(sim, dm_real, dm_imag, g->qb0); 
}

void Y_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    Y_GATE(sim, dm_real, dm_imag, g->qb0); 
}

void Z_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    Z_GATE(sim, dm_real, dm_imag, g->qb0); 
}

void H_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    H_GATE(sim, dm_real, dm_imag, g->qb0); 
}

void S_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    S_GATE(sim, dm_real, dm_imag, g->qb0); 
}

void SDG_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    SDG_GATE(sim, dm_real, dm_imag, g->qb0); 
}

void T_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    T_GATE(sim, dm_real, dm_imag, g->qb0); 
}

void TDG_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    TDG_GATE(sim, dm_real, dm_imag, g->qb0); 
}

void RX_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    RX_GATE(sim, dm_real, dm_imag, g->theta, g->qb0); 
}

void RY_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    RY_GATE(sim, dm_real, dm_imag, g->theta, g->qb0); 
}

void RZ_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    RZ_GATE(sim, dm_real, dm_imag, g->phi, g->qb0); 
}

//Composition Ops
void CZ_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    CZ_GATE(sim, dm_real, dm_imag, g->qb0, g->qb1); 
}

void CY_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    CY_GATE(sim, dm_real, dm_imag, g->qb0, g->qb1); 
}

void SWAP_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    SWAP_GATE(sim, dm_real, dm_imag, g->qb0, g->qb1); 
}

void CH_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    CH_GATE(sim, dm_real, dm_imag, g->qb0, g->qb1); 
}

void CCX_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    CCX_GATE(sim, dm_real, dm_imag, g->qb0, g->qb1, g->qb2); 
}

void CSWAP_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    CSWAP_GATE(sim, dm_real, dm_imag, g->qb0, g->qb1, g->qb2); 
}

void CRX_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    CRX_GATE(sim, dm_real, dm_imag, g->lambda, g->qb0, g->qb1);
}

void CRY_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    CRY_GATE(sim, dm_real, dm_imag, g->lambda, g->qb0, g->qb1);
}

void CRZ_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    CRZ_GATE(sim, dm_real, dm_imag, g->lambda, g->qb0, g->qb1);
}

void CU1_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    CU1_GATE(sim, dm_real, dm_imag, g->lambda, g->qb0, g->qb1);
}

void CU3_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    CU3_GATE(sim, dm_real, dm_imag, g->theta, g->phi, g->lambda, g->qb0, g->qb1);
}

void RXX_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    RXX_GATE(sim, dm_real, dm_imag, g->theta, g->qb0, g->qb1);
}

void RZZ_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    RZZ_GATE(sim, dm_real, dm_imag, g->theta, g->qb0, g->qb1);
}

void RCCX_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    RCCX_GATE(sim, dm_real, dm_imag, g->qb0, g->qb1, g->qb2);
}

void RC3X_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    RC3X_GATE(sim, dm_real, dm_imag, g->qb0, g->qb1, g->qb2, g->qb3);
}

void C3X_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    C3X_GATE(sim, dm_real, dm_imag, g->qb0, g->qb1, g->qb2, g->qb3);
}

void C3SQRTX_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    C3SQRTX_GATE(sim, dm_real, dm_imag, g->qb0, g->qb1, g->qb2, g->qb3);
}

void C4X_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    C4X_GATE(sim, dm_real, dm_imag, g->qb0, g->qb1, g->qb2, g->qb3, g->qb4);
}

void R_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    R_GATE(sim, dm_real, dm_imag, g->theta, g->qb0);
}
void SRN_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    SRN_GATE(sim, dm_real, dm_imag, g->qb0);
}
void W_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    W_GATE(sim, dm_real, dm_imag, g->qb0); 
}

void RYY_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    RYY_GATE(sim, dm_real, dm_imag, g->theta, g->qb0, g->qb1);
}

// ============================ Device Function Pointers ================================
 func_t pU3_OP = U3_OP;
 func_t pU2_OP = U2_OP;
 func_t pU1_OP = U1_OP;
 func_t pCX_OP = CX_OP;
 func_t pID_OP = ID_OP;
 func_t pX_OP = X_OP;
 func_t pY_OP = Y_OP;
 func_t pZ_OP = Z_OP;
 func_t pH_OP = H_OP;
 func_t pS_OP = S_OP;
 func_t pSDG_OP = SDG_OP;
 func_t pT_OP = T_OP;
 func_t pTDG_OP = TDG_OP;
 func_t pRX_OP = RX_OP;
 func_t pRY_OP = RY_OP;
 func_t pRZ_OP = RZ_OP;
 func_t pCZ_OP = CZ_OP;
 func_t pCY_OP = CY_OP;
 func_t pSWAP_OP = SWAP_OP;
 func_t pCH_OP = CH_OP;
 func_t pCCX_OP = CCX_OP;
 func_t pCSWAP_OP = CSWAP_OP;
 func_t pCRX_OP = CRX_OP;
 func_t pCRY_OP = CRY_OP;
 func_t pCRZ_OP = CRZ_OP;
 func_t pCU1_OP = CU1_OP;
 func_t pCU3_OP = CU3_OP;
 func_t pRXX_OP = RXX_OP;
 func_t pRZZ_OP = RZZ_OP;
 func_t pRCCX_OP = RCCX_OP;
 func_t pRC3X_OP = RC3X_OP;
 func_t pC3X_OP = C3X_OP;
 func_t pC3SQRTX_OP = C3SQRTX_OP;
 func_t pC4X_OP = C4X_OP;
 func_t pR_OP = R_OP;
 func_t pSRN_OP = SRN_OP;
 func_t pW_OP = W_OP;
 func_t pRYY_OP = RYY_OP;
//=====================================================================================

}; //namespace DMSim
#endif
