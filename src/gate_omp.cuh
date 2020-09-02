// ---------------------------------------------------------------------------
// DM-Sim: Density-Matrix quantum circuit simulator based on GPU clusters
// Version 2.0
// ---------------------------------------------------------------------------
// File: gate_omp.cuh
// OpenMP based implementation of the scale-up DM-Sim gates and 
// simulation runtime
// ---------------------------------------------------------------------------
// Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
// Homepage: http://www.angliphd.com
// GitHub repo: http://www.github.com/pnnl/DM-Sim
// PNNL-IPID: 31919-E, ECCN: EAR99, IR: PNNL-SA-143160
// BSD Lincese.
// ---------------------------------------------------------------------------
#ifndef GATE_OMP_CUH
#define GATE_OMP_CUH

#include <assert.h>
#include <cooperative_groups.h>
#include <vector>
#include <omp.h>

#include "config.hpp"

namespace DMSim
{

using namespace cooperative_groups;
using namespace std;
class Gate;
class Simulation;
using func_t = void (*)(const Gate*, const Simulation*, ValType*, ValType*);

//Simulation runtime, is_forward?
__global__ void simulation_kernel(Simulation*, bool);

//Current DMSim supported gates: 38
enum OP 
{
    U3, U2, U1, CX, ID, X, Y, Z, H, S, 
    SDG, T, TDG, RX, RY, RZ, CZ, CY, SWAP, CH, 
    CCX, CSWAP, CRX, CRY, CRZ, CU1, CU3, RXX, RZZ, RCCX, 
    RC3X, C3X, C3SQRTX, C4X, R, SRN, W, RYY
};

//Define gate function pointers
extern __device__ func_t pU3_OP;
extern __device__ func_t pU2_OP;
extern __device__ func_t pU1_OP;
extern __device__ func_t pCX_OP;
extern __device__ func_t pID_OP;
extern __device__ func_t pX_OP;
extern __device__ func_t pY_OP;
extern __device__ func_t pZ_OP;
extern __device__ func_t pH_OP;
extern __device__ func_t pS_OP;
extern __device__ func_t pSDG_OP;
extern __device__ func_t pT_OP;
extern __device__ func_t pTDG_OP;
extern __device__ func_t pRX_OP;
extern __device__ func_t pRY_OP;
extern __device__ func_t pRZ_OP;
extern __device__ func_t pCZ_OP;
extern __device__ func_t pCY_OP;
extern __device__ func_t pSWAP_OP;
extern __device__ func_t pCH_OP;
extern __device__ func_t pCCX_OP;
extern __device__ func_t pCSWAP_OP;
extern __device__ func_t pCRX_OP;
extern __device__ func_t pCRY_OP;
extern __device__ func_t pCRZ_OP;
extern __device__ func_t pCU1_OP;
extern __device__ func_t pCU3_OP;
extern __device__ func_t pRXX_OP;
extern __device__ func_t pRZZ_OP;
extern __device__ func_t pRCCX_OP;
extern __device__ func_t pRC3X_OP;
extern __device__ func_t pC3X_OP;
extern __device__ func_t pC3SQRTX_OP;
extern __device__ func_t pC4X_OP;
extern __device__ func_t pR_OP;
extern __device__ func_t pSRN_OP;
extern __device__ func_t pW_OP;
extern __device__ func_t pRYY_OP;

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

    //upload to a specific GPU
    Gate* upload(int dev) 
    {
        cudaSafeCall(cudaSetDevice(dev));
        Gate* gpu;
        SAFE_ALOC_GPU(gpu, sizeof(Gate)); 

#define GATE_BRANCH(GATE) case GATE: \
    cudaSafeCall(cudaMemcpyFromSymbol(&op, p ## GATE ## _OP, sizeof(func_t))); break;
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
        cudaSafeCall(cudaMemcpy(gpu, this, sizeof(Gate), cudaMemcpyHostToDevice));
        return gpu;
    }
    //applying the embedded gate operation on GPU side
    __device__ void exe_op(Simulation* sim, ValType* dm_real, ValType* dm_imag)
    {
        (*(this->op))(this, sim, dm_real, dm_imag);
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
    Simulation(IdxType _n_qubits, IdxType _n_gpus) 
        : n_qubits(_n_qubits), 
        n_gpus(_n_gpus),
        dim((IdxType)1<<(n_qubits)), 
        half_dim((IdxType)1<<(n_qubits-1)),
        gpu_mem(0), 
        n_gates(0), 
        gpu_scale(floor(log((double)_n_gpus+0.5)/log(2.0))),
        lg2_m_gpu(n_qubits-gpu_scale),
        m_gpu((IdxType)1<<(lg2_m_gpu)),
        n_tile((m_gpu+TILE-1)/TILE),
        dm_num(dim*dim), 
        dm_size(dm_num*(IdxType)sizeof(ValType)),
        dm_size_per_gpu(dm_size/n_gpus),
        circuit_gpu(NULL),
        sim_gpu(NULL),
        dm_real(NULL),
        dm_imag(NULL),
        dm_real_buf(NULL),
        dm_imag_buf(NULL)
    {
        //CPU side initialization
        assert(is_power_of_2(n_gpus));
        assert(dim % n_gpus == 0);
        SAFE_ALOC_HOST(dm_real_cpu, dm_size);
        SAFE_ALOC_HOST(dm_imag_cpu, dm_size);
        SAFE_ALOC_HOST(dm_real_res, dm_size);
        SAFE_ALOC_HOST(dm_imag_res, dm_size);

        memset(dm_real_cpu, 0, dm_size);
        memset(dm_imag_cpu, 0, dm_size);
        memset(dm_real_res, 0, dm_size);
        memset(dm_imag_res, 0, dm_size);
        //Density matrix initial state [0..0] = 1
        dm_real_cpu[0] = 1;

        SAFE_ALOC_HOST(dm_real_buf_ptr, sizeof(ValType*)*n_gpus);
        SAFE_ALOC_HOST(dm_imag_buf_ptr, sizeof(ValType*)*n_gpus);
        SAFE_ALOC_HOST(dm_real_ptr, sizeof(ValType*)*n_gpus);
        SAFE_ALOC_HOST(dm_imag_ptr, sizeof(ValType*)*n_gpus);
        SAFE_ALOC_HOST(circuit_copy, sizeof(vector<Gate*>*)*n_gpus);

        //GPU side initialization
        for (unsigned d=0; d<n_gpus; d++)
        {
            cudaSafeCall(cudaSetDevice(d));
            //GPU memory allocation
            SAFE_ALOC_GPU(dm_real_ptr[d], dm_size_per_gpu);
            SAFE_ALOC_GPU(dm_imag_ptr[d], dm_size_per_gpu);
            SAFE_ALOC_GPU(dm_real_buf_ptr[d], dm_size_per_gpu);
            SAFE_ALOC_GPU(dm_imag_buf_ptr[d], dm_size_per_gpu);
            gpu_mem += dm_size_per_gpu*4;
            //GPU memory initilization
            cudaSafeCall(cudaMemcpy(dm_real_ptr[d], &dm_real_cpu[d*dim*m_gpu], 
                        dm_size_per_gpu, cudaMemcpyHostToDevice));
            cudaSafeCall(cudaMemcpy(dm_imag_ptr[d], &dm_imag_cpu[d*dim*m_gpu], 
                        dm_size_per_gpu, cudaMemcpyHostToDevice));
            cudaSafeCall(cudaMemset(dm_real_buf_ptr[d], 0, dm_size_per_gpu));
            cudaSafeCall(cudaMemset(dm_imag_buf_ptr[d], 0, dm_size_per_gpu));
            //Enable direct interconnection
            for (unsigned g=0; g<n_gpus; g++)
            {
                if (g != d) cudaSafeCall(cudaDeviceEnablePeerAccess(g,0));
            }
        }
    }

    ~Simulation()
    {
        //Release circuit
        clear_circuit();
        //Release for GPU side
        for (unsigned d=0; d<n_gpus; d++)
        {
            cudaSafeCall(cudaSetDevice(d));
            SAFE_FREE_GPU(dm_real_ptr[d]);
            SAFE_FREE_GPU(dm_imag_ptr[d]);
            SAFE_FREE_GPU(dm_real_buf_ptr[d]);
            SAFE_FREE_GPU(dm_imag_buf_ptr[d]);
            for (unsigned g=0; g<n_gpus; g++)
            {
                if (g != d) cudaSafeCall(cudaDeviceDisablePeerAccess(g));
            }
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

        SAFE_FREE_HOST(circuit_copy);
    }


    //add a gate to the current circuit
    void append(Gate* g)
    {
        CHECK_NULL_POINTER(g); 
        //Be careful! PyBind11 will auto-release the object pointed by g, 
        //so we need to creat a new Gate object inside the code
        circuit.push_back(new Gate(*g));
        n_gates++;
    }
    Simulation* upload()
    {
        assert(n_gates == circuit.size());
        //Should be null after calling clear_circuit()
        assert(circuit_gpu == NULL);
        assert(sim_gpu == NULL);

        //std::cout << "DM-Sim:" << n_gates 
        //<< "gates uploaded to GPU for execution!" << std::endl;
        
        SAFE_ALOC_HOST(sim_gpu, sizeof(Simulation*)*n_gpus);
        for (unsigned d=0; d<n_gpus; d++)
        {
            cudaSafeCall(cudaSetDevice(d));
            for (IdxType t=0; t<n_gates; t++)
            {
                Gate* g_gpu = circuit[t]->upload(d);
                circuit_copy[d].push_back(g_gpu);
            }
            SAFE_ALOC_GPU(circuit_gpu, n_gates*sizeof(Gate*));
            cudaSafeCall(cudaMemcpy(circuit_gpu, circuit_copy[d].data(), 
                        n_gates*sizeof(Gate*), cudaMemcpyHostToDevice));
            dm_real = dm_real_ptr[d];
            dm_imag = dm_imag_ptr[d];
            dm_real_buf = dm_real_buf_ptr[d];
            dm_imag_buf = dm_imag_buf_ptr[d];
            SAFE_ALOC_GPU(sim_gpu[d], sizeof(Simulation));
            cudaSafeCall(cudaMemcpy(sim_gpu[d], this, 
                        sizeof(Simulation), cudaMemcpyHostToDevice));
        }
        return this;
    }

    //start dm simulation
    void sim()
    {
        double* sim_times;
        double* comm_times;
        SAFE_ALOC_HOST(sim_times, sizeof(double)*n_gpus);
        SAFE_ALOC_HOST(comm_times, sizeof(double)*n_gpus);

#pragma omp parallel num_threads (n_gpus) 
        {
            int d = omp_get_thread_num();
            cudaSafeCall(cudaSetDevice(d));
            gpu_timer sim_timer;
            gpu_timer comm_timer;
            cudaStream_t streams[n_gpus];
            for (unsigned g=0; g<n_gpus; g++) cudaSafeCall(cudaStreamCreate(&streams[g]));
            dim3 gridDim(1,1,1);
            cudaDeviceProp deviceProp;
            cudaSafeCall(cudaGetDeviceProperties(&deviceProp, d));
            int numBlocksPerSm;
            cudaSafeCall(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, 
                        simulation_kernel, THREADS_PER_BLOCK, 0));
            gridDim.x = numBlocksPerSm * deviceProp.multiProcessorCount;
            bool isforward = true;
            void* args[] = {&(sim_gpu[d]), &isforward};
            cudaSafeCall(cudaDeviceSynchronize());
            #pragma omp barrier
            //Forward Pass
            sim_timer.start_timer();
            cudaLaunchCooperativeKernel((void*)simulation_kernel,gridDim,
                    THREADS_PER_BLOCK,args,0);
            cudaSafeCall(cudaDeviceSynchronize());
            #pragma omp barrier

            comm_timer.start_timer();
            if (n_gpus > 1) //Need GPU-to-GPU communication only when there are multi-GPUs
            {
                //Transepose All2All Communication: In(dm_real_buf, dm_imag_buf) => Out(dm_real, dm_imag)
                for (unsigned g = 0; g<n_gpus; g++)
                {
                    unsigned dst = (d + g) % (n_gpus); 
                    //printf("i_gpu:%d,m_gpu:%d,d:%d,g:%d,dst:%d\n",d,m_gpu,d,g,dst);
                    cudaSafeCall( cudaMemcpyAsync(&(dm_real_ptr[dst][m_gpu*m_gpu*d]), 
                            &(dm_real_buf_ptr[d][dst*m_gpu*m_gpu]),
                            m_gpu*m_gpu*sizeof(ValType),
                            cudaMemcpyDefault, streams[dst]) );
                    cudaSafeCall( cudaMemcpyAsync(&(dm_imag_ptr[dst][m_gpu*m_gpu*d]), 
                            &(dm_imag_buf_ptr[d][dst*m_gpu*m_gpu]),
                            m_gpu*m_gpu*sizeof(ValType),
                            cudaMemcpyDefault, streams[dst]) );
                }
                cudaSafeCall(cudaStreamSynchronize(0));
            }

            comm_timer.stop_timer();
            #pragma omp barrier
            //Backward Pass
            isforward = false;
            cudaLaunchCooperativeKernel((void*)simulation_kernel,gridDim,THREADS_PER_BLOCK,args,0);
            sim_timer.stop_timer();
            cudaSafeCall(cudaDeviceSynchronize());
            sim_times[d] = sim_timer.measure();
            comm_times[d] = comm_timer.measure();
            #pragma omp barrier
            //Copy back
            if (n_gpus == 1)
            {
                swap_pointers(&dm_real_ptr[d], &dm_real_buf_ptr[d]);
                swap_pointers(&dm_imag_ptr[d], &dm_imag_buf_ptr[d]);
                cudaSafeCall(cudaMemcpy(dm_real_res, dm_real_ptr[d], dm_size, cudaMemcpyDeviceToHost));
                cudaSafeCall(cudaMemcpy(dm_imag_res, dm_imag_ptr[d], dm_size, cudaMemcpyDeviceToHost));
            }
            else
            {
                cudaSafeCall(cudaMemcpy(&dm_real_res[d*dim*m_gpu], dm_real_ptr[d], 
                            dm_size_per_gpu, cudaMemcpyDeviceToHost));
                cudaSafeCall(cudaMemcpy(&dm_imag_res[d*dim*m_gpu], dm_imag_ptr[d], 
                            dm_size_per_gpu, cudaMemcpyDeviceToHost));
            }
            cudaSafeCall(cudaDeviceSynchronize());

        } //end of OpenMP parallel

        double avg_comm_time = 0;
        double avg_sim_time = 0;
        for (unsigned d=0; d<n_gpus; d++)
        {
            avg_comm_time += comm_times[d];
            avg_sim_time += sim_times[d];
        }
        avg_comm_time /= (double)n_gpus;
        avg_sim_time /= (double)n_gpus;

        printf("\n============== DM-Sim ===============\n");
        printf("nqubits:%d, ngates:%d, ngpus:%d, comp:%.3lf ms, comm:%.3lf ms, sim:%.3lf ms, mem:%.3lf MB, mem_per_gpu:%.3lf MB\n",
                n_qubits, n_gates, n_gpus, avg_sim_time-avg_comm_time, avg_comm_time, 
                avg_sim_time, gpu_mem/1024/1024, gpu_mem/1024/1024/n_gpus);
        printf("=====================================\n");
    }

    void clear_circuit()
    {
        if (sim_gpu != NULL)
        {
            for (unsigned d=0; d<n_gpus; d++)
            {
                SAFE_FREE_GPU(sim_gpu[d]);
                for (unsigned i=0; i<n_gates; i++)
                    SAFE_FREE_GPU(circuit_copy[d][i]);
                circuit_copy[d].clear();
            }
        }
        SAFE_FREE_HOST(sim_gpu);
        SAFE_FREE_GPU(circuit_gpu);
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
        for (IdxType i=0; i<sv_num; i++)
            sv_diag[i] = abs(dm_real_res[i*dim+i]); //sv_diag[i] = dm_real_res[i*dim+i];
        ValType* sv_diag_scan = NULL;
        SAFE_ALOC_HOST(sv_diag_scan, (sv_num+1)*sizeof(ValType));
        sv_diag_scan[0] = 0;
        for (IdxType i=1; i<sv_num+1; i++)
            sv_diag_scan[i] = sv_diag_scan[i-1]+sv_diag[i-1];
        srand(RAND_SEED);
        IdxType* res_state = (IdxType*)malloc(repetition*sizeof(IdxType));
        memset(res_state, 0, (repetition*sizeof(IdxType)));
        for (unsigned i=0; i<repetition; i++)
        {
            ValType r = (ValType)rand()/(ValType)RAND_MAX;
            for (IdxType j=0; j<sv_num; j++)
                if (sv_diag_scan[j]<=r && r<sv_diag_scan[j+1])
                    res_state[i] = j;
        }
        if ( abs(sv_diag_scan[sv_num] - 1.0) > ERROR_BAR )
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
    // gpu_scale is 2^x of the number of GPUs, e.g., with 8 GPUs the gpu_scale is 3 (2^3=8)
    const IdxType gpu_scale;
    const IdxType n_gpus;
    const IdxType dim;
    const IdxType half_dim;
    const IdxType lg2_m_gpu;
    const IdxType m_gpu;
    const IdxType n_tile;

    const IdxType dm_num;
    const IdxType dm_size;
    const IdxType dm_size_per_gpu;

    IdxType n_gates;
    //CPU arrays
    ValType* dm_real_cpu;
    ValType* dm_imag_cpu;
    ValType* dm_real_res;
    ValType* dm_imag_res;

    //GPU pointers on CPU
    ValType** dm_real_ptr;
    ValType** dm_imag_ptr;
    ValType** dm_real_buf_ptr;
    ValType** dm_imag_buf_ptr;
   
    //GPU arrays, they are used as alias of particular pointers
    ValType* dm_real;
    ValType* dm_imag;
    ValType* dm_real_buf;
    ValType* dm_imag_buf;
    
    ValType gpu_mem;
    //hold the CPU-side gates
    vector<Gate*> circuit;
    //for freeing GPU-side gates in clear(), otherwise there can be GPU memory leak
    vector<Gate*>* circuit_copy;
    //hold the GPU-side gates
    Gate** circuit_gpu;
    //hold the GPU-side simulator instances
    Simulation** sim_gpu;
};

__device__ void circuit(Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    for (IdxType t=0; t<(sim->n_gates); t++)
    {
        ((sim->circuit_gpu)[t])->exe_op(sim, dm_real, dm_imag);
    }
}

//Blockwise transpose via shared memory
__device__ __inline__ void block_transpose(Simulation* sim, ValType* dm_real, ValType* dm_imag, 
        const ValType* real_buf, const ValType* imag_buf)
{
    __shared__ ValType smem_real[TILE][TILE+1];
    __shared__ ValType smem_imag[TILE][TILE+1];
    IdxType tlx = threadIdx.x % TILE;
    IdxType tly = threadIdx.x / TILE;
    for (IdxType bid = blockIdx.x; bid < (sim->n_tile)*(sim->n_tile)*(sim->n_gpus); 
            bid += gridDim.x)
    {
        IdxType bz = bid / ((sim->n_tile) * (sim->n_tile)); 
        IdxType by = (bid % ((sim->n_tile)*(sim->n_tile))) / (sim->n_tile);
        IdxType bx = bid % (sim->n_tile);
        IdxType tx = bx * TILE + tlx;
        IdxType ty = by * TILE + tly;

        if ((tlx < (sim->m_gpu)) && (tly < (sim->m_gpu)))
        {
            IdxType in_idx = ty*(sim->dim)+bz*(sim->m_gpu)+tx;
            smem_real[tly][tlx] = real_buf[in_idx];
            smem_imag[tly][tlx] = -imag_buf[in_idx];
        }
        __syncthreads(); //The two ifs cannot merge, looks like a cuda bug on Volta GPU
        if ((tlx < (sim->m_gpu)) && (tly < (sim->m_gpu)))
        {
            IdxType out_idx = (bx*TILE+tly)*(sim->dim)+ bz*(sim->m_gpu) + by*TILE+tlx;
            dm_real[out_idx] = smem_real[tlx][tly];
            dm_imag[out_idx] = smem_imag[tlx][tly];
        }
    } 
}

//Packing portions for all-to-all communication, see our paper for detail.
__device__ __inline__ void packing(Simulation* sim, const ValType* dm_real, const ValType* dm_imag,
        ValType* real_buf, ValType* imag_buf)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x; 
    for (IdxType i = tid; i < (sim->dim) * (sim->m_gpu); i+=blockDim.x*gridDim.x)
    {
        ////Original version with sementics
        //IdxType w_in_block = i / dim;
        //IdxType block_id = (i % dim) / m_gpu;
        //IdxType h_in_block = (i % dim) % m_gpu;
        //IdxType id_in_dm = w_in_block*dim+(i%dim);
        //IdxType id_in_buf = block_id * m_gpu * m_gpu + w_in_block * m_gpu + h_in_block;

        //Optimized version
        IdxType w_in_block = (i >> (sim->n_qubits));
        IdxType block_id = (i & (sim->dim-1)) >> (sim->lg2_m_gpu);
        IdxType h_in_block = (i & (sim->dim-1)) & (sim->m_gpu-1);
        IdxType id_in_dm = (w_in_block << (sim->n_qubits))+(i & (sim->dim-1));
        IdxType id_in_buf = (block_id << (sim->lg2_m_gpu+sim->lg2_m_gpu)) 
            + (w_in_block << (sim->lg2_m_gpu)) + h_in_block;

        real_buf[id_in_buf] = dm_real[id_in_dm];
        imag_buf[id_in_buf] = dm_imag[id_in_dm];
    }
}

//Unpacking portions after all-to-all communication, see our paper for detail.
__device__ __inline__ void unpacking(Simulation* sim, ValType* send_real, ValType* send_imag,
        const ValType* recv_real, const ValType* recv_imag)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x; 
    for (IdxType i = tid; i < (sim->dim) * (sim->m_gpu); i+=blockDim.x*gridDim.x)
    {
        ////Original version with sementics
        //IdxType j = i / dim; 
        //IdxType id_in_buf = j * dim + (i % dim);
        //IdxType block_id = id_in_buf / (m_gpu*m_gpu);
        //IdxType in_block_id = id_in_buf % (m_gpu*m_gpu);
        //IdxType w_in_block = in_block_id / m_gpu;
        //IdxType h_in_block = in_block_id % m_gpu;
        //IdxType dm_w = w_in_block;
        //IdxType dm_h = h_in_block + m_gpu*block_id;
        //IdxType id_in_dim = dm_w * dim + dm_h;

        //Optimized version
        IdxType j = (i >> (sim->n_qubits)); 
        IdxType id_in_buf = (j << (sim->n_qubits)) + (i & (sim->dim-0x1));
        IdxType block_id = (id_in_buf >> (sim->lg2_m_gpu+sim->lg2_m_gpu));
        IdxType in_block_id = (id_in_buf & ((sim->m_gpu)*(sim->m_gpu)-0x1));
        IdxType w_in_block = (in_block_id >> (sim->lg2_m_gpu));
        IdxType h_in_block = (in_block_id & (sim->m_gpu-1));
        IdxType dm_w = w_in_block;
        IdxType dm_h = h_in_block + (block_id<<(sim->lg2_m_gpu));
        IdxType id_in_dim = (dm_w << (sim->n_qubits)) + dm_h;

        send_real[id_in_dim] = recv_real[id_in_buf]; 
        send_imag[id_in_dim] = recv_imag[id_in_buf]; 
    }
}

__global__ void simulation_kernel(Simulation* sim, bool isforward)
{
    grid_group grid = this_grid(); 
    if (sim->n_gpus == 1)
    {
        //================ Single GPU version ===================
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
            grid.sync();
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
            grid.sync();
            block_transpose(sim, sim->dm_real, sim->dm_imag,
                    sim->dm_real_buf, sim->dm_imag_buf); //(out,in)
            grid.sync();
            circuit(sim, sim->dm_real, sim->dm_imag);
        }
    }
}

//=================================== Gate Definition ==========================================

//Define MG-BSP machine operation header (Original version with semantics)
// #define OP_HEAD_ORIGIN grid_group grid = this_grid(); \
    const int tid = blockDim.x * blockIdx.x + threadIdx.x; \
    const IdxType outer_bound = (1 << ( (sim->n_qubits) - qubit - 1)); \
    const IdxType inner_bound = (1 << qubit); \
        for (IdxType i = tid;i<outer_bound*inner_bound*(sim->m_gpu);\
                i+=blockDim.x*gridDim.x){ \
            IdxType col = i / (inner_bound * outer_bound); \
            IdxType outer = (i % (inner_bound * outer_bound)) / inner_bound; \
            IdxType inner =  i % inner_bound; \
            IdxType offset = (2 * outer) * inner_bound; \
            IdxType pos0 = col * (sim->dim) + offset + inner; \
            IdxType pos1 = pos0 + inner_bound; 

//Define MG-BSP machine operation header (Optimized version)
#define OP_HEAD grid_group grid = this_grid(); \
    const int tid = blockDim.x * blockIdx.x + threadIdx.x; \
        for (IdxType i=tid; i<((sim->half_dim)<<(sim->lg2_m_gpu));\
                i+=blockDim.x*gridDim.x){ \
            IdxType col = (i >> (sim->n_qubits-1)); \
            IdxType outer = ((i & ((sim->half_dim)-1)) >> qubit); \
            IdxType inner =  (i & ((1<<qubit)-1)); \
            IdxType offset = (outer << (qubit+1)); \
            IdxType pos0 = (col << (sim->n_qubits)) + offset + inner; \
            IdxType pos1 = pos0 + (1<<qubit); 

//Define MG-BSP machine operation footer
#define OP_TAIL  } grid.sync(); 

//============== Unified 1-qubit Gate ================
__device__ __inline__ void C1_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag, 
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
__device__ __inline__ void C2_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag, 
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
    grid_group grid = this_grid(); 
    const int tid = blockDim.x * blockIdx.x + threadIdx.x; 
    const IdxType outer_bound1 = (1 << ( (sim->n_qubits) - qubit1 - 1)); 
    const IdxType inner_bound1 = (1 << qubit1); 
    const IdxType outer_bound2 = (1 << ( (sim->n_qubits) - qubit2 - 1)); 
    const IdxType inner_bound2 = (1 << qubit2); 
    for (IdxType i = tid; i < outer_bound1* inner_bound2 * (sim->m_gpu); i++)
    { 
        IdxType col = i / (inner_bound1 * outer_bound1); 
        IdxType outer1 = (i % (inner_bound1 * outer_bound1)) / inner_bound1; 
        IdxType inner1 =  i % inner_bound1; 
        IdxType offset1 = (2 * outer1) * inner_bound1; 
        IdxType pos0 = col * (sim->dim) + offset1 + inner1; 
        IdxType pos1 = pos0 + inner_bound1; 
        const ValType el0_real = dm_real[pos0]; 
        const ValType el0_imag = dm_imag[pos0];
        const ValType el1_real = dm_real[pos1]; 
        const ValType el1_imag = dm_imag[pos1];

        for (IdxType i = tid; i < outer_bound2* inner_bound2; i+=blockDim.x*gridDim.x)
        { 
            IdxType outer2 = i / inner_bound2; 
            IdxType inner2 = i % inner_bound2;
            IdxType offset2 = (2 * outer2) * inner_bound2; 
            IdxType pos2 = col * (sim->dim) + offset2 + inner2; 
            IdxType pos3 = pos2 + inner_bound2; 
            const ValType el2_real = dm_real[pos2]; 
            const ValType el2_imag = dm_imag[pos2];
            const ValType el3_real = dm_real[pos3]; 
            const ValType el3_imag = dm_imag[pos3];
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
__device__ __inline__ void CX_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag, 
        const IdxType ctrl, const IdxType qubit)
{
    grid_group grid = this_grid(); 
    const int tid = blockDim.x * blockIdx.x + threadIdx.x; 
    const IdxType q0dim = (1 << max(ctrl, qubit) );
    const IdxType q1dim = (1 << min(ctrl, qubit) );
    assert (ctrl != qubit); //Non-cloning
    const IdxType outer_factor = ((sim->dim) + q0dim + q0dim - 1) >> (max(ctrl,qubit)+1);
    const IdxType mider_factor = (q0dim + q1dim + q1dim - 1) >> (min(ctrl,qubit)+1);
    const IdxType inner_factor = q1dim;
    const IdxType ctrldim = (1 << ctrl);

    for (IdxType i = tid; i < outer_factor * mider_factor * inner_factor * (sim->m_gpu); 
            i+=blockDim.x*gridDim.x)
    {
        IdxType col = i / (outer_factor * mider_factor * inner_factor);
        IdxType row = i % (outer_factor * mider_factor * inner_factor);
        IdxType outer = ((row/inner_factor) / (mider_factor)) * (q0dim+q0dim);
        IdxType mider = ((row/inner_factor) % (mider_factor)) * (q1dim+q1dim);
        IdxType inner = row % inner_factor;

        IdxType pos0 = col * (sim->dim) + outer + mider + inner + ctrldim;
        IdxType pos1 = col * (sim->dim) + outer + mider + inner + q0dim + q1dim;
        //assert (pos0 < dim*m_gpu); //ensure not out of bound
        //assert (pos1 < dim*m_gpu); //ensure not out of bound
        const ValType el0_real = dm_real[pos0]; 
        const ValType el0_imag = dm_imag[pos0];
        const ValType el1_real = dm_real[pos1]; 
        const ValType el1_imag = dm_imag[pos1];
        dm_real[pos0] = el1_real; 
        dm_imag[pos0] = el1_imag;
        dm_real[pos1] = el0_real; 
        dm_imag[pos1] = el0_imag;
    }
    grid.sync();
}

//============== X Gate ================
//Pauli gate: bit flip
/** X = [0 1]
        [1 0]
*/
__device__ __inline__ void X_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag, const IdxType qubit)
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
__device__ __inline__ void Y_GATE(const Simulation* sim, ValType* dm_real,
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
__device__ __inline__ void Z_GATE(const Simulation* sim, ValType* dm_real, 
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
__device__ __inline__ void H_GATE(const Simulation* sim, ValType* dm_real, 
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
__device__ __inline__ void SRN_GATE(const Simulation* sim, ValType* dm_real, 
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

//============== ID Gate ================
/** ID = [1 0]
         [0 1]
*/
__device__ __inline__ void ID_GATE(const Simulation* sim, ValType* dm_real,
        ValType* dm_imag, const IdxType qubit)
{
}

//============== R Gate ================
//Phase-shift gate, it leaves |0> unchanged
//and maps |1> to e^{i\psi}|1>
/** R = [1 0]
        [0 0+p*i]
*/
__device__ __inline__ void R_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag, 
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
__device__ __inline__ void S_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,  const IdxType qubit)
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
__device__ __inline__ void SDG_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,  const IdxType qubit)
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
__device__ __inline__ void T_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag, const IdxType qubit)
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
__device__ __inline__ void TDG_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag, const IdxType qubit)
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
__device__ __inline__ void D_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag, 
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

//============== U1 Gate ================
//1-parameter 0-pulse single qubit gate
__device__ __inline__ void U1_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
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
__device__ __inline__ void U2_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
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
__device__ __inline__ void U3_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
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

//============== RX Gate ================
//Rotation around X-axis
__device__ __inline__ void RX_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
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
__device__ __inline__ void RY_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
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

//============== RZ Gate ================
//Rotation around Z-axis
__device__ __inline__ void RZ_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
     const ValType phi, const IdxType qubit)
{
    U1_GATE(sim, dm_real, dm_imag, phi, qubit);
}

//============== CZ Gate ================
//Controlled-Phase
__device__ __inline__ void CZ_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
        const IdxType a, const IdxType b)
{
    H_GATE(sim, dm_real, dm_imag, b);
    CX_GATE(sim, dm_real, dm_imag, a, b);
    H_GATE(sim, dm_real, dm_imag, b);
}

//============== CY Gate ================
//Controlled-Y
__device__ __inline__ void CY_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
        const IdxType a, const IdxType b)
{
    SDG_GATE(sim, dm_real, dm_imag, b);
    CX_GATE(sim, dm_real, dm_imag, a, b);
    S_GATE(sim, dm_real, dm_imag, b);
}

//============== CH Gate ================
//Controlled-H
__device__ __inline__ void CH_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
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
__device__ __inline__ void CRZ_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
        const ValType lambda, const IdxType a, const IdxType b)
{
    U1_GATE(sim, dm_real, dm_imag, lambda/2, b);
    CX_GATE(sim, dm_real, dm_imag, a, b);
    U1_GATE(sim, dm_real, dm_imag, -lambda/2, b);
    CX_GATE(sim, dm_real, dm_imag, a, b);
}

//============== CU1 Gate ================
//Controlled phase rotation 
__device__ __inline__ void CU1_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
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
__device__ __inline__ void CU3_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
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
__device__ __inline__ void CCX_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
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
__device__ __inline__ void SWAP_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
        const IdxType a, const IdxType b)
{
    CX_GATE(sim, dm_real, dm_imag, a,b);
    CX_GATE(sim, dm_real, dm_imag, b,a);
    CX_GATE(sim, dm_real, dm_imag, a,b);
}

//========= Fredkin Gate ==========
__device__ __inline__ void CSWAP_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
        const IdxType a, const IdxType b, const IdxType c)
{
    CX_GATE(sim, dm_real, dm_imag, c,b);
    CCX_GATE(sim, dm_real, dm_imag, a,b,c);
    CX_GATE(sim, dm_real, dm_imag, c,b);
}

//============== CRX Gate ================
//Controlled RX rotation
__device__ __inline__ void CRX_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
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
__device__ __inline__ void CRY_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
       const ValType lambda, const IdxType a, const IdxType b)
{
    U3_GATE(sim, dm_real, dm_imag, lambda/2,0,0,b);
    CX_GATE(sim, dm_real, dm_imag, a,b);
    U3_GATE(sim, dm_real, dm_imag, -lambda/2,0,0,b);
    CX_GATE(sim, dm_real, dm_imag, a,b);
}
 
//============== RXX Gate ================
//2-qubit XX rotation
__device__ __inline__ void RXX_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
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
__device__ __inline__ void RZZ_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
       const ValType theta, const IdxType a, const IdxType b)
{
    CX_GATE(sim, dm_real, dm_imag, a,b);
    U1_GATE(sim, dm_real, dm_imag, theta,b);
    CX_GATE(sim, dm_real, dm_imag, a,b);
}
 
//============== RCCX Gate ================
//Relative-phase CCX
__device__ __inline__ void RCCX_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
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
__device__ __inline__ void RC3X_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
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
__device__ __inline__ void C3X_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
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
__device__ __inline__ void C3SQRTX_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
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
__device__ __inline__ void C4X_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
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

//============== W Gate ================
//W gate: e^(-i*pi/4*X)
/** W = [s2i    -s2i*i]
        [-s2i*i s2i   ]
*/
__device__ __inline__ void W_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag, const IdxType qubit)
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

//============== RYY Gate ================
//2-qubit YY rotation
__device__ __inline__ void RYY_GATE(const Simulation* sim, ValType* dm_real, ValType* dm_imag,
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

__device__ void U3_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    U3_GATE(sim, dm_real, dm_imag, g->theta, g->phi, g->lambda, g->qb0); 
}

__device__ void U2_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    U2_GATE(sim, dm_real, dm_imag, g->phi, g->lambda, g->qb0); 
}

__device__ void U1_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    U1_GATE(sim, dm_real, dm_imag, g->lambda, g->qb0); 
}

__device__ void CX_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    CX_GATE(sim, dm_real, dm_imag, g->qb0, g->qb1); 
}

__device__ void ID_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    ID_GATE(sim, dm_real, dm_imag, g->qb0); 
}

__device__ void X_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    X_GATE(sim, dm_real, dm_imag, g->qb0); 
}

__device__ void Y_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    Y_GATE(sim, dm_real, dm_imag, g->qb0); 
}

__device__ void Z_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    Z_GATE(sim, dm_real, dm_imag, g->qb0); 
}

__device__ void H_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    H_GATE(sim, dm_real, dm_imag, g->qb0); 
}

__device__ void S_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    S_GATE(sim, dm_real, dm_imag, g->qb0); 
}

__device__ void SDG_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    SDG_GATE(sim, dm_real, dm_imag, g->qb0); 
}

__device__ void T_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    T_GATE(sim, dm_real, dm_imag, g->qb0); 
}

__device__ void TDG_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    TDG_GATE(sim, dm_real, dm_imag, g->qb0); 
}

__device__ void RX_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    RX_GATE(sim, dm_real, dm_imag, g->theta, g->qb0); 
}

__device__ void RY_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    RY_GATE(sim, dm_real, dm_imag, g->theta, g->qb0); 
}

__device__ void RZ_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    RZ_GATE(sim, dm_real, dm_imag, g->phi, g->qb0); 
}

//Composition Ops
__device__ void CZ_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    CZ_GATE(sim, dm_real, dm_imag, g->qb0, g->qb1); 
}

__device__ void CY_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    CY_GATE(sim, dm_real, dm_imag, g->qb0, g->qb1); 
}

__device__ void SWAP_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    SWAP_GATE(sim, dm_real, dm_imag, g->qb0, g->qb1); 
}

__device__ void CH_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    CH_GATE(sim, dm_real, dm_imag, g->qb0, g->qb1); 
}

__device__ void CCX_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    CCX_GATE(sim, dm_real, dm_imag, g->qb0, g->qb1, g->qb2); 
}

__device__ void CSWAP_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    CSWAP_GATE(sim, dm_real, dm_imag, g->qb0, g->qb1, g->qb2); 
}

__device__ void CRX_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    CRX_GATE(sim, dm_real, dm_imag, g->lambda, g->qb0, g->qb1);
}

__device__ void CRY_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    CRY_GATE(sim, dm_real, dm_imag, g->lambda, g->qb0, g->qb1);
}

__device__ void CRZ_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    CRZ_GATE(sim, dm_real, dm_imag, g->lambda, g->qb0, g->qb1);
}

__device__ void CU1_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    CU1_GATE(sim, dm_real, dm_imag, g->lambda, g->qb0, g->qb1);
}

__device__ void CU3_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    CU3_GATE(sim, dm_real, dm_imag, g->theta, g->phi, g->lambda, g->qb0, g->qb1);
}

__device__ void RXX_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    RXX_GATE(sim, dm_real, dm_imag, g->theta, g->qb0, g->qb1);
}

__device__ void RZZ_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    RZZ_GATE(sim, dm_real, dm_imag, g->theta, g->qb0, g->qb1);
}

__device__ void RCCX_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    RCCX_GATE(sim, dm_real, dm_imag, g->qb0, g->qb1, g->qb2);
}

__device__ void RC3X_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    RC3X_GATE(sim, dm_real, dm_imag, g->qb0, g->qb1, g->qb2, g->qb3);
}

__device__ void C3X_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    C3X_GATE(sim, dm_real, dm_imag, g->qb0, g->qb1, g->qb2, g->qb3);
}

__device__ void C3SQRTX_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    C3SQRTX_GATE(sim, dm_real, dm_imag, g->qb0, g->qb1, g->qb2, g->qb3);
}

__device__ void C4X_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    C4X_GATE(sim, dm_real, dm_imag, g->qb0, g->qb1, g->qb2, g->qb3, g->qb4);
}

__device__ void R_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    R_GATE(sim, dm_real, dm_imag, g->theta, g->qb0);
}
__device__ void SRN_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    SRN_GATE(sim, dm_real, dm_imag, g->qb0);
}
__device__ void W_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    W_GATE(sim, dm_real, dm_imag, g->qb0); 
}

__device__ void RYY_OP(const Gate* g, const Simulation* sim, ValType* dm_real, ValType* dm_imag)
{
    RYY_GATE(sim, dm_real, dm_imag, g->theta, g->qb0, g->qb1);
}




// ============================ Device Function Pointers ================================
__device__ func_t pU3_OP = U3_OP;
__device__ func_t pU2_OP = U2_OP;
__device__ func_t pU1_OP = U1_OP;
__device__ func_t pCX_OP = CX_OP;
__device__ func_t pID_OP = ID_OP;
__device__ func_t pX_OP = X_OP;
__device__ func_t pY_OP = Y_OP;
__device__ func_t pZ_OP = Z_OP;
__device__ func_t pH_OP = H_OP;
__device__ func_t pS_OP = S_OP;
__device__ func_t pSDG_OP = SDG_OP;
__device__ func_t pT_OP = T_OP;
__device__ func_t pTDG_OP = TDG_OP;
__device__ func_t pRX_OP = RX_OP;
__device__ func_t pRY_OP = RY_OP;
__device__ func_t pRZ_OP = RZ_OP;
__device__ func_t pCZ_OP = CZ_OP;
__device__ func_t pCY_OP = CY_OP;
__device__ func_t pSWAP_OP = SWAP_OP;
__device__ func_t pCH_OP = CH_OP;
__device__ func_t pCCX_OP = CCX_OP;
__device__ func_t pCSWAP_OP = CSWAP_OP;
__device__ func_t pCRX_OP = CRX_OP;
__device__ func_t pCRY_OP = CRY_OP;
__device__ func_t pCRZ_OP = CRZ_OP;
__device__ func_t pCU1_OP = CU1_OP;
__device__ func_t pCU3_OP = CU3_OP;
__device__ func_t pRXX_OP = RXX_OP;
__device__ func_t pRZZ_OP = RZZ_OP;
__device__ func_t pRCCX_OP = RCCX_OP;
__device__ func_t pRC3X_OP = RC3X_OP;
__device__ func_t pC3X_OP = C3X_OP;
__device__ func_t pC3SQRTX_OP = C3SQRTX_OP;
__device__ func_t pC4X_OP = C4X_OP;
__device__ func_t pR_OP = R_OP;
__device__ func_t pSRN_OP = SRN_OP;
__device__ func_t pW_OP = W_OP;
__device__ func_t pRYY_OP = RYY_OP;
//=====================================================================================

}; //namespace DMSim
#endif
