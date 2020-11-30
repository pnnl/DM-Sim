## ---------------------------------------------------------------------------
## DM-Sim: Density-Matrix quantum circuit simulator based on GPU clusters
## Version 2.1
## ---------------------------------------------------------------------------
## File: adder_n10_mpi.py
## A 10-qubit adder example using Python API using MPI for GPU Cluster.
## Requires GPUDirect-RDMA support.
## Requires: GCC-9.1.0 (require latest GCC)
##           LLVM-10.0.1 (required by QIR on Summit)
##           CUDA-11.0 or newer (required by QIR on Summit)
##           MPI4PY (https://github.com/mpi4py/mpi4py) 
## ---------------------------------------------------------------------------
## Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
## Homepage: http://www.angliphd.com
## GitHub repo: http://www.github.com/pnnl/DM-Sim
## PNNL-IPID: 31919-E, ECCN: EAR99, IR: PNNL-SA-143160
## BSD Lincese.
## ---------------------------------------------------------------------------

from scipy.optimize import minimize
import commands
import time

#import mpi4py
#from mpi4py import MPI
#comm = MPI.COMM_WORLD
#rank = comm.Get_rank()
#size = comm.Get_size()

total_time = 0
 
def run_program(var_params):
    global total_time
    # TODO: call QIR program here
    # run parameterized quantum program for VQE algorithm

    ## MPI
    cmd = "jsrun -n4 -a1 -g1 -c1 --smpiargs='-gpu' ./vqe_qir_mpi " + str(var_params[0]) + " " + str(var_params[1]) + " " + str(var_params[2]) 
    ## OMP
    #cmd = "./vqe_qir_omp " + str(var_params[0]) + " " + str(var_params[1]) + " " + str(var_params[2]) 

    print (cmd)
    start = time.time()
    feedback = commands.getoutput(cmd)
    stop = time.time()
    total_time += stop - start
    print (feedback)
    #print (var_params)
    #return -1.1372704220924401
    return float(feedback)
 
def VQE(initial_var_params):
    """ Run VQE Optimization to find the optimal energy and the associated variational parameters """
 
    opt_result = minimize(run_program,
                          initial_var_params,
                          method="COBYLA",
                          tol=0.000001,
                          options={'disp': True, 'maxiter': 200,'rhobeg' : 0.05})
 
    return opt_result
 
if __name__ == "__main__":
    # Initial variational parameters
    var_params = [0.001, -0.001, 0.001]
 
   # Run VQE and print the results of the optimization process
    # A large number of samples is selected for higher accuracy
    #opt_result = VQE(var_params, jw_hamiltonian, n_samples=10)

    vqe_start = time.time()
    opt_result = VQE(var_params)
    vqe_stop = time.time()
    print(opt_result)
 
    # Print difference with exact FCI value known for this bond length
    fci_value = -1.1372704220924401
    print("Difference with exact FCI value :: ", abs(opt_result.fun - fci_value))
    print("Simulation Time:" + str(total_time))
    print("Overal Time:" + str(vqe_stop-vqe_start))

 
