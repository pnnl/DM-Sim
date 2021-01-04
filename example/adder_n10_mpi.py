# ---------------------------------------------------------------------------
# DM-Sim: Density-Matrix Quantum Circuit Simulation Environement
# ---------------------------------------------------------------------------
# Ang Li, Senior Computer Scientist
# Pacific Northwest National Laboratory(PNNL), U.S.
# Homepage: http://www.angliphd.com
# GitHub repo: http://www.github.com/pnnl/DM-Sim
# PNNL-IPID: 31919-E, ECCN: EAR99, IR: PNNL-SA-143160
# BSD Lincese.
# ---------------------------------------------------------------------------
# File: adder_n10_mpi.py
# A 10-qubit adder example using Python API using MPI.
# The NVGPU backend requires GPUDirect-RDMA support.
# Requires: PyBind11 (https://github.com/pybind/pybind11)
#           CUDA-10.0 or newer (required by pybind11 for Python API)
#           MPI4PY (https://github.com/mpi4py/mpi4py) 
# ---------------------------------------------------------------------------

import sys
import mpi4py
from mpi4py import MPI

# Using DM-Sim NVGPU backend
#import ../src/libdmsim_py_nvgpu_mpi as dmsim_mpi

# Using DM-Sim CPU backend
import ../src/libdmsim_py_cpu_mpi as dmsim_mpi

# Using DM-Sim AMDGPU backend
#import ../src/libdmsim_py_amdgpu_mpi as dmsim_mpi


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
#print ("Rank-" + str(rank) + " of " + str(size) + " processes.")

## Call via: $mpirun -np 4 python circuit.py num_of_qubits
if (len(sys.argv) != 2):
    print("Call using $python circuit.py n_qubits \n")
    exit()

n_qubits = int(sys.argv[1])

## Create simulator object
sim = dmsim_mpi.Simulation(n_qubits)

## Quantum ripple-carry adder from Cuccaro et al, quant-ph/0410184
## Define circuit module functions as below
def majority(sim, a, b, c):
	sim.append(sim.CX(c, b))
	sim.append(sim.CX(c, a))
	sim.append(sim.CCX(a, b, c))

def unmaj(sim, a, b, c):
	sim.append(sim.CCX(a, b, c))
	sim.append(sim.CX(c, a))
	sim.append(sim.CX(a, b))

## Add the gates to the circuit
sim.append(sim.X(1))
sim.append(sim.X(5))
sim.append(sim.X(6))
sim.append(sim.X(7))
sim.append(sim.X(8))

## You can upload, run and clear, then re-add gates and execute
sim.upload()
sim.run()
sim.clear_circuit()

## Add a new circuit for the current density matrix
majority(sim, 0, 5, 1)
majority(sim, 1, 6, 2)
majority(sim, 2, 7, 3)
majority(sim, 3, 8, 4)
sim.append(sim.CX(4, 9))
unmaj(sim, 3, 8, 4)
unmaj(sim, 2, 7, 3)
unmaj(sim, 1, 6, 2)
unmaj(sim, 0, 5, 1)

## Upload the new circuit to GPU
sim.upload()
## Run the new circuit
sim.run()
## Measure, 10 is the repetition or shots, return a list
res = sim.measure(10)
## Print measurement results
if rank == 0:
    print ("\n===============  Measurement (tests=" + str(len(res)) + ") ================")
    for i in range(len(res)):
        print ("Test-"+str(i)+": " + "{0:b}".format(res[i]).zfill(n_qubits))
