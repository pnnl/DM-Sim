# ---------------------------------------------------------------------------
# DM-Sim: Density-Matrix Quantum Circuit Simulation Environement
# ---------------------------------------------------------------------------
# Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
# Homepage: http://www.angliphd.com
# GitHub repo: http://www.github.com/pnnl/DM-Sim
# PNNL-IPID: 31919-E, ECCN: EAR99, IR: PNNL-SA-143160
# BSD Lincese.
# ---------------------------------------------------------------------------
# File: randomtest_n14.py
# A H-gate based random test using Python API (100K H-gates, 14 qubits).
# Single-node (1 or more GPUs); No inter-GPU communication required.
# Requires: PyBind11 (https://github.com/pybind/pybind11)
#           CUDA-10.0 or newer (required by pybind11 for Python API)
# ---------------------------------------------------------------------------

import sys
import dmsim_py_omp_wrapper as dmsim
import random

if (len(sys.argv) != 3):
    print("Call using $python circuit.py n_qubits n_gpus\n")
    exit()

sim = dmsim.Simulation(int(sys.argv[1]), int(sys.argv[2]))

## Test using 100K Hadamard gates with 14 qubits
for i in range(0,100000):
    sim.append(sim.H(random.randrange(14)))

sim.upload()
sim.run()
sim.measure(10)
