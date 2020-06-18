# ---------------------------------------------------------------------------
# File: dmsim_run_test.py
# Run test with different qubits, gates and ultra-deep circuits.
# It needs dmsim_gen_cir.py to generate the synthetic circuits.
# ---------------------------------------------------------------------------
# See our SC-20 paper for detail.
# Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
# Homepage: http://www.angliphd.com
# GitHub repo: http://www.github.com/pnnl/DM-Sim
# PNNL-IPID: 31919-E, ECCN: EAR99, IR: PNNL-SA-143160
# BSD Lincese.
# ---------------------------------------------------------------------------

import string
import commands
import os
import math
import sys
import argparse

#System Configurations
machine = "DGX-1"
arch = "V100"
N_GPUS = 4
N_QUBITS = 8
N_GATES = 256
TEST_TIMES = 1

# Generate synthetic circuits using "dmsim_gen_cir.py" 
def generate_circuit(n_qubit, n_gate, rand=False):
    cmd = "python dmsim_gen_cir.py -n " + str(n_qubit) + " -g " + str(n_gate)
    if rand is True:
        cmd += " -r Ture"
    if args.sim == 'omp':
        cmd += " -s omp"
    elif args.sim == 'mpi':
        cmd += " -s mpi"
    else:
        cmd += " -s sin"
    print (cmd)
    feedback = commands.getoutput(cmd)
    print (feedback)

# Update DM_Sim configuration.h
def make(n_gpu, n_qubit):
    outfile = open("configuration.h","w")
    infile = open("configuration_sample.h","r")
    for l in infile.readlines():
        if l.startswith('#define N_QUBITS'):
            s = "#define N_QUBITS " + str(n_qubit) + "\n"
        elif l.startswith('#deinfe GPU_SCALE'):
            v = int(math.log(n_gpu)/math.log(2))
            s = "#define GPU_SCALE " + str(v) + "\n"
            #print (v)
        else:
            s = l
        outfile.write(s)
    #print ("Making DM_Sim for n_qubits=" + str(n_qubit) + ", gpu_scale=" + str(v))
    outfile.close()
    infile.close()
    os.system("make -j 64")

# One test instance, we generate SQL for storing data into MySQL
def test(n_qubit, n_gate, ngpus, rand=False):
    os.system("make clean")
    generate_circuit(n_qubit, n_gate, rand)
    make(ngpus, n_qubit)
    cmd = ""
    if args.sim == 'omp':
        cmd = "./dmsim_omp"
    elif args.sim == 'mpi':
        cmd = "mpirun -np " + str(ngpus) + " ./dmsim_mpi"
    else:
        cmd = "./dmsim_sin"
    print (cmd)
    comp = 0
    comm = 0
    sim = 0
    mem = 0
    # Get average values
    for i in range(0,TEST_TIMES): 
        feedback = commands.getoutput(cmd).strip().split(',')
        print feedback
        comp += float(feedback[2][feedback[2].find(":")+1:])
        comm += float(feedback[3][feedback[3].find(":")+1:])
        sim += float(feedback[4][feedback[4].find(":")+1:])
        mem = float(feedback[5][feedback[5].find(":")+1:])
    comp /= float(TEST_TIMES)
    comm /= float(TEST_TIMES)
    sim /= float(TEST_TIMES)
    #machine, arch, gpus, qubits, gates, comp, comm, sim, mem
    sql =  "INSERT INTO qsim VALUES(" + "'" + machine +  "', " + "'" + arch + "', " + \
            str(ngpus) + ", " + str(n_qubit) + ", " + str(n_gate) + ", " +  \
            str(comp) + ", " + str(comm) + ", " + str(sim) + ", " + str(mem) +  ");"
    print (sql)
    return sql

# Testing different number of qubits from st_nq to ed_nq by +1
def qubit_test(st_nq, ed_nq, resfile):
    for q in range(st_nq, ed_nq):
        res = test(q, N_GATES, N_GPUS)
        resfile.write(str(res) + "\n") 

# Testing different number of gates from st_ng to ed_ng by *2
def gate_test(st_ng, ed_ng, resfile):
    g = st_ng
    while (g < ed_ng):
        res = test(N_QUBITS, g, N_GPUS)
        resfile.write(str(res) + "\n") 
        g = g * 2

# Testing different number of GPUs from 2 to N_GPUS by 2^{+1}
def gpu_test(resfile):
    p = 2
    while (p <= N_GPUS):
        res = test(N_QUBITS, N_GATES, p)
        resfile.write(str(res) + "\n") 
        p *= 2

# Testing very deep circuits
def deep_test(resfile):
    #gs = [10000, 100000, 1000000]
    gs = [10000]
    for g in gs:
        res = test(N_QUBITS, g, N_GPUS, rand=True)
        resfile.write(str(res) + "\n") 

# Main Program
resfile = open(str(machine) + "_result.txt","w")

parser = argparse.ArgumentParser(description='DM_Sim Run Testings: Testing DM_Sim using synthetic generated circuits.')

# Qubit test parameters
parser.add_argument('--starting_n_qubits', '-st_nq', type=int, default=4, help='Starting number of qubits in the qubit test(default=4).')
parser.add_argument('--ending_n_qubits', '-ed_nq', type=int, default=N_QUBITS, help='Ending number of qubits in the qubit test(default='+str(N_QUBITS)+').')

# Gate test parameters
parser.add_argument('--starting_n_gates', '-st_ng', type=int, default=8, help='Starting number of gates in the gate test(default=8).')
parser.add_argument('--ending_n_gates', '-ed_ng', type=int, default=N_GATES, help='Ending number of gates in the gate test(default='+str(N_GATES)+').')
parser.add_argument('--sim', '-s', default='omp', help="DM-Sim simulation mode: 'sin' for single-GPU, 'omp' for OpenMP scale-up, and 'mpi' for MPI scale-out.") 

resfile = open(str(machine) + "_result.txt","w")
args = parser.parse_args()

#qubit_test(args.starting_n_qubits, args.ending_n_qubits, resfile)
#gate_test(args.starting_n_gates, args.ending_n_gates, resfile)
#gpu_test(resfile)
deep_test(resfile)
resfile.close()

