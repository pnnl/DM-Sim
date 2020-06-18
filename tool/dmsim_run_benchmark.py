# ---------------------------------------------------------------------------
# File: dmsim_run_benchmark.py
# Run benchmarks listed in our SC-20 paper.
# Please also refer to our QASMBench Benchmark Suite for more benchmarks:
# QASMBench:
#       GitHub Repo: http://github.com/uuudown/QASMBench
#       arXiv: https://arxiv.org/abs/2005.13018
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

# Configurations
# Benchmark folder
algorithm_folder = "../benchmark/"
TEST_TIMES = 1
machine = "DGX-1"
arch = "Volta"

# Benchmark applications
# Format: ['header_file.cuh', n_qubits, n_gates, max_n_gpus]
adder = ['adder_n9.qasm', 9, 139, 16]
deutsch = ['deutsch_n5.qasm', 5, 5, 16]

grover = ['grover_n3.qasm', 3, 123, 8]
iswap = ['iswap_n5.qasm', 5, 9, 16]
pea3pi8 = ['pea3pi8_n5.qasm', 5, 74, 16]
qec = ['qec_n5.qasm', 5, 5, 16]
qv5 = ['qv5_n5.qasm', 5, 100, 16]
w3test = ['w3test_n5.qasm', 5, 10, 16]
wstate = ['wstate_n3.qasm', 3, 30, 8]
bv = ['bv_n15.qasm', 15, 44, 16]
cc = ['cc_n15.qasm', 15, 28, 16]
qft = ['qft_n15.qasm', 15, 540, 16]
sat = ['sat_n15.qasm', 10, 679, 16]

apps = []
apps.append(deutsch)
apps.append(grover)
apps.append(wstate)
apps.append(iswap)
apps.append(pea3pi8)
apps.append(qec)
apps.append(qv5)
apps.append(w3test)
apps.append(adder)
apps.append(sat)
#apps.append(bv)
#apps.append(cc)
#apps.append(qft)


def copy_algorithm_header(header):
    cmd = "cp " + algorithm_folder + header  + " circuit.qasm"
    print (cmd)
    os.system(cmd)
    cmd = "python dmsim_qasm_ass.py -i circuit.qasm -o circuit.cuh -s omp "
    print (cmd)
    os.system(cmd)

def make(n_gpu, n_qubit):
    os.system("make clean")
    outfile = open("configuration.h","w")
    infile = open("configuration_sample.h","r")
    for l in infile.readlines():
        if l.startswith('#define N_QUBITS'):
            s = "#define N_QUBITS " + str(n_qubit) + "\n"
        elif l.startswith('#deinfe GPU_SCALE'):
            v = int(math.log(n_gpu)/math.log(2))
            s = "#define GPU_SCALE " + str(v) + "\n"
        else:
            s = l
        outfile.write(s)
    #print ("Making DM_Sim for n_qubits=" + str(n_qubit) + ", gpu_scale=" + str(v))
    outfile.close()
    infile.close()
    os.system("make -j 16")

def test(n_qubit, n_gate, n_gpus, header=None):
    copy_algorithm_header(header)
    make(n_gpus, n_qubit)
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
    for i in range(0,TEST_TIMES):
        feedback = commands.getoutput(cmd).strip().split(',')
        #print feedback
        comp += float(feedback[2][feedback[2].find(":")+1:])
        comm += float(feedback[3][feedback[3].find(":")+1:])
        sim += float(feedback[4][feedback[4].find(":")+1:])
        mem = float(feedback[5][feedback[5].find(":")+1:])
    comp /= float(TEST_TIMES)
    comm /= float(TEST_TIMES)
    sim /= float(TEST_TIMES)

    #app, machine, arch, gpus, qubits, gates, comp, comm, sim, mem
    sql =  "INSERT INTO qsim VALUES(" + "'" + header +  "', " + \
            "'" + machine + "', " + \
            "'" + arch + "', " + \
            str(n_gpus) + ", " + str(n_qubit) + ", " + str(n_gate) + ", " +  \
            str(comp) + ", " + str(comm) + ", " + str(sim) + ", " + str(mem) + \
            ");"
    print (sql)
    return sql

def algorithm_test(resfile):
    for app in apps:
        app_header = app[0]
        qubits = app[1]
        gates = app[2]
        max_gpus = app[3]

        if args.sim == 'sin':
            res = test(qubits, gates, 1, app_header)
            resfile.write(str(res) + "\n") 
        else:
            gpus = 2
            while (gpus <= max_gpus):
                res = test(qubits, gates, gpus, app_header)
                resfile.write(str(res) + "\n") 
                gpus = gpus * 2

# Main Program
parser = argparse.ArgumentParser(description='DM_Sim Run Benchmark: Testing DM_Sim using real quantum circuits.')
parser.add_argument('--sim', '-s', default='omp', help="DM-Sim simulation mode: 'sin' for single-GPU, 'omp' for OpenMP scale-up, and 'mpi' for MPI scale-out.") 
args = parser.parse_args()

resfile = open(str(machine) + "_result.txt","w")
algorithm_test(resfile)

