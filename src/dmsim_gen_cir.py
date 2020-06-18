# ---------------------------------------------------------------------------
# File: dmsim_gen_cir.py
# Generating synthetic circuit for profiling and performance measurement.
# It generates DM-Sim native circuit file (".cuh") rather than OpenQASM code.
# ---------------------------------------------------------------------------
# See our SC-20 paper for detail.
# Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
# Homepage: http://www.angliphd.com
# GitHub repo: http://www.github.com/pnnl/DM-Sim
# PNNL-IPID: 31919-E, ECCN: EAR99, IR: PNNL-SA-143160
# BSD Lincese.
# ---------------------------------------------------------------------------

import string
import os
import math
import sys
import random
import argparse
from scipy.stats import unitary_group

# Selective gate collection to choose a gate from
gate_collection = ['X','Y','Z','H','ID','S','T','R','N','CX']

# Main Program
parser = argparse.ArgumentParser(description='DM_Sim synthetic circuit generator: generating DM_Sim native simulation circuit according to user-defined rules.')
parser.add_argument('--n_qubits', '-n', type=int, default=8, help='Number of qubits of the Circuit(default=8).')
parser.add_argument('--n_gates', '-g', type=int, default=128, help='Number of gates of the Circuit(default=128).') 
parser.add_argument('--threshold', '-t', type=int, default=512, help='Number of gates per circuit file(default=512).') 
parser.add_argument('--seed', '-e', type=int, default=5, help='Seed for the random generator(default=5).') 
parser.add_argument('--sim', '-s', default='omp', help="DM-Sim simulation mode: 'sin' for single-GPU, 'omp' for OpenMP scale-up, and 'mpi' for MPI scale-out.") 
parser.add_argument('--random', '-r', type=bool, default=False, help='Generate purely random matrix gate or C gate (default=False).') #See our paper for detail

args = parser.parse_args()
#print (args.input)

N_QUBITS = args.n_qubits
N_GATES = args.n_gates
M_BAR = args.threshold
random.seed(args.seed)
SIM = "dmsim_" + args.sim

# Generate random unitary matrix for C1 gate
def print_random_unitary_gates(n_gates,fp):
    for i in range(0,n_gates):
        q = random.randrange(N_QUBITS)
        x = unitary_group.rvs(2)
        s = str('\tC1(') \
                + str(x[0][0].real) + ", " + str(x[0][0].imag) + ", " \
                + str(x[0][1].real) + ", " + str(x[0][1].imag) + ", " \
                + str(x[1][0].real) + ", " + str(x[1][0].imag) + ", " \
                + str(x[1][1].real) + ", " + str(x[1][1].imag) + ", " \
                + str(q) + ");\n"
        fp.write(s)

# Generate gates randomly selected from the defined gate collection.
def print_gates(n_gates,fp):
    for gate in [random.choice(gate_collection) for g in range(0,n_gates)]:
        q = random.randrange(N_QUBITS)
        s = ""
        if gate == 'CX':
            ctrl = random.randrange(N_QUBITS)
            while (ctrl == q):
                ctrl = random.randrange(N_QUBITS)
            s = "\t" + str(gate) + "(" + str(q) + ", " + str(ctrl) + ");\n"
        elif gate == 'R':
            phase = random.random()
            s = "\t" + str(gate) + "(" + str(phase) + ", " + str(q) + ");\n"
        else:
            s = "\t" + str(gate) + "(" + str(q) + ");\n"
        fp.write(s)

# Writing to the circuit file
circuit_file = open("circuit.cuh","w")
circuit_file.write("#ifndef CIRCUIT_CUH \n#define CIRCUIT_CUH \n\n")

# Writing to the makefile file
make_file = open("Makefile","w")
make_file.write("CC = nvcc\nFLAGS = -O3 -arch=sm_70 -rdc=true\nLIBS = -lm\n\nall: "\
        + SIM + "\n\n")

# For deep circuit when the depth is over the threshold
if (N_GATES > M_BAR):
    n_steps = int(math.ceil(N_GATES / M_BAR))
    for i in range(0,n_steps):
        #write header file
        s0 = str("__global__ void simulation_") + str(i) \
                + ("(double* dm_real, double* dm_imag);\n\n")
        circuit_file.write(s0)
        #write each sub-circuit file
        srcfile = open("circuit_" + str(i) + ".cu","w")
        srcfile.write('#include "gate.cuh"\n\n')
        s1 = str("__global__ void simulation_") + str(i) \
                + ("(double* dm_real, double* dm_imag)\n{\n")
        srcfile.write(s1)
        #generate random unitary gates or C gates
        if args.random is True:
            print_random_unitary_gates(min(M_BAR, N_GATES-M_BAR*i), srcfile)
        #ganerate gates sampled from defined gate collection
        else:
            print_gates(min(M_BAR, N_GATES-M_BAR*i), srcfile)
        srcfile.write("}\n\n")
        srcfile.close()
        #write corresponding command to makefile
        s2 = ''
        if args.sim == 'omp':
            s2 = str("circuit_")+str(i)+".o: circuit_" +str(i) + ".cu\n" \
                    + "\t$(CC) $(FLAGS) $(LIBS) -Xcompiler -fopenmp -c $^\n"
        elif args.sim == 'mpi':
            s2 = str("circuit_")+str(i)+".o: circuit_" +str(i) + ".cu\n" \
                    + "\t$(CC) $(FLAGS) $(LIBS) -ccbin mpicc -c $^\n"
        else:
            s2 = str("circuit_")+str(i)+".o: circuit_" +str(i) + ".cu\n" \
                    + "\t$(CC) $(FLAGS) $(LIBS) -c $^\n"
        make_file.write(s2)

    #To define an empty function for successful compilation
    s = "__device__ __inline__ void circuit(double* dm_real, double* dm_imag){}\n\n"
    circuit_file.write(s)

    #Define kernel invocation function
    s = "void deep_simulation(double* dm_real, double* dm_imag, dim3 gridDim)\n" \
            + "{\n\tvoid* args_step[] = {&dm_real, &dm_imag};\n"
    circuit_file.write(s)
    for i in range(0,n_steps):
        s = str("\tcudaLaunchCooperativeKernel((void*)simulation_") + str(i) \
                + str(",gridDim,THREADS_PER_BLOCK,args_step,0);\n")
        circuit_file.write(s)
    circuit_file.write("}\n")
    #Compile overall function

    s = ""
    if args.sim == 'omp':
        s = "\n"+SIM+".o: " + SIM + ".cu gate.cuh configuration.h circuit.cuh util.cuh\n"\
                + "\t$(CC) $(FLAGS) $(LIBS) -Xcompiler -fopenmp -c " + SIM + ".cu\n\n"
    elif args.sim == 'mpi':
        s = "\n"+SIM+".o: " + SIM + ".cu gate.cuh configuration.h circuit.cuh util.cuh\n"\
                + "\t$(CC) $(FLAGS) $(LIBS) -ccbin mpicc -c " + SIM + ".cu\n\n"
    else:
        s = "\n"+SIM+".o: " + SIM + ".cu gate.cuh configuration.h circuit.cuh util.cuh\n"\
                + "\t$(CC) $(FLAGS) $(LIBS) -c " + SIM + ".cu\n\n"
    make_file.write(s)

    
    #Separate compilation (use "make -j X") for parallel accelration of the making process
    s = SIM + ": " + SIM + ".o "
    for i in range(0,n_steps):
        s += "circuit_"+ str(i) + ".o "
    make_file.write(s+"\n")

    #Linking
    s = ""
    if args.sim == 'omp':
        s = "\t$(CC) $(FLAGS) $(LIBS) -Xcompiler -fopenmp *.o -o $@\n" + \
                "\nclean:\n\trm -rf circuit_*.cu *.o dm_sim\n\n"
    elif args.sim == 'mpi':
        s = "\t$(CC) $(FLAGS) $(LIBS) -ccbin mpicc *.o -o $@\n" + \
                "\nclean:\n\trm -rf circuit_*.cu *.o dm_sim\n\n"
    else:
        s = "\t$(CC) $(FLAGS) $(LIBS) *.o -o $@\n" + \
                "\nclean:\n\trm -rf circuit_*.cu *.o dm_sim\n\n"
    make_file.write(s)

# For shallow circuit 
else:
    s = "__device__ __inline__ void circuit(double* dm_real, double* dm_imag)\n{\n"
    circuit_file.write(s)
    
    #generate random unitary gates or C gates
    if args.random is True:
        print_random_unitary_gates(N_GATES, circuit_file)
    #ganerate gates sampled from defined gate collection
    else:
        print_gates(N_GATES, circuit_file)
    circuit_file.write("}\n\n")

    s = "void deep_simulation(double* dm_real, double* dm_imag, dim3 gridDim){}\n" 
    circuit_file.write(s)
    #write to makefile
    s = ""
    if args.sim == 'omp':
        s = "\n" + SIM +": " + SIM + ".cu gate.cuh configuration.h circuit.cuh util.cuh\n"\
                + "\t$(CC) $(FLAGS) $(LIBS) -Xcompiler -fopenmp " + SIM + ".cu -o $@\n\n"
    elif args.sim == 'mpi':
        s = "\n" + SIM +": " + SIM + ".cu gate.cuh configuration.h circuit.cuh util.cuh\n"\
                + "\t$(CC) $(FLAGS) $(LIBS) -ccbin mpicc " + SIM + ".cu -o $@\n\n"
    else:
        s = "\n" + SIM +": " + SIM + ".cu gate.cuh configuration.h circuit.cuh util.cuh\n"\
                + "\t$(CC) $(FLAGS) $(LIBS) " + SIM + ".cu -o $@\n\n"
    make_file.write(s)
    s = "\nclean:\n\trm -rf *.o " + SIM + "\n"
    make_file.write(s)

#Finalize
circuit_file.write("\n#endif\n")
make_file.close()
circuit_file.close()

