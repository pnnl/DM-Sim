#!/bin/bash
#BSUB -P {PROJECT_ID}
#BSUB -W 1
#BSUB -nnodes 2
#BSUB -o out_cc.txt -e err_cc.txt

module load xl
module load cuda/10.1.105
module load spectrum-mpi
module load python/2.7.15-anaconda2-5.3.0

source activate /ccs/home/angli/.conda/envs/dmsim

export LD_LIBRARY_PATH=/autofs/nccs-svm1_sw/summit/cuda/10.1.105/lib64/:$LD_LIBRARY_PATH
date

## "--smpiargs=-gpu" is for enabling GPU-Direct RDMA
jsrun -n8 -a1 -g1 -c1 --smpiargs="-gpu" python -m mpi4py adder_n10_mpi.py 10 
jsrun -n8 -a1 -g1 -c1 --smpiargs="-gpu" ./adder_n10_mpi
