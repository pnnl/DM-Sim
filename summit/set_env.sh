## ---------------------------------------------------------------------------
## DM-Sim: Density-Matrix quantum circuit simulator based on GPU clusters
## Version 2.0
## ---------------------------------------------------------------------------
## File: set_env_summit.sh
## Environment settings for ORNL summit supercopmuter.
## ---------------------------------------------------------------------------
## Ang Li, Scientist, Pacific Northwest National Laboratory(PNNL), U.S.
## Homepage: http://www.angliphd.com
## GitHub repo: http://www.github.com/pnnl/DM-Sim
## PNNL-IPID: 31919-E, ECCN: EAR99, IR: PNNL-SA-143160
## BSD Lincese.
## ---------------------------------------------------------------------------

## ---------------------------------------------------------------------------
## You need to create a conda environment first, use
## $ module load python/2.7.15-anaconda2-5.3.0
## $ conda create -name dmsim python=2.7
## $ source activate dmsim
## Then you need to install pybind11
## $ pip install pybind11
## Then you need to install mpi4py, this requires gcc to build
## $ module load gcc
## $ pip install --no-binary mpi4py install mpi4py
## Then you can load the running environment, note the compile 
## of pybind11 requries CUDA-10.0 or newer
## $ module purge
## $ module load xl
## $ module load cuda/10.1.105
## $ module load spectrum-mpi
## $ module load python/2.7.15-anaconda2-5.3.0
## $ source activate dmsim
## $ make
## $ bsub summit_dmsim.lsf
## We provide a sample Summit lsf file for your reference. 
## Have fun on Summit!

module load xl
module load cuda/10.1.105
module load spectrum-mpi
module load python/2.7.15-anaconda2-5.3.0

