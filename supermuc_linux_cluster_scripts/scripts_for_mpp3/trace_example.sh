#!/bin/bash
#SBATCH -J ttt
#SBATCH -o ./%x.%j.out
#SBATCH -e ./%x.%j.err
#Initial working directory (also --chdir):
#SBATCH -D ./
#Notification and type
#SBATCH --mail-type=end,fail,timeout
#SBATCH --mail-user=yakup.paradox@gmail.com
# Wall clock limit:
#SBATCH --time=00:32:00
#SBATCH --no-requeue
#Setup of execution environment
#SBATCH --export=NONE
#SBATCH --get-user-env
#SBATCH --clusters=mpp3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64

module load slurm_setup
module unload intel-mpi intel-mkl intel
module load parallel-studio
module load netcdf
module load metis

export PATH=$PATH:/dss/dsshome1/lxc05/ge69xij2/upcxx-intel-parallelstudio/bin
export GASNET_PHYSMEM_MAX='41 GB'
export GASNET_MAX_SEGSIZE="512MB/P"
export UPCXX_INSTALL="/dss/dsshome1/lxc05/ge69xij2/upcxx-intel-parallelstudio"
export UPCXX_SHARED_HEAP_SIZE="512 MB"
export LD_PRELOAD=$VT_ROOT/intel64/slib/libVT.so
export VT_FLUSH_PREFIX=${SCRATCH}/trace/tmp
export VT_LOGFILE_PREFIX=${SCRATCH}/trace

#when compiled with intel-mpi
/lrz/sys/intel/impi2019u6/impi/2019.6.154/intel64/bin/mpiexec \
-np 64 -ppn 64 \
/dss/dsshome1/lxc05/ge69xij2/actor-upcxx/./pond \
-x 8000 -y 8000 -p 500 -c 10 --scenario 3 \
-o /gpfs/scratch/pr63so/ge69xij2/out/NetCdf -e 0.01