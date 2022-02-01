#!/bin/bash
#SBATCH -J debug-test
#SBATCH -o ${SCRATCH}/testdir/debug-test/%x.%j.out
#SBATCH -e ${SCRATCH}/testdir/debug-test/%x.%j.err
#SBATCH -D ./
#Notification and type
#SBATCH --mail-type=end,fail,timeout
#SBATCH --mail-user=yakup.paradox@gmail.com
# Wall clock limit:
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#Setup of execution environment
#SBATCH --export=NONE
#SBATCH --get-user-env
#SBATCH --clusters=cm2_tiny
#SBATCH --partition=cm2_tiny
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=28

module load slurm_setup

module load netcdf-hdf5-all/4.7_hdf5-1.10-intel19-serial
module load metis/5.1.0-intel19-i64-r64
module load ddt
module load valgrind

export UPCXX_INSTALL=~/upcxx-intel-mpp2
export PATH=$PATH:~/upcxx-intel-mpp2/bin
export GASNET_PHYSMEM_MAX='40 GB'
export GASNET_BACKTRACE=1

export GASNET_MAX_SEGSIZE="1000MB/P"
export UPCXX_INSTALL=/dss/dsshome1/lxc05/ge69xij2/upcxx-intel-mpp2
export UPCXX_SHARED_HEAP_SIZE="1000 MB"

export GASNET_PSHM_NODES=28
export REPARTITIONING_INTERVAL=50.0

 /dss/dsshome1/lrz/sys/spack/release/21.1.1/opt/x86_64/intel-mpi/2019.10.317-gcc-6ool5so/compilers_and_libraries_2020.4.317/linux/mpi/intel64/bin/mpirun \
 -np 28 -ppn 28 \
 /dss/dsshome1/lrz/sys/tools/ddt/21.0.2/bin/forge-client --ddtsessionfile /dss/dsshome1/lxc05/ge69xij2/.allinea/session/cm2login3-1 \
 /dss/dsshome1/lxc05/ge69xij2/ActorHPC/./pond-static-debug-1 \
 -x 5000 -y 5000 -p 250 -c 10 --scenario 3 -o /gpfs/scratch/pr63so/ge69xij2/work/out/f -e 0.1

/dss/dsshome1/lrz/sys/tools/ddt/21.0.2/bin/forge-client upcxx-run -n 28 -shared-heap 2048MB ./pond-debug-static-1 -x 8000 -y 8000 -p 250 -c 10 --scenario 3 -o /gpfs/scratch/pr63so/ge69xij2/work/out/f -e 5

/dss/dsshome1/lxc05/ge69xij2/.allinea/session/cm2login3-1
/dss/dsshome1/lrz/sys/tools/ddt/21.0.2/bin/forge-client

 -np 28 -ppn 28 /dss/dsshome1/lrz/sys/tools/ddt/21.0.2/bin/forge-client /dss/dsshome1/lxc05/ge69xij2/ActorHPC/./pond-debug-static-1 -x 8000 -y 8000 -p 250 -c 10 --scenario 3 -o /gpfs/scratch/pr63so/ge69xij2/work/out/f -e 0.5

upcxx-run -show -N 1 -n 28 ./pond-static-debug-1 -x 8000 -y 8000 -p 250 -c 10 --scenario 3 -o /gpfs/scratch/pr63so/ge69xij2/work/out/f -e 0.5

export GASNET_PSHM_NODES=28
export REPARTITIONING_INTERVAL=60.0

/dss/dsshome1/lrz/sys/spack/release/21.1.1/opt/x86_64/intel-mpi/2019.10.317-gcc-6ool5so/compilers_and_libraries_2020.4.317/linux/mpi/intel64/bin/mpirun -np 28 -ppn 28 \
valgrind --leak-check=full \
         --show-leak-kinds=all \
         --track-origins=yes \
         --show-reachable=yes --num-callers=50 --track-fds=yes \
            /dss/dsshome1/lxc05/ge69xij2/ActorHPC/./pond-static-debug-1 -x 4000 -y 4000 -p 250 -c 10 --scenario 2 -o /tmp/o -e 0.001


/dss/dsshome1/lrz/sys/tools/ddt/21.0.2/bin/forge-client --ddtsessionfile /dss/dsshome1/lxc05/ge69xij2/.allinea/session/cm2login3-1 \
./pond-static-debug-1 -x 8000 -y 8000 -p 250 -c 10 --scenario 3 \
-o /gpfs/scratch/pr63so/ge69xij2/work/out/f -e 0.1


export UPCXX_INSTALL=~/upcxx-intel-mpp2
export PATH=$PATH:~/upcxx-intel-mpp2/bin
export GASNET_PHYSMEM_MAX='40 GB'
export GASNET_BACKTRACE=1

export GASNET_MAX_SEGSIZE="256MB/P"
export UPCXX_INSTALL=/dss/dsshome1/lxc05/ge69xij2/upcxx-intel-mpp2
export UPCXX_SHARED_HEAP_SIZE="256 MB"

module unload intel-mpi intel intel-mkl
module load intel-parallel-studio

/dss/dsshome1/lrz/sys/spack/release/21.1.1/opt/x86_64/intel-mpi/2019.10.317-gcc-6ool5so/compilers_and_libraries_2020.4.317/linux/mpi/intel64/bin/mpirun -np 28 -ppn 28 /dss/dsshome1/lxc05/ge69xij2/ActorHPC/./pond-static-debug-1 -x 8000 -y 8000 -p 250 -c 10 --scenario 3 -o /gpfs/scratch/pr63so/ge69xij2/work/out/f -e 0.5

inspxe-cl -collect=mi3 \
/dss/dsshome1/lrz/sys/spack/release/21.1.1/opt/x86_64/intel-mpi/2019.10.317-gcc-6ool5so/compilers_and_libraries_2020.4.317/linux/mpi/intel64/bin/mpirun -np 28 -ppn 28 /dss/dsshome1/lxc05/ge69xij2/ActorHPC/./pond-static-debug-1 -x 8000 -y 8000 -p 250 -c 10 --scenario 3 -o /gpfs/scratch/pr63so/ge69xij2/work/out/f -e 0.1