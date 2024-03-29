#!/bin/bash

#script for building and copying right executables
#~/upcxx-intel-mpp2

module load cmake
module unload intel-mpi/2019-intel
module unload intel-oneapi-compilers/2021.4.0
module load intel-oneapi-compilers/2022.0.1
module load intel-oneapi-mpi/2021-intel
module load metis

mkdir build
cd build 

#JOBTYPES is readas an environment variable, the user has to make sure that the needed joytypes are compiled!!!!
#no loop for joytypes here because every job type will have a different set of arguments so, copy by yourself

cmake \
-DCMAKE_C_COMPILER=mpiicc \
-DCMAKE_CXX_COMPILER=mpiicpc \
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
-DENABLE_FILE_OUTPUT=OFF \
-DBUILD_RELEASE=ON \
-DENABLE_O3_UPCXX_BACKEND=ON \
-DENABLE_MEMORY_SANITATION=OFF \
-DIS_CROSS_COMPILING=OFF \
-DINVASION=OFF \
-DTIME=OFF \
-DTRACE=OFF \
-DMIGRATION=0 \
-DREPORT_MAIN_ACTIONS=OFF \
-DTHREAD_SANITIZER=OFF \
-DLAZY_ACTIVATION=ON \
-DANALYZE=OFF \
-DINTERRUPT=OFF \
-DSTEAL_ONLY_ACTABLE_ACTOR=OFF \
-DSTEAL_FROM_BUSY_RANK=OFF \
-DGLOBAL_MIGRATION=OFF \
-DEXTRA_MEASURES= OFF \
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
..

make -j pond
mv pond ../pond-static

rm -r *

cmake \
-DCMAKE_C_COMPILER=mpiicc \
-DCMAKE_CXX_COMPILER=mpiicpc \
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
-DENABLE_FILE_OUTPUT=OFF \
-DBUILD_RELEASE=ON \
-DENABLE_O3_UPCXX_BACKEND=ON \
-DENABLE_MEMORY_SANITATION=OFF \
-DIS_CROSS_COMPILING=OFF \
-DINVASION=OFF \
-DTIME=OFF \
-DTRACE=OFF \
-DMIGRATION=2 \
-DREPORT_MAIN_ACTIONS=OFF \
-DTHREAD_SANITIZER=OFF \
-DLAZY_ACTIVATION=ON \
-DANALYZE=OFF \
-DINTERRUPT=OFF \
-DSTEAL_ONLY_ACTABLE_ACTOR=OFF \
-DSTEAL_FROM_BUSY_RANK=OFF \
-DGLOBAL_MIGRATION=OFF \
-DEXTRA_MEASURES= OFF \
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
..

make -j pond
mv pond ../pond-local-random

rm -r *
cmake \
-DCMAKE_C_COMPILER=mpiicc \
-DCMAKE_CXX_COMPILER=mpiicpc \
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
-DENABLE_FILE_OUTPUT=OFF \
-DBUILD_RELEASE=ON \
-DENABLE_O3_UPCXX_BACKEND=ON \
-DENABLE_MEMORY_SANITATION=OFF \
-DIS_CROSS_COMPILING=OFF \
-DINVASION=OFF \
-DTIME=OFF \
-DTRACE=OFF \
-DMIGRATION=2 \
-DREPORT_MAIN_ACTIONS=OFF \
-DTHREAD_SANITIZER=OFF \
-DLAZY_ACTIVATION=ON \
-DANALYZE=OFF \
-DINTERRUPT=OFF \
-DSTEAL_ONLY_ACTABLE_ACTOR=OFF \
-DSTEAL_FROM_BUSY_RANK=ON \
-DGLOBAL_MIGRATION=OFF \
-DEXTRA_MEASURES= OFF \
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
..

make -j pond
mv pond ../pond-local-busy

rm -r *

cmake \
-DCMAKE_C_COMPILER=mpiicc \
-DCMAKE_CXX_COMPILER=mpiicpc \
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
-DENABLE_FILE_OUTPUT=OFF \
-DBUILD_RELEASE=ON \
-DENABLE_O3_UPCXX_BACKEND=ON \
-DENABLE_MEMORY_SANITATION=OFF \
-DIS_CROSS_COMPILING=OFF \
-DINVASION=OFF \
-DTIME=OFF \
-DTRACE=OFF \
-DMIGRATION=2 \
-DREPORT_MAIN_ACTIONS=OFF \
-DTHREAD_SANITIZER=OFF \
-DLAZY_ACTIVATION=ON \
-DANALYZE=OFF \
-DINTERRUPT=OFF \
-DSTEAL_ONLY_ACTABLE_ACTOR=OFF \
-DSTEAL_FROM_BUSY_RANK=OFF \
-DGLOBAL_MIGRATION=ON \
-DEXTRA_MEASURES= OFF \
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
..

make -j pond
mv pond ../pond-global-random

rm -r *
cmake \
-DCMAKE_C_COMPILER=mpiicc \
-DCMAKE_CXX_COMPILER=mpiicpc \
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
-DENABLE_FILE_OUTPUT=OFF \
-DBUILD_RELEASE=ON \
-DENABLE_O3_UPCXX_BACKEND=ON \
-DENABLE_MEMORY_SANITATION=OFF \
-DIS_CROSS_COMPILING=OFF \
-DINVASION=OFF \
-DTIME=OFF \
-DTRACE=OFF \
-DMIGRATION=2 \
-DREPORT_MAIN_ACTIONS=OFF \
-DTHREAD_SANITIZER=OFF \
-DLAZY_ACTIVATION=ON \
-DANALYZE=OFF \
-DINTERRUPT=OFF \
-DSTEAL_ONLY_ACTABLE_ACTOR=OFF \
-DSTEAL_FROM_BUSY_RANK=ON \
-DGLOBAL_MIGRATION=ON \
-DEXTRA_MEASURES= OFF \
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
..

make -j pond
mv pond ../pond-global-busy

rm -r *
