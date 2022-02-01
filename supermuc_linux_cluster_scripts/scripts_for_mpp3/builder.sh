#!/bin/bash
module load cmake/3.16.5

make clean
rm CMakeCache.txt
rm -r CMakeFiles
rm cmake_install.cmake
rm Makefile

#JOBTYPES is readas an environment variable, the user has to make sure that the needed joytypes are compiled!!!!

#no loop for joytypes here because every job type will have a different set of arguments so, copy by yourself

#no loop for joytypes here because every job type will have a different set of arguments so, copy by yourself
cmake . -DCMAKE_C_COMPILER=mpiicc -DCMAKE_CXX_COMPILER=mpiicpc -DCMAKE_PREFIX_PATH=${UPCXX_INSTALL} -DENABLE_FILE_OUTPUT=OFF -DBUILD_RELEASE=OFF \
-DENABLE_LOGGING=OFF -DENABLE_O3_UPCXX_BACKEND=OFF -DENABLE_PARALLEL_UPCXX_BACKEND=OFF -DENABLE_MEMORY_SANITATION=OFF -DIS_CROSS_COMPILING=OFF \
-DINVASION=OFF -DTIME=OFF -DANALYZE=OFF -DTRACE=OFF -DMIGRATION=1 -DREPORT_MAIN_ACTIONS=ON \
-DGLOBAL_MIGRATION=OFF -DLAZY_ACTIVATION=OFF

make actorlib -j 16
make pond -j 16
#pond-* should be same as pond-${jobtypes[@]} that is an environment variable
mv pond pond-interrupt-1

#no loop for joytypes here because every job type will have a different set of arguments so, copy by yourself
cmake . -DCMAKE_C_COMPILER=mpiicc -DCMAKE_CXX_COMPILER=mpiicpc -DCMAKE_PREFIX_PATH=${UPCXX_INSTALL} -DENABLE_FILE_OUTPUT=OFF -DBUILD_RELEASE=OFF \
-DENABLE_LOGGING=OFF -DENABLE_O3_UPCXX_BACKEND=OFF -DENABLE_PARALLEL_UPCXX_BACKEND=OFF -DENABLE_MEMORY_SANITATION=OFF -DIS_CROSS_COMPILING=OFF \
-DINVASION=OFF -DTIME=OFF -DANALYZE=OFF -DTRACE=OFF -DMIGRATION=0 -DREPORT_MAIN_ACTIONS=ON \
-DGLOBAL_MIGRATION=OFF -DLAZY_ACTIVATION=OFF

make actorlib -j 16
make pond -j 16
#pond-* should be same as pond-${jobtypes[@]} that is an environment variable
mv pond pond-interrupt-0

