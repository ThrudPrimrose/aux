#!/usr/bin/env bash

echo download upc++...
FILE=upcxx-2021.9.0.tar.gz
if [ -f "$FILE" ]
then
    echo "$FILE exists"
else
    wget https://bitbucket.org/berkeleylab/upcxx/downloads/upcxx-2021.9.0.tar.gz
fi
tar xvf upcxx-2021.9.0.tar.gz
mkdir ~/upcxx-intel-mpp3
cd upcxx-2021.9.0/

echo load modules...
module load gcc
module load netcdf
module load metis 
module load cmake


echo start building upc++...
export UPATH=~/upcxx-intel-mpp3
./configure --prefix=$UPATH --with-cc=mpiicc --with-cxx=mpiicpc --with-mpi-cc=mpiicc --with-mpi-cxx=mpiicpc \
--with-default-network=ibv
make -j16 all
make install


