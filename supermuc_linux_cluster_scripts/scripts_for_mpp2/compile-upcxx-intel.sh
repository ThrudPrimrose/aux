#!/usr/bin/env bash

echo download upc++...
FILE=upcxx-2021.3.0.tar.gz
if [ -f "$FILE" ]
then
    echo "$FILE exists"
else
    wget https://bitbucket.org/berkeleylab/upcxx/downloads/upcxx-2021.9.0.tar.gz
fi
tar xvf upcxx-2021.3.0.tar.gz
rm -rf ~/upcxx-intel-mpp2
mkdir ~/upcxx-intel-mpp2
cd upcxx-2021.3.0/

echo load modules...
module unload cmake
module unload intel-mpi
module unload intel
module load cmake/3.16.5
module load boost/1.75.0-intel19
module load intel/20.0
module load intel-mpi/2019-intel
module load netcdf-hdf5-all 
module load metis/5.1.0-intel19-i64-r64 


echo start building upc++...
export UPATH=~/upcxx-intel-mpp2
./configure --prefix=$UPATH --with-cc=mpiicc --with-cxx=mpiicpc --with-mpi-cc=mpiicc --with-mpi-cxx=mpiicpc \
--with-default-network=ibv --enable-ibv
make -j8 all
make install
