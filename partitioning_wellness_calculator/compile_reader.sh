#!/bin/bash

#compile with
h5c++ reader.cpp

#export the settings for omp
export OMP_NUM_THREAD=4
export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_PROC_BIND=close

#read with
./a.out output/tpv5_cell.h5