syclcc -DSYCL --opensycl-targets=omp sycl.cpp -o sycl 
nvcc -DCUDA -x cu cuda.cpp -o cuda
#g++ hwloc.cpp -o hwloc  $(pkg-config --cflags hwloc) $(pkg-config --libs hwloc)
#nvcc -x cu cuda2.cpp -o cuda2