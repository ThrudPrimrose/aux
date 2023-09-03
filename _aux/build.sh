syclcc --opensycl-targets=omp sycl.cpp -o sycl 
nvcc -x cu cuda.cpp -o cuda
g++ hwloc.cpp -o hwloc  $(pkg-config --cflags hwloc) $(pkg-config --libs hwloc)
nvcc -x cu cuda2.cpp -o cuda2