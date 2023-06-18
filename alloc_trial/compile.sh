nvc++ -mp=gpu -stdpar=gpu -std=c++20 alloc_isocpp.cpp -o allocisocpp
nvcc alloc_cuda.cu -o alloccuda
nvc++ -mp=gpu -std=c++20 alloc_omp.cpp -o allocomp