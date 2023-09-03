#include <cuda_runtime.h>
#include <stdio.h>

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

void cuda_hello_launcher(){
    cuda_hello<<<1,1>>>(); 
    cudaDeviceSynchronize();
}

