#include <cuda_runtime.h>
#include <stdio.h>

__global__ void cuda_hello2(){
    printf("Hello World from GPU!\n");
}

void cuda_hello_launcher2(){
    cuda_hello2<<<1,1>>>(); 
    cudaDeviceSynchronize();
}

