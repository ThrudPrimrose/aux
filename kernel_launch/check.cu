#include <stdio.h>

__global__ void check()
{
    printf("Good\n");
}


void entry()
{
    printf("Entry\n");
    check<<<1,1>>>();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
}