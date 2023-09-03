#include <stdio.h>
#include <iostream>

#include "macros2.h"

__global__ void kernel(){
    printf("Hello from GPU\n");
}

void entry(){
    kernel<<<1,1>>>();
    put();
    cudaDeviceSynchronize();
}

int main(){
    entry();
}