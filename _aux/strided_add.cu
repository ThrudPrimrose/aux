#include <stdio.h>

__global__ void vectorAdd(int* a, int* b, int* result, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        result[tid] = a[tid] + b[tid];
    }
}

__global__ void stridedVectorAdd(int* a, int* b, int* result, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size / 2) {
        result[tid*2] = a[tid*2] + b[tid*2];
    }
}

int main() {
    int size = 100000000;
    int* a, * b, * result;
    int* d_a, * d_b, * d_result;

    // Allocate memory on host
    a = (int*)malloc(size * sizeof(int));
    b = (int*)malloc(size * sizeof(int));
    result = (int*)malloc(size * sizeof(int));

    // Initialize vectors to zero
    memset(a, 0, size * sizeof(int));
    memset(b, 0, size * sizeof(int));

    // Allocate memory on device
    cudaMalloc((void**)&d_a, size * sizeof(int));
    cudaMalloc((void**)&d_b, size * sizeof(int));
    cudaMalloc((void**)&d_result, size * sizeof(int));

    // Copy vectors from host to device
    cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    // Create CUDA events for start and end
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Record the start event
    cudaEventRecord(start);
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, size);
    // Record the end event
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);

    // Print the elapsed time
    printf("Elapsed time for unit-stride add: %.5f ms\n", milliseconds);

    // Copy result from device to host
    cudaMemcpy(result, d_result, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    cudaMalloc((void**)&d_result, size * sizeof(int));
    
    // Create CUDA events for start and end
    cudaEvent_t start2, end2;
    cudaEventCreate(&start2);
    cudaEventCreate(&end2);

    // Record the start event
    cudaEventRecord(start2);
    // Launch kernel
    threadsPerBlock = 256;
    blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    stridedVectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, size);
    
    // Record the end event
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    // Calculate the elapsed time
    float milliseconds2 = 0;
    cudaEventElapsedTime(&milliseconds2, start2, end2);

    // Copy result from device to host
    cudaMemcpy(result, d_result, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the elapsed time
    printf("Elapsed time for stridedAdd: %.5f ms\n", milliseconds);

    // Free memory on host and device
    free(a);
    free(b);
    free(result);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    return 0;
}
