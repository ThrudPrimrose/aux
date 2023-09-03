#include <stdio.h>
#include <cuda_runtime.h>

int main()
{
    int deviceID = 0; // Use 0 if you have a single GPU
    cudaSetDevice(deviceID);

    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceID);

    int coresPerSM;
    cudaDeviceGetAttribute(&coresPerSM, cudaDevAttrCudaCoresPerMultiprocessor, deviceID);

    int totalCudaCores = numSMs * coresPerSM;

    printf("Total CUDA Cores: %d\n", totalCudaCores);

    return 0;
}