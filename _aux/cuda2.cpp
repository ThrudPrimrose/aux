#include <iostream>
#include <cuda_runtime.h>

void printDeviceProperties(int deviceID)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceID);

    std::cout << "Device Name: " << deviceProp.name << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "Memory Clock Rate (KHz): " << deviceProp.memoryClockRate << std::endl;
    std::cout << "Memory Bus Width (bits): " << deviceProp.memoryBusWidth << std::endl;
    std::cout << "Peak Memory Bandwidth (GB/s): " << 2.0 * deviceProp.memoryClockRate * (deviceProp.memoryBusWidth / 8) / 1.0e6 << std::endl;
    std::cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "Multiprocessor Count: " << deviceProp.multiProcessorCount << std::endl;
    std::cout << "Max Threads Per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "Max Threads Per Multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Max Threads Dim: (" << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "Max Grid Size: (" << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << ")" << std::endl;
    std::cout << "Shared Memory Per Block (bytes): " << deviceProp.sharedMemPerBlock << std::endl;
    std::cout << "Total Global Memory (bytes): " << deviceProp.totalGlobalMem << std::endl;
    std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
    std::cout << "Memory Pitch (bytes): " << deviceProp.memPitch << std::endl;
    std::cout << "Max Threads Per Block (Dimension 0): " << deviceProp.maxThreadsDim[0] << std::endl;
    std::cout << "Max Threads Per Block (Dimension 1): " << deviceProp.maxThreadsDim[1] << std::endl;
    std::cout << "Max Threads Per Block (Dimension 2): " << deviceProp.maxThreadsDim[2] << std::endl;
    std::cout << "Total Constant Memory (bytes): " << deviceProp.totalConstMem << std::endl;
    std::cout << "Max Registers Per Block: " << deviceProp.regsPerBlock << std::endl;
    std::cout << "Clock Rate (KHz): " << deviceProp.clockRate << std::endl;
    std::cout << "Texture Alignment: " << deviceProp.textureAlignment << std::endl;
    std::cout << "Device Overlap: " << deviceProp.deviceOverlap << std::endl;
    std::cout << "Kernel Execution Timeout: " << deviceProp.kernelExecTimeoutEnabled << std::endl;
    std::cout << "Integrated GPU: " << deviceProp.integrated << std::endl;
    std::cout << "Can Map Host Memory: " << deviceProp.canMapHostMemory << std::endl;
    std::cout << "Concurrent Kernels: " << deviceProp.concurrentKernels << std::endl;
    std::cout << "ECC Enabled: " << deviceProp.ECCEnabled << std::endl;
    std::cout << "PCI Bus ID: " << deviceProp.pciBusID << std::endl;
    std::cout << "PCI Device ID: " << deviceProp.pciDeviceID << std::endl;
    std::cout << "TCC Driver: " << deviceProp.tccDriver << std::endl;
    std::cout << "Async Engine Count: " << deviceProp.asyncEngineCount << std::endl;
    std::cout << "Unified Addressing: " << deviceProp.unifiedAddressing << std::endl;
    std::cout << "Memory Clock Rate (KHz): " << deviceProp.memoryClockRate << std::endl;
    std::cout << "Memory Bus Width (bits): " << deviceProp.memoryBusWidth << std::endl;
    std::cout << "L2 Cache Size: " << deviceProp.l2CacheSize << std::endl;
    std::cout << "Max Threads Per MultiProcessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Stream Priorities Supported: " << deviceProp.streamPrioritiesSupported << std::endl;
    std::cout << "Global L1 Cache Supported: " << deviceProp.globalL1CacheSupported << std::endl;
    std::cout << "Local L1 Cache Supported: " << deviceProp.localL1CacheSupported << std::endl;
    std::cout << "Shared Memory Per Multiprocessor (bytes): " << deviceProp.sharedMemPerMultiprocessor << std::endl;
    std::cout << "Registers Per Multiprocessor: " << deviceProp.regsPerMultiprocessor << std::endl;
    std::cout << "Managed Memory: " << deviceProp.managedMemory << std::endl;
    std::cout << "Is Multi-GPU Board: " << deviceProp.isMultiGpuBoard << std::endl;
    std::cout << "Multi-GPU Board Group ID: " << deviceProp.multiGpuBoardGroupID << std::endl;
    std::cout << "Host Native Atomic Supported: " << deviceProp.hostNativeAtomicSupported << std::endl;
    std::cout << "Single-Double Precision Perf Ratio: " << deviceProp.singleToDoublePrecisionPerfRatio << std::endl;
    std::cout << "Pageable Memory Access: " << deviceProp.pageableMemoryAccess << std::endl;
    std::cout << "Concurrent Managed Access: " << deviceProp.concurrentManagedAccess << std::endl;
    std::cout << "Compute Preemption Supported: " << deviceProp.computePreemptionSupported << std::endl;
    std::cout << "Can Use Host Pointer For Registered Memory: " << deviceProp.canUseHostPointerForRegisteredMem << std::endl;
    std::cout << "Cooperative Launch: " << deviceProp.cooperativeLaunch << std::endl;
    std::cout << "Cooperative Multi-Device Launch: " << deviceProp.cooperativeMultiDeviceLaunch << std::endl;
    std::cout << "Shared Memory Per Block Optin: " << deviceProp.sharedMemPerBlockOptin << std::endl;
    std::cout << "Pageable Memory Access Uses Host Page Tables: " << deviceProp.pageableMemoryAccessUsesHostPageTables << std::endl;
    std::cout << "Direct Managed Memory Access From Host: " << deviceProp.directManagedMemAccessFromHost << std::endl;
    std::cout << "Max Blocks Per MultiProcessor: " << deviceProp.maxBlocksPerMultiProcessor << std::endl;
}

int main()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        std::cout << "No CUDA devices found" << std::endl;
        return 0;
    }

    std::cout << "Number of CUDA devices: " << deviceCount << std::endl;
    std::cout << "----------------------------------------------" << std::endl;

    for (int i = 0; i < deviceCount; ++i)
    {
        std::cout << "CUDA Device #" << i << std::endl;
        std::cout << "----------------------------------------------" << std::endl;
        printDeviceProperties(i);
        std::cout << std::endl;
    }

    return 0;
}