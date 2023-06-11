#if defined(CUDA)
#include <sstream>
#include <iostream>
#include <cuda.h>

namespace cuda
{
   std::string getDeviceInfo()
   {
      std::stringstream ss;
      int numberOfDevices;
      cudaGetDeviceCount(&numberOfDevices);
      // TODO: TABLE OUTPUT
      ss << "Using CUDA GPU backend, with " << numberOfDevices << " CUDA capable devices:\n";
      for (unsigned int i = 0; i < numberOfDevices; i++)
      {
         cudaDeviceProp props;
         cudaGetDeviceProperties(&props, i);
         size_t free, total;
         cudaMemGetInfo(&free, &total);
         constexpr unsigned int kb = 1024;
         constexpr unsigned int mb = kb * kb;
         ss << "Device-" << i << ": " << props.name << "\n";
         ss << "  Compute architecture:                       " << props.major << props.minor << "\n";
         ss << "  Global memory:                              " << props.totalGlobalMem / mb << " mb"
            << "\n";
         ss << "  Shared memory:                              " << props.sharedMemPerBlock / kb << " kb"
            << "\n";
         ss << "  Constant memory:                            " << props.totalConstMem / kb << " kb"
            << "\n";
         ss << "  Free memory:                                " << free << " b"
            << " (" << free / mb << " mb)"
            << "\n";
         ss << "  Clock frequency:                            " << props.clockRate / 1000 << " mHz\n";
         ss << "  Compute units (Streaming Multiprocessors):  " << props.multiProcessorCount << "\n";
         ss << "  Warp size:                                  " << props.warpSize << "\n";
         ss << "  Block size:                                 "
            << "<" << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1] << ", " << props.maxThreadsDim[2] << ">"
            << "\n";
         ss << "  Threads per block:                          " << props.maxThreadsPerBlock << "\n";
         ss << "  Grid size:                                  "
            << "<" << props.maxGridSize[0] << ", " << props.maxGridSize[1] << ", " << props.maxGridSize[2] << ">"
            << "\n";
      }
      ss << std::endl;
      return ss.str();
   }
};

int main()
{
   std::stringstream ss;
   int numberOfDevices;
   cudaGetDeviceCount(&numberOfDevices);
   // TODO: TABLE OUTPUT
   ss << "Using CUDA GPU backend, with " << numberOfDevices << " CUDA capable devices:\n";
   for (unsigned int i = 0; i < numberOfDevices; i++)
   {
      cudaDeviceProp props;
      cudaGetDeviceProperties(&props, i);
      size_t free, total;
      cudaMemGetInfo(&free, &total);
      constexpr unsigned int kb = 1024;
      constexpr unsigned int mb = kb * kb;
      ss << "Device-" << i << ": " << props.name << "\n";
      ss << "  Compute architecture:                       " << props.major << props.minor << "\n";
      ss << "  Global memory:                              " << props.totalGlobalMem / mb << " mb"
         << "\n";
      ss << "  Shared memory:                              " << props.sharedMemPerBlock / kb << " kb"
         << "\n";
      ss << "  Constant memory:                            " << props.totalConstMem / kb << " kb"
         << "\n";
      ss << "  Free memory:                                " << free << " b"
         << " (" << free / mb << " mb)"
         << "\n";
      ss << "  Clock frequency:                            " << props.clockRate / 1000 << " mHz\n";
      ss << "  Compute units (Streaming Multiprocessors):  " << props.multiProcessorCount << "\n";
      ss << "  Warp size:                                  " << props.warpSize << "\n";
      ss << "  Block size:                                 "
         << "<" << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1] << ", " << props.maxThreadsDim[2] << ">"
         << "\n";
      ss << "  Threads per block:                          " << props.maxThreadsPerBlock << "\n";
      ss << "  Grid size:                                  "
         << "<" << props.maxGridSize[0] << ", " << props.maxGridSize[1] << ", " << props.maxGridSize[2] << ">"
         << "\n";
   }
   ss << std::endl;
   std::cout << ss.str();

   /*
  for (unsigned int i = 1; i <= 1e9; i *= 1e1)
  {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    constexpr size_t N = 1;
    const size_t bytes = N * sizeof(int) * i;
    constexpr unsigned int byte_to_gb = 1e9;
    std::cout << "Allocate and copy: " << float(bytes) / float(byte_to_gb) << std::endl;
    int *h_a = (int *)malloc(bytes);
    int *d_a;
    cudaMalloc((int **)&d_a, bytes);
    memset(h_a, 0, bytes);
    cudaEventRecord(start);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(h_a, d_a, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    cudaFree(d_a);

    float in_gb = float(bytes) / float(byte_to_gb);
    float s = milliseconds / 1000;
    float gb_per_s = in_gb / s;
    float gb_per_s_2 = bytes / milliseconds / 1e6;
    std::cout << gb_per_s_2 << std::endl;
    free(h_a);
  }
  */

   return 0;
}

#endif