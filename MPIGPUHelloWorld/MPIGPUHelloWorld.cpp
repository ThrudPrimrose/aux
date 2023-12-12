
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <mpi.h>
#include <stdexcept>
#include <vector>
#include <iostream>

#define CUDA_CHECK_ERROR()                                                     \
  {                                                                            \
    cudaError_t err = cudaGetLastError();                                      \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line "    \
                << __LINE__ << std::endl;                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

extern void vectorAddLauncher(float *d_a, float *d_b);

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  printf("Hello from process (global) %d of %d\n", world_rank, world_size);

  // Determine the type of split (in this case, by node)
  int split_type = MPI_COMM_TYPE_SHARED;

  // Split the communicator based per ComputeNode
  MPI_Comm node_local;
  MPI_Comm_split_type(MPI_COMM_WORLD, split_type, world_rank, MPI_INFO_NULL,
                      &node_local);

  // Get the rank and size of the new communicator
  int node_local_size, node_local_rank;
  MPI_Comm_size(node_local, &node_local_size);
  MPI_Comm_rank(node_local, &node_local_rank);

  // Print a message from each process in the new communicator
  printf("Global Rank %d, Global Size %d | Node-Local Rank %d, Node-Local Size "
         "%d\n",
         world_rank, world_size, node_local_rank, node_local_size);

  cudaError_t cuda_status = cudaSetDevice(0);

  if (cuda_status != cudaSuccess) {
    throw std::runtime_error(
        "cudaSetDevice failed! CUDA cannot be initialized.");
  }

  // Get the number of available GPUs
  int device_count = 0;
  cuda_status = cudaGetDeviceCount(&device_count);

  if (device_count == 0 || cuda_status != cudaSuccess) {
    throw std::runtime_error("No CUDA-compatible GPU device found.");
  }

  if (node_local_size != device_count) {
    throw std::runtime_error(
        "Number of Ranks in Node is not equal to the number of GPU Devices.");
  }

  cudaSetDevice(node_local_rank);

  constexpr size_t vector_size = 256;
  std::vector<float> h_a_v(vector_size, 1.0f);
  std::vector<float> h_b_v(vector_size, 0.0f);
  std::vector<float> reduced_h_b_v(vector_size, 0.0f);
  float *h_a = h_a_v.data();
  float *h_b = h_b_v.data();
  float *reduced_h_b = reduced_h_b_v.data();

  // Allocate memory for vectors on GPU
  float *d_a, *d_b, *d_result;
  cudaMalloc((void **)&d_a, vector_size * sizeof(float));
  cudaMalloc((void **)&d_b, vector_size * sizeof(float));

  // Copy vectors from CPU to GPU
  cudaMemcpy(d_a, h_a, vector_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, vector_size * sizeof(float), cudaMemcpyHostToDevice);
  CUDA_CHECK_ERROR();

  // Launch CUDA kernel
  vectorAddLauncher(d_a, d_b);
  CUDA_CHECK_ERROR();

  // Copy result from GPU to CPU
  cudaMemcpy(h_b, d_b, vector_size * sizeof(float), cudaMemcpyDeviceToHost);
  CUDA_CHECK_ERROR();

  // Sum results across MPI ranks
  MPI_Reduce(reduced_h_b, h_b, vector_size, MPI_FLOAT, MPI_SUM, 0,
             MPI_COMM_WORLD);

  if (world_rank == 0) {
    bool incorrect = false;
    for (int i = 0; i < vector_size; i++) {
      if (std::abs(reduced_h_b[i] - static_cast<float>(world_size)) > 0.1) {
        printf("Wrong reduction result at %d: is %.4f should be %.4f.\n", i,
               reduced_h_b[i], static_cast<float>(world_size));
        printf("Unreduced results at %d: is %.4f should be %.4f.\n", i, h_b[i],
               h_a[i]);
        incorrect = true;
        break;
      }
    }
    if (incorrect) {
      throw std::runtime_error("Incorrect results after Reduction");
    } else {
      printf("Results correct after reduction.\n");
    }
  }

  cudaFree(d_a);
  cudaFree(d_b);

  MPI_Finalize();

  return 0;
}