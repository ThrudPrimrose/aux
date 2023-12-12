__global__ void vectorAdd(float* a, float* b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  b[i] += a[i];
}

void vectorAddLauncher(float* d_a, float* d_b){
  int blockSize = 256;
  int numBlocks = 1;
  vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b);
  cudaDeviceSynchronize();
}
