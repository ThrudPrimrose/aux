__global__ void vectorAdd(float* a, float* b, int vector_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < vector_size) {
    b[i] += a[i];
  }
}

void vectorAddLauncher(float* d_a, float* d_b, int vector_size){
  int blockSize = 256;
  int numBlocks = (vector_size + blockSize - 1) / blockSize;
  vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, vector_size);
}
