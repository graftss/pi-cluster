#include <stdio.h>

const int N = 100000;
const int MASK_SIZE = 5;
const int TRIALS = 100000;

const int THREADS_PER_BLOCK = 1000;
const int NUM_BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

__global__ void convolution_kernel(int *in, int *out, int mask_width, int in_width, int* dev_mask) {
  for (int trial = 0; trial < TRIALS; trial++) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int radius = mask_width / 2;

    int result = 0;
    for (int i = 0, j = tid - radius; i < mask_width && j < in_width; i++, j++) {
      if (j >= 0) {
        result += dev_mask[i] * in[j];
      }
    }

    out[tid] = result;
  }
}

int main() {
  float elapsed_ms;

  int in[N], out[N], *dev_in, *dev_out, *mask, *dev_mask;

  for (int i = 0; i < N; i++) in[i] = i;
  memset(out, 0, N * sizeof(int));
  mask = (int*)malloc(MASK_SIZE * sizeof(int));
  for (int i = 0; i < MASK_SIZE; i++) mask[i] = i + 1;

  cudaMalloc((void**)&dev_in, N * sizeof(int));
  cudaMemcpy(dev_in, in, N * sizeof(int), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&dev_out, N * sizeof(int));
  cudaMemcpy(dev_out, out, N * sizeof(int), cudaMemcpyHostToDevice);

  cudaMalloc((void**)&dev_mask, MASK_SIZE * sizeof(int));
  cudaMemcpy(dev_mask, mask, MASK_SIZE * sizeof(int), cudaMemcpyHostToDevice);
  free(mask);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  convolution_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
    dev_in, dev_out,
    MASK_SIZE, N,
    dev_mask
  );

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_ms, start, stop);

  cudaMemcpy(out, dev_out, N * sizeof(int), cudaMemcpyDeviceToHost);

  printf("Input array size: %d\n", N);
  printf("Convolution duration: %3.3fms (%d trials)\n", elapsed_ms, TRIALS);

  return 0;
}