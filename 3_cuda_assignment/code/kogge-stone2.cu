#include <stdio.h>

const int TRIALS = 1;
const int N = 1024;
const int THREADS_PER_BLOCK = N;
const int NUM_BLOCKS = 1;

__global__ void kogge_stone_scan(int *in, int *out) {
  __shared__ int partials[THREADS_PER_BLOCK];
  __shared__ int partials_next[THREADS_PER_BLOCK];

  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  for (int trial = 0; trial < TRIALS; trial++) {
    if (tid < N) {
      partials[threadIdx.x] = in[tid];
    }

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
      __syncthreads();

      if (threadIdx.x >= stride) {
        partials_next[threadIdx.x] =
          partials[threadIdx.x] + partials[threadIdx.x - stride];
      } else {
        partials_next[threadIdx.x] = partials[threadIdx.x];
      }

      __syncthreads();

      partials[threadIdx.x] = partials_next[threadIdx.x];
    }
  }

  out[tid] = partials_next[threadIdx.x];
}

int main() {
  int *in, *out, *dev_in, *dev_out;
  float elapsed_ms;
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  in = (int*)malloc(N * sizeof(int));
  out = (int*)malloc(N * sizeof(int));

  for (int i = 0; i < N; i++) {
    in[i] = 1;
  }

  cudaMalloc((void**)&dev_in, N * sizeof(int));
  cudaMemcpy(dev_in, in, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&dev_out, N * sizeof(int));

  cudaEventRecord(start, 0);

  kogge_stone_scan<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(dev_in, dev_out);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&elapsed_ms, start, stop);

  cudaDeviceSynchronize();
  cudaMemcpy(out, dev_out, N * sizeof(int), cudaMemcpyDeviceToHost);

  // verify results by checking that `out[i] == i + 1` for all i.
  int result = 1;
  for (int i = 0; i < N; i += 1) {
    result = result && (out[i] == i + 1);
  }

  printf("Results valid: %s\n", result ? "yes" : "no");
  printf("Input array size: %d\n", N);
  printf("Elapsed time: %.3fms (%d trials)\n", elapsed_ms, TRIALS);
}
