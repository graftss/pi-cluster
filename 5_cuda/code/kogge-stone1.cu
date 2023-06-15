#include <stdio.h>

// Note: this implementation of the Kogge-Stone-inspired prefix sum
// has a race condition. It should start to produce incorrect results for
// large values of N. The exact value that incorrect results become possible
// is hardware dependent.

const int N = 32;
const int THREADS_PER_BLOCK = N;
const int NUM_BLOCKS = 1;

__global__ void kogge_stone_scan(int *in, int *out) {
  __shared__ int partials[N];

  // we're assuming there's exactly 1 thread block,
  // so the thread id is equal to the thread index.
  int tid = threadIdx.x;

  partials[tid] = in[tid];

  for (int stride = 1; stride < N; stride *= 2) {
    __syncthreads();
    if (tid >= stride) {
      partials[tid] += partials[tid - stride];
    }
  }

  out[tid] = partials[tid];
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
  printf("Elapsed time: %.6fms\n", elapsed_ms);
}
