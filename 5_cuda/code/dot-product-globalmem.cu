#include <stdio.h>

const int N = 500000;
const int BLOCKS = 32;
const int THREADS_PER_BLOCK = 256;

__global__ void kernel_dot(float* a, float* b, float* partials) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int totalThreads = blockDim.x * gridDim.x;

  int i = tid;
  while (i < N) {
    partials[i] = a[i] * b[i];
    i += totalThreads;
  }
}

int main() {
  // allocate host memory
  float *a, *b, *partials;
  a = (float*) malloc(N * sizeof(float));
  b = (float*) malloc(N * sizeof(float));
  partials = (float*) malloc(N * sizeof(float));
  for (int i = 0; i < N; i++) {
    a[i] = 7;
    b[i] = 2;
  }

  // allocate device memory
  float *dev_a, *dev_b, *dev_partials;
  cudaMalloc((void**) &dev_a, N * sizeof(float));
  cudaMalloc((void**) &dev_b, N * sizeof(float));
  cudaMalloc((void**) &dev_partials, N * sizeof(float));

  // copy host data to device
  cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

  // set up timing events
  cudaEvent_t start, stop;
  float elapsed;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // compute dot product on device
  kernel_dot<<<BLOCKS, THREADS_PER_BLOCK>>>(dev_a, dev_b, dev_partials);

  // stop timing and compute elapsed time
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);

  // copy block partials back to host and compute their sum on the host
  cudaMemcpy(partials, dev_partials,
    N * sizeof(float), cudaMemcpyDeviceToHost);
  float result = 0;
  for (int i = 0; i < N; i++) {
    result += partials[i];
  }
  printf("result = %3.3f, duration = %3.5f ms\n", result, elapsed);
}
