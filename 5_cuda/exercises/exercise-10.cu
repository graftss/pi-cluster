#include <stdio.h>
#include <time.h>

const int TRIALS = 1000;
const int N = 1000000;

void profile_unpinned_memory(int *dev_a) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // allocate unpinned memory
  int *a;
  a = (int*)malloc(N * sizeof(int));

  for (int i = 0; i < N; i++) {
    a[i] = rand();
  }

  cudaEventRecord(start);

  // copy data from host memory to device memory
  for (int trial = 0; trial < TRIALS; trial++) {
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Unpinned transfer duration: %3.4fms\n", milliseconds);

  // free the unpinned memory
  free(a);
}

void profile_pinned_memory(int *dev_a) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // allocate pinned memory
  int *a_pinned;
  cudaMallocHost((void**)&a_pinned, N * sizeof(int));

  for (int i = 0; i < N; i++) {
    a_pinned[i] = rand();
  }

  cudaEventRecord(start);

  // copy data from host memory to device memory
  for (int trial = 0; trial < TRIALS; trial++) {
    cudaMemcpy(dev_a, a_pinned, N * sizeof(int), cudaMemcpyHostToDevice);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Pinned transfer duration: %3.4fms\n", milliseconds);

  // free the pinned host memory.
  // note: while it's not necessary to manually free memory in a small
  // program like this, it's a very important habit to develop in general.
  cudaFreeHost(a_pinned);
}

int main() {
  // allocate device memory
  int *dev_a;
  cudaMalloc((void**)&dev_a, N * sizeof(int));

  profile_unpinned_memory(dev_a);
  profile_pinned_memory(dev_a);

  // free the device memory
  cudaFree(dev_a);
}
