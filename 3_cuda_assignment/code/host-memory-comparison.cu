#include <stdio.h>
#include <time.h>

const int TRIALS = 1000;
const int N = 1000000;

void profile_unpinned_memory(int *dev_a) {
  // allocate unpinned memory
  int *a;
  a = (int*)malloc(N * sizeof(int));

  clock_t start = clock();

  // copy data from host memory to device memory
  for (int trial = 0; trial < TRIALS; trial++) {
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(a, dev_a, N * sizeof(int), cudaMemcpyDeviceToHost);
  }

  clock_t end = clock();
  printf("Unpinned transfer duration: %3.4f seconds\n",
    (float)(end - start) / CLOCKS_PER_SEC);

  // free the unpinned memory
  free(a);
}

void profile_pinned_memory(int *dev_a) {
  // allocate pinned memory
  int *a_pinned;
  cudaMallocHost((void**)&a_pinned, N * sizeof(int));

  clock_t start = clock();

  // copy data from host memory to device memory
  for (int trial = 0; trial < TRIALS; trial++) {
    cudaMemcpy(dev_a, a_pinned, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_a, a_pinned, N * sizeof(int), cudaMemcpyDeviceToHost);
  }

  clock_t end = clock();
  printf("Pinned transfer duration: %3.4f seconds\n",
    (float)(end - start) / CLOCKS_PER_SEC);

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
