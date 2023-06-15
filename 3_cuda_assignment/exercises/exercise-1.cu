#include <stdio.h>

__global__ void print_tid() {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  printf("block index=%d, thread index=%d, tid=%d\n",
    blockIdx.x, threadIdx.x, tid);
}

int main() {
  print_tid<<<4,3>>>();
  cudaDeviceSynchronize();
}