#include <stdio.h>

const int N = 100;
const int ARRAY_SIZE = N * 2;

__global__ void kernel_add(int* a, int* b, int* c) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int other_idx = tid + N;

  c[tid] = a[tid] + b[tid];
  c[other_idx] = a[other_idx] + b[other_idx];
}

int main() {
  // allocate and initialize host memory
  int *a, *b, *c;
  a = (int*)malloc(ARRAY_SIZE * sizeof(int));
  b = (int*)malloc(ARRAY_SIZE * sizeof(int));
  c = (int*)malloc(ARRAY_SIZE * sizeof(int));
  for (int i = 0; i < ARRAY_SIZE; i++) {
    a[i] = 1;
    b[i] = 2;
  }

  // allocate device memory
  int *dev_a, *dev_b, *dev_c;
  cudaMalloc((void**)&dev_a, ARRAY_SIZE * sizeof(int));
  cudaMalloc((void**)&dev_b, ARRAY_SIZE * sizeof(int));
  cudaMalloc((void**)&dev_c, ARRAY_SIZE * sizeof(int));

  // copy data from host memory to device memory
  cudaMemcpy(dev_a, a, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);

  // perform a computation on the device
  kernel_add<<<1, N>>>(dev_a, dev_b, dev_c);

  // copy results from the device to the host
  cudaMemcpy(c, dev_c, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

  // verify that the kernel has correctly added a and b into c
  for (int i = 0; i < ARRAY_SIZE; i++) printf("%d ", c[i]);
  printf("\n");
}
