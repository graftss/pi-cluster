#include <stdio.h>

const int N = 100;

__global__ void kernel_add(int* a, int* b, int* c) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  c[tid] = a[tid] + b[tid];
}

int main() {
  // allocate and initialize host memory
  int *a, *b, *c;
  a = (int*)malloc(N * sizeof(int));
  b = (int*)malloc(N * sizeof(int));
  c = (int*)malloc(N * sizeof(int));
  for (int i = 0; i < N; i++) {
    a[i] = 1;
    b[i] = 2;
  }

  // allocate device memory
  int *dev_a, *dev_b, *dev_c;
  cudaMalloc((void**)&dev_a, N * sizeof(int));
  cudaMalloc((void**)&dev_b, N * sizeof(int));
  cudaMalloc((void**)&dev_c, N * sizeof(int));

  // copy data from host memory to device memory
  cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

  // perform a computation on the device
  kernel_add<<<1, N>>>(dev_a, dev_b, dev_c);

  // copy results from the device to the host
  cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);
    
  // verify that the kernel has correctly added a and b into c
  for (int i = 0; i < N; i++) printf("%d ", c[i]);
  printf("\n");
}