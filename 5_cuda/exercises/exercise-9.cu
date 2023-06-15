#include <stdio.h>
#include <time.h>
#include <stdlib.h>

const int N = 100000;
const int BINS = 10;
const int THREADS_PER_BLOCK = 1000;
const int BLOCKS_PER_GRID = 32;

__global__ void histogram_sharedmem(int *values, int *histo) {
  __shared__ int histo_block[BINS];
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int totalThreads = blockDim.x * gridDim.x;

  // initialize shared memory only once per block
  for (int i = threadIdx.x; i < BINS; i += blockDim.x) {
    histo_block[i] = 0;
  }

  // make other threads wait until the initialization completes
  __syncthreads();

  // process values into the block-wide histogram
  for (int i = tid; i < N; i += totalThreads) {
    atomicAdd(&(histo_block[values[i]]), 1);
  }

  // wait for all threads to finish processing their values
  __syncthreads();

  // add the block's partial results to the global result array
  // that was passed in as an argument
  for (int i = threadIdx.x; i < BINS; i += blockDim.x)  {
    atomicAdd(&(histo[i]), histo_block[i]);
  }
}

int main() {
  int values[N], histo[BINS], *dev_values, *dev_histo;

  cudaMalloc((void**)&dev_values, N * sizeof(int));
  cudaMalloc((void**)&dev_histo, BINS * sizeof(int));
  cudaMemset(dev_histo, 0, BINS * sizeof(int));

  //srand(time(NULL));
  memset(histo, 0, BINS * sizeof(int));
  for (int i = 0; i < N; i++) values[i] = rand() % BINS;

  // copy host data to device
  cudaMemcpy(dev_values, values, N * sizeof(int), cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // perform computation on device
  histogram_sharedmem<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(dev_values, dev_histo);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float duration_ms;
  cudaEventElapsedTime(&duration_ms, start, stop);

  // copy device results to host
  cudaMemcpy(histo, dev_histo, BINS * sizeof(int), cudaMemcpyDeviceToHost);

  // print results
  for (int i = 0; i < BINS; i++) {
    printf("Bin %d: %d\n", i, histo[i]);
  }
  printf("duration: %3.3fms\n", duration_ms);

  return 0;
}
