#include <stdio.h>

const int N = 500000;
const int BLOCKS = 32;
const int THREADS_PER_BLOCK = 256;

__global__ void kernel_dot(int* a, int* b, int* blockPartials) {
  // declare memory that's shared among all threads in the block
  __shared__ int threadPartials[THREADS_PER_BLOCK];

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int totalThreads = blockDim.x * gridDim.x;

  // compute this thread's partial
  int i = tid;
  int threadPartial = 0;
  while (i < N) {
    threadPartial += a[i] * b[i];
    i += totalThreads;
  }

  // store this thread's partial in shared memory
  threadPartials[threadIdx.x] = threadPartial;

  // wait until all threads in the block have computed their partial
  __syncthreads();

  // add together the partials from each thread in the block
  if (threadIdx.x == 0) {
    int blockPartial = 0;
    for (int i = 0; i < THREADS_PER_BLOCK; i++) {
      blockPartial += threadPartials[i];
    }
    blockPartials[blockIdx.x] = blockPartial;
  }
}

int main() {
  // allocate host memory
  int *a, *b, *blockPartials;
  a = (int*) malloc(N * sizeof(int));
  b = (int*) malloc(N * sizeof(int));
  blockPartials = (int*) malloc(BLOCKS * sizeof(int));
  for (int i = 0; i < N; i++) {
    a[i] = 7;
    b[i] = 2;
  }

  // allocate device memory
  int *dev_a, *dev_b, *dev_blockPartials;
  cudaMalloc((void**) &dev_a, N * sizeof(int));
  cudaMalloc((void**) &dev_b, N * sizeof(int));
  cudaMalloc((void**) &dev_blockPartials, BLOCKS * sizeof(int));

  // copy host data to device
  cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

  // set up timing events
  cudaEvent_t start, stop;
  float elapsed;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // compute dot product on device
  kernel_dot<<<BLOCKS, THREADS_PER_BLOCK>>>(dev_a, dev_b, dev_blockPartials);

  // stop timing and compute elapsed time
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);

  // copy block partials back to host and compute their sum on the host
  cudaMemcpy(blockPartials, dev_blockPartials,
    BLOCKS * sizeof(int), cudaMemcpyDeviceToHost);
  int result = 0;
  for (int i = 0; i < BLOCKS; i++) {
    result += blockPartials[i];
  }
  printf("result = %d, duration = %3.5f ms\n", result, elapsed);
}
