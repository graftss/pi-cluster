#include <stdio.h>
#include <stdlib.h>

const int N = 10000000;
const int THREADS_PER_BLOCK = 100;
const int NUM_BLOCKS = 32;

// merge `m` elements from `a` and `n` elements from `b` into `out`.
__device__ void serial_merge(int *a, int m, int *b, int n, int *out) {
  int i, j, k;

  // step 1
  // merge from both input arrays until one of them is fully merged
  for (i = 0, j = 0, k = 0; i < m && j < n; k++) {
    if (a[i] <= b[j]) {
      out[k] = a[i];
      i++;
    } else {
      out[k] = b[j];
      j++;
    }
  }

  // step 2
  // once one input array is fully merged, merge all remaining elements
  // from the second input array into the output array.
  if (i == m) {
    while (k < m + n) {
      out[k++] = b[j++];
    }
  } else {
    while (k < m + n) {
      out[k++] = a[i++];
    }
  }
}

__device__ int co_rank(int k, int *a, int m, int *b, int n) {
  // initialize i and j so that i + j == k.
  int i = min(k, m);
  int j = k - i;

  // if k > n, then at most n elements can come from array b,
  // meaning that at least k-n would need to come from array a.
  // in this case, the lowest possible value for the co-rank i is k-n.
  int i_min = max(0, k - n);

  // likewise, if k > m, then at least k-m elements will come from array b.
  int j_min = max(0, k - m);

  int delta;

  while (1) {
    if (i > 0 && j < n && a[i-1] > b[j]) {
      // if a[i-1] > b[j], then i is too high.
      // decrease i to halfway between its current value and the lowest possible value.
      // increase j by the same amount.
      delta = (i - i_min + 1) / 2;
      j_min = j;
      j = j + delta;
      i = i - delta;
    } else if (j > 0 && i < m && b[j-1] >= a[i]) {
      // likewise, if b[j-1] >= a[i], then j is too high.
      // decrease j to halfway between its current value and the lowest possible value.
      // increase i by the same amount.
      delta = (j - j_min + 1) / 2;
      i_min = i;
      i = i + delta;
      j = j - delta;
    } else {
      return i;
    }
  }
}

__global__ void parallel_merge_kernel(int *a, int m, int *b, int n, int *out) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_threads = blockDim.x * gridDim.x;
  int elts_per_thread = ceilf((m + n) / total_threads);

  // compute the lowest output index of this thread and the next thread
  int k = tid * elts_per_thread;
  int k_next = min(m + n, k + elts_per_thread);

  // compute the co-ranks of this thread's lowest output index
  int i = co_rank(k, a, m, b, n);
  int j = k - i;

  // compute the co-ranks of the next thread's lowest output index
  int i_next = co_rank(k_next, a, m, b, n);
  int j_next = k_next - i_next;

  int merged_elts_of_a = i_next - i;
  int merged_elts_of_b = j_next - j;

  serial_merge(&(a[i]), merged_elts_of_a,
               &(b[j]), merged_elts_of_b,
               &(out[k]));
}

int main() {
  int *a, *b, *out, *dev_a, *dev_b, *dev_out;
  float elapsed_ms;
  cudaEvent_t start, stop;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  a = (int*)malloc(N * sizeof(int));
  b = (int*)malloc(N * sizeof(int));
  out = (int*)malloc(2 * N * sizeof(int));

  // `a` and `b` must be initialized to sorted arrays to satisfy the
  // assumptions of the merging phase.
  for (int i = 0; i < N; i++) {
    a[i] = 2 * i;
    b[i] = 2 * i + 1;
  }

  cudaMalloc((void**)&dev_a, N * sizeof(int));
  cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&dev_b, N * sizeof(int));
  cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMalloc((void**)&dev_out, 2 * N * sizeof(int));

  cudaEventRecord(start, 0);

  parallel_merge_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(dev_a, N, dev_b, N, dev_out);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&elapsed_ms, start, stop);

  cudaDeviceSynchronize();
  cudaMemcpy(out, dev_out, 2 * N * sizeof(int), cudaMemcpyDeviceToHost);

  // verify results by checking that `out[i] < out[i+1]` for all i.
  int is_sorted = 1;
  for (int i = 0; i < 2 * N - 1; i += 1) {
    is_sorted = is_sorted && (out[i] <= out[i+1]);
  }

  printf("Output array sorted: %s\n", is_sorted ? "yes" : "no");
  printf("Elements merged: %d\n", 2 * N);
  printf("Elapsed time: %.3fms\n", elapsed_ms);
}
