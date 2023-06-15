#include <stdio.h>
#include <stdlib.h>

// defines the input sparse matrix
const int ROWS = 1000000;
const int COLS = 1000;
const int NONZERO_ELTS_PER_ROW = 20;
const int NONZERO_ELTS = ROWS * NONZERO_ELTS_PER_ROW;

// defines the thread allocation on the device
const int THREADS_PER_BLOCK = 100;
const int NUM_BLOCKS = 10;

void generate_csr_matrix(float *data, int *col_index, int *row_ptr) {
  for (int i = 0; i < NONZERO_ELTS; i++) {
    // choose a random value for the element
    data[i] = (float)rand() / RAND_MAX;

    // choose a random column index for the element
    col_index[i] = rand() % COLS;
  }

  for (int i = 0; i <= ROWS; i++) {
    row_ptr[i] = i * NONZERO_ELTS_PER_ROW;
  }
}

__global__ void spmv_csr(float *data, int *col_index, int *row_ptr,
                         float *x, float *y) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_threads = blockDim.x * gridDim.x;
  int row = tid;

  while (row < ROWS) {
    float partial = 0;
    int min_idx = row_ptr[row];
    int max_idx = row_ptr[row + 1];
    for (int i = min_idx; i < max_idx; i++) {
      partial += data[i] * x[col_index[i]];
    }

    y[row] = partial;
    row += total_threads;
  }
}

int main() {
  float elapsed_ms;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // allocate and generate a CSR-encoded matrix and two vectors on the host
  float *data = (float*)malloc(NONZERO_ELTS * sizeof(float));
  int *col_index = (int*)malloc(NONZERO_ELTS * sizeof(int));
  int *row_ptr = (int*)malloc((ROWS + 1) * sizeof(int));
  generate_csr_matrix(data, col_index, row_ptr);

  float *x = (float*)malloc(COLS * sizeof(float));
  for (int i = 0; i < COLS; i++) {
    x[i] = (float)rand() / RAND_MAX;
  }

  float *y = (float*)malloc(ROWS * sizeof(float));

  // allocate a CSR-encoded matrix and two vectors on the device
  float *dev_data, *dev_x, *dev_y;
  int *dev_col_index, *dev_row_ptr;
  cudaMalloc((void**)&dev_data, NONZERO_ELTS * sizeof(float));
  cudaMalloc((void**)&dev_col_index, NONZERO_ELTS * sizeof(int));
  cudaMalloc((void**)&dev_row_ptr, (ROWS + 1) * sizeof(int));
  cudaMalloc((void**)&dev_x, COLS * sizeof(float));
  cudaMalloc((void**)&dev_y, ROWS * sizeof(float));

  // copy the CSR-encoded matrix and the input vector from the host to the device
  cudaMemcpy(dev_data, data, NONZERO_ELTS * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_col_index, col_index, NONZERO_ELTS * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_row_ptr, row_ptr, (ROWS + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_x, x, COLS * sizeof(float), cudaMemcpyHostToDevice);

  cudaEventRecord(start, 0);

  spmv_csr<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(dev_data, dev_col_index, dev_row_ptr, dev_x, dev_y);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&elapsed_ms, start, stop);

  cudaDeviceSynchronize();
  cudaMemcpy(y, dev_y, ROWS * sizeof(float), cudaMemcpyDeviceToHost);

  printf("Input matrix: %dx%d, %d nonzero elements per row.\n", ROWS, COLS, NONZERO_ELTS_PER_ROW);
  printf("Elapsed time: %.6fms\n", elapsed_ms);
}
