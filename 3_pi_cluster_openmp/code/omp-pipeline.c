#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "omp.h"

#define N 1000
const int DATA_SIZE = 100;

int *data_ptrs[N];
int results[N];

int* read_data(int data_size) {
  // simulate an asynchronous I/O operation by briefly sleeping
  usleep(1000);

  // allocate and initialize the data
  int* data = (int*)malloc(data_size * sizeof(int));
  for (int i = 0; i < data_size; i++) {
    data[i] = rand() % 100;
  }

  return data;
}

// compute the sum of the values in `data`
int process_data(int data_size, int *data) {
  int result = 0;
  for (int i = 0; i < data_size; i++) {
    result += data[i];
  }
  return result;
}

int main() {
  double start = omp_get_wtime();

  #pragma omp parallel shared(data_ptrs, results)
  #pragma omp single
  {
    for (int i = 0; i < N; i++) {
      #pragma omp task depend(out: data_ptrs[i])
      {
        data_ptrs[i] = read_data(DATA_SIZE);
      }

      #pragma omp task depend(in: data_ptrs[i])
      {
        results[i] = process_data(DATA_SIZE, data_ptrs[i]);
      }
    }
  }

  double duration = omp_get_wtime() - start;

  printf("Duration: %.5f\n", duration);
}
