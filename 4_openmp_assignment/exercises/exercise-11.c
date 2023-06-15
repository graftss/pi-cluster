#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "omp.h"

// the number of elements to be sorted
const int N = 200000;

void swap(int *a, int *b) {
  int temp = *a;
  *a = *b;
  *b = temp;
}

int partition(int *data, int start, int end) {
  int pivot_idx = (start + end) / 2;
  int pivot_value = data[pivot_idx];

  // swap the pivot to the end of the array, out of the way
  swap(&data[end], &data[pivot_idx]);

  int next_swap_idx = start;

  for (int i = start; i < end; i++) {
    if (data[i] < pivot_value) {
      swap(&data[i], &data[next_swap_idx]);
      next_swap_idx++;
    }
  }

  // swap the pivot back to between the low and high subarrays
  swap(&data[next_swap_idx], &data[end]);

  // return the final index of the pivot
  return next_swap_idx;
}

void serial_quicksort(int *data, int start, int end) {
  if (start < end) {
    int pivot_idx = partition(data, start, end);
    serial_quicksort(data, start, pivot_idx - 1);
    serial_quicksort(data, pivot_idx + 1, end);
  }
}

void quicksort(int *data, int start, int end) {
  if (start < end) {
    if (end - start < 1000) {
      serial_quicksort(data, start, end);
      return;
    }

    int pivot_idx = partition(data, start, end);

    #pragma omp task
    quicksort(data, start, pivot_idx - 1);

    #pragma omp task
    quicksort(data, pivot_idx + 1, end);
  }
}

int main() {
  // initialize the input array to all random values
  int* data = (int*)malloc(N * sizeof(int));
  for (int i = 0; i < N; i++) {
    data[i] = rand() % 100;
  }

  double start = omp_get_wtime();

  #pragma omp parallel
  #pragma omp single
  quicksort(data, 0, N - 1);

  double duration = omp_get_wtime() - start;

  // check that the array is sorted in increasing order
  // by comparing each pair of consecutive elements
  int valid = 1;
  for (int i = 1; i < N; i++) {
    valid = valid && (data[i] >= data[i - 1]);
  }

  printf("Result valid: %s\n", valid ? "yes" : "no");
  printf("Values sorted: %d\n", N);
  printf("Duration: %.6f seconds\n", duration);
}
