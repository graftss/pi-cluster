#include <stdio.h>
#include <stdlib.h>
#include "omp.h"

const int MIN_PARALLEL_COUNT = 1000;

#define N 1000000
int data[N];

#define MAX_THREADS 8
int invocations[MAX_THREADS];

int serial_sum(int *data, int count) {
  int result = 0;
  for (int i = 0; i < count; i++) {
    result += data[i];
  }
  return result;
}

int parallel_sum(int *data, int count) {
  invocations[omp_get_thread_num()]++;

  if (count <= MIN_PARALLEL_COUNT) {
    return serial_sum(data, count);
  }

  int left_sum, right_sum, sum;
  int middle = count / 2;

  #pragma omp task shared(left_sum)
  left_sum = parallel_sum(data, middle);

  #pragma omp task shared(right_sum)
  right_sum = parallel_sum(data + middle, count - middle);

  #pragma omp taskwait
  return left_sum + right_sum;
}

int main() {
  for (int i = 0; i < MAX_THREADS; i++) invocations[i] = 0;
  for (int i = 0; i < N; i++) data[i] = rand() % 5;

  int sum;
  double start = omp_get_wtime();

  #pragma omp parallel
  #pragma omp single
  sum = parallel_sum(data, N);

  double duration = omp_get_wtime() - start;

  printf("Invocations per thread number: { ");
  for (int i = 0; i < MAX_THREADS; i++) {
    if (invocations[i] == 0) break;
    printf("%d", invocations[i]);
    if (i < MAX_THREADS - 1 && invocations[i+1] > 0) printf(", ");
  }
  printf(" }\n");

  printf("Result sum: %d\n", sum);
  printf("Duration: %.6f\n", duration);
}
