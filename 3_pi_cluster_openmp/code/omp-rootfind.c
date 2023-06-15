#include <stdio.h>
#include <math.h>
#include "omp.h"

// the number of times to run the main loop of the program.
const int TRIALS = 10000;

const float PRECISION = 0.000001f;
const float X_STEP = 0.500001;
const float X_MIN = -20;
const float X_MAX = 20;

float roots[100];
int next_root_idx = 0;

void record_root(float root) {
  roots[next_root_idx] = root;
  next_root_idx += 1;
}

// returns the sign of `num`: -1 if negative, 1 if positive, 0 if zero.
int sign(float num) {
  if (num < 0) return -1;
  if (num > 0) return 1;
  return 0;
}

float eval_polynomial(float *poly, int degree, float x) {
  float result = 0;
  for (int i = 0; i < degree + 1; i++) {
    result += poly[i];
    if (i < degree) result *= x;
  }
  return result;
}

void find_roots(float *poly, int degree, float min, float max) {
  float f_min = eval_polynomial(poly, degree, min);
  int min_sign = sign(f_min);

  float f_max = eval_polynomial(poly, degree, max);
  int max_sign = sign(f_max);

  float mid = (min + max) / 2;
  float f_mid = eval_polynomial(poly, degree, mid);
  int mid_sign = sign(f_mid);
  float mid_width = mid - min;

  if (min_sign == -1 * mid_sign) {
    if (mid_width < PRECISION) {
      #pragma omp critical (root)
      record_root((min + mid) / 2);
    } else {
      #pragma omp task
      find_roots(poly, degree, min, mid);
    }
  } else if (max_sign == -1 * mid_sign) {
    if (mid_width < PRECISION) {
      #pragma omp critical (root)
      record_root((mid + max) / 2);
    } else {
      #pragma omp task
      find_roots(poly, degree, mid, max);
    }
  }
}

int main(int argc, char** argv) {
  // polynomial to find the roots of
  float poly[9] = { 1.0f, -5.75f, -223.41f, 669.618f, 13791.5f,
                    -2934.76f, -80463.2f, 72925.3f, 3846.97f };
  int degree = 8;

  double start_time = omp_get_wtime();

  for (int trial = 0; trial < TRIALS; trial++) {
    // reset the root count before each trial;
    // this way, the results of the previous trial are just
    // overwritten by the next trial.
    next_root_idx = 0;

    #pragma omp parallel
    #pragma omp single
    {
      // initialize the searched interval's min and max
      float next_min = X_MIN;
      float next_max = X_MIN + X_STEP;

      while (next_max <= X_MAX) {
        find_roots(poly, degree, next_min, next_max);
        next_min = next_max;
        next_max += X_STEP;
      }
    }
  }

  for (int i = 0; i < next_root_idx; i++) {
    printf("Root %d: %.3f\n", i + 1, roots[i]);
  }

  double duration = omp_get_wtime() - start_time;
  printf("Duration: %.5f seconds\n", duration);
}
