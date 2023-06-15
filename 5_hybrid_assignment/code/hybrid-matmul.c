#include <omp.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

const int N = 1000;

#define TAG 13

int main(int argc, char *argv[]) {
  double **A, **B, **C, *tmp;
  double startTime, endTime;
  int numElements, offset, stripSize, myrank, numnodes, i, j, k;
  int chunkSize = 10;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &numnodes);

  // allocate A, B, and C --- note that you want these to be
  // contiguously allocated.  Workers need less memory allocated.

  if (myrank == 0) {
    tmp = (double *) malloc (sizeof(double ) * N * N);
    A = (double **) malloc (sizeof(double *) * N);
    for (i = 0; i < N; i++)
      A[i] = &tmp[i * N];
  }
  else {
    tmp = (double *) malloc (sizeof(double ) * N * N / numnodes);
    A = (double **) malloc (sizeof(double *) * N / numnodes);
    for (i = 0; i < N / numnodes; i++)
      A[i] = &tmp[i * N];
  }


  tmp = (double *) malloc (sizeof(double ) * N * N);
  B = (double **) malloc (sizeof(double *) * N);
  for (i = 0; i < N; i++)
    B[i] = &tmp[i * N];


  if (myrank == 0) {
    tmp = (double *) malloc (sizeof(double ) * N * N);
    C = (double **) malloc (sizeof(double *) * N);
    for (i = 0; i < N; i++)
      C[i] = &tmp[i * N];
  }
  else {
    tmp = (double *) malloc (sizeof(double ) * N * N / numnodes);
    C = (double **) malloc (sizeof(double *) * N / numnodes);
    for (i = 0; i < N / numnodes; i++)
      C[i] = &tmp[i * N];
  }

  if (myrank == 0) {
    // initialize A and B
    for (i=0; i<N; i++) {
      for (j=0; j<N; j++) {
        A[i][j] = 1.0;
        B[i][j] = 1.0;
      }
    }
  }

  // start timer
  if (myrank == 0) {
    startTime = MPI_Wtime();
  }

  stripSize = N/numnodes;

  // send each node its piece of A -- note could be done via MPI_Scatter
  if (myrank == 0) {
    offset = stripSize;
    numElements = stripSize * N;
    for (i=1; i<numnodes; i++) {
      MPI_Send(A[offset], numElements, MPI_DOUBLE, i, TAG, MPI_COMM_WORLD);
      offset += stripSize;
    }
  }
  else {  // receive my part of A
    MPI_Recv(A[0], stripSize * N, MPI_DOUBLE, 0, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // everyone gets B
  MPI_Bcast(B[0], N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Let each process initialize C to zero
  for (i=0; i<stripSize; i++) {
    for (j=0; j<N; j++) {
      C[i][j] = 0.0;
    }
  }

  // do the work---this is the primary difference from the pure MPI program
  #pragma omp parallel for shared(A,B,C)
  for (int i=0; i<stripSize; i++) {
    for (int j=0; j<N; j++) {
      for (int k=0; k<N; k++) {
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  // master receives from workers  -- note could be done via MPI_Gather
  if (myrank == 0) {
    offset = stripSize;
    numElements = stripSize * N;
    for (i=1; i<numnodes; i++) {
      MPI_Recv(C[offset], numElements, MPI_DOUBLE, i, TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      offset += stripSize;
    }
  }
  else { // send my contribution to C
    MPI_Send(C[0], stripSize * N, MPI_DOUBLE, 0, TAG, MPI_COMM_WORLD);
  }

  // stop timer
  if (myrank == 0) {
    endTime = MPI_Wtime();
    printf("Time is %f\n", endTime-startTime);
  }

  // check that results are correct
  // every value in the result matrix should equal N
  if (myrank == 0) {
    int valid = 1;
    for (i=0; i<N-1; i++) {
      for (j=0; j<N; j++) {
        if ((int)C[i][j] != N) {
          printf("value: i=%d, j=%d, %d\n", i, j, C[i][j]);
          valid = 0;
        }
      }
    }
    printf("Results valid: %s\n", valid ? "yes" : "no");
  }

  MPI_Finalize();
  return 0;
}
