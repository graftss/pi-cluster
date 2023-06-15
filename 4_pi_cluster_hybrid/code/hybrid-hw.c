#include <stdio.h>
#include <mpi.h>
#include "omp.h"

int main(int argc, char **argv) {
  int mpi_rank, size, hostname_length;
  char hostname[MPI_MAX_PROCESSOR_NAME];

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Get_processor_name(hostname, &hostname_length);

  #pragma omp parallel for
  for (int i = 0; i < 4; i++) {
    int omp_tid = omp_get_thread_num();
    printf("hostname: %s, MPI rank: %d, OpenMP id: %d\n",
           hostname, mpi_rank, omp_tid);
  }

  MPI_Finalize();
}
