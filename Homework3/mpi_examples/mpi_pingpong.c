#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#define PING_PONG_LIMIT 10

int main(int argc, char** argv) {
  
  MPI_Init(NULL, NULL);
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // We are assuming at least 2 processes for this task
  if (size != 2) {
    fprintf(stderr, "World size must be two for %s\n", argv[0]);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int count = 0;
  int partner_rank = (rank + 1) % 2; /* 1 for rank 0, 0 for rank 1 */
  while (count < PING_PONG_LIMIT) {
    if (rank == count % 2) {
      count++;
      MPI_Send(&count, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD);
      printf("%d sent and incremented count %d to %d\n", rank, count, partner_rank);
    } else {
      MPI_Recv(&count, 1, MPI_INT, partner_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printf("%d received count %d from %d\n", rank, count, partner_rank);
    }
  }
  MPI_Finalize();
}
