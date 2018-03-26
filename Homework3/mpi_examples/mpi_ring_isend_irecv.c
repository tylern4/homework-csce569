#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  MPI_Init(NULL, NULL);
  // Find out rank, size
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int token;
  MPI_Request request;
  // Receive from the lower process and send to the higher process. Take care
  // of the special case when you are the first process to prevent deadlock.
  if (rank != 0) {
    MPI_Irecv(&token, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, &request);
    printf("waiting for the massge to arrive so I can pass on %d\n", rank); 
    /* this may delay the ring performance */
    MPI_Wait(&request, MPI_STATUS_IGNORE);
    printf("Process %d received token %d from process %d\n", rank, token, rank - 1);
  } else {
    // Set the token's value if you are process 0
    token = -1;
  }
  MPI_Isend(&token, 1, MPI_INT, (rank + 1) % size, 0, MPI_COMM_WORLD, &request);
  // Now process 0 can receive from the last process. This makes sure that at
  // least one MPI_Send is initialized before all MPI_Recvs (again, to prevent
  // deadlock)
  printf("msg is sent out and waiting for the msg to pass on  %d\n", rank);
  if (rank == 0) {
    MPI_Recv(&token, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Process %d received token %d from process %d\n", rank, token, size - 1);
  } {
  }
  MPI_Wait(&request, MPI_STATUS_IGNORE);
  MPI_Finalize();
}
