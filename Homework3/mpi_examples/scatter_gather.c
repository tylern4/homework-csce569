#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/*
 * ! This program shows how to use MPI_Scatter and MPI_Gather
 * ! Each processor gets different data from the root processor
 * ! by way of mpi_scatter.  The data is summed and then sent back
 * ! to the root processor using MPI_Gather.  The root processor
 * ! then prints the global sum. 
 * */

int main(int argc, char *argv[]) {
    int *myray, *send_ray, *back_ray;
    int count;
    int size, mysize, i, k, j, total;
    int numprocs, myrank;
    int root = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    /* each processor will get count elements from the root */
    count = 4;
    myray = (int *) malloc(count * sizeof(int));
    /* create the data to be sent on the root */
    if (myrank == root) {
        size = count * numprocs;
        send_ray = (int *) malloc(size * sizeof(int));
        back_ray = (int *) malloc(numprocs * sizeof(int));
        for (i = 0; i < size; i++)
            send_ray[i] = i;
    }
    /* send different data to each processor */
    MPI_Scatter(send_ray, count, MPI_INT, myray, count, MPI_INT, root, MPI_COMM_WORLD);

    /* each processor does a local sum */
    total = 0;
    for (i = 0; i < count; i++)
        total = total + myray[i];
    printf("myid= %d total= %d\n", myrank, total);
    /* send the local sums back to the root */
    MPI_Gather(&total, 1, MPI_INT, back_ray, 1, MPI_INT, root, MPI_COMM_WORLD);
    /* the root prints the global sum */
    if (myrank == root) {
        total = 0;
        for (i = 0; i < numprocs; i++)
            total = total + back_ray[i];
        printf("results from all processors= %d \n", total);
    }
    MPI_Finalize();
}
