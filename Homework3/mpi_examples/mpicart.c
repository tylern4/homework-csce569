#include <stdio.h>
#include "mpi.h"

void main(int argc, char *argv[]) {
    int nrow, mcol, root, Iam, ndim, p, rank;
    int dims[2], coords[2], cyclic[2], reorder;
    MPI_Comm comm2D, comm2Dp;
    /* Starts MPI processes ... */
    MPI_Init(&argc, &argv);               /* starts MPI */
    MPI_Comm_rank(MPI_COMM_WORLD, &Iam);  /* get current process id */
    MPI_Comm_size(MPI_COMM_WORLD, &p);    /* get number of processes */

    nrow = 3;
    mcol = 2;
    ndim = 2;
    root = 0;
    cyclic[0] = 0;
    cyclic[1] = 0;
    reorder = 1;

    if (Iam == root) {
        printf("\n");
        printf("There are six (6) processes\n");
        printf("use all 6 to create 3x2 cartesian topology\n");
        printf("    Cart. Coords.     Cart\n");
        printf("       i        j     rank      Iam\n");
    }
    comm2D = MPI_COMM_NULL;
    comm2Dp = MPI_COMM_NULL;
    MPI_Barrier(MPI_COMM_WORLD);
    /* first, create 3x2 cartesian topology for processes */
    dims[0] = nrow;      /* rows */
    dims[1] = mcol;      /* columns */
    MPI_Cart_create(MPI_COMM_WORLD, ndim, dims, cyclic, reorder, &comm2D);

    if (comm2D != MPI_COMM_NULL) {
        MPI_Cart_coords(comm2D, Iam, ndim, coords);
        MPI_Cart_rank(comm2D, coords, &rank);
        printf("%8d %8d %8d %8d\n", coords[0], coords[1], rank, Iam);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    /* next, create 2x2 cartesian topology for processes */
    if (Iam == root) {
        printf("\n");
        printf("There are six (6) processes\n");
        printf("use 4 to create 2x2 cartesian topology\n");
        printf("    Cart. Coords.     Cart\n");
        printf("       i        j     rank      Iam\n");
    }
#if 0
    MPI_Barrier(MPI_COMM_WORLD);
    dims[0] = 2;        /* rows */
    dims[1] = 2;        /* columns */
    MPI_Cart_create(MPI_COMM_WORLD, ndim, dims, cyclic, reorder, &comm2Dp);

    if (comm2Dp != MPI_COMM_NULL) {
        MPI_Cart_coords(comm2Dp, Iam, ndim, coords);
        MPI_Cart_rank(comm2Dp, coords, &rank);
        printf("%8d %8d %8d %8d\n", coords[0], coords[1], rank, Iam);
    }
#endif
    MPI_Finalize();                  /* let MPI finish up ...  */
}
