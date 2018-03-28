#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/timeb.h>
#include <mpi.h>
#define REAL float

/* compile and run the program using the following command
 *    mpicc jacobi.c -lm -o jacobi
 *    mpirun -np 4 ./jacobi
 */

/* read timer in second */
double read_timer() {
  struct timeb tm;
  ftime(&tm);
  return (double)tm.time + (double)tm.millitm / 1000.0;
}

/* read timer in ms */
double read_timer_ms() {
  struct timeb tm;
  ftime(&tm);
  return (double)tm.time * 1000.0 + (double)tm.millitm;
}

/************************************************************
 * program to solve a finite difference
 * discretization of Helmholtz equation :
 * (d2/dx2)u + (d2/dy2)u - alpha u = f
 * using Jacobi iterative method.
 *
 * Input :  n - grid dimension in x direction
 *          m - grid dimension in y direction
 *          alpha - Helmholtz constant (always greater than 0.0)
 *          tol   - error tolerance for iterative solver
 *          relax - Successice over relaxation parameter
 *          mits  - Maximum iterations for iterative solver
 *
 * On output
 *       : u(n,m) - Dependent variable (solutions)
 *       : f(n,m) - Right hand side function
 *************************************************************/

// flexible between REAL and double
#define DEFAULT_DIMSIZE 256

void print_array(char *title, char *name, REAL *A, long n, long m) {
  printf("%s:\n", title);
  long i, j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
      printf("%s[%ld][%ld]:%f  ", name, i, j, A[i * m + j]);
    }
    printf("\n");
  }
  printf("\n");
}

/*      subroutine initialize (n,m,alpha,dx,dy,u,f)
 ******************************************************
 * Initializes data
 * Assumes exact solution is u(x,y) = (1-x^2)*(1-y^2)
 *
 ******************************************************/
void initialize(long n, long m, REAL alpha, REAL dx, REAL dy, REAL *u_p, REAL *f_p) {
  long i;
  long j;
  long xx;
  long yy;
  REAL(*u)[m] = (REAL(*)[m])u_p;
  REAL(*f)[m] = (REAL(*)[m])f_p;

  // double PI=3.1415926;
  /* Initialize initial condition and RHS */
  //#pragma omp parallel for private(xx,yy,j,i)
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++) {
      xx = ((int)(-1.0 + (dx * (i - 1))));
      yy = ((int)(-1.0 + (dy * (j - 1))));
      u[i][j] = 0.0;
      f[i][j] = (((((-1.0 * alpha) * (1.0 - (xx * xx))) * (1.0 - (yy * yy))) - (2.0 * (1.0 - (xx * xx)))) -
                 (2.0 * (1.0 - (yy * yy))));
    }
}

/*  subroutine error_check (n,m,alpha,dx,dy,u,f)
 implicit none
 ************************************************************
 * Checks error between numerical and exact solution
 *
 ************************************************************/
double error_check(long n, long m, REAL alpha, REAL dx, REAL dy, REAL *u_p, REAL *f_p) {
  int i;
  int j;
  REAL xx;
  REAL yy;
  REAL temp;
  double error;
  error = 0.0;
  REAL(*u)[m] = (REAL(*)[m])u_p;
  REAL(*f)[m] = (REAL(*)[m])f_p;
  //#pragma omp parallel for private(xx,yy,temp,j,i) reduction(+:error)
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++) {
      xx = (-1.0 + (dx * (i - 1)));
      yy = (-1.0 + (dy * (j - 1)));
      temp = (u[i][j] - ((1.0 - (xx * xx)) * (1.0 - (yy * yy))));
      error = (error + (temp * temp));
    }
  error = (sqrt(error) / (n * m));
  return error;
}
void jacobi_seq(long n, long m, REAL dx, REAL dy, REAL alpha, REAL relax, REAL *u_p, REAL *f_p, REAL tol, int mits);
void jacobi_mpi(long n, long m, REAL dx, REAL dy, REAL alpha, REAL relax, REAL *u_p, REAL *f_p, REAL tol, int mits);
int numprocs;
int myrank;

int main(int argc, char *argv[]) {
  long n = DEFAULT_DIMSIZE;
  long m = DEFAULT_DIMSIZE;
  REAL alpha = 0.0543;
  REAL tol = 0.0000000001;
  REAL relax = 1.0;
  int mits = 5000;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  if (argc == 2) {
    sscanf(argv[1], "%ld", &n);
    m = n;
  } else if (argc == 3) {
    sscanf(argv[1], "%ld", &n);
    sscanf(argv[2], "%ld", &m);
  } else if (argc == 4) {
    sscanf(argv[1], "%ld", &n);
    sscanf(argv[2], "%ld", &m);
    sscanf(argv[3], "%g", &alpha);
  } else if (argc == 5) {
    sscanf(argv[1], "%ld", &n);
    sscanf(argv[2], "%ld", &m);
    sscanf(argv[3], "%g", &alpha);
    sscanf(argv[4], "%g", &tol);
  } else if (argc == 6) {
    sscanf(argv[1], "%ld", &n);
    sscanf(argv[2], "%ld", &m);
    sscanf(argv[3], "%g", &alpha);
    sscanf(argv[4], "%g", &tol);
    sscanf(argv[5], "%g", &relax);
  } else if (argc == 7) {
    sscanf(argv[1], "%ld", &n);
    sscanf(argv[2], "%ld", &m);
    sscanf(argv[3], "%g", &alpha);
    sscanf(argv[4], "%g", &tol);
    sscanf(argv[5], "%g", &relax);
    sscanf(argv[6], "%d", &mits);
  } else {
    /* the rest of arg ignored */
    fprintf(stderr, "Usage: jacobi [<n> <m> <alpha> <tol> <relax> <mits>]\n");
    fprintf(stderr, "\tn - grid dimension in x direction, default: %ld\n", n);
    fprintf(stderr, "\tm - grid dimension in y direction, default: n if provided or %ld\n", m);
    fprintf(stderr, "\talpha - Helmholtz constant (always greater than 0.0), default: %g\n", alpha);
    fprintf(stderr, "\ttol   - error tolerance for iterative solver, default: %g\n", tol);
    fprintf(stderr, "\trelax - Successice over relaxation parameter, default: %g\n", relax);
    fprintf(stderr, "\tmits  - Maximum iterations for iterative solver, default: %d\n", mits);
  }
  REAL dx; /* grid spacing in x direction */
  REAL dy; /* grid spacing in y direction */
  dx = (2.0 / (n - 1));
  dy = (2.0 / (m - 1));

  REAL *u;
  REAL *f;
  REAL *umpi;
  REAL *fmpi;
  REAL *uptr;
  REAL *fptr;

  double elapsed_mpi = 0.0;
  double elapsed_seq = 0.0;
  MPI_Request request;

  int rows_to_process = n / numprocs;
  int rows_to_send = rows_to_process + 2;
  int rts;

  if (myrank == 0) {
    printf("jacobi %ld %ld %g %g %g %d\n", n, m, alpha, tol, relax, mits);
    printf("--------------------------------------------------------\n");
    /** init the array */

    u = (REAL *)malloc(sizeof(REAL) * n * m);
    f = (REAL *)malloc(sizeof(REAL) * n * m);

    umpi = (REAL *)malloc(sizeof(REAL) * n * m);
    fmpi = (REAL *)malloc(sizeof(REAL) * n * m);

    initialize(n, m, alpha, dx, dy, u, f);

    memcpy(umpi, u, n * m * sizeof(REAL));
    memcpy(fmpi, f, n * m * sizeof(REAL));

    printf("================ Sequential Execution ================\n");
    elapsed_seq = read_timer_ms();
    jacobi_seq(n, m, dx, dy, alpha, relax, u, f, tol, mits);
    elapsed_seq = read_timer_ms() - elapsed_seq;
    printf("\n");
  }

  MPI_Barrier(MPI_COMM_WORLD);
  /* TODO #1: process 0 performs data decomposition and distribution of the umpi
   *and fmpi arrays to
   * other processes using MPI_Send/Recv. MPI_Scatter may work, but the
   *recommendation is to implement using
   * MPI_Send/Recv first and then to see whether you can convert to MPI_Scatter.
   *
   * Row-wise decomposition should be used. Memory needs to be allocated for
   *holding the umpi and
   * fmpi data distributed to each process.
   *
   * Assuming N is dividable by numprocs. Considering the boundary row(s) that
   * need to be exchanged between processes, you will need to properly set the
   *address of MPI_Send/Recv
   * of of each process for sending data to its neighbour(s).
   * Some processes (0, numprocs-1) have only one neighbour, and others have two
   *neighbours.
   *
   */
  int temp = n / numprocs;
  int num_rows;

  if (myrank == 0 || myrank == numprocs - 1)
    num_rows = temp + 1;
  else
    num_rows = temp + 2;

  if (myrank == 0) {
    int send_to_rank;
    for (send_to_rank = 1; send_to_rank < numprocs; send_to_rank++) {
      int num_rows_to_send;
      if (send_to_rank == numprocs - 1)
        num_rows_to_send = temp + 1;
      else
        num_rows_to_send = temp + 2;

      uptr = umpi + (send_to_rank * temp - 1) * m;
      fptr = umpi + (send_to_rank * temp - 1) * m;
      /*
            MPI_Send(uptr, rts * m, MPI_FLOAT, send_to_rank, send_to_rank,
                     MPI_COMM_WORLD);
            MPI_Send(fptr, rts * m, MPI_FLOAT, send_to_rank, send_to_rank,
                     MPI_COMM_WORLD);
      */

      MPI_Isend(uptr, num_rows_to_send * m, MPI_FLOAT, send_to_rank, send_to_rank, MPI_COMM_WORLD, &request);
      MPI_Isend(fptr, num_rows_to_send * m, MPI_FLOAT, send_to_rank, send_to_rank, MPI_COMM_WORLD, &request);
    }
  } else {
    umpi = malloc(num_rows * m * sizeof(REAL));
    fmpi = malloc(num_rows * m * sizeof(REAL));
    MPI_Irecv(umpi, num_rows * m, MPI_FLOAT, 0, myrank, MPI_COMM_WORLD, &request);
    MPI_Irecv(fmpi, num_rows * m, MPI_FLOAT, 0, myrank, MPI_COMM_WORLD, &request);
    /*
    MPI_Recv(umpi, rts * m, MPI_FLOAT, 0, myrank, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    MPI_Recv(fmpi, rts * m, MPI_FLOAT, 0, myrank, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
             */
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (myrank == 0) {
    printf("================ Parallel MPI Execution (%d processes) ================\n", numprocs);
    elapsed_mpi = read_timer_ms();
  }
  /* TODO #2: perform jacobi iterative method for computation, check more
   *details in jacobi_mpi function
   *comments
   *
   * You do not need to use exactly the same function and its arguments,
   * feel free to adjust as you see fit
   */
  //// jacobi_mpi(n, m, dx, dy, alpha, relax, umpi, fmpi, tol, mits);

  MPI_Barrier(MPI_COMM_WORLD); /* this may be unnecessnary */
  if (myrank == 0) {
    elapsed_mpi = read_timer_ms() - elapsed_mpi;
    printf("\n");
  }

  /* TODO #3: Each process sends data (umpi) to process 0 using MPI_Send/Recv,
   * MPI_Gather may work as well. */
  MPI_Barrier(MPI_COMM_WORLD);

  if (myrank != 0) {
    int num_rows_to_send;
    if (myrank == numprocs - 1)
      num_rows_to_send = temp + 1;
    else
      num_rows_to_send = temp + 2;

    MPI_Isend(uptr, num_rows_to_send * m, MPI_FLOAT, 0, myrank, MPI_COMM_WORLD, &request);
  } else {
    int recv_rank;
    for (recv_rank = 1; recv_rank < numprocs; recv_rank++) {
      int num_rows_to_recv;
      if (recv_rank == numprocs - 1)
        num_rows_to_recv = temp + 1;
      else
        num_rows_to_recv = temp + 2;

      MPI_Irecv(umpi, num_rows_to_recv * m, MPI_FLOAT, recv_rank, 0, MPI_COMM_WORLD, &request);
    }
  }

  if (myrank == 0) {
#ifdef CORRECTNESS_CHECK
    print_array("Sequential Run", "u", (REAL *)u, n, m);
    print_array("MPI Parallel ", "umpi", (REAL *)umpi, n, m);
#endif

    double flops = mits * (n - 2) * (m - 2) * 13;

    printf("--------------------------------------------------------\n");
    printf("Performance:\tRuntime(ms)\tMFLOPS\t\tError\n");
    printf("--------------------------------------------------------\n");
    printf("base:\t\t%.2f\t\t%.2f\t\t%g\n", elapsed_seq, flops / (1.0e3 * elapsed_seq), 0.0);
    // error_check(n, m, alpha, dx, dy, u, f);
    printf("mpi(%d processes):\t%.2f\t\t%.2f\t\t%g\n", numprocs, elapsed_mpi, flops / (1.0e3 * elapsed_mpi), 0.0);
    // error_check(n, m, alpha, dx, dy, umpi, fmpi));

    free(u);
    free(f);
    // free(umpi);
    // free(fmpi);
  }
  MPI_Finalize();
  return 0;
}

/*      subroutine jacobi (n,m,dx,dy,alpha,omega,u,f,tol,mits)
 ******************************************************************
 * Subroutine HelmholtzJ
 * Solves poisson equation on rectangular grid assuming :
 * (1) Uniform discretization in each direction, and
 * (2) Dirichlect boundary conditions
 *
 * Jacobi method is used in this routine
 *
 * Input : n,m   Number of grid points in the X/Y directions
 *         dx,dy Grid spacing in the X/Y directions
 *         alpha Helmholtz eqn. coefficient
 *         omega Relaxation factor
 *         f(n,m) Right hand side function
 *         u(n,m) Dependent variable/Solution
 *         tol    Tolerance for iterative solver
 *         mits  Maximum number of iterations
 *
 * Output : u(n,m) - Solution
 *****************************************************************/
void jacobi_seq(long n, long m, REAL dx, REAL dy, REAL alpha, REAL omega, REAL *u_p, REAL *f_p, REAL tol, int mits) {
  long i, j, k;
  REAL error;
  REAL ax;
  REAL ay;
  REAL b;
  REAL resid;
  REAL uold[n][m];
  REAL(*u)[m] = (REAL(*)[m])u_p;
  REAL(*f)[m] = (REAL(*)[m])f_p;
  /*
   * Initialize coefficients */
  /* X-direction coef */
  ax = (1.0 / (dx * dx));
  /* Y-direction coef */
  ay = (1.0 / (dy * dy));
  /* Central coeff */
  b = (((-2.0 / (dx * dx)) - (2.0 / (dy * dy))) - alpha);
  error = (10.0 * tol);
  k = 1;
  while ((k <= mits) && (error > tol)) {
    error = 0.0;

    /* Copy new solution into old */
    for (i = 0; i < n; i++)
      for (j = 0; j < m; j++) uold[i][j] = u[i][j];

    for (i = 1; i < (n - 1); i++)
      for (j = 1; j < (m - 1); j++) {
        resid =
            (ax * (uold[i - 1][j] + uold[i + 1][j]) + ay * (uold[i][j - 1] + uold[i][j + 1]) + b * uold[i][j] - f[i][j]) / b;
        // printf("i: %d, j: %d, resid: %f\n", i, j, resid);

        u[i][j] = uold[i][j] - omega * resid;
        error = error + resid * resid;
      }
    /* Error check */
    if (k % 500 == 0) printf("Finished %ld iteration with error: %g\n", k, error);
    error = sqrt(error) / (n * m);

    k = k + 1;
  } /*  End iteration loop */
  printf("Total Number of Iterations: %ld\n", k);
  printf("Residual: %.15g\n", error);
}

/**
 * TODO #2: MPI implementation of Jacobi methods using row-wise data
 *distribution. For each iteration of the
 *while loop,
 * each process needs to exchange boundary data with its neighbor(s) using
 *MPI_Send and MPI_Recv, and then
 *perform computation.
 * You can also use MPI_Isend and MPI_Irecv for boundary exchange.
 *
 * In each iteration, error computed by each process needs to be reduced using
 *add ops by process 0 and
 * then broadcasted to all processes, each of which will then check whether they
 *need terminate the while
 *loop. You can use
 * MPI_Allreduce or MPI_Reduce+MPI_Bcast to do that.
 *
 * printf calls should only be performed by process 0
 *
 * You do not need to use the same arguments, feel free to change as you think
 *reasonable.
 *
 * Since this function is called by each process who computes u subarray(the
 *portion of u array distributed
 *from process 0),
 * u_p and f_p is actually the pointer of the subarray, and n and m are for the
 *size of the subarray.
 */
void jacobi_mpi(long n, long m, REAL dx, REAL dy, REAL alpha, REAL omega, REAL *u_p, REAL *f_p, REAL tol, int mits) {
  long i, j, k;
  REAL error, gerror;
  REAL ax;
  REAL ay;
  REAL b;
  REAL resid;
  REAL uold[n][m];
  REAL(*u)[m] = (REAL(*)[m])u_p;
  REAL(*f)[m] = (REAL(*)[m])f_p;

  //// recompute temp and num of rows from above
  /*
   * Initialize coefficients */
  /* X-direction coef */
  ax = (1.0 / (dx * dx));
  /* Y-direction coef */
  ay = (1.0 / (dy * dy));
  /* Central coeff */
  b = (((-2.0 / (dx * dx)) - (2.0 / (dy * dy))) - alpha);
  error = (10.0 * tol);
  k = 1;
  while ((k <= mits) && (gerror > tol)) {
    error = 0.0;

    /* Copy new solution into old */
    /* TODO #2.a: since u and f are pointers to the subarray that contains
     * boundary data, you will need to
     * adjust the
     * start and end iteration of i and j depending on whether the process has
     * one neighbour or two
     * neighbours.
     * You need to do similar for the next for loops.
     */
    for (i = 0; i < n; i++)
      for (j = 0; j < m; j++) uold[i][j] = u[i][j];
    /* TODO #2.b: boundary exchange with neighbour process(es) using
     * MPI_Send/Recv.
     * The memory address of the boundary data should be correctly specified.
     */
    for (i = 1; i < (n - 1); i++)
      for (j = 1; j < (m - 1); j++) {
        resid =
            (ax * (uold[i - 1][j] + uold[i + 1][j]) + ay * (uold[i][j - 1] + uold[i][j + 1]) + b * uold[i][j] - f[i][j]) / b;
        // printf("i: %d, j: %d, resid: %f\n", i, j, resid);

        u[i][j] = uold[i][j] - omega * resid;
        error = error + resid * resid;
      }

    /* TODO #2.c: compute the global error using MPI_Allreduce or
     * MPI_Reduce+MPI_Bcast
     */
    MPI_Allreduce(&error, &gerror, numprocs, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

    if (myrank == 0) {
      if (k % 500 == 0) printf("Finished %ld iteration with error: %g\n", k, error);

      gerror = sqrt(gerror) / (n * m);
      k = k + 1;
    }
  } /*  End iteration loop */
  printf("Total Number of Iterations: %ld\n", k);
  printf("Residual: %.15g\n", gerror);
}
