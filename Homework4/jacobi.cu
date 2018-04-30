#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/timeb.h>
#define REAL float
#define BLOCK_SIZE 16

/* compile the program using the following command
 *    nvcc jacobi.cu -lpthread -o jacobi
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
 * Modified: Sanjiv Shah,       Kuck and Associates, Inc. (KAI), 1998
 * Author:   Joseph Robicheaux, Kuck and Associates, Inc. (KAI), 1998
 *
 * This c version program is translated by
 * Chunhua Liao, University of Houston, Jan, 2005
 *
 * Directives are used in this code to achieve parallelism.
 * All do loops are parallelized with default 'static' scheduling.
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
void initialize(long n, long m, REAL alpha, REAL *dx, REAL *dy, REAL *u_p, REAL *f_p) {
  long i;
  long j;
  long xx;
  long yy;
  REAL(*u)[m] = (REAL(*)[m])u_p;
  REAL(*f)[m] = (REAL(*)[m])f_p;

  // double PI=3.1415926;
  *dx = (2.0 / (n - 1));
  *dy = (2.0 / (m - 1));
  /* Initialize initial condition and RHS */
  //#pragma omp parallel for private(xx,yy,j,i)
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++) {
      xx = ((int)(-1.0 + (*dx * (i - 1))));
      yy = ((int)(-1.0 + (*dy * (j - 1))));
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
// REAL(*f)[m] = (REAL(*)[m])f_p;
#pragma omp parallel for private(xx, yy, temp, j, i) reduction(+ : error)
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
void jacobi_cuda(long n, long m, REAL dx, REAL dy, REAL alpha, REAL relax, REAL *u_p, REAL *f_p, REAL tol, int mits);

int main(int argc, char *argv[]) {
  long n = DEFAULT_DIMSIZE;
  long m = DEFAULT_DIMSIZE;
  REAL alpha = 0.0543;
  REAL tol = 0.0000000001;
  REAL relax = 1.0;
  int mits = 5000;

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
    fprintf(stderr, "Usage: jacobi [<n> <m> <alpha> <tol> <relax> <mits>]\n");
    fprintf(stderr, "\tn - grid dimension in x direction, default: %ld\n", n);
    fprintf(stderr, "\tm - grid dimension in y direction, default: n if provided or %ld\n", m);
    fprintf(stderr, "\talpha - Helmholtz constant (always greater than 0.0), default: %g\n", alpha);
    fprintf(stderr, "\ttol   - error tolerance for iterative solver, default: %g\n", tol);
    fprintf(stderr, "\trelax - Successice over relaxation parameter, default: %g\n", relax);
    fprintf(stderr, "\tmits  - Maximum iterations for iterative solver, default: %d\n", mits);
  }

  printf("jacobi %ld %ld %g %g %g %d\n", n, m, alpha, tol, relax, mits);
  printf("---------------------------------------------------------------\n");
  /** init the array */

  REAL *u = (REAL *)malloc(sizeof(REAL) * n * m);
  REAL *f = (REAL *)malloc(sizeof(REAL) * n * m);

  REAL *ucuda = (REAL *)malloc(sizeof(REAL) * n * m);
  REAL *fcuda = (REAL *)malloc(sizeof(REAL) * n * m);

  REAL dx; /* grid spacing in x direction */
  REAL dy; /* grid spacing in y direction */

  initialize(n, m, alpha, &dx, &dy, u, f);

  memcpy(ucuda, u, n * m * sizeof(REAL));
  memcpy(fcuda, f, n * m * sizeof(REAL));

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  // Free speed
  REAL *cuda_temp = (REAL *)malloc(sizeof(REAL));
  cudaMalloc(&cuda_temp, (sizeof(REAL)));

  printf("===================== Sequential Execution =====================\n");
  double elapsed_seq = read_timer_ms();
  jacobi_seq(n, m, dx, dy, alpha, relax, u, f, tol, mits);
  elapsed_seq = read_timer_ms() - elapsed_seq;
  printf("\n");

  printf("===================== GPU CUDA Execution =====================\n");

  double elapsed_cuda = read_timer_ms();
  jacobi_cuda(n, m, dx, dy, alpha, relax, ucuda, fcuda, tol, mits);
  elapsed_cuda = read_timer_ms() - elapsed_cuda;
  printf("\n");

#if CORRECTNESS_CHECK
  print_array("Sequential Run", "u", (REAL *)u, n, m);
  print_array("GPU Run       ", "ucuda", (REAL *)ucuda, n, m);
#endif

  double flops = mits * (n - 2) * (m - 2) * 13;
  printf("---------------------------------------------------------------\n");
  printf("Performance:\tRuntime(ms)\tMFLOPS\t\tError\n");
  printf("---------------------------------------------------------------\n");
  printf("base:\t\t%.2f\t\t%.2f\t\t%g\n", elapsed_seq, flops / (1.0e3 * elapsed_seq),
         error_check(n, m, alpha, dx, dy, u, f));
  printf("GPU :\t\t%.2f\t\t%.2f\t\t%g\n", elapsed_cuda, flops / (1.0e3 * elapsed_cuda),
         error_check(n, m, alpha, dx, dy, ucuda, fcuda));

  free(u);
  free(f);
  free(ucuda);
  free(fcuda);
  cudaFree(cuda_temp);

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
 * TODO #1: jacobi_kernel implementation of the double-nested loop for
 * computation
 */
//__constant__ REAL *cuda_f;
__global__ void jacobi_kernel(long n, long m, REAL ax, REAL ay, REAL b, REAL omega, REAL *u, REAL *uold, REAL *resid,
                              REAL *cuda_f) {
  long i, j;
  i = blockIdx.x * blockDim.x + threadIdx.x;
  j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i == 0 || j == 0) return;
  if (i >= (n - 1) || j >= (m - 1)) return;

  // if (i > 0 && i < (n - 1)) {
  //  if (i > 0 && j < (m - 1)) {
  resid[i * n + j] = (ax * (uold[(i - 1) * n + j] + uold[(i + 1) * n + j]) +
                      ay * (uold[i * n + (j - 1)] + uold[i * n + (j + 1)]) + b * uold[i * n + j] - cuda_f[i * n + j]) /
                     b;
  u[i * n + j] = uold[i * n + j] - omega * resid[i * n + j];
  //  }
  //}
}
void jacobi_cuda(long n, long m, REAL dx, REAL dy, REAL alpha, REAL omega, REAL *u_p, REAL *f_p, REAL tol, int mits) {
  long i, j, k;
  REAL error;
  REAL ax;
  REAL ay;
  REAL b;
  REAL *resid = (REAL *)malloc((sizeof(REAL) * n * m));
  REAL *uold = (REAL *)malloc((sizeof(REAL) * n * m));
  REAL *temp;
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
  /* TODO #2: CUDA memory allocation for u, f and uold and copy data for u and f
   * from host memory to GPU memory, depending on how error
   * will be calculated (see below), a [n][m] array or a one-element array need
   * to be allocated as well. */
  int size = (sizeof(REAL) * n * m);
  REAL *cuda_u;
  REAL *cuda_f;
  REAL *cuda_uold;
  REAL *cuda_resid;

  // Copy u to cuda memory
  cudaMalloc((void **)&cuda_u, size);
  cudaMemcpy(cuda_u, u, size, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&cuda_f, size);
  cudaMemcpy(cuda_f, f, size, cudaMemcpyHostToDevice);

  cudaMalloc((void **)&cuda_uold, size);
  cudaMalloc((void **)&cuda_resid, size);

  /* TODO #4: set 16x16 threads/block and n/16 x m/16 blocks/grid for GPU
   * computation (assuming n and m are dividable by 16 */
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(n / dimBlock.x, m / dimBlock.y);

  while ((k <= mits) && (error > tol)) {
    error = 0.0;

    /* TODO #3: swap u and uold */
    temp = cuda_u;
    cuda_u = cuda_uold;
    cuda_uold = temp;
    /* TODO #5: launch jacobi_kernel */
    jacobi_kernel << <dimGrid, dimBlock>>> (n, m, ax, ay, b, omega, cuda_u, cuda_uold, cuda_resid, cuda_f);
    /* TODO #6: compute error on CPU or GPU. error is calculated by accumulating
    *          resid*resid computed by each thread. There are multiple
    * approaches to compute the error. E.g. 1). A array of resid[n][m]
    *          could be allocated and store the resid computed by each thread.
    * After the computation, all the resids in the array are
    *          accumulated on either CPU or GPU. 2). A simpler implementation
    * could be just using CUDA atomicAdd, check.
    *
 (http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
    */
    /* Error check */
    /* TODO #7: copy the error from GPU to CPU */
    cudaMemcpy(resid, cuda_resid, size, cudaMemcpyDeviceToHost);
    for (i = 1; i < (n - 1); i++)
      for (j = 1; j < (m - 1); j++) {
        error = error + resid[i * n + j] * resid[i * n + j];
      }

    if (k % 500 == 0) printf("Finished %ld iteration with error: %g\n", k, error);
    error = sqrt(error) / (n * m);
    k = k + 1;
  } /*  End iteration loop */
    /* TODO #8: GPU memory deallocation */
  cudaFree(cuda_u);
  cudaFree(cuda_f);
  cudaFree(cuda_uold);
  cudaFree(cuda_resid);
  printf("Total Number of Iterations: %ld\n", k);
  printf("Residual: %.15g\n", error);
}
