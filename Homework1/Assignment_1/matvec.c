/*
 * matrix vector multiplication: Y[] = A[][] * B[]
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/timeb.h>
#include <string.h>

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

#define REAL float

void zero(REAL A[], int n) {
  int i, j;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      A[i * n + j] = 0.0;
    }
  }
}

void init(int N, REAL A[]) {
  int i;
  for (i = 0; i < N; i++) {
    A[i] = (REAL)drand48();
  }
}

void mv(int N, int M, REAL *A, REAL *B, REAL *C, int A_rowMajor);

int main(int argc, char *argv[]) {
  int N, M = N = 512;

  if (argc < 3) {
    fprintf(stderr, "Usage: ./matvec <N>\n");
    fprintf(stderr, "\n\tY[M] = A[N x M] * B[N]\n");
    fprintf(stderr, "\n\tExample: ./matvec %d %d (default)\n", N, M);
  } else {
    N = atoi(argv[1]);
    M = atoi(argv[2]);
  }

  REAL A[N * M];
  REAL B[N];
  REAL C[M];
  /* more C matrix needed */

  srand48((1 << 12));
  init(N * M, (REAL *)A);
  init(N, B);

  double elapsed_mv_R = read_timer();
  mv(N, M, A, B, C, 1);
  elapsed_mv_R = (read_timer() - elapsed_mv_R);

  double elapsed_mv_C = read_timer();
  mv(N, M, A, B, C, 0);
  elapsed_mv_C = (read_timer() - elapsed_mv_C);

  /* you should add the call to each function and time the execution */
  printf("=================================================================\n");
  printf("Matrix Vector Multiplication: Y[%d] = A[%d x %d] * B[%d]\n", M, N, M,
         N);
  printf("-----------------------------------------------------------------\n");
  printf("Performance:\tRuntime (ms)\t MFLOPS \n");
  printf("-----------------------------------------------------------------\n");
  printf("mv_R:\t\t%4f\t%4f\n", elapsed_mv_R * 1.0e3,
         (2.0 * N * M) / (1.0e6 * elapsed_mv_R));
  printf("mv_C:\t\t%4f\t%4f\n", elapsed_mv_C * 1.0e3,
         (2.0 * N * M) / (1.0e6 * elapsed_mv_C));
  return 0;
}

void mv(int N, int M, REAL *A, REAL *B, REAL *C, int A_rowMajor) {
  int i, j;
  if (A_rowMajor > 0) {
    for (i = 0; i < M; i++) {        // Loop over rows
      REAL temp = 0.0;               // Create temp to for new element
      for (j = 0; j < N; j++) {      // Loop over cols
        temp += A[i * N + j] * B[j]; // Add the product of elements
      }
      C[i] = temp; // Put element into output vector
    }
  } else if (A_rowMajor == 0) {
    for (i = 0; i < M; i++) {
      REAL temp = 0.0;
      for (j = 0; j < N; j++) {
        temp += A[j * N + i] * B[j];
      }
      C[i] = temp;
    }
  }
}
