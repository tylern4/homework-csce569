#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>

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
#define VECTOR_LENGTH 512

/* initialize a vector with random floating point numbers */
void init(REAL A[], int N) {
  int i;
  for (i = 0; i < N; i++) {
    A[i] = (double)drand48();
  }
}

/* C[N][M] = A[N][K] * B[K][M] */
void mm(int N, int K, int M, REAL *A, REAL *B, REAL *C, int A_rowMajor,
        int B_rowMajor);

/**
 * To compile: gcc mm.c -o mm
 */
int main(int argc, char *argv[]) {
  int N = VECTOR_LENGTH;
  int M = N;
  int K = N;
  double elapsed; /* for timing */
  if (argc < 4) {
    fprintf(stderr, "Usage: mm [<N(%d)>] [<K(%d)>] [<M(%d)>] \n", N, K, M);
    fprintf(stderr, "\n\tC[N][M] = A[N][K] * B[K][M]\n");
    fprintf(stderr, "\n\tExample: ./mm %d %d %d (default)\n", N, K, M);
  } else {
    N = atoi(argv[1]);
    K = atoi(argv[2]);
    M = atoi(argv[3]);
  }

  REAL *A = malloc(sizeof(REAL) * N * K);
  REAL *B = malloc(sizeof(REAL) * K * M);
  REAL *C = malloc(sizeof(REAL) * N * M);
  REAL *D = malloc(sizeof(REAL) * N);

  srand48((1 << 12));
  init(A, N * K);
  init(B, K * M);

  /* example run */
  double elapsed_mm_RR = read_timer();
  mm(N, K, M, A, B, C, 1, 1);
  elapsed_mm_RR = (read_timer() - elapsed_mm_RR);

  double elapsed_mm_RC = read_timer();
  mm(N, K, M, A, B, C, 1, 0);
  elapsed_mm_RC = (read_timer() - elapsed_mm_RC);

  double elapsed_mm_CR = read_timer();
  mm(N, K, M, A, B, C, 0, 1);
  elapsed_mm_CR = (read_timer() - elapsed_mm_CR);

  double elapsed_mm_CC = read_timer();
  mm(N, K, M, A, B, C, 0, 0);
  elapsed_mm_CC = (read_timer() - elapsed_mm_CC);

  /* you should add the call to each function and time the execution */
  printf("=================================================================\n");
  printf("\tC[%d][%d] = A[%d][%d] * B[%d][%d]\n", N, M, N, K, K, M);
  printf("-----------------------------------------------------------------\n");
  printf("Performance:\t\t\tRuntime (ms)\t MFLOPS \n");
  printf("-----------------------------------------------------------------\n");
  printf("mm row row:\t\t\t\t%4f\t%4f\n", elapsed_mm_RR * 1.0e3,
         M * N * K / (1.0e6 * elapsed_mm_RR));
  printf("mm row col:\t\t\t\t%4f\t%4f\n", elapsed_mm_RC * 1.0e3,
         M * N * K / (1.0e6 * elapsed_mm_RC));
  printf("mm col row:\t\t\t\t%4f\t%4f\n", elapsed_mm_CR * 1.0e3,
         M * N * K / (1.0e6 * elapsed_mm_CR));
  printf("mm col col:\t\t\t\t%4f\t%4f\n", elapsed_mm_CC * 1.0e3,
         M * N * K / (1.0e6 * elapsed_mm_CC));
  
  
  
  free(A);
  free(B);
  free(C);
  return 0;
}

void mm(int N, int K, int M, REAL *A, REAL *B, REAL *C, int A_rowMajor,
        int B_rowMajor) {
  int i, j, w;
  // Both rowmajor
  if (A_rowMajor > 0 && B_rowMajor > 0) {
    for (i = 0; i < N; i++)
      for (j = 0; j < M; j++) {
        REAL temp = 0.0;
        for (w = 0; w < K; w++)
          temp += A[i * K + w] * B[w * M + j];
        C[i * M + j] = temp;
      }
  }
  // A columbmajor B rowmajor
  // swap N <-> K
  else if (A_rowMajor == 0 && B_rowMajor > 0) {
    for (i = 0; i < K; i++)
      for (j = 0; j < M; j++) {
        REAL temp = 0.0;
        for (w = 0; w < N; w++)
          temp += A[i * N + w] * B[w * M + j];
        C[i * M + j] = temp;
      }
  }
  // A rowmajor B columbmajor
  // swap M <-> K
  else if (A_rowMajor > 0 && B_rowMajor == 0) {
    for (i = 0; i < N; i++)
      for (j = 0; j < K; j++) {
        REAL temp = 0.0;
        for (w = 0; w < M; w++)
          temp += A[i * K + w] * B[w * K + j];
        C[i * M + j] = temp;
      }
  }
  // Both columbmajor
  // ???
  else if (A_rowMajor == 0 && B_rowMajor == 0) {
    for (i = 0; i < N; i++)
      for (j = 0; j < M; j++) {
        REAL temp = 0.0;
        for (w = 0; w < K; w++)
          temp += A[i * K + w] * B[w * M + j];
        C[i * M + j] = temp;
      }
  }
}
