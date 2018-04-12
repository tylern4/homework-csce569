/*
 * Rectangular matrix multiplication
 * A[M][K] * B[k][N] = C[M][N]
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/timeb.h>
#include <string.h>
#include <cublas_v2.h>

#define REAL float
#define BLOCK_SIZE 16

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

void init(int M, int N, REAL *A) {
  int i, j;

  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      A[i * N + j] = (REAL)drand48();
    }
  }
}

double maxerror(int M, int N, REAL *A, REAL *B) {
  int i, j;
  double error = 0.0;

  for (i = 0; i < M; i++) {
    for (j = 0; j < N; j++) {
      double diff = (A[i * N + j] - B[i * N + j]) / A[i * N + j];
      if (diff < 0)
        diff = -diff;
      if (diff > error)
        error = diff;
    }
  }
  return error;
}

void matmul_base(int N, REAL *A, REAL *B, REAL *C);
void matmul_openmp(int N, REAL *A, REAL *B, REAL *C, int num_tasks);
void matmul_cuda_v1_vanilla(int N, REAL *A, REAL *B, REAL *C);
void matmul_cuda_v2_shmem(int N, REAL *A, REAL *B, REAL *C);
void matmul_cuda_v3_cublas(int N, REAL *A, REAL *B, REAL *C);

int main(int argc, char *argv[]) {
  int N;
  int num_tasks = 5; /* 5 is default number of tasks */
  double elapsed_base, elapsed_openmp;
  double elapsed_cuda_v1, elapsed_cuda_v2, elapsed_cuda_v3; /* for timing */
  if (argc < 2) {
    fprintf(stderr, "Usage: matmul <n> [<#tasks(%d)>]\n", num_tasks);
    exit(1);
  }
  N = atoi(argv[1]);
  if (argc > 2)
    num_tasks = atoi(argv[2]);
  REAL *heap_buffer = (REAL *)malloc(sizeof(REAL) * N * N * 8);
  /* we use 5 matrix in this example */
  /* below is a cast from memory buffer to a 2-d row-major array */
  REAL *A = heap_buffer;
  REAL *B = &heap_buffer[N * N];
  REAL *C_base = &heap_buffer[2 * N * N];
  REAL *C_openmp = &heap_buffer[3 * N * N];
  REAL *C_v1 = &heap_buffer[4 * N * N];
  REAL *C_v2 = &heap_buffer[5 * N * N];
  REAL *C_v3 = &heap_buffer[6 * N * N];

  srand48((1 << 12));
  init(N, N, A);
  init(N, N, B);

  /* example run */
  elapsed_base = read_timer();
  matmul_base(N, A, B, C_base);
  elapsed_base = (read_timer() - elapsed_base);

  elapsed_openmp = read_timer();
  matmul_openmp(N, A, B, C_openmp, num_tasks);
  elapsed_openmp = (read_timer() - elapsed_openmp);

  /* call and timing for the three CUDA versions */
  // TODO: call and time for
  elapsed_cuda_v1 = read_timer();
  matmul_cuda_v1_vanilla(N, A, B, C_v1);
  elapsed_cuda_v1 = (read_timer() - elapsed_cuda_v1);

  // TODO: call and time for matmul_cuda_v1_shmem(int N, REAL *A, REAL *B, REAL
  // *C);
  elapsed_cuda_v2 = read_timer();
  matmul_cuda_v2_shmem(N, A, B, C_v2);
  elapsed_cuda_v2 = (read_timer() - elapsed_cuda_v2);

  // TODO: call and time for matmul_cuda_v1_cublas(int N, REAL *A, REAL *B, REAL
  // *C);
  elapsed_cuda_v3 = read_timer();
  matmul_cuda_v3_cublas(N, A, B, C_v3);
  elapsed_cuda_v3 = (read_timer() - elapsed_cuda_v3);

  printf("===============================================================\n");
  printf("Matrix Multiplication: A[M][K] * B[k][N] = C[M][N]\n");
  printf("\tM=K=N=%d, %d threads/tasks\n", N, num_tasks);
  printf("---------------------------------------------------------------\n");
  printf("Performance:\tRuntime (ms)\t MFLOPS\tError (compared to base)\n");
  printf("---------------------------------------------------------------\n");
  printf("matmul_base:\t%4f\t%4f\t%g\n", elapsed_base * 1.0e3,
         ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_base)),
         maxerror(N, N, C_base, C_base));
  printf("matmul_openmp:\t%4f\t%4f\t%g\n", elapsed_openmp * 1.0e3,
         ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_openmp)),
         maxerror(N, N, C_base, C_openmp));
  printf("matmul_cuda_v1:\t%4f\t%4f\t%g\n", elapsed_cuda_v1 * 1.0e3,
         ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_cuda_v1)),
         maxerror(N, N, C_base, C_v1));
  printf("matmul_cuda_v2:\t%4f\t%4f\t%g\n", elapsed_cuda_v2 * 1.0e3,
         ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_cuda_v2)),
         maxerror(N, N, C_base, C_v2));
  printf("matmul_cuda_v3:\t%4f\t%4f\t%g\n", elapsed_cuda_v3 * 1.0e3,
         ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_cuda_v3)),
         maxerror(N, N, C_base, C_v3));
  free(heap_buffer);
  return 0;
}

void matmul_base(int N, REAL *A, REAL *B, REAL *C) {
  int i, j, k;
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      REAL temp = 0.0;
      for (k = 0; k < N; k++) {
        temp += A[i * N + k] * B[k * N + j];
      }
      C[i * N + j] = temp;
    }
  }
}

void matmul_openmp(int N, REAL *A, REAL *B, REAL *C, int num_tasks) {
  int i, j, k;
#pragma omp parallel for shared(N, A, B, C, num_tasks) private(i, j, k)        \
    num_threads(num_tasks)
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      REAL temp = 0.0;
      for (k = 0; k < N; k++) {
        temp += A[i * N + k] * B[k * N + j];
      }
      C[i * N + j] = temp;
    }
  }
}

/**
  * TODO: kernel implementation
  */
__global__ void matmul_cuda_v1_vanilla_kernel(int N, REAL *A, REAL *B,
                                              REAL *C) {
  // Each thread computes one element of C
  // by accumulating results into Cvalue
  float Cvalue = 0;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  for (int e = 0; e < N; ++e)
    Cvalue += A[row * N + e] * B[e * N + col];

  C[row * N + col] = Cvalue;
}
/*
 * call to kernel that uses GPU global memory
 */
void matmul_cuda_v1_vanilla(int N, REAL *A, REAL *B, REAL *C) {
  // Determine size of matrix
  size_t size = N * N * sizeof(REAL);

  // Make cuda matricies
  REAL *cuda_A = (REAL *)malloc(size);
  REAL *cuda_B = (REAL *)malloc(size);
  REAL *cuda_C = (REAL *)malloc(size);

  // Copy A to cuda memory
  cudaMalloc(&cuda_A, size);
  cudaMemcpy(cuda_A, A, size, cudaMemcpyHostToDevice);

  // Copy B to cuda memory
  cudaMalloc(&cuda_B, size);
  cudaMemcpy(cuda_B, B, size, cudaMemcpyHostToDevice);

  // Allocate C to cuda memory
  cudaMalloc(&cuda_C, size);

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid(N / dimBlock.x, N / dimBlock.y);
  matmul_cuda_v1_vanilla_kernel << <dimGrid, dimBlock>>>
      (N, cuda_A, cuda_B, cuda_C);

  cudaMemcpy(C, cuda_C, size, cudaMemcpyDeviceToHost);

  cudaFree(cuda_A);
  cudaFree(cuda_B);
  cudaFree(cuda_C);
}

/**
  * TODO: kernel implementation
  */
__global__ void matmul_cuda_v2_shmem_kernel(int N, REAL *A, REAL *B, REAL *C) {}
/*
 * call to kernel that use GPU shared memory
 */
void matmul_cuda_v2_shmem(int N, REAL *A, REAL *B, REAL *C) {}

/*
 * call to sgemm of cublas library
 */
void matmul_cuda_v3_cublas(int N, REAL *A, REAL *B, REAL *C) {

  float alf = 1;
  float bet = 0;
  float *alpha = &alf;
  float *beta = &bet;

  // Determine size of matrix
  size_t size = N * N * sizeof(REAL);

  // Make cuda matricies
  REAL *cuda_A = (REAL *)malloc(size);
  REAL *cuda_B = (REAL *)malloc(size);
  REAL *cuda_C = (REAL *)malloc(size);

  // Copy A to cuda memory
  cudaMalloc(&cuda_A, size);
  cudaMemcpy(cuda_A, A, size, cudaMemcpyHostToDevice);

  // Copy B to cuda memory
  cudaMalloc(&cuda_B, size);
  cudaMemcpy(cuda_B, B, size, cudaMemcpyHostToDevice);

  // Allocate C to cuda memory
  cudaMalloc(&cuda_C, size);
  // Create a handle for CUBLAS
  cublasHandle_t handle;
  cublasCreate(&handle);

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, alpha, cuda_A, N,
              cuda_B, N, beta, cuda_C, N);

  cublasDestroy(handle);

  cudaMemcpy(C, cuda_C, size, cudaMemcpyDeviceToHost);

  cudaFree(cuda_A);
  cudaFree(cuda_B);
  cudaFree(cuda_C);
}
