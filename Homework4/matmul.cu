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

/* read timer in second */
double read_timer() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time + (double) tm.millitm / 1000.0;
}

/* read timer in ms */
double read_timer_ms() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time * 1000.0 + (double) tm.millitm;
}

#define REAL float

void init(int M, int N, REAL * A) {
    int i, j;

    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            A[i*N+j] = (REAL) drand48();
        }
    }
}

double maxerror(int M, int N, REAL * A, REAL *B) {
    int i, j;
    double error = 0.0;

    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            double diff = (A[i*N+j] - B[i*N+j]) / A[i*N+j];
            if (diff < 0)
                diff = -diff;
            if (diff > error)
                error = diff;
        }
    }
    return error;
}

void matmul_base(int N, REAL *A, REAL * B, REAL *C);
void matmul_openmp(int N, REAL *A, REAL *B, REAL *C, int num_tasks);
void matmul_cuda_v1_vanilla(int N, REAL *A, REAL *B, REAL *C);
void matmul_cuda_v1_shmem(int N, REAL *A, REAL *B, REAL *C);
void matmul_cuda_v1_cublas(int N, REAL *A, REAL *B, REAL *C);

int main(int argc, char *argv[]) {
    int N;
    int num_tasks = 5; /* 5 is default number of tasks */
    double elapsed_base, elapsed_openmp;
    //double elapsed_cuda_v1, elapsed_cuda_v2, elapsed_cuda_v3; /* for timing */
    if (argc < 2) {
        fprintf(stderr, "Usage: matmul <n> [<#tasks(%d)>]\n", num_tasks);
        exit(1);
    }
    N = atoi(argv[1]);
    if (argc > 2) num_tasks = atoi(argv[2]);
    REAL * heap_buffer = (REAL*)malloc(sizeof(REAL)*N*N*4); /* we use 5 matrix in this example */
    /* below is a cast from memory buffer to a 2-d row-major array */
    REAL *A = heap_buffer;
    REAL *B = &heap_buffer[N*N];
    REAL *C_base = &heap_buffer[2*N*N];
    REAL *C_openmp = &heap_buffer[3*N*N];

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
    //TODO: call and time for matmul_cuda_v1_vanilla(int N, REAL *A, REAL *B, REAL *C);

    //TODO: call and time for matmul_cuda_v1_shmem(int N, REAL *A, REAL *B, REAL *C);

    //TODO: call and time for matmul_cuda_v1_cublas(int N, REAL *A, REAL *B, REAL *C);

    printf("======================================================================================================\n");
    printf("Matrix Multiplication: A[M][K] * B[k][N] = C[M][N], M=K=N=%d, %d threads/tasks\n", N, num_tasks);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\tRuntime (ms)\t MFLOPS \t\tError (compared to base)\n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("matmul_base:\t\t%4f\t%4f \t\t%g\n", elapsed_base * 1.0e3, ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_base)), maxerror(N, N, C_base, C_base));
    printf("matmul_openmp:\t\t%4f\t%4f \t\t%g\n", elapsed_openmp * 1.0e3, ((((2.0 * N) * N) * N) / (1.0e6 * elapsed_openmp)), maxerror(N, N, C_base, C_openmp));
    /* TODO: put other printf statements for outputing results for GPU execution */
    free(heap_buffer);
    return 0;
}

void matmul_base(int N, REAL *A, REAL * B, REAL *C) {
    int i, j, k;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            REAL temp = 0.0;
            for (k = 0; k < N; k++) {
                temp += A[i*N+k] * B[k*N+j];
            }
            C[i*N+j] = temp;
        }
    }
}

void matmul_openmp(int N, REAL *A, REAL *B, REAL *C, int num_tasks) {
    int i, j, k;
#pragma omp parallel for shared(N,A,B,C,num_tasks) private(i,j,k) num_threads(num_tasks)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            REAL temp = 0.0;
            for (k = 0; k < N; k++) {
                temp += A[i*N+k] * B[k*N+j];
            }
            C[i*N+j] = temp;
        }
    }
}

/** 
  * TODO: kernel implementation 
  */
__global__ matmul_cuda_v1_vanilla_kernel( ... ) {

}
/*
 * call to kernel that uses GPU global memory
 */
void matmul_cuda_v1_vanilla(int N, REAL *A, REAL *B, REAL *C) {

}

/** 
  * TODO: kernel implementation 
  */
__global__ matmul_cuda_v2_shmem_kernel( ... ) {

}
/*
 * call to kernel that use GPU shared memory
 */
void matmul_cuda_v2_shmem(int N, REAL *A, REAL *B, REAL *C) {

}

/*
 * call to sgemm of cublas library 
 */
void matmul_cuda_v3_cublas(int N, REAL *A, REAL *B, REAL *C) {

}
