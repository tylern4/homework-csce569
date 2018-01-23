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
    return (double) tm.time + (double) tm.millitm / 1000.0;
}

/* read timer in ms */
double read_timer_ms() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time * 1000.0 + (double) tm.millitm;
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
        A[i] = (REAL) drand48();
    }
}

double check(REAL A[], REAL B[], int N) {
    int i;
    double sum = 0.0;
    for (i = 0; i < N; i++) {
        sum += A[i] - B[i];
    }
    return sum;
}

void matvec_base(int M, int N, REAL Y[], REAL A[][N], REAL B[]);
void matvec_omp_parallel(int M, int N, REAL Y[], REAL A[][N], REAL B[], int num_tasks);
void matvec_omp_parallel_for(int M, int N, REAL Y[], REAL A[][N], REAL B[], int num_tasks);

int main(int argc, char *argv[]) {
    int N;
    int num_tasks = 5; /* 5 is default number of tasks */
    double elapsed; /* for timing */
    double elapsed_omp_parallel;
    if (argc < 2) {
        fprintf(stderr, "Usage: matvec <n> [<#tasks(%d)>]\n", num_tasks);
        exit(1);
    }
    N = atoi(argv[1]);
    if (argc > 2) num_tasks = atoi(argv[2]);
    REAL A[N][N];
    REAL B[N];
    REAL Y_base[N];
    REAL Y_omp_parallel[N];
    /* more C matrix needed */

    srand48((1 << 12));
    init(N * N, (REAL *) A);
    init(N, B);

    /* example run */
    elapsed = read_timer();
    matvec_base(N, N, Y_base, A, B);
    elapsed = (read_timer() - elapsed);

    elapsed_omp_parallel = read_timer();
    matvec_omp_parallel(N, N, Y_omp_parallel, A, B, num_tasks);
    elapsed_omp_parallel = (read_timer() - elapsed_omp_parallel);

    /* you should add the call to each function and time the execution */
    printf("======================================================================================================\n");
    printf("\tMatrix Vector Multiplication: Y[N] = A[N][N] * B[N], N=%d, %d tasks for omp_parallel\n", N, num_tasks);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\tRuntime (ms)\t MFLOPS \t\tError (compared to base)\n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("matvec_base:\t\t%4f\t%4f \t\t%g\n", elapsed * 1.0e3, (2.0 * N * N) / (1.0e6 * elapsed), check(Y_base,Y_base, N));
    printf("matvec_omp_parallel:\t%4f\t%4f \t\t%g\n", elapsed_omp_parallel * 1.0e3, (2.0 * N * N) / (1.0e6 * elapsed_omp_parallel), check(Y_base, Y_omp_parallel, N));
    return 0;

}

void matvec_base(int M, int N, REAL Y[], REAL A[][N], REAL B[]) {
    int i, j;
    for (i = 0; i < M; i++) {
        REAL temp = 0.0;
        for (j = 0; j < N; j++) {
            temp += A[i][j] * B[j];
        }
        Y[i] = temp;
    }
}

void matvec_base_sub(int i_start, int Mt, int M, int N, REAL Y[], REAL A[][N], REAL B[]) {
    int i, j;
    for (i = i_start; i < i_start + Mt; i++) {
        REAL temp = 0.0;
        for (j = 0; j < N; j++) {
            temp += A[i][j] * B[j];
        }
        Y[i] = temp;
    }
}

void matvec_omp_parallel(int M, int N, REAL Y[], REAL A[][N], REAL B[], int num_tasks) {
    int tid;
    for (tid = 0; tid < num_tasks; tid++) {
        int Mt = N/num_tasks;
	int i_start = tid*Mt;
        matvec_base_sub(i_start, Mt, M, N, Y, A, B);
    }
}

void matvec_omp_parallel_for(int M, int N, REAL Y[], REAL A[][N], REAL B[], int num_tasks) {

}
