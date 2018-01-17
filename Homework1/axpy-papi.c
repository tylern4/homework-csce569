/*
 * AXPY  Y[N] = Y[N] + a*X[N]
 *
 * compile with gcc axpy-papi.c -o axpy -lpapi
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>
#include "papi.h"

#include <pthread.h>

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
#define VECTOR_LENGTH 102400

/* initialize a vector with random floating point numbers */
void init(REAL A[], int N) {
    int i;
    for (i = 0; i < N; i++) {
        A[i] = (double) drand48();
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

void axpy_base(int N, REAL Y[], REAL X[], REAL a);

void axpy_base_sub(int i_start, int Nt, int N, REAL Y[], REAL X[], REAL a);

void axpy_dist(int N, REAL Y[], REAL X[], REAL a, int num_tasks);

void axpy_omp_parallel(int N, REAL Y[], REAL X[], REAL a, int num_tasks);
 
int main(int argc, char *argv[]) {
    int N = VECTOR_LENGTH;
    int num_tasks = 4; /* 4 is default number of tasks */
    double elapsed; /* for timing */
    double elapsed_dist; /* for timing */
    if (argc < 2) {
        fprintf(stderr, "Usage: axpy <n> [<#tasks(%d)>] (n should be dividable by #tasks)\n", num_tasks);
        exit(1);
    }
    N = atoi(argv[1]);
    if (argc > 2) num_tasks = atoi(argv[2]);
    REAL a = 123.456;
    REAL Y_base[N];
    REAL Y_dist[N];
    REAL X[N];


     #define NUM_EVENTS 3 
     long_long values[NUM_EVENTS];
     unsigned int Events[NUM_EVENTS]={PAPI_TOT_INS,PAPI_TOT_CYC, PAPI_L1_DCM};

    srand48((1 << 12));
    init(X, N);
    init(Y_base, N);
    memcpy(Y_dist, Y_base, N * sizeof(REAL));

    /* example run */
    elapsed = read_timer();
    PAPI_start_counters((int*)Events,NUM_EVENTS);
    axpy_base(N, Y_base, X, a);
    PAPI_stop_counters(values, NUM_EVENTS);
    elapsed = (read_timer() - elapsed);
    printf("INS: %d, CYC: %d, L1 Misses: %d, CPI: %f\n", values[0], values[1], values[2], ((double)values[1])/((double)values[0]));
 
    elapsed_dist = read_timer();
    axpy_omp_parallel(N, Y_dist, X, a, num_tasks);
    elapsed_dist = (read_timer() - elapsed_dist);

    /* you should add the call to each function and time the execution */
    printf("======================================================================================================\n");
    printf("\tAXPY: Y[N] = Y[N] + a*X[N], N=%d, %d tasks for dist\n", N, num_tasks);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\tRuntime (ms)\t MFLOPS \t\tError (compared to base)\n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("axpy_base:\t\t%4f\t%4f \t\t%g\n", elapsed * 1.0e3, (2.0 * N) / (1.0e6 * elapsed), check(Y_base, Y_base, N));
    printf("axpy_dist:\t\t%4f\t%4f \t\t%g\n", elapsed_dist * 1.0e3, (2.0 * N) / (1.0e6 * elapsed_dist), check(Y_base, Y_dist, N));
    return 0;
}

void axpy_base(int N, REAL Y[], REAL X[], REAL a) {
    int i;
    for (i = 0; i < N; ++i)
        Y[i] += a * X[i];
}

void axpy_base_sub(int i_start, int Nt, int N, REAL Y[], REAL X[], REAL a) {
    int i;
    for (i = i_start; i < i_start + Nt; ++i)
        Y[i] += a * X[i];
}

void axpy_dist(int N, REAL Y[], REAL X[], REAL a, int num_tasks) {
    int tid;
    for (tid = 0; tid < num_tasks; tid++) {
        int Nt, start;
	Nt = N/num_tasks;
	start = tid*Nt;
        axpy_base_sub(start, Nt, N, Y, X, a);
    }
}

/* replace the for loop for task decomposition with "omp parallel" */
void axpy_omp_parallel(int N, REAL Y[], REAL X[], REAL a, int num_tasks) {
}
