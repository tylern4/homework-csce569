/*
 * AXPY  Y[N] = Y[N] + a*X[N]
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>

#include <pthread.h>
#include <omp.h>

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
void axpy_openmp(int N, REAL Y[], REAL X[], REAL a);

int main(int argc, char *argv[]) {
    int N = VECTOR_LENGTH;
    double elapsed, elapsed_openmp; /* for timing */
    if (argc < 2) {
        fprintf(stderr, "Usage: axpy <n>\n");
        exit(1);
    }
    N = atoi(argv[1]);
    REAL a = 123.456;
    REAL Y_base[N];
    REAL Y_openmp[N];
    REAL X[N];

    srand48((1 << 12));
    init(X, N);
    init(Y_base, N);
    memcpy(Y_openmp, Y_base, N * sizeof(REAL));

    /* example run */
    elapsed = read_timer();
    axpy_base(N, Y_base, X, a);
    elapsed = (read_timer() - elapsed);

    elapsed_openmp = read_timer();
    axpy_openmp(N, Y_openmp, X, a);
    elapsed_openmp = (read_timer() - elapsed_openmp);
    
    int num_threads;
    #pragma omp parallel
    {
        #pragma omp master 
	num_threads = omp_get_num_threads();
    }

    /* you should add the call to each function and time the execution */
    printf("======================================================================================================\n");
    printf("\tAXPY: Y[N] = Y[N] + a*X[N], N=%d, %d threads for OpenMP\n", N, num_threads);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\tRuntime (ms)\t MFLOPS \t\tError (compared to base)\n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("axpy_base:\t\t%4f\t%4f \t\t%g\n", elapsed * 1.0e3, (2.0 * N) / (1.0e6 * elapsed), check(Y_base, Y_base, N));
    printf("axpy_openmp:\t\t%4f\t%4f \t\t%g\n", elapsed_openmp * 1.0e3, (2.0 * N) / (1.0e6 * elapsed_openmp), check(Y_base, Y_openmp, N));
    return 0;
}

void axpy_base(int N, REAL Y[], REAL X[], REAL a) {
    int i;
    for (i = 0; i < N; ++i)
        Y[i] += a * X[i];
}

void axpy_openmp(int N, REAL Y[], REAL X[], REAL a) {

}

void axpy_openmp_parallel_for(int N, REAL Y[], REAL X[], REAL a) {

}
