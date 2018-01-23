/*
 * Sum of a*X[N]
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>

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

REAL sum(int N, REAL X[], REAL a);
REAL sumaxpy(int N, REAL X[], REAL Y[], REAL a); 

int main(int argc, char *argv[]) {
    int N = VECTOR_LENGTH;
    double elapsed; /* for timing */
    if (argc < 2) {
        fprintf(stderr, "Usage: sum <n> (default %d)\n", N);
        exit(1);
    }
    N = atoi(argv[1]);
    REAL X[N];
    REAL Y[N];

    srand48((1 << 12));
    init(X, N);
    init(Y, N);
    REAL a = 0.1234;
    /* example run */
    elapsed = read_timer();
    REAL result = sum(N, X, a);
    elapsed = (read_timer() - elapsed);
    
    int num_ths = omp_get_num_threads();

    #pragma omp parallel 
    {
	#pragma omp master
        num_ths = omp_get_num_threads(); 
    }
    
    double elapsed_2 = read_timer();
    result = sumaxpy(N, X, Y, a);
    elapsed_2 = (read_timer() - elapsed_2);

    /* you should add the call to each function and time the execution */
    printf("======================================================================================================\n");
    printf("\tSum %d numbers\n", N);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\tRuntime (ms)\t MFLOPS \n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Sum:\t\t\t%4f\t%4f\n", elapsed * 1.0e3, 2*N / (1.0e6 * elapsed));
    printf("SumAXPY:\t\t\t%4f\t%4f\n", elapsed_2 * 1.0e3, 3*N / (1.0e6 * elapsed_2));
    return 0;
}

REAL sum(int N, REAL X[], REAL a) {
    int i;
    REAL result = 0.0;
    #pragma omp parallel for reduction(+:result)
    for (i = 0; i < N; ++i)
        result += a * X[i];
    return result;
}

REAL sum_reduce(int N, REAL X[], REAL a) {
    int i;
    REAL * results;
    int num_threads;
    #pragma omp parallel
    {
      #pragma omp master
      {
        num_threads = omp_get_num_threads();
        results = malloc(sizeof(REAL)*num_threads);
      }
      #pragma omp barrier
 
      int id = omp_get_thread_num();
      REAL tmp = 0.0;
      #pragma omp for
      for (i = 0; i < N; ++i)
        tmp += a * X[i];

      results[id] = tmp;
    }

    REAL tmp = 0;
    for (i=0; i<num_threads; i++)
      tmp += results[i];

    return tmp;
}

/*
 * sum: a*X[]+Y[]
 */
REAL sumaxpy(int N, REAL X[], REAL Y[], REAL a) {
    int i;
    REAL result = 0.0;
    for (i = 0; i < N; ++i)
        result += a * X[i] + Y[i];
    return result;
}
