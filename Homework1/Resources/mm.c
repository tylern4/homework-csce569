#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/timeb.h>
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
#define VECTOR_LENGTH 512

/* initialize a vector with random floating point numbers */
void init(REAL A[], int N) {
    int i;
    for (i = 0; i < N; i++) {
        A[i] = (double) drand48();
    }
}
    
int num_ths = 4;

void mm(int N, int K, int M, REAL * A, REAL * B, REAL * C); 
void mm_omp_parallel(int N, int K, int M, REAL * A, REAL * B, REAL * C); 
void mm_omp_parallel_for(int N, int K, int M, REAL * A, REAL * B, REAL * C); 

/**
 * To compile: gcc mm.c -fopenmp -o mm
 */
int main(int argc, char *argv[]) {
    int N = VECTOR_LENGTH;
    int M = N;
    int K = N;
    double elapsed; /* for timing */
    if (argc < 5) {
        fprintf(stderr, "Usage: mm [<N(%d)>] <K(%d) [<M(%d)>] [<num_threads(%d)]\n", N,K,M, num_ths);
        fprintf(stderr, "\t Example: ./mm %d %d %d %d (default)\n", N,K,M,num_ths);
    } else {
    	N = atoi(argv[1]);
    	K = atoi(argv[2]);
    	M = atoi(argv[3]);
    	num_ths = atoi(argv[4]);
    }
//    printf("\tC[%d][%d] = A[%d][%d] * B[%d][%d] with %d threads\n", N, M, N, K, K, M, num_ths);
    REAL * A = malloc(sizeof(REAL)*N*K);
    REAL * B = malloc(sizeof(REAL)*K*M);
    REAL * C = malloc(sizeof(REAL)*N*M);

    srand48((1 << 12));
    init(A, N*K);
    init(B, K*M);

    /* example run */
    double elapsed_mm = read_timer();
    mm(N, K, M, A, B, C);
    elapsed_mm  = (read_timer() - elapsed_mm);
    
    double elapsed_mm_omp_parallel = read_timer();
    mm_omp_parallel(N, K, M, A, B, C);
    elapsed_mm_omp_parallel  = (read_timer() - elapsed_mm_omp_parallel);

    double elapsed_mm_omp_parallel_for = read_timer();
    mm_omp_parallel_for(N, K, M, A, B, C);
    elapsed_mm_omp_parallel_for  = (read_timer() - elapsed_mm_omp_parallel_for);

    /* you should add the call to each function and time the execution */
    printf("======================================================================================================\n");
    printf("\tC[%d][%d] = A[%d][%d] * B[%d][%d] with %d threads for OpenMP\n", N, M, N, K, K, M, num_ths);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\t\tRuntime (ms)\t MFLOPS \n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("mm:\t\t\t\t%4f\t%4f\n",  elapsed_mm * 1.0e3, M*N*K / (1.0e6 *  elapsed_mm));
    printf("mm_omp_parallel:\t\t%4f\t%4f\n",  elapsed_mm_omp_parallel * 1.0e3, M*N*K / (1.0e6 *  elapsed_mm_omp_parallel));
    printf("mm_omp_paralell_for:\t\t%4f\t%4f\n",  elapsed_mm_omp_parallel * 1.0e3, M*N*K / (1.0e6 *  elapsed_mm_omp_parallel_for));
    
    free(A);
    free(B);
    free(C);
    return 0;
}

void mm(int N, int K, int M, REAL * A, REAL * B, REAL * C) {
    int i, j, w;
    for (i=0; i<N; i++) 
        for (j=0; j<M; j++) {
	    REAL temp = 0.0;
	    for (w=0; w<K; w++) 
	        temp += A[i*K+w]*B[w*M+j];
	    C[i*M+j] = temp;
	}
}

void mm_omp_parallel(int N, int K, int M, REAL * A, REAL * B, REAL * C) {
}

void mm_omp_parallel_for(int N, int K, int M, REAL * A, REAL * B, REAL * C) {
}
