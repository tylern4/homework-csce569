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
#define VECTOR_LENGTH 512

/* initialize a vector with random floating point numbers */
void init(REAL A[], int N) {
    int i;
    for (i = 0; i < N; i++) {
        A[i] = (double) drand48();
    }
}

/* A, B, and C are NxM matrix */
void matrix_addition(int N, int M, REAL * A, REAL * B, REAL * C, int A_rowMajor, int B_rowMajor);


/**
 * To compile: gcc assignment1.c -o assignment1
 * To run: ./assignment1 256 128
 *
 */
int main(int argc, char *argv[]) {
    int N = VECTOR_LENGTH;
    int M = M;
    double elapsed; /* for timing */
    if (argc < 3) {
        fprintf(stderr, "Usage: assignment1 <n> <m> (default %d)\n", N);
        exit(1);
    }
    N = atoi(argv[1]);
    M = atoi(argv[2]);
    //REAL A[N][M]; /* wrong, you should do the following */
    REAL * A = malloc(sizeof(REAL)*N*M);
    REAL * B = malloc(sizeof(REAL)*N*M);
    REAL * C = malloc(sizeof(REAL)*N*M);

    srand48((1 << 12));
    init(A, N*M);
    init(B, N*M);
    /* example run */
    double elapsed_matrixAdd_row_row = read_timer();
    matrix_addition(N, M, A, B, C, 1, 1);
    elapsed_matrixAdd_row_row  = (read_timer() - elapsed_matrixAdd_row_row);
    
    double elapsed_matrixAdd_row_col = read_timer();
    matrix_addition(N, M, A, B, C, 1, 0);
    elapsed_matrixAdd_row_col  = (read_timer() - elapsed_matrixAdd_row_col);
    
    double elapsed_matrixAdd_col_row = read_timer();
    matrix_addition(N, M, A, B, C, 0, 1);
    elapsed_matrixAdd_col_row = (read_timer() - elapsed_matrixAdd_col_row);
    
    double elapsed_matrixAdd_col_col = read_timer();
    matrix_addition(N, M, A, B, C, 0, 0);
    elapsed_matrixAdd_col_col = (read_timer() - elapsed_matrixAdd_col_col);
    
    free(A);
    free(B);
    free(C);

    /* you should add the call to each function and time the execution */
    printf("======================================================================================================\n");
    printf("\tN: %d, M: %d, K: %d\n", N, M, N);
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("Performance:\t\t\t\tRuntime (ms)\t MFLOPS \n");
    printf("------------------------------------------------------------------------------------------------------\n");
    printf("matrix addition row row:\t\t%4f\t%4f\n",  elapsed_matrixAdd_row_row * 1.0e3, 
		    M*N / (1.0e6 *  elapsed_matrixAdd_row_row));
    printf("matrix addition row col:\t\t%4f\t%4f\n",  elapsed_matrixAdd_row_col * 1.0e3, 
		    M*N / (1.0e6 *  elapsed_matrixAdd_row_col));
    printf("matrix addition col row:\t\t%4f\t%4f\n",  elapsed_matrixAdd_col_row * 1.0e3, 
		    M*N / (1.0e6 *  elapsed_matrixAdd_col_row));
    printf("matrix addition col col:\t\t%4f\t%4f\n",  elapsed_matrixAdd_col_col * 1.0e3, 
		    M*N / (1.0e6 *  elapsed_matrixAdd_col_col));
    return 0;
}

void matrix_addition(int N, int M, REAL * A, REAL * B, REAL * C, int A_rowMajor, int B_rowMajor) {
	if (A_rowMajor != 0 && B_rowMajor != 0) { /* A is row major, B is row major */
		int i, j;
		for (i=0; i<N; i++) 
			for (j=0; j<M; j++) {
				/* the offset of matrix A[i][j] in memory based on A */
			       int offset = i*M+j;
		       		C[offset] = A[offset] + B[offset];		       
			}
		return;
	}

	if (A_rowMajor != 0 && B_rowMajor == 0) { /* A is row major and B is col major */
		int i, j;
		for (i=0; i<N; i++) 
			for (j=0; j<M; j++) {
				/* the offset of matrix A[i][j] in memory based on A */
			       int rowMajor_offset = i*M+j;
			       int colMajor_offset = j*N+i;
		       		C[rowMajor_offset] = A[rowMajor_offset] + B[colMajor_offset];		       
			}
		return;
	}
}
