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
        REAL *C_RR = malloc(sizeof(REAL) * N * M);
        REAL *C_RC = malloc(sizeof(REAL) * N * M);
        REAL *C_CR = malloc(sizeof(REAL) * N * M);
        REAL *C_CC = malloc(sizeof(REAL) * N * M);

        srand48((1 << 12));
        init(A, N * K);
        init(B, K * M);

        /* example run */
        double elapsed_mm_RR = read_timer();
        mm(N, K, M, A, B, C_RR, 1, 1);
        elapsed_mm_RR = (read_timer() - elapsed_mm_RR);

        double elapsed_mm_RC = read_timer();
        mm(N, K, M, A, B, C_RC, 1, 0);
        elapsed_mm_RC = (read_timer() - elapsed_mm_RC);

        double elapsed_mm_CR = read_timer();
        mm(N, K, M, A, B, C_CR, 0, 1);
        elapsed_mm_CR = (read_timer() - elapsed_mm_CR);

        double elapsed_mm_CC = read_timer();
        mm(N, K, M, A, B, C_CC, 0, 0);
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

        /*printf("\n");
           for (int i = 0; i < M * N; i++) {
           printf("%f,%f,%f,%f\n", C_RR[i], C_RC[i], C_CR[i], C_CC[i]);
           }
           printf("\n");
         */
        free(A);
        free(B);
        free(C_RR);
        free(C_RC);
        free(C_CR);
        free(C_CC);
        return 0;
}

void mm(int N, int K, int M, REAL *A, REAL *B, REAL *C, int A_rowMajor,
        int B_rowMajor) {
        int i, j, w;
        // Both rowmajor
        if (A_rowMajor > 0 && B_rowMajor > 0) {
                for (i = 0; i < N; i++) {      // Loop over cols
                        for (j = 0; j < M; j++) { // Loop over rows
                                REAL temp = 0.0; // Create temp to for new element
                                for (w = 0; w < K; w++) // Walk through the row/column
                                        temp += A[i * K + w] * B[w * M + j]; // Add the product of elements
                                C[i * M + j] = temp; // Put element into output matrix
                        }
                }
        }
        // A columbmajor B rowmajor
        else if (A_rowMajor == 0 && B_rowMajor > 0) {
                for (i = 0; i < N; i++)
                        for (j = 0; j < M; j++) {
                                REAL temp = 0.0;
                                for (w = 0; w < K; w++)
                                        temp += A[w * K + i] * B[w * M + j];
                                C[i * M + j] = temp;
                        }
        }
        // A rowmajor B columbmajor
        else if (A_rowMajor > 0 && B_rowMajor == 0) {
                for (i = 0; i < N; i++)
                        for (j = 0; j < M; j++) {
                                REAL temp = 0.0;
                                for (w = 0; w < K; w++)
                                        temp += A[w * K + i] * B[j * M + w];
                                C[i * M + j] = temp;
                        }
        }
        // Both columbmajor
        else if (A_rowMajor == 0 && B_rowMajor == 0) {
                for (i = 0; i < N; i++)
                        for (j = 0; j < M; j++) {
                                REAL temp = 0.0;
                                for (w = 0; w < K; w++)
                                        temp += A[i * K + w] * B[j * M + w];
                                C[i * M + j] = temp;
                        }
        }
}
