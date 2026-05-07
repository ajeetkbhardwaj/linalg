#include "../include/linalg.h"
#include "../include/blas.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    printf("=== Performance Regression Monitor ===\n");
    int N = 512;
    float *A = malloc(N * N * sizeof(float));
    float *B = malloc(N * N * sizeof(float));
    float *C = malloc(N * N * sizeof(float));
    
    for (int i = 0; i < N * N; i++) { A[i] = 1.0f; B[i] = 1.0f; C[i] = 0.0f; }
    
    double t0 = get_time_sec();
    sgemm('N', 'N', N, N, N, 1.0f, A, B, 0.0f, C);
    double t1 = get_time_sec();
    
    double gflops = (2.0 * N * N * N * 1e-9) / (t1 - t0);
    printf("SGEMM %dx%d: %.2f GFLOPS\n", N, N, gflops);
    
    /* Assuming an M4 chip should clear 10 GFLOPS effortlessly */
    if (gflops < 10.0) {
        printf("[WARN] Performance is suspiciously low (< 10 GFLOPS). Check compiler flags & OpenMP.\n");
    } else {
        printf("[PASS] Performance cleared the baseline threshold.\n");
    }
    
    free(A); free(B); free(C);
    return 0;
}