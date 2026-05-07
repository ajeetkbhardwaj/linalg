/* tests/bench.c - Performance comparison vs Apple Accelerate [[19]][[21]] */
#include "../include/m4linalg.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <Accelerate/Accelerate.h>  /* Apple's optimized BLAS/LAPACK */

double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void fill_random(float *arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    }
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    
    m4_autotune_init();
    
    printf("🔬 m4linalg Benchmark: %dx%d matrix multiply\n", N, N);
    printf("   Auto-Tuned Cache: L1=%d, L2=%d\n", m4_get_l1_tile_size(), m4_get_l2_tile_size());
    
    /* Allocate matrices (row-major) */
    float *A = aligned_alloc(64, N * N * sizeof(float));
    float *B = aligned_alloc(64, N * N * sizeof(float));
    float *C = aligned_alloc(64, N * N * sizeof(float));
    float *C_acc = aligned_alloc(64, N * N * sizeof(float));
    
    if (!A || !B || !C || !C_acc) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }
    
    fill_random(A, N*N);
    fill_random(B, N*N);
    
    /* === m4linalg SGEMM === */
    printf("\n[1] m4linalg SGEMM...\n");
    for (int i = 0; i < N*N; i++) C[i] = 0.0f;
    
    double t0 = get_time_sec();
    m4_sgemm('N', 'N', N, N, N, 1.0f, A, N, B, N, 0.0f, C, N);
    double t1 = get_time_sec();
    double gflops = (2.0 * N * N * N * 1e-9) / (t1 - t0);
    printf("   Time: %.4f s | Performance: %.2f GFLOPS\n", 
           t1 - t0, gflops);
    
    /* === Apple Accelerate SGEMM (baseline) [[19]][[21]] === */
    printf("\n[2] Apple Accelerate SGEMM...\n");
    for (int i = 0; i < N*N; i++) C_acc[i] = 0.0f;
    
    t0 = get_time_sec();
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0f, A, N, B, N, 0.0f, C_acc, N);
    #pragma clang diagnostic pop
    t1 = get_time_sec();
    double gflops_acc = (2.0 * N * N * N * 1e-9) / (t1 - t0);
    printf("   Time: %.4f s | Performance: %.2f GFLOPS\n", 
           t1 - t0, gflops_acc);
    
    /* === Verify correctness === */
    printf("\n[3] Correctness check...\n");
    float max_err = 0.0f;
    for (int i = 0; i < N*N; i++) {
        float err = fabsf(C[i] - C_acc[i]);
        if (err > max_err) max_err = err;
    }
    printf("   Max absolute error: %.2e %s\n", 
           max_err, max_err < 1e-4 ? "✓ PASS" : "✗ FAIL");
    
    /* === Summary === */
    printf("\n📊 Results:\n");
    printf("   m4linalg:  %.2f GFLOPS\n", gflops);
    printf("   Accelerate: %.2f GFLOPS\n", gflops_acc);
    printf("   Ratio:     %.2fx %s\n", 
           gflops / gflops_acc,
           gflops >= gflops_acc * 0.9 ? "✓ Competitive" : "⚠ Room for improvement");
    
    free(A); free(B); free(C); free(C_acc);
    return 0;
}
