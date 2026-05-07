#include "m4linalg.h"
#include <stdlib.h>
#include <math.h>

/* Helper for Double Precision Matrix-Vector Product */
static void m4_internal_dgemv(const m4_matrix_t *A, const double *x, double *y) {
    int n = A->cols;
    for (int i = 0; i < A->rows; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += (double)A->data[i * n + j] * x[j];
        }
        y[i] = sum;
    }
}

m4_error_t m4_solve_mixed_precision_ir(m4_matrix_t *A, const double *b, double *x, int max_iter, double tol, int *out_iters) {
    if (!A || !A->data || !b || !x || A->rows != A->cols) return M4_ERR_INVALID_ARGS;
    int n = A->rows;
    
    int *ipiv = (int*)malloc(n * sizeof(int));
    float *bs = (float*)malloc(n * sizeof(float));
    double *r = (double*)malloc(n * sizeof(double));
    
    if (!ipiv || !bs || !r) {
        free(ipiv); free(bs); free(r);
        return M4_ERR_OUT_OF_MEMORY;
    }
    
    /* We must preserve A for double precision residual calculation.
       So we copy A to As for the single-precision LU factorization. */
    m4_matrix_t As = { .data = (float*)malloc(n * n * sizeof(float)), .rows = n, .cols = n, .owns_data = true };
    if (!As.data) { free(ipiv); free(bs); free(r); return M4_ERR_OUT_OF_MEMORY; }
    
    for (int i = 0; i < n * n; i++) As.data[i] = A->data[i];
    
    /* 1. Factorize As in single precision (FP32) */
    m4_error_t err = m4_getrf(&As, ipiv);
    if (err != M4_SUCCESS) {
        free(ipiv); free(bs); free(r); free(As.data);
        return err;
    }
    
    /* 2. Initial solve in single precision */
    for (int i = 0; i < n; i++) bs[i] = (float)b[i];
    m4_getrs(&As, ipiv, bs);
    for (int i = 0; i < n; i++) x[i] = (double)bs[i];
    
    /* 3. Iterative Refinement in Double Precision (FP64) */
    double b_norm = 0.0;
    for (int i = 0; i < n; i++) b_norm += b[i] * b[i];
    b_norm = sqrt(b_norm);
    if (b_norm == 0.0) b_norm = 1.0;
    
    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        m4_internal_dgemv(A, x, r);
        double r_norm = 0.0;
        for (int i = 0; i < n; i++) {
            r[i] = b[i] - r[i];
            r_norm += r[i] * r[i];
        }
        r_norm = sqrt(r_norm);
        if (r_norm / b_norm < tol) break; /* Converged */
        
        for (int i = 0; i < n; i++) bs[i] = (float)r[i];
        m4_getrs(&As, ipiv, bs);
        for (int i = 0; i < n; i++) x[i] += (double)bs[i];
    }
    
    if (out_iters) *out_iters = iter;
    free(ipiv); free(bs); free(r); free(As.data);
    return (iter < max_iter) ? M4_SUCCESS : M4_ERR_NON_CONVERGENCE;
}

