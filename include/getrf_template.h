/* src/lapack/getrf_template.h - Generic LU Decomposition & Solve */
#ifndef SCALAR
#error "SCALAR must be defined"
#endif

#define CONCAT2(a, b) a ## b
#define CONCAT(a, b) CONCAT2(a, b)

#ifdef _OPENMP
#include <omp.h>
#endif

#define GETRF_FUNC CONCAT(PREFIX, getrf)
#define GETRS_FUNC CONCAT(PREFIX, getrs)

/* LU Decomposition with Partial Pivoting */
int GETRF_FUNC(MATRIX_T *A, int *pivot) {
    if (!A || !A->data || !pivot) return -1;
    int m = A->rows;
    int n = A->cols;
    int min_mn = (m < n) ? m : n;
    
    for (int i = 0; i < min_mn; i++) pivot[i] = i;
    
    for (int j = 0; j < min_mn; j++) {
        int p = j;
        REAL_SCALAR max_val = -1.0;
        for (int i = j; i < m; i++) {
            REAL_SCALAR val = ABS_MACRO(A->data[i * n + j]);
            if (val > max_val) { max_val = val; p = i; }
        }
        
        if (max_val == 0.0) return j + 1; /* Singular */
        
        if (p != j) {
            pivot[j] = p;
            for (int k = 0; k < n; k++) {
                SCALAR tmp = A->data[p * n + k];
                A->data[p * n + k] = A->data[j * n + k];
                A->data[j * n + k] = tmp;
            }
        }
        
        SCALAR inv_diag = SCALAR_ONE / A->data[j * n + j];
        for (int i = j + 1; i < m; i++) {
            A->data[i * n + j] *= inv_diag;
        }
        
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = j + 1; i < m; i++) {
            SCALAR factor = A->data[i * n + j];
            for (int k = j + 1; k < n; k++) {
                A->data[i * n + k] -= factor * A->data[j * n + k];
            }
        }
    }
    return 0;
}


/* LU Forward/Backward Solve */
int GETRS_FUNC(const MATRIX_T *LU, const int *pivot, SCALAR *b, int nrhs) {
    if (!LU || !LU->data || !pivot || !b) return -1;
    int n = LU->rows;
    if (LU->cols != n) return -1; /* Matrix must be square */
    
    /* Apply row permutations to all RHS simultaneously (optimized) */
    for (int i = 0; i < n; i++) {
        int p = pivot[i];
        if (p != i) {
            for (int k = 0; k < nrhs; k++) {
                SCALAR tmp = b[k * n + i];
                b[k * n + i] = b[k * n + p];
                b[k * n + p] = tmp;
            }
        }
    }
    
    for (int k = 0; k < nrhs; k++) {
        SCALAR *bk = &b[k * n];
        
        for (int i = 0; i < n; i++)
            for (int j = 0; j < i; j++) bk[i] -= LU->data[i * n + j] * bk[j];
            
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i + 1; j < n; j++) bk[i] -= LU->data[i * n + j] * bk[j];
            bk[i] /= LU->data[i * n + i];
        }
    }
    return 0;
}

