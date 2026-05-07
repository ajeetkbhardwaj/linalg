#include "m4linalg.h"
#include <math.h>
#include <dispatch/dispatch.h>

m4_error_t m4_getrf(m4_matrix_t *A, int *ipiv) {
    if (!A || !A->data || !ipiv) {
        return M4_ERR_INVALID_ARGS;
    }

    int m = A->rows;
    int n = A->cols;
    int lda = n; /* Row-major stride */
    int min_mn = (m < n) ? m : n;

    for (int i = 0; i < min_mn; i++) {
        /* 1. Find pivot in column i, from row i to m */
        int pivot_row = i;
        float max_val = fabsf(A->data[i * lda + i]);
        
        for (int k = i + 1; k < m; k++) {
            float val = fabsf(A->data[k * lda + i]);
            if (val > max_val) {
                max_val = val;
                pivot_row = k;
            }
        }
        
        /* Record pivot index (1-based for LAPACK compatibility, but 0-based internally here) */
        ipiv[i] = pivot_row;

        /* Check for exact singularity */
        if (max_val == 0.0f) {
            return M4_ERR_SINGULAR;
        }

        /* 2. Swap rows if necessary */
        if (pivot_row != i) {
            for (int j = 0; j < n; j++) {
                float tmp = A->data[i * lda + j];
                A->data[i * lda + j] = A->data[pivot_row * lda + j];
                A->data[pivot_row * lda + j] = tmp;
            }
        }

        /* 3. Compute multipliers for the current column */
        float pivot_inv = 1.0f / A->data[i * lda + i];
        for (int k = i + 1; k < m; k++) {
            A->data[k * lda + i] *= pivot_inv;
        }

        /* 4. Rank-1 update of the trailing submatrix: A[i+1:m, i+1:n] */
        int row_count = m - (i + 1);
        int col_count = n - (i + 1);
        
        if (row_count > 0 && col_count > 0) {
            if (row_count * col_count > 4096) {
                /* Parallelized update using Apple's Grand Central Dispatch */
                dispatch_apply(row_count, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t idx) {
                    int k = i + 1 + (int)idx;
                    float factor = A->data[k * lda + i];
                    for (int j = i + 1; j < n; j++) {
                        A->data[k * lda + j] -= factor * A->data[i * lda + j];
                    }
                });
            } else {
                /* Serial fallback for small submatrices to avoid thread overhead */
                for (int k = i + 1; k < m; k++) {
                    float factor = A->data[k * lda + i];
                    for (int j = i + 1; j < n; j++) {
                        A->data[k * lda + j] -= factor * A->data[i * lda + j];
                    }
                }
            }
        }
    }
    
    return M4_SUCCESS;
}

m4_error_t m4_getrs(const m4_matrix_t *LU, const int *ipiv, float *b) {
    if (!LU || !LU->data || !ipiv || !b || LU->rows != LU->cols) {
        return M4_ERR_INVALID_ARGS;
    }
    int n = LU->rows;
    
    /* Forward substitution: solve Ly = Pb */
    for (int i = 0; i < n; i++) {
        int pivot_row = ipiv[i];
        if (pivot_row != i) {
            float tmp = b[i];
            b[i] = b[pivot_row];
            b[pivot_row] = tmp;
        }
    }
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
            b[i] -= LU->data[i * n + j] * b[j];
        }
    }
    
    /* Backward substitution: solve Ux = y */
    for (int i = n - 1; i >= 0; i--) {
        for (int j = i + 1; j < n; j++) {
            b[i] -= LU->data[i * n + j] * b[j];
        }
        b[i] /= LU->data[i * n + i];
    }
    
    return M4_SUCCESS;
}
