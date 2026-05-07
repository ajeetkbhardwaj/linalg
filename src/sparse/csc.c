#include "sparse.h"
#include "blas.h"
#include <stdlib.h>
#include <string.h>

/* Allocate CSC matrix */
sparse_csc_t sparse_csc_alloc(int rows, int cols, int nnz) {
    sparse_csc_t A;
    A.rows = rows;
    A.cols = cols;
    A.nnz = nnz;
    A.owns_data = true;
    
    A.values = nnz > 0 ? calloc(nnz, sizeof(float)) : NULL;
    A.row_idx = nnz > 0 ? calloc(nnz, sizeof(int)) : NULL;
    A.col_ptr = calloc(cols + 1, sizeof(int));
    
    return A;
}

/* Free CSC matrix */
void sparse_csc_free(sparse_csc_t *A) {
    if (!A) return;
    if (A->owns_data) {
        free(A->values);
        free(A->row_idx);
    }
    free(A->col_ptr);
    A->values = NULL;
    A->row_idx = NULL;
    A->col_ptr = NULL;
}

/* CSC SpMV: y = alpha*A*x + beta*y */
void spmv_csc(char trans, float alpha, const sparse_csc_t *A,
                 const float *x, float beta, float *y) {
    
    if (!A || !A->values || !x || !y) return;
    
    if (trans == 'N' || trans == 'n') {
        /* y = alpha * A * x + beta * y */
        if (beta != 1.0f) {
            if (beta == 0.0f) memset(y, 0, A->rows * sizeof(float));
            else for (int i = 0; i < A->rows; i++) y[i] *= beta;
        }
        
        /* CSC SpMV: column-wise */
        for (int j = 0; j < A->cols; j++) {
            float xj = x[j];
            if (xj != 0.0f) {
                for (int i = A->col_ptr[j]; i < A->col_ptr[j + 1]; i++) {
                    y[A->row_idx[i]] += alpha * A->values[i] * xj;
                }
            }
        }
    } else {
        /* y = alpha * A^T * x + beta * y */
        if (beta != 1.0f) {
            if (beta == 0.0f) memset(y, 0, A->cols * sizeof(float));
            else for (int j = 0; j < A->cols; j++) y[j] *= beta;
        }
        
        /* Transposed: row-wise accumulation */
        for (int j = 0; j < A->cols; j++) {
            float sum = 0.0f;
            for (int i = A->col_ptr[j]; i < A->col_ptr[j + 1]; i++) {
                sum += A->values[i] * x[A->row_idx[i]];
            }
            y[j] += alpha * sum;
        }
    }
}

/* Convert CSR to CSC */
sparse_csc_t csr_to_csc(const sparse_csr_t *A) {
    sparse_csc_t B = sparse_csc_alloc(A->rows, A->cols, A->nnz);
    
    /* Count entries per column */
    for (int i = 0; i < A->nnz; i++) {
        B.col_ptr[A->col_idx[i] + 1]++;
    }
    
    /* Cumulative sum */
    for (int j = 0; j < A->cols; j++) {
        B.col_ptr[j + 1] += B.col_ptr[j];
    }
    
    /* Copy data */
    int *next = malloc((A->cols + 1) * sizeof(int));
    memcpy(next, B.col_ptr, (A->cols + 1) * sizeof(int));
    
    for (int i = 0; i < A->rows; i++) {
        for (int j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++) {
            int col = A->col_idx[j];
            int pos = next[col]++;
            B.values[pos] = A->values[j];
            B.row_idx[pos] = i;
        }
    }
    
    free(next);
    return B;
}
