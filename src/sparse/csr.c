#include "m4linalg.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

m4_sparse_csr_t m4_dense_to_csr(const m4_matrix_t *A, float tol) {
    m4_sparse_csr_t S = {0};
    if (!A || !A->data) return S;
    
    S.rows = A->rows;
    S.cols = A->cols;
    S.owns_data = true;
    
    int nnz = 0;
    for (int i = 0; i < A->rows * A->cols; i++) {
        if (fabsf(A->data[i]) > tol) nnz++;
    }
    
    S.nnz = nnz;
    S.values = (float*)malloc(nnz * sizeof(float));
    S.col_idx = (int*)malloc(nnz * sizeof(int));
    S.row_ptr = (int*)calloc(A->rows + 1, sizeof(int));
    
    if (!S.values || !S.col_idx || !S.row_ptr) {
        m4_sparse_csr_free(&S);
        return S;
    }
    
    int idx = 0;
    for (int i = 0; i < A->rows; i++) {
        S.row_ptr[i] = idx;
        for (int j = 0; j < A->cols; j++) {
            float val = A->data[i * A->cols + j];
            if (fabsf(val) > tol) {
                S.values[idx] = val;
                S.col_idx[idx] = j;
                idx++;
            }
        }
    }
    S.row_ptr[A->rows] = nnz;
    
    return S;
}

void m4_sparse_csr_free(m4_sparse_csr_t *A) {
    if (!A) return;
    if (A->owns_data) {
        if (A->values) free(A->values);
        if (A->col_idx) free(A->col_idx);
        if (A->row_ptr) free(A->row_ptr);
    }
    A->values = NULL;
    A->col_idx = NULL;
    A->row_ptr = NULL;
    A->rows = 0; A->cols = 0; A->nnz = 0;
}

m4_error_t m4_spmv_csr(char trans, float alpha, const m4_sparse_csr_t *A, const float *x, float beta, float *y) {
    if (!A || !A->values || !x || !y) return M4_ERR_INVALID_ARGS;
    
    if (trans == 'N' || trans == 'n') {
        if (beta != 1.0f) {
            if (beta == 0.0f) memset(y, 0, A->rows * sizeof(float));
            else for (int i = 0; i < A->rows; i++) y[i] *= beta;
        }
        
        for (int i = 0; i < A->rows; i++) {
            float sum = 0.0f;
            for (int j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++) {
                sum += A->values[j] * x[A->col_idx[j]];
            }
            y[i] += alpha * sum;
        }
    } else if (trans == 'T' || trans == 't') {
        if (beta != 1.0f) {
            if (beta == 0.0f) memset(y, 0, A->cols * sizeof(float));
            else for (int j = 0; j < A->cols; j++) y[j] *= beta;
        }
        
        for (int i = 0; i < A->rows; i++) {
            for (int j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++) {
                y[A->col_idx[j]] += alpha * A->values[j] * x[i];
            }
        }
    } else {
        return M4_ERR_INVALID_ARGS;
    }
    
    return M4_SUCCESS;
}
