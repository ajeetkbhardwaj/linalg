#include "../../include/sparse.h"
#include <stdlib.h>
#include <string.h>

/* === Coordinate (COO) Format === */
sparse_coo_t sparse_coo_alloc(int rows, int cols, int nnz) {
    sparse_coo_t A;
    A.rows = rows;
    A.cols = cols;
    A.nnz = nnz;
    A.owns_data = true;
    A.values = (nnz > 0) ? calloc(nnz, sizeof(float)) : NULL;
    A.row_idx = (nnz > 0) ? calloc(nnz, sizeof(int)) : NULL;
    A.col_idx = (nnz > 0) ? calloc(nnz, sizeof(int)) : NULL;
    return A;
}

void sparse_coo_free(sparse_coo_t *A) {
    if (!A) return;
    if (A->owns_data) {
        free(A->values);
        free(A->row_idx);
        free(A->col_idx);
    }
    A->values = NULL; A->row_idx = NULL; A->col_idx = NULL;
    A->nnz = A->rows = A->cols = 0;
}

void spmv_coo(char trans, float alpha, const sparse_coo_t *A, const float *x, float beta, float *y) {
    if (!A || !A->values || !x || !y) return;
    
    int out_len = (trans == 'N' || trans == 'n') ? A->rows : A->cols;
    if (beta != 1.0f) {
        if (beta == 0.0f) memset(y, 0, out_len * sizeof(float));
        else for (int i = 0; i < out_len; i++) y[i] *= beta;
    }
    
    if (trans == 'N' || trans == 'n') {
        for (int i = 0; i < A->nnz; i++) {
            y[A->row_idx[i]] += alpha * A->values[i] * x[A->col_idx[i]];
        }
    } else {
        for (int i = 0; i < A->nnz; i++) {
            y[A->col_idx[i]] += alpha * A->values[i] * x[A->row_idx[i]];
        }
    }
}

/* === Diagonal (DIA) Format === */
sparse_dia_t sparse_dia_alloc(int rows, int cols, int num_diags) {
    sparse_dia_t A;
    A.rows = rows;
    A.cols = cols;
    A.num_diags = num_diags;
    A.owns_data = true;
    A.values = (num_diags > 0) ? calloc(rows * num_diags, sizeof(float)) : NULL;
    A.offsets = (num_diags > 0) ? calloc(num_diags, sizeof(int)) : NULL;
    return A;
}

void sparse_dia_free(sparse_dia_t *A) {
    if (!A) return;
    if (A->owns_data) {
        free(A->values);
        free(A->offsets);
    }
    A->values = NULL; A->offsets = NULL;
    A->num_diags = A->rows = A->cols = 0;
}

void spmv_dia(char trans, float alpha, const sparse_dia_t *A, const float *x, float beta, float *y) {
    if (!A || !A->values || !x || !y) return;
    
    int out_len = (trans == 'N' || trans == 'n') ? A->rows : A->cols;
    if (beta != 1.0f) {
        if (beta == 0.0f) memset(y, 0, out_len * sizeof(float));
        else for (int i = 0; i < out_len; i++) y[i] *= beta;
    }
    
    if (trans == 'N' || trans == 'n') {
        for (int d = 0; d < A->num_diags; d++) {
            int k = A->offsets[d];
            int i_start = (k < 0) ? -k : 0;
            int i_end = A->rows;
            if (i_end + k > A->cols) i_end = A->cols - k;
            
            for (int i = i_start; i < i_end; i++) {
                int j = i + k;
                y[i] += alpha * A->values[d * A->rows + i] * x[j];
            }
        }
    } else {
        for (int d = 0; d < A->num_diags; d++) {
            int k = A->offsets[d];
            int i_start = (k < 0) ? -k : 0;
            int i_end = A->rows;
            if (i_end + k > A->cols) i_end = A->cols - k;
            
            for (int i = i_start; i < i_end; i++) {
                int j = i + k;
                y[j] += alpha * A->values[d * A->rows + i] * x[i];
            }
        }
    }
}

/* === Block Sparse Row (BSR) Format === */
sparse_bsr_t sparse_bsr_alloc(int block_rows, int block_cols, int block_size, int nnz_blocks) {
    sparse_bsr_t A;
    A.rows = block_rows;
    A.cols = block_cols;
    A.block_size = block_size;
    A.nnz_blocks = nnz_blocks;
    A.owns_data = true;
    
    int block_elements = block_size * block_size;
    A.values = (nnz_blocks > 0) ? calloc(nnz_blocks * block_elements, sizeof(float)) : NULL;
    A.col_idx = (nnz_blocks > 0) ? calloc(nnz_blocks, sizeof(int)) : NULL;
    A.row_ptr = calloc(block_rows + 1, sizeof(int));
    
    return A;
}

void sparse_bsr_free(sparse_bsr_t *A) {
    if (!A) return;
    if (A->owns_data) {
        free(A->values);
        free(A->col_idx);
    }
    free(A->row_ptr);
    A->values = NULL; A->col_idx = NULL; A->row_ptr = NULL;
    A->nnz_blocks = A->rows = A->cols = A->block_size = 0;
}

void spmv_bsr(char trans, float alpha, const sparse_bsr_t *A, const float *x, float beta, float *y) {
    if (!A || !A->values || !x || !y) return;
    
    int bs = A->block_size;
    int out_len = ((trans == 'N' || trans == 'n') ? A->rows : A->cols) * bs;
    
    if (beta != 1.0f) {
        if (beta == 0.0f) memset(y, 0, out_len * sizeof(float));
        else for (int i = 0; i < out_len; i++) y[i] *= beta;
    }
    
    if (trans == 'N' || trans == 'n') {
        for (int i = 0; i < A->rows; i++) {
            for (int p = A->row_ptr[i]; p < A->row_ptr[i+1]; p++) {
                int j = A->col_idx[p];
                float *block = &A->values[p * bs * bs];
                for (int bi = 0; bi < bs; bi++) {
                    float sum = 0.0f;
                    for (int bj = 0; bj < bs; bj++) sum += block[bi * bs + bj] * x[j * bs + bj];
                    y[i * bs + bi] += alpha * sum;
                }
            }
        }
    } else {
        for (int i = 0; i < A->rows; i++) {
            for (int p = A->row_ptr[i]; p < A->row_ptr[i+1]; p++) {
                int j = A->col_idx[p];
                float *block = &A->values[p * bs * bs];
                for (int bj = 0; bj < bs; bj++) {
                    float sum = 0.0f;
                    for (int bi = 0; bi < bs; bi++) {
                        sum += block[bi * bs + bj] * x[i * bs + bi];
                    }
                    y[j * bs + bj] += alpha * sum;
                }
            }
        }
    }
}
