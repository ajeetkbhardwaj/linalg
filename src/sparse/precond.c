#include "../../include/sparse.h"
#include <stdlib.h>
#include <math.h>

/* Incomplete Cholesky (IC0) Preconditioner (No fill-in) */
sparse_csr_t ic0_precond(const sparse_csr_t *A) {
    sparse_csr_t M;
    M.rows = A->rows; M.cols = A->cols; M.nnz = A->nnz;
    M.owns_data = true;
    M.row_ptr = malloc((M.rows + 1) * sizeof(int));
    M.col_idx = malloc(M.nnz * sizeof(int));
    M.values = malloc(M.nnz * sizeof(float));
    
    for (int i = 0; i <= M.rows; i++) M.row_ptr[i] = A->row_ptr[i];
    for (int i = 0; i < M.nnz; i++) {
        M.col_idx[i] = A->col_idx[i];
        M.values[i] = A->values[i];
    }
    
    for (int i = 0; i < M.rows; i++) {
        int diag_idx = -1;
        for (int j = M.row_ptr[i]; j < M.row_ptr[i+1]; j++) {
            if (M.col_idx[j] == i) { diag_idx = j; break; }
        }
        if (diag_idx == -1 || M.values[diag_idx] <= 0.0f) continue; /* Not SPD fallback */
        
        M.values[diag_idx] = sqrtf(M.values[diag_idx]);
        
        for (int j = diag_idx + 1; j < M.row_ptr[i+1]; j++) {
            M.values[j] /= M.values[diag_idx];
        }
        
        for (int k = diag_idx + 1; k < M.row_ptr[i+1]; k++) {
            int col_k = M.col_idx[k];
            float L_ik = M.values[k];
            int ptr_k = M.row_ptr[col_k];
            int end_k = M.row_ptr[col_k+1];
            
            for (int j = k; j < M.row_ptr[i+1]; j++) {
                int col_j = M.col_idx[j];
                float L_ij = M.values[j];
                
                for (int p = ptr_k; p < end_k; p++) {
                    if (M.col_idx[p] == col_j) {
                        M.values[p] -= L_ik * L_ij;
                        break;
                    }
                }
            }
        }
    }
    return M;
}

/* Solve M * z = r where M = L * L^T */
void apply_ic0(const sparse_csr_t *M, const float *r, float *z) {
    int n = M->rows;
    float *y = malloc(n * sizeof(float));
    
    /* Forward substitution L * y = r */
    for (int i = 0; i < n; i++) {
        float sum = r[i];
        int diag_idx = -1;
        for (int j = M->row_ptr[i]; j < M->row_ptr[i+1]; j++) {
            if (M->col_idx[j] < i) sum -= M->values[j] * y[M->col_idx[j]];
            else if (M->col_idx[j] == i) diag_idx = j;
        }
        y[i] = (diag_idx != -1) ? (sum / M->values[diag_idx]) : sum;
    }
    
    for (int i = 0; i < n; i++) z[i] = y[i];
    for (int i = n - 1; i >= 0; i--) {
        int diag_idx = -1;
        for (int j = M->row_ptr[i]; j < M->row_ptr[i+1]; j++) if (M->col_idx[j] == i) diag_idx = j;
        if (diag_idx != -1) z[i] /= M->values[diag_idx];
        for (int j = M->row_ptr[i]; j < M->row_ptr[i+1]; j++) if (M->col_idx[j] < i) z[M->col_idx[j]] -= M->values[j] * z[i];
    }
    free(y);
}

/* === Jacobi Preconditioner (Diagonal) === */
sparse_csr_t jacobi_precond(const sparse_csr_t *A) {
    sparse_csr_t M;
    M.rows = A->rows; M.cols = A->cols; M.nnz = A->rows;
    M.owns_data = true;
    M.row_ptr = malloc((M.rows + 1) * sizeof(int));
    M.col_idx = malloc(M.nnz * sizeof(int));
    M.values = malloc(M.nnz * sizeof(float));
    
    for (int i = 0; i < M.rows; i++) {
        M.row_ptr[i] = i;
        M.col_idx[i] = i;
        M.values[i] = 1.0f; /* Fallback if diagonal element is missing */
        
        for (int j = A->row_ptr[i]; j < A->row_ptr[i+1]; j++) {
            if (A->col_idx[j] == i) {
                M.values[i] = A->values[j];
                break;
            }
        }
    }
    M.row_ptr[M.rows] = M.rows;
    return M;
}

void apply_jacobi(const sparse_csr_t *M, const float *r, float *z) {
    for (int i = 0; i < M->rows; i++) {
        z[i] = (M->values[i] != 0.0f) ? (r[i] / M->values[i]) : r[i];
    }
}

/* === Incomplete LU Preconditioner (ILU0) - No Fill-in === */
sparse_csr_t ilu0_precond(const sparse_csr_t *A) {
    sparse_csr_t M;
    M.rows = A->rows; M.cols = A->cols; M.nnz = A->nnz;
    M.owns_data = true;
    M.row_ptr = malloc((M.rows + 1) * sizeof(int));
    M.col_idx = malloc(M.nnz * sizeof(int));
    M.values = malloc(M.nnz * sizeof(float));
    
    for (int i = 0; i <= M.rows; i++) M.row_ptr[i] = A->row_ptr[i];
    for (int i = 0; i < M.nnz; i++) {
        M.col_idx[i] = A->col_idx[i];
        M.values[i] = A->values[i];
    }
    
    for (int i = 1; i < M.rows; i++) {
        for (int k = M.row_ptr[i]; k < M.row_ptr[i+1]; k++) {
            int col_k = M.col_idx[k];
            if (col_k >= i) break; /* Process only the strictly lower triangular part of row i */
            
            /* Find M(col_k, col_k) for division */
            int diag_k = -1;
            for (int p = M.row_ptr[col_k]; p < M.row_ptr[col_k+1]; p++) {
                if (M.col_idx[p] == col_k) { diag_k = p; break; }
            }
            if (diag_k != -1 && M.values[diag_k] != 0.0f) {
                M.values[k] /= M.values[diag_k];
            }
            
            /* Update remaining elements M(i, j) for j > col_k */
            for (int j = k + 1; j < M.row_ptr[i+1]; j++) {
                int col_j = M.col_idx[j];
                /* Find M(col_k, col_j) */
                for (int p = M.row_ptr[col_k]; p < M.row_ptr[col_k+1]; p++) {
                    if (M.col_idx[p] == col_j) {
                        M.values[j] -= M.values[k] * M.values[p];
                        break;
                    }
                }
            }
        }
    }
    return M;
}

/* Solve M * z = r where M = L * U (stored sequentially in CSR) */
void apply_ilu0(const sparse_csr_t *M, const float *r, float *z) {
    int n = M->rows;
    float *y = malloc(n * sizeof(float));
    
    /* Forward substitution: L * y = r (L has 1s on diagonal) */
    for (int i = 0; i < n; i++) {
        float sum = r[i];
        for (int j = M->row_ptr[i]; j < M->row_ptr[i+1]; j++) {
            if (M->col_idx[j] < i) sum -= M->values[j] * y[M->col_idx[j]];
        }
        y[i] = sum;
    }
    
    /* Backward substitution: U * z = y */
    for (int i = n - 1; i >= 0; i--) {
        float sum = y[i];
        int diag_idx = -1;
        for (int j = M->row_ptr[i]; j < M->row_ptr[i+1]; j++) {
            if (M->col_idx[j] > i) sum -= M->values[j] * z[M->col_idx[j]];
            else if (M->col_idx[j] == i) diag_idx = j;
        }
        z[i] = (diag_idx != -1 && M->values[diag_idx] != 0.0f) ? (sum / M->values[diag_idx]) : sum;
    }
    
    free(y);
}
