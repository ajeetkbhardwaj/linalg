#ifndef SPARSE_H
#define SPARSE_H

#include "linalg.h"

#ifdef __cplusplus
extern "C" {
#endif

/* === Compressed Sparse Row (CSR) Format === */
typedef struct {
    float *values;      /* Non-zero values (length nnz) */
    int *col_idx;       /* Column index for each value (length nnz) */
    int *row_ptr;       /* Row pointers (length rows+1), row_ptr[i] = start of row i */
    int nnz;            /* Number of non-zero elements */
    int rows, cols;     /* Matrix dimensions */
    bool owns_data;     /* Whether to free arrays on sparse_free */
} sparse_csr_t;

/* === Compressed Sparse Column (CSC) Format === */
typedef struct {
    float *values;      /* Non-zero values (length nnz) */
    int *row_idx;       /* Row index for each value (length nnz) */
    int *col_ptr;       /* Column pointers (length cols+1) */
    int nnz;            /* Number of non-zero elements */
    int rows, cols;     /* Matrix dimensions */
    bool owns_data;
} sparse_csc_t;

/* === Coordinate (COO) Format === */
typedef struct {
    float *values;      /* Non-zero values */
    int *row_idx;       /* Row indices */
    int *col_idx;       /* Column indices */
    int nnz;            /* Number of non-zeros */
    int rows, cols;     /* Matrix dimensions */
    bool owns_data;
} sparse_coo_t;

/* === Diagonal (DIA) Format === */
typedef struct {
    float *values;      /* Padded diagonals, size: rows * num_diags */
    int *offsets;       /* Offset of each diagonal (0=main, -k=sub, +k=super) */
    int num_diags;      /* Number of diagonals stored */
    int rows, cols;     /* Matrix dimensions */
    bool owns_data;
} sparse_dia_t;

/* === Block Sparse Row (BSR) Format === */
typedef struct {
    float *values;      /* Block values, size: nnz_blocks * block_size * block_size */
    int *col_idx;       /* Column index for each block */
    int *row_ptr;       /* Row pointers for blocks */
    int nnz_blocks;     /* Number of non-zero blocks */
    int rows, cols;     /* Matrix dimensions in terms of BLOCKS */
    int block_size;     /* Size of each square block (e.g., 2, 4, 8) */
    bool owns_data;
} sparse_bsr_t;

/* === CSR Operations === */

/**
 * @brief Allocate CSR matrix with given sparsity pattern
 */
sparse_csr_t sparse_csr_alloc(int rows, int cols, int nnz);

/**
 * @brief Free CSR matrix resources
 */
void sparse_csr_free(sparse_csr_t *A);

/**
 * @brief Sparse matrix-vector multiply: y = alpha*A*x + beta*y (CSR)
 */
void spmv_csr(char trans, float alpha, const sparse_csr_t *A,
                const float *x, float beta, float *y);

/**
 * @brief Convert dense matrix to CSR format
 */
sparse_csr_t dense_to_csr(const matrix_t *A, float tol);

/**
 * @brief Extract diagonal of sparse matrix
 */
void sparse_diag_csr(const sparse_csr_t *A, float *diag);

/* === CSC Operations === */

sparse_csc_t sparse_csc_alloc(int rows, int cols, int nnz);
void sparse_csc_free(sparse_csc_t *A);
void spmv_csc(char trans, float alpha, const sparse_csc_t *A,
                const float *x, float beta, float *y);
sparse_csc_t csr_to_csc(const sparse_csr_t *A);

/* === Additional Sparse Format Operations === */
sparse_coo_t sparse_coo_alloc(int rows, int cols, int nnz);
void sparse_coo_free(sparse_coo_t *A);
void spmv_coo(char trans, float alpha, const sparse_coo_t *A, const float *x, float beta, float *y);

sparse_dia_t sparse_dia_alloc(int rows, int cols, int num_diags);
void sparse_dia_free(sparse_dia_t *A);
void spmv_dia(char trans, float alpha, const sparse_dia_t *A, const float *x, float beta, float *y);

sparse_bsr_t sparse_bsr_alloc(int block_rows, int block_cols, int block_size, int nnz_blocks);
void sparse_bsr_free(sparse_bsr_t *A);
void spmv_bsr(char trans, float alpha, const sparse_bsr_t *A, const float *x, float beta, float *y);

/* === Iterative Solver Preconditioners === */

/**
 * @brief Incomplete Cholesky preconditioner (IC(0)) for SPD matrices
 */
sparse_csr_t ic0_precond(const sparse_csr_t *A);

/**
 * @brief Apply IC(0) preconditioner: solve M*z = r
 */
void apply_ic0(const sparse_csr_t *M, const float *r, float *z);

/**
 * @brief Jacobi (Diagonal) preconditioner
 */
sparse_csr_t jacobi_precond(const sparse_csr_t *A);
void apply_jacobi(const sparse_csr_t *M, const float *r, float *z);

/**
 * @brief Incomplete LU preconditioner (ILU(0)) with no fill-in
 */
sparse_csr_t ilu0_precond(const sparse_csr_t *A);
void apply_ilu0(const sparse_csr_t *M, const float *r, float *z);

/* === Sparse Iterative Solvers === */

/**
 * @brief Conjugate Gradient solver optimized for sparse CSR format matrices
 */
int solve_sparse_cg(const sparse_csr_t *A, const float *b, float *x, int max_iter, float tol);
linalg_error_t solve_sparse_cg_ext(const sparse_csr_t *A, const float *b, float *x, 
                                   int max_iter, float tol, cg_workspace_t *ws, int *out_iters);

/**
 * @brief Preconditioned Conjugate Gradient solver optimized for sparse CSR matrices
 */
int solve_sparse_cg_precond(const sparse_csr_t *A, const float *b, float *x, const sparse_csr_t *M, int max_iter, float tol);
linalg_error_t solve_sparse_cg_precond_ext(const sparse_csr_t *A, const float *b, float *x,
                                           const sparse_csr_t *M, int max_iter, float tol,
                                           cg_workspace_t *ws, int *out_iters);

/**
 * @brief GMRES solver optimized for sparse CSR format matrices (for non-symmetric systems)
 */
int solve_sparse_gmres(const sparse_csr_t *A, const float *b, float *x, int restart, int max_iter, float tol);
linalg_error_t solve_sparse_gmres_ext(const sparse_csr_t *A, const float *b, float *x, 
                                      int restart, int max_iter, float tol, gmres_workspace_t *ws, int *out_iters);

/* === Sparse Direct Solvers & Utilities === */

/**
 * @brief Symbolic factor ordering (Minimum Degree Algorithm) to minimize fill-in
 */
int* sparse_minimum_degree_ordering(const sparse_csr_t *A);

/**
 * @brief Symbolic Factorization: Computes the elimination tree for Sparse Cholesky/LU
 */
int* sparse_symbolic_factorization(const sparse_csr_t *A, const int *perm, int *nnz_L);

#ifdef __cplusplus
}
#endif

#endif /* SPARSE_H */
