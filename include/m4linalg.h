#ifndef M4LINALG_H
#define M4LINALG_H

#include <stdbool.h>
#include <stddef.h>

/* ========================================================================= *
 * Unified Core Types
 * ========================================================================= */

typedef struct {
    float *data;      /* Row-major memory layout: data[row * cols + col] */
    int rows, cols;
    bool owns_data;   /* If true, auto-free data on m4_matrix_free() */
} m4_matrix_t;

/* Standardized Library Error Codes */
typedef enum {
    M4_SUCCESS = 0,
    M4_ERR_INVALID_ARGS = -1,
    M4_ERR_SINGULAR = -2,
    M4_ERR_NON_CONVERGENCE = -3,
    M4_ERR_OUT_OF_MEMORY = -4
} m4_error_t;

/* ========================================================================= *
 * BLAS Level 3 Operations
 * ========================================================================= */

/**
 * Performs matrix-matrix multiplication: C = alpha * op(A) * op(B) + beta * C
 */
m4_error_t m4_sgemm(char transa, char transb, int m, int n, int k,
                    float alpha, const float *A, int lda,
                    const float *B, int ldb,
                    float beta, float *C, int ldc);

/**
 * Performs a batch of matrix-matrix multiplications concurrently using Grand Central Dispatch.
 * Ideal for deep learning workloads (e.g., multi-head attention blocks).
 */
m4_error_t m4_sgemm_batched(char transa, char transb, int m, int n, int k,
                            float alpha, const float *A, int lda, long long stride_a,
                            const float *B, int ldb, long long stride_b,
                            float beta, float *C, int ldc, long long stride_c,
                            int batch_count);


/* ========================================================================= *
 * LAPACK Factorizations
 * ========================================================================= */

/**
 * Computes the QR factorization of a real M-by-N matrix A using Householder reflections.
 * 
 * @param A   Pointer to the matrix (overwritten with Q and R factors).
 * @param tau Array of dimension min(M,N) containing scalar factors of the reflectors.
 */
m4_error_t m4_geqrf(m4_matrix_t *A, float *tau);

/**
 * Computes the LU factorization of a general M-by-N matrix A using partial pivoting.
 * 
 * @param A    Pointer to the matrix (overwritten with L and U factors).
 * @param ipiv Array of dimension min(M,N) containing pivot indices.
 */
m4_error_t m4_getrf(m4_matrix_t *A, int *ipiv);

/**
 * Solves a system of linear equations Ax = b using the LU factorization from m4_getrf.
 * 
 * @param LU   Pointer to the LU factored matrix.
 * @param ipiv Array of pivot indices from m4_getrf.
 * @param b    Right-hand side vector (overwritten with solution).
 */
m4_error_t m4_getrs(const m4_matrix_t *LU, const int *ipiv, float *b);

/**
 * Computes the Cholesky factorization of a real symmetric positive definite matrix A.
 * 
 * @param A Pointer to the matrix (overwritten with the lower triangular factor L).
 */
m4_error_t m4_potrf(m4_matrix_t *A);

/* ========================================================================= *
 * Eigenvalue Solvers
 * ========================================================================= */

/**
 * Computes all eigenvalues of a real symmetric matrix A using the Jacobi algorithm.
 * @param w Array of size A->cols to store the eigenvalues.
 */
m4_error_t m4_syev(m4_matrix_t *A, float *w);

/**
 * Computes all eigenvalues and eigenvectors of a real symmetric matrix A using the Jacobi algorithm.
 * @param V Pointer to the matrix (overwritten with orthonormal eigenvectors).
 */
m4_error_t m4_syev_evec(m4_matrix_t *A, float *w, m4_matrix_t *V);

/* ========================================================================= *
 * Solvers & Workspace Management
 * ========================================================================= */

/* Workspace for Conjugate Gradient Solver */
typedef struct {
    float *r;   /* Residual vector */
    float *p;   /* Search direction vector */
    float *Ap;  /* Matrix-vector product vector */
    float *z;   /* Preconditioned residual vector (for PCG) */
    int size;
} m4_cg_workspace_t;

/**
 * Allocates workspace memory for the Conjugate Gradient solver.
 * Prevents memory fragmentation during iterative loops.
 */
m4_error_t m4_cg_query_workspace(int n, m4_cg_workspace_t *work);
void m4_cg_free_workspace(m4_cg_workspace_t *work);

/**
 * Solves the symmetric positive definite linear system Ax = b using Conjugate Gradient.
 * 
 * @param work Pre-allocated workspace (must be initialized via m4_cg_query_workspace).
 * @param out_iters Number of iterations executed.
 */
m4_error_t m4_solve_cg(const m4_matrix_t *A, const float *b, float *x, int max_iter, float tol, m4_cg_workspace_t *work, int *out_iters);

/**
 * Solves the symmetric positive definite linear system Ax = b using Conjugate Gradient
 * with a Jacobi (diagonal) preconditioner.
 * 
 * @param work Pre-allocated workspace (must be initialized via m4_cg_query_workspace).
 * @param out_iters Number of iterations executed.
 */
m4_error_t m4_solve_pcg_jacobi(const m4_matrix_t *A, const float *b, float *x, int max_iter, float tol, m4_cg_workspace_t *work, int *out_iters);

/* Workspace for GMRES Solver */
typedef struct {
    float *V;   /* Krylov basis vectors */
    float *H;   /* Hessenberg matrix */
    float *cs;  /* Givens rotations (cos) */
    float *sn;  /* Givens rotations (sin) */
    float *s;   /* RHS for least squares */
    float *r;   /* Residual vector */
    float *w;   /* Temporary vector */
    float *y;   /* Least squares solution */
    float *z;   /* Preconditioned vector */
    int n;
    int restart;
} m4_gmres_workspace_t;

/**
 * Allocates workspace memory for the GMRES solver.
 */
m4_error_t m4_gmres_query_workspace(int n, int restart, m4_gmres_workspace_t *work);
void m4_gmres_free_workspace(m4_gmres_workspace_t *work);

/**
 * Solves the non-symmetric linear system Ax = b using Restarted GMRES.
 */
m4_error_t m4_solve_gmres(const m4_matrix_t *A, const float *b, float *x, int restart, int max_iter, float tol, m4_gmres_workspace_t *work, int *out_iters);

/**
 * Computes the Incomplete LU (ILU) factorization of a matrix A with a drop tolerance.
 * Useful as a preconditioner for iterative solvers.
 * 
 * @param A Pointer to the matrix (overwritten with incomplete L and U factors).
 * @param drop_tol Tolerance for dropping small elements to maintain sparsity/speed.
 */
m4_error_t m4_ilu_factor(m4_matrix_t *A, float drop_tol);

/**
 * Solves the preconditioned system Mx = b where M = L*U is from m4_ilu_factor.
 */
m4_error_t m4_ilu_solve(const m4_matrix_t *ILU, float *b);

/**
 * Solves the non-symmetric linear system Ax = b using Restarted GMRES with ILU preconditioning.
 * Handles highly asymmetric and ill-conditioned systems.
 */
m4_error_t m4_solve_pgmres_ilu(const m4_matrix_t *A, const float *b, float *x, int restart, int max_iter, float tol, m4_gmres_workspace_t *ws, float drop_tol, int *out_iters);

/**
 * Mixed-Precision Iterative Refinement.
 * Uses FP32 for the computationally heavy factorization and FP64 (double) for 
 * residual computation to achieve high accuracy with FP32 performance.
 *
 * @param x Output solution vector in double precision.
 */
m4_error_t m4_solve_mixed_precision_ir(m4_matrix_t *A, const double *b, double *x, int max_iter, double tol, int *out_iters);

/* ========================================================================= *
 * Hardware Auto-Tuning (Cache Blocking)
 * ========================================================================= */

void m4_autotune_init(void);
int m4_get_l1_tile_size(void);
int m4_get_l2_tile_size(void);

/* ========================================================================= *
 * Sparse Matrix Operations (CSR Format)
 * ========================================================================= */

typedef struct {
    float *values;      /* Non-zero values */
    int *col_idx;       /* Column indices */
    int *row_ptr;       /* Row start indices */
    int rows, cols, nnz;
    bool owns_data;
} m4_sparse_csr_t;

m4_sparse_csr_t m4_dense_to_csr(const m4_matrix_t *A, float tol);
void m4_sparse_csr_free(m4_sparse_csr_t *A);
m4_error_t m4_spmv_csr(char trans, float alpha, const m4_sparse_csr_t *A, const float *x, float beta, float *y);

#endif /* M4LINALG_H */
