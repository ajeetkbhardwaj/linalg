#ifndef LAPACK_H
#define LAPACK_H

#include "linalg.h"

#ifdef __cplusplus
extern "C" {
#endif

/* === Factorization Routines === */

/**
 * @brief LU decomposition with partial pivoting: PA = LU
 * @param A Input/output: on exit contains L (unit lower) and U (upper)
 * @param pivot Output: pivot indices (1-based, size min(rows,cols))
 * @return 0 on success, >0 if U[i,i] == 0 (singular at step i)
 */
int getrf(matrix_t *A, int *pivot);

/**
 * @brief Solve Ax = b using LU factorization
 * @param LU LU factorization from getrf
 * @param pivot Pivot array from getrf
 * @param b Right-hand side (modified to contain solution x)
 * @param nrhs Number of right-hand sides (columns of b)
 * @return 0 on success
 */
int getrs(const matrix_t *LU, const int *pivot, float *b, int nrhs);

/**
 * @brief Householder QR decomposition: A = Q*R
 * @param A Input/output: on exit contains R (upper) and Householder vectors
 * @param Q Output: orthogonal matrix Q (m×m), pass NULL to skip
 * @param tau Output: scalar factors of Householder reflectors (size min(m,n))
 * @return 0 on success
 */
int geqrf(matrix_t *A, matrix_t *Q, float *tau);

/**
 * @brief Cholesky decomposition: A = L*L^T (A must be SPD)
 * @param A Input/output: on exit contains lower triangular L
 * @return 0 on success, >0 if not positive definite at step i
 */
int potrf(matrix_t *A);

/**
 * @brief Solve Ax = b using Cholesky factorization
 * @param L Lower triangular Cholesky factor
 * @param b Right-hand side (modified to contain solution)
 * @return 0 on success
 */
int potrs(const matrix_t *L, float *b);

/**
 * @brief One-sided Jacobi SVD: A = U * diag(S) * V^T
 * @param A Input matrix (m×n)
 * @param U Output: left singular vectors (m×m), pass NULL to skip
 * @param S Output: singular values (length min(m,n), descending)
 * @param V Output: right singular vectors (n×n), pass NULL to skip
 * @param max_iter Maximum Jacobi iterations (0 = auto)
 * @param tol Convergence tolerance (0 = machine epsilon)
 * @return 0 on success, -1 if not converged
 */
int gesvd_jacobi(const matrix_t *A, matrix_t *U, float *S, 
                    matrix_t *V, int max_iter, float tol);

/* === Eigenvalue Routines === */

/**
 * @brief Symmetric eigenvalue decomposition via QR algorithm
 * @param A Input/output: symmetric matrix, on exit contains eigenvectors
 * @param w Output: eigenvalues (length n, ascending)
 * @return 0 on success
 */
int syev(matrix_t *A, float *w);

/* === Utility === */

/**
 * @brief Generate Householder vector and scalar
 * @param x Input vector (length n), on exit contains reflector
 * @param n Length of vector
 * @param tau Output: scalar factor
 * @return beta such that H*x = [beta; 0; ...; 0]
 */
float larfg(int n, float *x, float *tau);

#ifdef __cplusplus
}
#endif

#endif /* LAPACK_H */
