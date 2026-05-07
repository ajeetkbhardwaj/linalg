/* src/blas/gemm_template.h - Generic GEMM Implementation */
#ifndef SCALAR
#error "SCALAR must be defined before including gemm_template.h"
#endif
#ifndef PREFIX
#error "PREFIX must be defined before including gemm_template.h"
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#define CONCAT2(a, b) a ## b
#define CONCAT(a, b) CONCAT2(a, b)

#define GEMM_FUNC CONCAT(PREFIX, gemm)
#define GEMM_BLOCKED CONCAT(GEMM_FUNC, _blocked)
#define GEMM_BLOCKED_NT CONCAT(GEMM_FUNC, _blocked_nt)

/* Forward declarations */
static void GEMM_BLOCKED(int m, int n, int k, SCALAR alpha, 
                          const SCALAR *A, const SCALAR *B, SCALAR beta, SCALAR *C);
static void GEMM_BLOCKED_NT(int m, int n, int k, SCALAR alpha, 
                             const SCALAR *A, const SCALAR *B, SCALAR beta, SCALAR *C);

/* GEMM: C = alpha * A * B + beta * C */
void GEMM_FUNC(char trans_a, char trans_b, int m, int n, int k,
              SCALAR alpha, const SCALAR *A, const SCALAR *B, SCALAR beta, SCALAR *C) {
    
    if (m == 0 || n == 0 || k == 0) return;
    
    if (alpha == SCALAR_ZERO) {
        if (beta == SCALAR_ZERO) {
            /* Portable zeroing (safer than memset for complex types) */
            for (int i = 0; i < m * n; i++) C[i] = SCALAR_ZERO;
        } else if (beta != SCALAR_ONE) {
            for (int i = 0; i < m * n; i++) C[i] *= beta;
        }
        return;
    }
    
    if (trans_a == 'N' && trans_b == 'N') {
        GEMM_BLOCKED(m, n, k, alpha, A, B, beta, C);
    } else if (trans_a == 'N' && trans_b == 'T') {
        GEMM_BLOCKED_NT(m, n, k, alpha, A, B, beta, C);
    } else {
        /* Generic fallback for T,N and T,T */
        if (beta != SCALAR_ONE) {
            for (int i = 0; i < m * n; i++) C[i] = (beta == SCALAR_ZERO) ? SCALAR_ZERO : beta * C[i];
        }
        int lda = (trans_a == 'N' || trans_a == 'n') ? k : m;
        int ldb = (trans_b == 'N' || trans_b == 'n') ? n : k;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                SCALAR sum = SCALAR_ZERO;
                for (int p = 0; p < k; p++) {
                    SCALAR a_val = (trans_a == 'N' || trans_a == 'n') ? A[i * lda + p] : A[p * lda + i];
                    SCALAR b_val = (trans_b == 'N' || trans_b == 'n') ? B[p * ldb + j] : B[j * ldb + p];
                    sum += a_val * b_val;
                }
                C[i * n + j] += alpha * sum;
            }
        }
    }
}

/* Cache-blocked GEMM */
static void GEMM_BLOCKED(int m, int n, int k, SCALAR alpha, 
                          const SCALAR *A, const SCALAR *B, SCALAR beta, SCALAR *C) {
    
    const int TM = TILE_M, TN = TILE_N, TK = TILE_K;
    
    if (beta != SCALAR_ONE) {
        for (int i = 0; i < m * n; i++) {
            C[i] = (beta == SCALAR_ZERO) ? SCALAR_ZERO : beta * C[i];
        }
    }
    
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(dynamic)
#endif
    for (int ii = 0; ii < m; ii += TM) {
        for (int jj = 0; jj < n; jj += TN) {
            /* Stack-allocated tile buffer per thread avoids malloc overhead and enhances cache locality */
            SCALAR tile_buf[TILE_M * TILE_N];
            for (int i = 0; i < TM * TN; i++) tile_buf[i] = SCALAR_ZERO;
            
            for (int kk = 0; kk < k; kk += TK) {
                for (int i = ii; i < ii + TM && i < m; i++) {
                    for (int p = kk; p < kk + TK && p < k; p++) {
                        SCALAR a_val = alpha * A[i * k + p];
                        for (int j = jj; j < jj + TN && j < n; j++) {
                            tile_buf[(i - ii) * TN + (j - jj)] += a_val * B[p * n + j];
                        }
                    }
                }
            }
            
            for (int i = ii; i < ii + TM && i < m; i++) {
                for (int j = jj; j < jj + TN && j < n; j++) {
                    C[i * n + j] += tile_buf[(i - ii) * TN + (j - jj)];
                }
            }
        }
    }
}

/* Cache-blocked GEMM for A(Not Transposed) * B(Transposed) */
static void GEMM_BLOCKED_NT(int m, int n, int k, SCALAR alpha, 
                             const SCALAR *A, const SCALAR *B, SCALAR beta, SCALAR *C) {
    const int TM = TILE_M, TN = TILE_N, TK = TILE_K;
    if (beta != SCALAR_ONE) {
        for (int i = 0; i < m * n; i++) {
            C[i] = (beta == SCALAR_ZERO) ? SCALAR_ZERO : beta * C[i];
        }
    }
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(dynamic)
#endif
    for (int ii = 0; ii < m; ii += TM) {
        for (int jj = 0; jj < n; jj += TN) {
            SCALAR tile_buf[TILE_M * TILE_N];
            for (int i = 0; i < TM * TN; i++) tile_buf[i] = SCALAR_ZERO;
            for (int kk = 0; kk < k; kk += TK) {
                for (int i = ii; i < ii + TM && i < m; i++) {
                    for (int p = kk; p < kk + TK && p < k; p++) {
                        SCALAR a_val = alpha * A[i * k + p];
                        for (int j = jj; j < jj + TN && j < n; j++) tile_buf[(i - ii) * TN + (j - jj)] += a_val * B[j * k + p];
                    }
                }
            }
            for (int i = ii; i < ii + TM && i < m; i++)
                for (int j = jj; j < jj + TN && j < n; j++) C[i * n + j] += tile_buf[(i - ii) * TN + (j - jj)];
        }
    }
}

#undef GEMM_FUNC
#undef GEMM_BLOCKED
#undef GEMM_BLOCKED_NT
#undef CONCAT
#undef CONCAT2


