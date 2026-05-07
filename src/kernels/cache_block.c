#include "config.h"
#include "blas.h"
#include <string.h>

/* Copy matrix tile with prefetching */
__attribute__((unused))
static inline void copy_tile_prefetch(float *dst, int dst_ld, 
                                       const float *src, int src_ld,
                                       int m, int n) {
    for (int i = 0; i < m; i++) {
        /* Prefetch next row */
        if (i + 4 < m) {
            PREFETCH_READ(&src[(i + 4) * src_ld]);
        }
        for (int j = 0; j < n; j++) {
            dst[i * dst_ld + j] = src[i * src_ld + j];
        }
    }
}

/* Machine Learning / Heuristic based auto-tuning for block sizes */
static void auto_tune_gemm_block_size(int m, int n, int k, int *tm, int *tn, int *tk) {
    /* Default to L1/L2 cache macros from config.h */
    *tm = TILE_M;
    *tn = TILE_N;
    *tk = TILE_K;
    
    /* Heuristic adjustment based on matrix dimensions */
    if (m <= 64 && n <= 64) {
        *tm = m; *tn = n; *tk = k; /* No blocking needed for tiny matrices */
    } else if (k > 1024) {
        *tk = 64; /* Increase K-tiling depth for memory-bound inner loops */
    }
    
    /* Hardware specific: Apple M-series features 128KB+ L1D, expand tiles safely */
    if (L1D_CACHE_SIZE >= 131072 && m >= 128 && n >= 128) {
        *tm = 64; *tn = 64;
    }
}

/* Cache-blocked matrix multiply with prefetching */
void sgemm_blocked_prefetch(int m, int n, int k, float alpha,
                                const float *A, int lda, const float *B, int ldb,
                                float beta, float *C, int ldc) {
    
    int TM, TN, TK;
    auto_tune_gemm_block_size(m, n, k, &TM, &TN, &TK);
    
    /* Initialize C */
    if (beta != 1.0f) {
        for (int i = 0; i < m * n; i++) {
            C[i] = (beta == 0.0f) ? 0.0f : beta * C[i];
        }
    }
    
    /* Blocked multiplication with prefetching */
    for (int ii = 0; ii < m; ii += TM) {
        for (int jj = 0; jj < n; jj += TN) {
            /* Tile accumulators in L1 */
            float tile[TM][TN];
            memset(tile, 0, TM * TN * sizeof(float));
            
            for (int kk = 0; kk < k; kk += TK) {
                /* Prefetch A and B tiles */
                if (kk + TK < k) {
                    for (int i = ii; i < ii + TM && i < m; i++) {
                        PREFETCH_READ(&A[i * lda + kk + TK]);
                    }
                    for (int p = kk; p < kk + TK && p < k; p++) {
                        PREFETCH_READ(&B[p * ldb + jj + TN]);
                    }
                }
                
                /* Compute tile product */
                for (int i = ii; i < ii + TM && i < m; i++) {
                    for (int p = kk; p < kk + TK && p < k; p++) {
                        float a_val = alpha * A[i * lda + p];
                        for (int j = jj; j < jj + TN && j < n; j++) {
                            tile[i - ii][j - jj] += a_val * B[p * ldb + j];
                        }
                    }
                }
            }
            
            /* Write back with prefetching */
            for (int i = ii; i < ii + TM && i < m; i++) {
                if (i + 4 < m) {
                    PREFETCH_WRITE(&C[(i + 4) * ldc + jj]);
                }
                for (int j = jj; j < jj + TN && j < n; j++) {
                    C[i * ldc + j] += tile[i - ii][j - jj];
                }
            }
        }
    }
}

/* Transpose matrix with cache blocking */
void transpose_blocked(float *dst, const float *src, int m, int n) {
    const int TILE = 32;
    
    for (int ii = 0; ii < n; ii += TILE) {
        for (int jj = 0; jj < m; jj += TILE) {
            for (int i = ii; i < ii + TILE && i < n; i++) {
                for (int j = jj; j < jj + TILE && j < m; j++) {
                    dst[i * m + j] = src[j * n + i];
                }
            }
        }
    }
}
