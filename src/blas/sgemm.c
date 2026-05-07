#include "m4linalg.h"
#include <dispatch/dispatch.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

m4_error_t m4_sgemm(char transa, char transb, int m, int n, int k,
                    float alpha, const float *A, int lda,
                    const float *B, int ldb,
                    float beta, float *C, int ldc) {
    
    if (transa != 'N' || transb != 'N') return M4_ERR_INVALID_ARGS;
    
    /* Pre-scale C matrix by beta */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (beta == 0.0f) {
                C[i * ldc + j] = 0.0f;
            } else if (beta != 1.0f) {
                C[i * ldc + j] *= beta;
            }
        }
    }
    
    if (alpha == 0.0f) return M4_SUCCESS;

    /* Retrieve dynamically tuned cache block sizes */
    int l1_tile = m4_get_l1_tile_size();
    int l2_tile = m4_get_l2_tile_size();
    if (l1_tile <= 0) l1_tile = 32;
    if (l2_tile <= 0) l2_tile = 64;

    /* GCD Parallelization over independent column blocks (j dimension) */
    int j_blocks = (n + l2_tile - 1) / l2_tile;
    
    dispatch_apply(j_blocks, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t j_idx) {
        int j0 = (int)j_idx * l2_tile;
        int j_end = (j0 + l2_tile > n) ? n : j0 + l2_tile;
        
        /* Cache-Blocked loops */
        for (int k0 = 0; k0 < k; k0 += l2_tile) {
            int k_end = (k0 + l2_tile > k) ? k : k0 + l2_tile;
            for (int i0 = 0; i0 < m; i0 += l1_tile) {
                int i_end = (i0 + l1_tile > m) ? m : i0 + l1_tile;
                
                /* Inner i-k-j loop for contiguous row-major access */
                int i = i0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
                /* 8x4 Register-Blocked NEON Microkernel */
                for (; i <= i_end - 8; i += 8) {
                    int j = j0;
                    for (; j <= j_end - 4; j += 4) {
                        float32x4_t c0 = vdupq_n_f32(0.0f);
                        float32x4_t c1 = vdupq_n_f32(0.0f);
                        float32x4_t c2 = vdupq_n_f32(0.0f);
                        float32x4_t c3 = vdupq_n_f32(0.0f);
                        float32x4_t c4 = vdupq_n_f32(0.0f);
                        float32x4_t c5 = vdupq_n_f32(0.0f);
                        float32x4_t c6 = vdupq_n_f32(0.0f);
                        float32x4_t c7 = vdupq_n_f32(0.0f);

                        for (int l = k0; l < k_end; l++) {
                            float32x4_t b_vec = vld1q_f32(&B[l * ldb + j]);
                            
                            c0 = vmlaq_n_f32(c0, b_vec, alpha * A[(i+0) * lda + l]);
                            c1 = vmlaq_n_f32(c1, b_vec, alpha * A[(i+1) * lda + l]);
                            c2 = vmlaq_n_f32(c2, b_vec, alpha * A[(i+2) * lda + l]);
                            c3 = vmlaq_n_f32(c3, b_vec, alpha * A[(i+3) * lda + l]);
                            c4 = vmlaq_n_f32(c4, b_vec, alpha * A[(i+4) * lda + l]);
                            c5 = vmlaq_n_f32(c5, b_vec, alpha * A[(i+5) * lda + l]);
                            c6 = vmlaq_n_f32(c6, b_vec, alpha * A[(i+6) * lda + l]);
                            c7 = vmlaq_n_f32(c7, b_vec, alpha * A[(i+7) * lda + l]);
                        }

                        vst1q_f32(&C[(i+0) * ldc + j], vaddq_f32(vld1q_f32(&C[(i+0) * ldc + j]), c0));
                        vst1q_f32(&C[(i+1) * ldc + j], vaddq_f32(vld1q_f32(&C[(i+1) * ldc + j]), c1));
                        vst1q_f32(&C[(i+2) * ldc + j], vaddq_f32(vld1q_f32(&C[(i+2) * ldc + j]), c2));
                        vst1q_f32(&C[(i+3) * ldc + j], vaddq_f32(vld1q_f32(&C[(i+3) * ldc + j]), c3));
                        vst1q_f32(&C[(i+4) * ldc + j], vaddq_f32(vld1q_f32(&C[(i+4) * ldc + j]), c4));
                        vst1q_f32(&C[(i+5) * ldc + j], vaddq_f32(vld1q_f32(&C[(i+5) * ldc + j]), c5));
                        vst1q_f32(&C[(i+6) * ldc + j], vaddq_f32(vld1q_f32(&C[(i+6) * ldc + j]), c6));
                        vst1q_f32(&C[(i+7) * ldc + j], vaddq_f32(vld1q_f32(&C[(i+7) * ldc + j]), c7));
                    }
                    
                    /* Remaining columns (j edge cases) */
                    for (; j < j_end; j++) {
                        float sum[8] = {0};
                        for (int l = k0; l < k_end; l++) {
                            float b_val = B[l * ldb + j];
                            sum[0] += alpha * A[(i+0) * lda + l] * b_val;
                            sum[1] += alpha * A[(i+1) * lda + l] * b_val;
                            sum[2] += alpha * A[(i+2) * lda + l] * b_val;
                            sum[3] += alpha * A[(i+3) * lda + l] * b_val;
                            sum[4] += alpha * A[(i+4) * lda + l] * b_val;
                            sum[5] += alpha * A[(i+5) * lda + l] * b_val;
                            sum[6] += alpha * A[(i+6) * lda + l] * b_val;
                            sum[7] += alpha * A[(i+7) * lda + l] * b_val;
                        }
                        for(int x = 0; x < 8; x++) C[(i+x) * ldc + j] += sum[x];
                    }
                }
#endif
                /* Serial Fallback / Remaining Rows (i edge cases) */
                for (; i < i_end; i++) {
                    for (int l = k0; l < k_end; l++) {
                        float a_val = alpha * A[i * lda + l];
                        for (int j = j0; j < j_end; j++) {
                            C[i * ldc + j] += a_val * B[l * ldb + j];
                        }
                    }
                }
            }
        }
    });
    
    return M4_SUCCESS;
}

m4_error_t m4_sgemm_batched(char transa, char transb, int m, int n, int k,
                            float alpha, const float *A, int lda, long long stride_a,
                            const float *B, int ldb, long long stride_b,
                            float beta, float *C, int ldc, long long stride_c,
                            int batch_count) {
                                
    if (batch_count <= 0 || !A || !B || !C) return M4_ERR_INVALID_ARGS;
    
    /* Dispatch the workload across all available hardware cores concurrently */
    dispatch_apply(batch_count, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t batch_idx) {
        /* Calculate memory offset for the current batch */
        const float *A_batch = A + (batch_idx * stride_a);
        const float *B_batch = B + (batch_idx * stride_b);
        float *C_batch = C + (batch_idx * stride_c);
        
        /* Process the GEMM for this specific matrix */
        m4_sgemm(transa, transb, m, n, k, 
                 alpha, A_batch, lda, 
                 B_batch, ldb, 
                 beta, C_batch, ldc);
    });
    
    return M4_SUCCESS;
}
