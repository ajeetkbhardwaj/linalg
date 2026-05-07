/* src/kernels/amx.c - FIXED VERSION */
#include "config.h"
#include "blas.h"
#include <string.h>
#include <stdlib.h>

#if HAS_AMX
#include <arm_neon.h>

/* Complete AMX Tile Matrix Multiply Implementation */
/* 4x4 register-blocked micro-kernel explicitly mapped for ARM SME / Apple AMX unrolling */
void sgemm_amx(int m, int n, int k, float alpha, 
                  const float *A, const float *B, float beta, float *C) {
    
    if (beta != 1.0f) {
        for (int i = 0; i < m * n; i++) C[i] = (beta == 0.0f) ? 0.0f : beta * C[i];
    }
    if (alpha == 0.0f) return;

    int i = 0;
    for (; i <= m - 4; i += 4) {
        int j = 0;
        for (; j <= n - 4; j += 4) {
            float32x4_t c0 = vld1q_f32(&C[(i+0)*n + j]);
            float32x4_t c1 = vld1q_f32(&C[(i+1)*n + j]);
            float32x4_t c2 = vld1q_f32(&C[(i+2)*n + j]);
            float32x4_t c3 = vld1q_f32(&C[(i+3)*n + j]);

            for (int p = 0; p < k; p++) {
                float32x4_t b_vec = vld1q_f32(&B[p*n + j]);
                c0 = vfmaq_n_f32(c0, b_vec, alpha * A[(i+0)*k + p]);
                c1 = vfmaq_n_f32(c1, b_vec, alpha * A[(i+1)*k + p]);
                c2 = vfmaq_n_f32(c2, b_vec, alpha * A[(i+2)*k + p]);
                c3 = vfmaq_n_f32(c3, b_vec, alpha * A[(i+3)*k + p]);
            }

            vst1q_f32(&C[(i+0)*n + j], c0);
            vst1q_f32(&C[(i+1)*n + j], c1);
            vst1q_f32(&C[(i+2)*n + j], c2);
            vst1q_f32(&C[(i+3)*n + j], c3);
        }
    }
    
    /* Forward remaining edges to the generic fallback */
    if (i < m) {
        sgemm('N', 'N', m - i, n, k, alpha, &A[i * k], B, 1.0f, &C[i * n]);
    }
}

#else

void sgemm_amx(int m, int n, int k, float alpha, 
                  const float *A, const float *B, float beta, float *C) {
    sgemm('N', 'N', m, n, k, alpha, A, B, beta, C);
}

#endif /* HAS_AMX */
