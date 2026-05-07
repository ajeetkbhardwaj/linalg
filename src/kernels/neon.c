#include "config.h"
#include <arm_neon.h>
#include <string.h>

#if HAS_NEON

/* NEON-optimized vector add */
void vec_add_neon(int n, const float *a, const float *b, float *res) {
    int i = 0;
    
    /* Process 8 floats at a time with unrolling */
    for (; i <= n - 8; i += 8) {
        float32x4_t va0 = vld1q_f32(&a[i]);
        float32x4_t vb0 = vld1q_f32(&b[i]);
        float32x4_t va1 = vld1q_f32(&a[i+4]);
        float32x4_t vb1 = vld1q_f32(&b[i+4]);
        
        vst1q_f32(&res[i], vaddq_f32(va0, vb0));
        vst1q_f32(&res[i+4], vaddq_f32(va1, vb1));
    }
    
    /* Remainder */
    for (; i < n; i++) {
        res[i] = a[i] + b[i];
    }
}

/* NEON SAXPY with FMA */
void saxpy_neon(int n, float alpha, const float *x, float *y) {
    int i = 0;
    float32x4_t valpha = vdupq_n_f32(alpha);
    
    for (; i <= n - 8; i += 8) {
        float32x4_t vx0 = vld1q_f32(&x[i]);
        float32x4_t vy0 = vld1q_f32(&y[i]);
        float32x4_t vx1 = vld1q_f32(&x[i+4]);
        float32x4_t vy1 = vld1q_f32(&y[i+4]);
        
        vst1q_f32(&y[i], vfmaq_f32(vy0, valpha, vx0));
        vst1q_f32(&y[i+4], vfmaq_f32(vy1, valpha, vx1));
    }
    
    for (; i < n; i++) {
        y[i] = FMA(alpha, x[i], y[i]);
    }
}

/* NEON dot product with horizontal add */
float sdot_neon(int n, const float *x, const float *y) {
    float32x4_t vsum = vdupq_n_f32(0.0f);
    int i = 0;
    
    for (; i <= n - 8; i += 8) {
        float32x4_t vx0 = vld1q_f32(&x[i]);
        float32x4_t vy0 = vld1q_f32(&y[i]);
        float32x4_t vx1 = vld1q_f32(&x[i+4]);
        float32x4_t vy1 = vld1q_f32(&y[i+4]);
        
        vsum = vfmaq_f32(vsum, vx0, vy0);
        vsum = vfmaq_f32(vsum, vx1, vy1);
    }
    
    float sum = vaddvq_f32(vsum);
    
    for (; i < n; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}

/* NEON matrix-vector multiply (row-major A) */
void sgemv_neon(int m, int n, const float *A, const float *x, float *y) {
    for (int i = 0; i < m; i++) {
        const float *row = &A[i * n];
        float sum = 0.0f;
        int j = 0;
        
        /* NEON inner product */
        for (; j <= n - 4; j += 4) {
            float32x4_t va = vld1q_f32(&row[j]);
            float32x4_t vx = vld1q_f32(&x[j]);
            sum += vaddvq_f32(vmulq_f32(va, vx));
        }
        
        for (; j < n; j++) {
            sum += row[j] * x[j];
        }
        y[i] = sum;
    }
}

#endif /* HAS_NEON */
