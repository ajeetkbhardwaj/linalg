#include "m4linalg.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

/* Helper: Row-major dense matrix-vector product y = alpha * A * x + beta * y */
static void m4_internal_gemv(int m, int n, float alpha, const float *A, const float *x, float beta, float *y) {
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        int j = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        for (; j <= n - 4; j += 4) {
            sum_vec = vmlaq_f32(sum_vec, vld1q_f32(&A[i * n + j]), vld1q_f32(&x[j]));
        }
        sum = vaddvq_f32(sum_vec);
#endif
        for (; j < n; j++) sum += A[i * n + j] * x[j];
        y[i] = beta * y[i] + alpha * sum;
    }
}

static float m4_internal_dot(int n, const float *x, const float *y) {
    float sum = 0.0f;
    int i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    for (; i <= n - 4; i += 4) {
        sum_vec = vmlaq_f32(sum_vec, vld1q_f32(&x[i]), vld1q_f32(&y[i]));
    }
    sum = vaddvq_f32(sum_vec);
#endif
    for (; i < n; i++) sum += x[i] * y[i];
    return sum;
}

static float m4_internal_nrm2(int n, const float *x) {
    float sum = 0.0f;
    int i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    for (; i <= n - 4; i += 4) {
        float32x4_t vx = vld1q_f32(&x[i]);
        sum_vec = vmlaq_f32(sum_vec, vx, vx);
    }
    sum = vaddvq_f32(sum_vec);
#endif
    for (; i < n; i++) sum += x[i] * x[i];
    return sqrtf(sum);
}

static void m4_internal_axpy(int n, float alpha, const float *x, float *y) {
    int i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    float32x4_t valpha = vdupq_n_f32(alpha);
    for (; i <= n - 4; i += 4) {
        float32x4_t vy = vld1q_f32(&y[i]);
        vy = vmlaq_f32(vy, valpha, vld1q_f32(&x[i]));
        vst1q_f32(&y[i], vy);
    }
#endif
    for (; i < n; i++) y[i] += alpha * x[i];
}

static void m4_internal_scal(int n, float alpha, float *x) {
    int i = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    float32x4_t valpha = vdupq_n_f32(alpha);
    for (; i <= n - 4; i += 4) {
        float32x4_t vx = vld1q_f32(&x[i]);
        vx = vmulq_f32(vx, valpha);
        vst1q_f32(&x[i], vx);
    }
#endif
    for (; i < n; i++) x[i] *= alpha;
}

m4_error_t m4_gmres_query_workspace(int n, int restart, m4_gmres_workspace_t *ws) {
    if (!ws || n <= 0 || restart <= 0) return M4_ERR_INVALID_ARGS;
    ws->n = n;
    ws->restart = restart;
    ws->V = (float*)malloc(n * (restart + 1) * sizeof(float));
    ws->H = (float*)malloc(restart * (restart + 1) * sizeof(float));
    ws->cs = (float*)malloc(restart * sizeof(float));
    ws->sn = (float*)malloc(restart * sizeof(float));
    ws->s = (float*)malloc((restart + 1) * sizeof(float));
    ws->r = (float*)malloc(n * sizeof(float));
    ws->w = (float*)malloc(n * sizeof(float));
    ws->y = (float*)malloc(restart * sizeof(float));
    ws->z = (float*)malloc(n * sizeof(float));
    
    if (!ws->V || !ws->H || !ws->cs || !ws->sn || !ws->s || !ws->r || !ws->w || !ws->y || !ws->z) {
        m4_gmres_free_workspace(ws);
        return M4_ERR_OUT_OF_MEMORY;
    }
    return M4_SUCCESS;
}

void m4_gmres_free_workspace(m4_gmres_workspace_t *ws) {
    if (!ws) return;
    free(ws->V); ws->V = NULL;
    free(ws->H); ws->H = NULL;
    free(ws->cs); ws->cs = NULL;
    free(ws->sn); ws->sn = NULL;
    free(ws->s); ws->s = NULL;
    free(ws->r); ws->r = NULL;
    free(ws->w); ws->w = NULL;
    free(ws->y); ws->y = NULL;
    free(ws->z); ws->z = NULL;
    ws->n = 0;
    ws->restart = 0;
}

m4_error_t m4_solve_gmres(const m4_matrix_t *A, const float *b, float *x,
                          int restart, int max_iter, float tol, m4_gmres_workspace_t *ws, int *out_iters) {
    if (!A || !A->data || !b || !x || !ws) return M4_ERR_INVALID_ARGS;
    
    int n = A->rows;
    if (A->cols != n) return M4_ERR_INVALID_ARGS;
    if (ws->n < n || ws->restart < restart) return M4_ERR_INVALID_ARGS;
    
    if (max_iter <= 0) max_iter = n;
    if (tol <= 0.0f) tol = 1e-6f; 
    
    float *V = ws->V;
    float *H = ws->H;
    float *cs = ws->cs;
    float *sn = ws->sn;
    float *s = ws->s;
    float *r = ws->r;
    
    memcpy(r, b, n * sizeof(float));
    m4_internal_gemv(n, n, -1.0f, A->data, x, 1.0f, r);
    
    float beta = m4_internal_nrm2(n, r);
    if (beta < tol) {
        if (out_iters) *out_iters = 0;
        return M4_SUCCESS; 
    }
    
    int total_iter = 0;
    
    /* Restarted GMRES loop */
    while (total_iter < max_iter) {
        int m = (total_iter + restart <= max_iter) ? restart : max_iter - total_iter;
        
        m4_internal_scal(n, 1.0f / beta, r);
        memcpy(&V[0 * n], r, n * sizeof(float));
        
        memset(s, 0, (restart + 1) * sizeof(float));
        s[0] = beta;
        
        /* Arnoldi process */
        int j;
        for (j = 0; j < m; j++) {
            /* w = A * v_j */
            float *v_j = &V[j * n];
            float *w = &V[(j + 1) * n];
            
            for (int i=0; i<n; i++) w[i] = 0.0f;
            m4_internal_gemv(n, n, 1.0f, A->data, v_j, 0.0f, w);
            
            /* Modified Gram-Schmidt */
            for (int i = 0; i <= j; i++) {
                float *v_i = &V[i * n];
                H[i + j * (restart + 1)] = m4_internal_dot(n, v_i, w);
                m4_internal_axpy(n, -H[i + j * (restart + 1)], v_i, w);
            }
            H[j + 1 + j * (restart + 1)] = m4_internal_nrm2(n, w);
            
            /* Breakdown check */
            if (H[j + 1 + j * (restart + 1)] < 1e-12f) {
                j++;  /* Lucky breakdown */
                break;
            }
            
            m4_internal_scal(n, 1.0f / H[j + 1 + j * (restart + 1)], w);
            
            /* Apply previous Givens rotations */
            for (int i = 0; i < j; i++) {
                float tmp = cs[i] * H[i + j * (restart + 1)] + 
                           sn[i] * H[i + 1 + j * (restart + 1)];
                H[i + 1 + j * (restart + 1)] = -sn[i] * H[i + j * (restart + 1)] + 
                                               cs[i] * H[i + 1 + j * (restart + 1)];
                H[i + j * (restart + 1)] = tmp;
            }
            
            /* Compute new Givens rotation */
            float denom = sqrtf(H[j + j * (restart + 1)] * H[j + j * (restart + 1)] + 
                                H[j + 1 + j * (restart + 1)] * H[j + 1 + j * (restart + 1)]);
            if (denom < 1e-12f) denom = 1e-12f;
            cs[j] = H[j + j * (restart + 1)] / denom;
            sn[j] = H[j + 1 + j * (restart + 1)] / denom;
            
            /* Apply to RHS */
            float tmp = cs[j] * s[j] + sn[j] * s[j + 1];
            s[j + 1] = -sn[j] * s[j] + cs[j] * s[j + 1];
            s[j] = tmp;
            
            /* Update residual norm estimate */
            H[j + j * (restart + 1)] = cs[j] * H[j + j * (restart + 1)] + 
                                       sn[j] * H[j + 1 + j * (restart + 1)];
            H[j + 1 + j * (restart + 1)] = 0.0f;
            
            if (fabsf(s[j + 1]) < tol) {
                j++;  /* Converged */
                break;
            }
        }
        
        /* Solve least squares problem: min ||H*y - s|| */
        /* Back-substitute in upper triangular H */
        for (int i = j - 1; i >= 0; i--) {
            s[i] /= H[i + i * (restart + 1)];
            for (int k = 0; k < i; k++) {
                s[k] -= H[k + i * (restart + 1)] * s[i];
            }
        }
        
        /* Update solution: x = x + V*y */
        for (int i = 0; i < j; i++) {
            m4_internal_axpy(n, s[i], &V[i * n], x);
        }
        
        total_iter += j;
        
        /* Check convergence */
        if (fabsf(s[j]) < tol) {
            if (out_iters) *out_iters = total_iter;
            return M4_SUCCESS;
        }
        
        /* Compute new residual for restart */
        memcpy(r, b, n * sizeof(float));
        m4_internal_gemv(n, n, -1.0f, A->data, x, 1.0f, r);
        beta = m4_internal_nrm2(n, r);
        
        if (beta < tol) {
            if (out_iters) *out_iters = total_iter;
            return M4_SUCCESS;
        }
    }
    
    if (out_iters) *out_iters = total_iter;
    return M4_ERR_NON_CONVERGENCE;
}

m4_error_t m4_solve_pgmres_ilu(const m4_matrix_t *A, const float *b, float *x,
                               int restart, int max_iter, float tol, m4_gmres_workspace_t *ws, float drop_tol, int *out_iters) {
    if (!A || !A->data || !b || !x || !ws) return M4_ERR_INVALID_ARGS;
    
    int n = A->rows;
    if (A->cols != n) return M4_ERR_INVALID_ARGS;
    if (ws->n < n || ws->restart < restart) return M4_ERR_INVALID_ARGS;
    if (max_iter <= 0) max_iter = n;
    if (tol <= 0.0f) tol = 1e-6f; 
    
    /* Create ILU Preconditioner Matrix */
    m4_matrix_t ILU = { .data = (float*)malloc(n * n * sizeof(float)), .rows = n, .cols = n, .owns_data = true };
    if (!ILU.data) return M4_ERR_OUT_OF_MEMORY;
    memcpy(ILU.data, A->data, n * n * sizeof(float));
    
    m4_error_t err = m4_ilu_factor(&ILU, drop_tol);
    if (err != M4_SUCCESS) {
        free(ILU.data);
        return err;
    }
    
    float *V = ws->V, *H = ws->H, *cs = ws->cs, *sn = ws->sn;
    float *s = ws->s, *r = ws->r, *z = ws->z;
    
    memcpy(r, b, n * sizeof(float));
    m4_internal_gemv(n, n, -1.0f, A->data, x, 1.0f, r);
    
    float beta = m4_internal_nrm2(n, r);
    if (beta < tol) {
        if (out_iters) *out_iters = 0;
        free(ILU.data);
        return M4_SUCCESS; 
    }
    
    int total_iter = 0;
    
    while (total_iter < max_iter) {
        int m = (total_iter + restart <= max_iter) ? restart : max_iter - total_iter;
        
        m4_internal_scal(n, 1.0f / beta, r);
        memcpy(&V[0 * n], r, n * sizeof(float));
        
        memset(s, 0, (restart + 1) * sizeof(float));
        s[0] = beta;
        
        int j;
        for (j = 0; j < m; j++) {
            float *v_j = &V[j * n];
            float *w = &V[(j + 1) * n];
            
            /* Right Preconditioning Application: w = A * M^{-1} * v_j */
            memcpy(z, v_j, n * sizeof(float));
            m4_ilu_solve(&ILU, z);
            
            for (int i=0; i<n; i++) w[i] = 0.0f;
            m4_internal_gemv(n, n, 1.0f, A->data, z, 0.0f, w);
            
            for (int i = 0; i <= j; i++) {
                float *v_i = &V[i * n];
                H[i + j * (restart + 1)] = m4_internal_dot(n, v_i, w);
                m4_internal_axpy(n, -H[i + j * (restart + 1)], v_i, w);
            }
            H[j + 1 + j * (restart + 1)] = m4_internal_nrm2(n, w);
            
            if (H[j + 1 + j * (restart + 1)] < 1e-12f) { j++; break; }
            
            m4_internal_scal(n, 1.0f / H[j + 1 + j * (restart + 1)], w);
            
            for (int i = 0; i < j; i++) {
                float tmp = cs[i] * H[i + j * (restart + 1)] + sn[i] * H[i + 1 + j * (restart + 1)];
                H[i + 1 + j * (restart + 1)] = -sn[i] * H[i + j * (restart + 1)] + cs[i] * H[i + 1 + j * (restart + 1)];
                H[i + j * (restart + 1)] = tmp;
            }
            
            float denom = sqrtf(H[j + j * (restart + 1)] * H[j + j * (restart + 1)] + H[j + 1 + j * (restart + 1)] * H[j + 1 + j * (restart + 1)]);
            if (denom < 1e-12f) denom = 1e-12f;
            cs[j] = H[j + j * (restart + 1)] / denom;
            sn[j] = H[j + 1 + j * (restart + 1)] / denom;
            
            float tmp = cs[j] * s[j] + sn[j] * s[j + 1];
            s[j + 1] = -sn[j] * s[j] + cs[j] * s[j + 1];
            s[j] = tmp;
            
            H[j + j * (restart + 1)] = cs[j] * H[j + j * (restart + 1)] + sn[j] * H[j + 1 + j * (restart + 1)];
            H[j + 1 + j * (restart + 1)] = 0.0f;
            
            if (fabsf(s[j + 1]) < tol) { j++; break; }
        }
        
        for (int i = j - 1; i >= 0; i--) {
            s[i] /= H[i + i * (restart + 1)];
            for (int k = 0; k < i; k++) s[k] -= H[k + i * (restart + 1)] * s[i];
        }
        
        /* Update solution: x = x + M^{-1} * V * y */
        memset(z, 0, n * sizeof(float));
        for (int i = 0; i < j; i++) m4_internal_axpy(n, s[i], &V[i * n], z);
        m4_ilu_solve(&ILU, z);
        for (int i = 0; i < n; i++) x[i] += z[i];
        
        total_iter += j;
        if (fabsf(s[j]) < tol) break;
        
        memcpy(r, b, n * sizeof(float));
        m4_internal_gemv(n, n, -1.0f, A->data, x, 1.0f, r);
        beta = m4_internal_nrm2(n, r);
        if (beta < tol) break;
    }
    
    if (out_iters) *out_iters = total_iter;
    free(ILU.data);
    return (total_iter < max_iter) ? M4_SUCCESS : M4_ERR_NON_CONVERGENCE;
}
