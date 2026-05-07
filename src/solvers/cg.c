#include "m4linalg.h"
#include <stdlib.h>
#include <math.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

/* Helper: Vector Dot Product */
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

/* Helper: Dense Matrix-Vector Product (Row-Major) */
static void m4_internal_gemv(const m4_matrix_t *A, const float *x, float *y) {
    int n = A->cols;
    for (int i = 0; i < A->rows; i++) {
        float sum = 0.0f;
        int j = 0;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        for (; j <= n - 4; j += 4) {
            sum_vec = vmlaq_f32(sum_vec, vld1q_f32(&A->data[i * n + j]), vld1q_f32(&x[j]));
        }
        sum = vaddvq_f32(sum_vec);
#endif
        for (; j < n; j++) sum += A->data[i * n + j] * x[j];
        y[i] = sum;
    }
}

m4_error_t m4_cg_query_workspace(int n, m4_cg_workspace_t *work) {
    if (!work || n <= 0) return M4_ERR_INVALID_ARGS;
    
    work->size = n;
    work->r = (float*)malloc(n * sizeof(float));
    work->p = (float*)malloc(n * sizeof(float));
    work->Ap = (float*)malloc(n * sizeof(float));
    work->z = (float*)malloc(n * sizeof(float));
    
    if (!work->r || !work->p || !work->Ap || !work->z) {
        m4_cg_free_workspace(work);
        return M4_ERR_OUT_OF_MEMORY;
    }
    
    return M4_SUCCESS;
}

void m4_cg_free_workspace(m4_cg_workspace_t *work) {
    if (!work) return;
    if (work->r)  { free(work->r);  work->r = NULL; }
    if (work->p)  { free(work->p);  work->p = NULL; }
    if (work->Ap) { free(work->Ap); work->Ap = NULL; }
    if (work->z)  { free(work->z);  work->z = NULL; }
    work->size = 0;
}

m4_error_t m4_solve_cg(const m4_matrix_t *A, const float *b, float *x, int max_iter, float tol, m4_cg_workspace_t *work, int *out_iters) {
    if (!A || !b || !x || !work || A->rows != A->cols || A->rows != work->size) {
        return M4_ERR_INVALID_ARGS;
    }

    int n = A->rows;
    float *r = work->r;
    float *p = work->p;
    float *Ap = work->Ap;

    /* r = b - A*x */
    m4_internal_gemv(A, x, Ap);
    for (int i = 0; i < n; i++) {
        r[i] = b[i] - Ap[i];
        p[i] = r[i]; /* p = r */
    }

    float rs_old = m4_internal_dot(n, r, r);
    
    /* Early exit if already converged */
    if (sqrtf(rs_old) < tol) {
        if (out_iters) *out_iters = 0;
        return M4_SUCCESS;
    }

    for (int iter = 0; iter < max_iter; iter++) {
        m4_internal_gemv(A, p, Ap); /* Ap = A * p */
        
        float alpha = rs_old / m4_internal_dot(n, p, Ap);
        
        for (int i = 0; i < n; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
        
        float rs_new = m4_internal_dot(n, r, r);
        if (sqrtf(rs_new) < tol) {
            if (out_iters) *out_iters = iter + 1;
            return M4_SUCCESS;
        }
        
        for (int i = 0; i < n; i++) p[i] = r[i] + (rs_new / rs_old) * p[i];
        rs_old = rs_new;
    }

    return M4_ERR_NON_CONVERGENCE;
}

m4_error_t m4_solve_pcg_jacobi(const m4_matrix_t *A, const float *b, float *x, int max_iter, float tol, m4_cg_workspace_t *work, int *out_iters) {
    if (!A || !b || !x || !work || A->rows != A->cols || A->rows != work->size) {
        return M4_ERR_INVALID_ARGS;
    }

    int n = A->rows;
    float *r = work->r;
    float *p = work->p;
    float *Ap = work->Ap;
    float *z = work->z;

    /* r = b - A*x */
    m4_internal_gemv(A, x, Ap);
    for (int i = 0; i < n; i++) {
        r[i] = b[i] - Ap[i];
        /* Apply Jacobi preconditioner: M = diag(A) */
        float diag = A->data[i * n + i];
        z[i] = r[i] / (fabsf(diag) > 1e-12f ? diag : 1e-12f);
        p[i] = z[i];
    }

    float rs_old = m4_internal_dot(n, r, z);
    
    if (sqrtf(m4_internal_dot(n, r, r)) < tol) {
        if (out_iters) *out_iters = 0;
        return M4_SUCCESS;
    }

    for (int iter = 0; iter < max_iter; iter++) {
        m4_internal_gemv(A, p, Ap);
        float alpha = rs_old / m4_internal_dot(n, p, Ap);
        for (int i = 0; i < n; i++) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
        if (sqrtf(m4_internal_dot(n, r, r)) < tol) {
            if (out_iters) *out_iters = iter + 1;
            return M4_SUCCESS;
        }
        for (int i = 0; i < n; i++) {
            float diag = A->data[i * n + i];
            z[i] = r[i] / (fabsf(diag) > 1e-12f ? diag : 1e-12f);
        }
        float rs_new = m4_internal_dot(n, r, z);
        for (int i = 0; i < n; i++) p[i] = z[i] + (rs_new / rs_old) * p[i];
        rs_old = rs_new;
    }

    return M4_ERR_NON_CONVERGENCE;
}
