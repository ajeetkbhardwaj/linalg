/* src/lapack/geqrf_template.h - Generic QR Decomposition */
#include <stdlib.h>

#ifndef SCALAR
#error "SCALAR must be defined"
#endif

#define CONCAT2(a, b) a ## b
#define CONCAT(a, b) CONCAT2(a, b)

#define GEQRF_FUNC CONCAT(PREFIX, geqrf)

int GEQRF_FUNC(MATRIX_T *A, MATRIX_T *Q, SCALAR *tau) {
    if (!A || !A->data) return -1;
    int m = A->rows, n = A->cols;
    int min_mn = (m < n) ? m : n;
    
    SCALAR *v = (SCALAR *)malloc(m * sizeof(SCALAR));
    if (!v) return -1;
    
    if (Q) {
        if (Q->rows != m || Q->cols != n) { free(v); return -1; }
        for (int i = 0; i < m * n; i++) Q->data[i] = SCALAR_ZERO;
        for (int i = 0; i < min_mn; i++) Q->data[i * n + i] = SCALAR_ONE;
    }
    if (tau) { for (int i = 0; i < min_mn; i++) tau[i] = SCALAR_ZERO; }
    
    for (int j = 0; j < min_mn; j++) {
        REAL_SCALAR x_norm = 0.0;
        for (int i = j; i < m; i++) {
#if defined(COMPLEX_MATH)
            REAL_SCALAR vr = CREAL_MACRO(A->data[i * n + j]);
            REAL_SCALAR vi = CIMAG_MACRO(A->data[i * n + j]);
            x_norm += vr*vr + vi*vi;
#else
            x_norm += A->data[i * n + j] * A->data[i * n + j];
#endif
        }
        x_norm = SQRT_MACRO(x_norm);
        
        if (x_norm < 1e-15) {
            if (tau) tau[j] = SCALAR_ZERO;
            continue;
        }
        
        SCALAR a_jj = A->data[j * n + j];
#if defined(COMPLEX_MATH)
        REAL_SCALAR a_jj_real = CREAL_MACRO(a_jj);
#else
        REAL_SCALAR a_jj_real = a_jj;
#endif
        REAL_SCALAR sign = (a_jj_real >= 0.0) ? 1.0 : -1.0;
        SCALAR v_first = a_jj + sign * x_norm;
        
        v[j] = SCALAR_ONE;
        for (int i = j + 1; i < m; i++) {
            v[i] = A->data[i * n + j] / v_first;
            A->data[i * n + j] = SCALAR_ZERO;
        }
        A->data[j * n + j] = -sign * x_norm;
        
        REAL_SCALAR v_sq = 1.0;
        for (int i = j + 1; i < m; i++) {
#if defined(COMPLEX_MATH)
            REAL_SCALAR vr = CREAL_MACRO(v[i]);
            REAL_SCALAR vi = CIMAG_MACRO(v[i]);
            v_sq += vr*vr + vi*vi;
#else
            v_sq += v[i] * v[i];
#endif
        }
        SCALAR t = (SCALAR)(2.0 / v_sq);
        if (tau) tau[j] = t;
        
        /* Apply H to remaining columns of A */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int k = j + 1; k < n; k++) {
            SCALAR sum = SCALAR_ZERO;
            for (int i = j; i < m; i++) {
#if defined(COMPLEX_MATH)
                sum += CONJ_MACRO(v[i]) * A->data[i * n + k];
#else
                sum += v[i] * A->data[i * n + k];
#endif
            }
            sum *= t;
            for (int i = j; i < m; i++) A->data[i * n + k] -= sum * v[i];
        }
        
        /* Accumulate orthogonal Q matrix */
        if (Q) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < m; i++) {
                SCALAR sum = SCALAR_ZERO;
                for (int k = j; k < m; k++) {
                    if (k < n) {
#if defined(COMPLEX_MATH)
                        sum += Q->data[i * n + k] * v[k];
#else
                        sum += Q->data[i * n + k] * v[k];
#endif
                    }
                }
                sum *= t;
                for (int k = j; k < m; k++) {
                    if (k < n) Q->data[i * n + k] -= sum * v[k];
                }
            }
        }
    }
    free(v);
    return 0;
}

#undef GEQRF_FUNC