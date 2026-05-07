#include "../../include/lapack.h"
#include "../../include/blas.h"
#include <math.h>
#include <stdlib.h>

/* One-sided Jacobi SVD: A * V = U * Sigma */
int gesvd_jacobi(const matrix_t *A, matrix_t *U, float *S, 
                 matrix_t *V, int max_iter, float tol) {
    if (!A || !A->data || !S) return -1;
    
    int m = A->rows, n = A->cols;
    if (max_iter <= 0) max_iter = 30;
    if (tol <= 0.0f) tol = 1e-6f;
    
    matrix_t tempA = matrix_alloc(m, n);
    for (int i = 0; i < m * n; i++) tempA.data[i] = A->data[i];
    
    if (V) {
        for (int i = 0; i < n * n; i++) V->data[i] = (i % (n + 1) == 0) ? 1.0f : 0.0f;
    }
    
    int converged = 0;
    for (int iter = 0; iter < max_iter; iter++) {
        converged = 1;
        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) {
                float app = 0.0f, aqq = 0.0f, apq = 0.0f;
                for (int i = 0; i < m; i++) {
                    app += tempA.data[i * n + p] * tempA.data[i * n + p];
                    aqq += tempA.data[i * n + q] * tempA.data[i * n + q];
                    apq += tempA.data[i * n + p] * tempA.data[i * n + q];
                }
                
                if (fabsf(apq) > tol * sqrtf(app * aqq)) {
                    converged = 0;
                    float tau = (aqq - app) / (2.0f * apq);
                    float t = (tau >= 0.0f) ? 1.0f / (tau + sqrtf(1.0f + tau * tau)) 
                                            : -1.0f / (-tau + sqrtf(1.0f + tau * tau));
                    float c = 1.0f / sqrtf(1.0f + t * t);
                    float s = t * c;
                    
                    /* Apply Jacobi Rotation */
                    for (int i = 0; i < m; i++) {
                        float temp_p = tempA.data[i * n + p];
                        float temp_q = tempA.data[i * n + q];
                        tempA.data[i * n + p] = c * temp_p - s * temp_q;
                        tempA.data[i * n + q] = s * temp_p + c * temp_q;
                    }
                    if (V) {
                        for (int i = 0; i < n; i++) {
                            float vp = V->data[i * n + p];
                            float vq = V->data[i * n + q];
                            V->data[i * n + p] = c * vp - s * vq;
                            V->data[i * n + q] = s * vp + c * vq;
                        }
                    }
                }
            }
        }
        if (converged) break;
    }
    
    for (int j = 0; j < n; j++) {
        float norm = 0.0f;
        for (int i = 0; i < m; i++) norm += tempA.data[i * n + j] * tempA.data[i * n + j];
        S[j] = sqrtf(norm);
        if (U && S[j] > 1e-15f) {
            for (int i = 0; i < m; i++) U->data[i * n + j] = tempA.data[i * n + j] / S[j];
        }
    }
    
    /* Sort singular values and corresponding columns of U and V in descending order */
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (S[i] < S[j]) {
                float tmpS = S[i];
                S[i] = S[j];
                S[j] = tmpS;
                if (U) {
                    for (int k = 0; k < m; k++) {
                        float tmpU = U->data[k * n + i];
                        U->data[k * n + i] = U->data[k * n + j];
                        U->data[k * n + j] = tmpU;
                    }
                }
                if (V) {
                    for (int k = 0; k < n; k++) {
                        float tmpV = V->data[k * n + i];
                        V->data[k * n + i] = V->data[k * n + j];
                        V->data[k * n + j] = tmpV;
                    }
                }
            }
        }
    }

    matrix_free(&tempA);
    return converged ? 0 : -1;
}
