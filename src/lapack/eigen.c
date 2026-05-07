#include "m4linalg.h"
#include <math.h>

/* Robust Jacobi Eigenvalue Algorithm for Symmetric Matrices */
m4_error_t m4_syev(m4_matrix_t *A, float *w) {
    if (!A || !A->data || !w || A->rows != A->cols) return M4_ERR_INVALID_ARGS;
    int n = A->rows;
    
    /* Initialize eigenvalues to diagonal elements */
    for (int i = 0; i < n; i++) {
        w[i] = A->data[i * n + i];
    }
    
    int max_iter = 50;
    int iter = 0;
    for (; iter < max_iter; iter++) {
        float sm = 0.0f;
        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) {
                sm += fabsf(A->data[p * n + q]);
            }
        }
        if (sm == 0.0f) break; /* Converged */
        
        float tresh = (iter < 3) ? 0.2f * sm / (n * n) : 0.0f;
        
        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) {
                float g = 100.0f * fabsf(A->data[p * n + q]);
                if (iter > 3 && fabsf(w[p]) + g == fabsf(w[p]) && fabsf(w[q]) + g == fabsf(w[q])) {
                    A->data[p * n + q] = 0.0f;
                } else if (fabsf(A->data[p * n + q]) > tresh) {
                    float h = w[q] - w[p];
                    float t;
                    if (fabsf(h) + g == fabsf(h)) {
                        t = A->data[p * n + q] / h;
                    } else {
                        float theta = 0.5f * h / A->data[p * n + q];
                        t = 1.0f / (fabsf(theta) + sqrtf(1.0f + theta * theta));
                        if (theta < 0.0f) t = -t;
                    }
                    float c = 1.0f / sqrtf(1.0f + t * t);
                    float s = t * c;
                    float tau = s / (1.0f + c);
                    float h_val = t * A->data[p * n + q];
                    
                    w[p] -= h_val;
                    w[q] += h_val;
                    A->data[p * n + q] = 0.0f;
                    
                    for (int j = 0; j < p; j++) {
                        g = A->data[j * n + p]; h = A->data[j * n + q];
                        A->data[j * n + p] = g - s * (h + g * tau); A->data[j * n + q] = h + s * (g - h * tau);
                    }
                    for (int j = p + 1; j < q; j++) {
                        g = A->data[p * n + j]; h = A->data[j * n + q];
                        A->data[p * n + j] = g - s * (h + g * tau); A->data[j * n + q] = h + s * (g - h * tau);
                    }
                    for (int j = q + 1; j < n; j++) {
                        g = A->data[p * n + j]; h = A->data[q * n + j];
                        A->data[p * n + j] = g - s * (h + g * tau); A->data[q * n + j] = h + s * (g - h * tau);
                    }
                }
            }
        }
    }
    
    /* Sort eigenvalues in ascending order */
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (w[i] > w[j]) {
                float tmp = w[i]; w[i] = w[j]; w[j] = tmp;
            }
        }
    }
    return (iter < max_iter) ? M4_SUCCESS : M4_ERR_NON_CONVERGENCE;
}

m4_error_t m4_syev_evec(m4_matrix_t *A, float *w, m4_matrix_t *V) {
    if (!A || !A->data || !w || !V || !V->data || A->rows != A->cols || V->rows != V->cols || A->rows != V->rows) {
        return M4_ERR_INVALID_ARGS;
    }
    int n = A->rows;
    
    for (int i = 0; i < n; i++) {
        w[i] = A->data[i * n + i];
        for (int j = 0; j < n; j++) {
            V->data[i * n + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
    
    int max_iter = 50;
    int iter = 0;
    for (; iter < max_iter; iter++) {
        float sm = 0.0f;
        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) sm += fabsf(A->data[p * n + q]);
        }
        if (sm == 0.0f) break; /* Converged */
        
        float tresh = (iter < 3) ? 0.2f * sm / (n * n) : 0.0f;
        
        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) {
                float g = 100.0f * fabsf(A->data[p * n + q]);
                if (iter > 3 && fabsf(w[p]) + g == fabsf(w[p]) && fabsf(w[q]) + g == fabsf(w[q])) {
                    A->data[p * n + q] = 0.0f;
                } else if (fabsf(A->data[p * n + q]) > tresh) {
                    float h = w[q] - w[p];
                    float t;
                    if (fabsf(h) + g == fabsf(h)) t = A->data[p * n + q] / h;
                    else {
                        float theta = 0.5f * h / A->data[p * n + q];
                        t = 1.0f / (fabsf(theta) + sqrtf(1.0f + theta * theta));
                        if (theta < 0.0f) t = -t;
                    }
                    float c = 1.0f / sqrtf(1.0f + t * t);
                    float s = t * c;
                    float tau = s / (1.0f + c);
                    float h_val = t * A->data[p * n + q];
                    
                    w[p] -= h_val; w[q] += h_val; A->data[p * n + q] = 0.0f;
                    
                    for (int j = 0; j < p; j++) {
                        g = A->data[j * n + p]; h = A->data[j * n + q];
                        A->data[j * n + p] = g - s * (h + g * tau); A->data[j * n + q] = h + s * (g - h * tau);
                    }
                    for (int j = p + 1; j < q; j++) {
                        g = A->data[p * n + j]; h = A->data[j * n + q];
                        A->data[p * n + j] = g - s * (h + g * tau); A->data[j * n + q] = h + s * (g - h * tau);
                    }
                    for (int j = q + 1; j < n; j++) {
                        g = A->data[p * n + j]; h = A->data[q * n + j];
                        A->data[p * n + j] = g - s * (h + g * tau); A->data[q * n + j] = h + s * (g - h * tau);
                    }
                    for (int i = 0; i < n; i++) {
                        float vi = V->data[i * n + p]; float vj = V->data[i * n + q];
                        V->data[i * n + p] = c * vi - s * vj; V->data[i * n + q] = s * vi + c * vj;
                    }
                }
            }
        }
    }
    
    /* Sort eigenvalues and eigenvectors in ascending order */
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (w[i] > w[j]) {
                float tmp = w[i]; w[i] = w[j]; w[j] = tmp;
                for (int k = 0; k < n; k++) {
                    float tmp_v = V->data[k * n + i];
                    V->data[k * n + i] = V->data[k * n + j];
                    V->data[k * n + j] = tmp_v;
                }
            }
        }
    }
    return (iter < max_iter) ? M4_SUCCESS : M4_ERR_NON_CONVERGENCE;
}
