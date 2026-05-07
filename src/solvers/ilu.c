#include "m4linalg.h"
#include <math.h>

m4_error_t m4_ilu_factor(m4_matrix_t *A, float drop_tol) {
    if (!A || !A->data || A->rows != A->cols) return M4_ERR_INVALID_ARGS;
    int n = A->rows;
    
    for (int i = 0; i < n; i++) {
        if (fabsf(A->data[i * n + i]) < 1e-12f) return M4_ERR_SINGULAR;
        float pivot_inv = 1.0f / A->data[i * n + i];
        
        for (int k = i + 1; k < n; k++) {
            if (fabsf(A->data[k * n + i]) > drop_tol) {
                A->data[k * n + i] *= pivot_inv;
                float factor = A->data[k * n + i];
                
                for (int j = i + 1; j < n; j++) {
                    /* Only update if the source U-element wasn't dropped */
                    if (fabsf(A->data[i * n + j]) > drop_tol) {
                        A->data[k * n + j] -= factor * A->data[i * n + j];
                    }
                }
            } else {
                A->data[k * n + i] = 0.0f; /* Drop small element */
            }
        }
    }
    return M4_SUCCESS;
}

m4_error_t m4_ilu_solve(const m4_matrix_t *ILU, float *b) {
    if (!ILU || !ILU->data || !b || ILU->rows != ILU->cols) return M4_ERR_INVALID_ARGS;
    int n = ILU->rows;
    
    /* Forward substitution for L (implicit unit diagonal) */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) b[i] -= ILU->data[i * n + j] * b[j];
    }
    
    /* Backward substitution for U */
    for (int i = n - 1; i >= 0; i--) {
        for (int j = i + 1; j < n; j++) b[i] -= ILU->data[i * n + j] * b[j];
        b[i] /= ILU->data[i * n + i];
    }
    
    return M4_SUCCESS;
}
