#include "m4linalg.h"
#include <math.h>

m4_error_t m4_geqrf(m4_matrix_t *A, float *tau) {
    if (!A || !A->data || !tau) {
        return M4_ERR_INVALID_ARGS;
    }

    int m = A->rows;
    int n = A->cols;
    int lda = n; /* Row-major stride */
    int min_mn = (m < n) ? m : n;

    for (int i = 0; i < min_mn; i++) {
        /* 1. Compute the norm of the vector v = A[i:m, i] */
        float norm = 0.0f;
        for (int k = i; k < m; k++) {
            float val = A->data[k * lda + i];
            norm += val * val;
        }
        norm = sqrtf(norm);

        /* 2. Compute the Householder reflector */
        float alpha = A->data[i * lda + i];
        if (norm > 0.0f) {
            float beta = (alpha < 0.0f) ? norm : -norm;
            tau[i] = (beta - alpha) / beta;

            /* Scale the first element of the vector down temporarily */
            A->data[i * lda + i] = alpha - beta;

            /* 3. Apply the reflector to the trailing submatrix A[i:m, i+1:n] */
            /* A_trailing -= tau[i] * v * (v^T * A_trailing) */
            for (int j = i + 1; j < n; j++) {
                float sum = 0.0f;
                
                /* Compute dot product: sum = v^T * A[i:m, j] */
                for (int k = i; k < m; k++) {
                    sum += A->data[k * lda + i] * A->data[k * lda + j];
                }
                sum *= tau[i];

                /* Rank-1 update: A[i:m, j] -= sum * v */
                for (int k = i; k < m; k++) {
                    A->data[k * lda + j] -= sum * A->data[k * lda + i];
                }
            }
            /* Store the diagonal element of R */
            A->data[i * lda + i] = beta;
        } else {
            /* Column is already zero */
            tau[i] = 0.0f;
        }
    }
    
    return M4_SUCCESS;
}
