#include "../../include/linalg.h"
#include "../../include/blas.h"
#include <stdlib.h>

/* Solve overdetermined least squares system Ax = b using QR decomposition */
int solve_least_squares(const matrix_t *A, float *b) {
    if (!A || !A->data || !b || A->rows < A->cols) return -1;
    int m = A->rows, n = A->cols;
    
    matrix_t R = matrix_alloc(m, n);
    matrix_t Q = matrix_alloc(m, m);
    for(int i = 0; i < m * n; i++) R.data[i] = A->data[i];
    
    float *tau = malloc((m < n ? m : n) * sizeof(float));
    
    if (sgeqrf(&R, &Q, tau) != 0) {
        matrix_free(&R); matrix_free(&Q); free(tau);
        return -1;
    }
    
    float *y = malloc(m * sizeof(float));
    sgemv('T', m, m, 1.0f, Q.data, b, 0.0f, y); /* y = Q^T * b */
    
    /* Backward substitution: R(1:n, 1:n) * x = y(1:n) */
    for (int i = n - 1; i >= 0; i--) {
        for (int j = i + 1; j < n; j++) y[i] -= R.data[i * n + j] * y[j];
        if (R.data[i * n + i] == 0.0f) {
            free(y); matrix_free(&R); matrix_free(&Q); free(tau);
            return -1;
        }
        y[i] /= R.data[i * n + i];
    }
    
    /* Store result in first n indices of b */
    for (int i = 0; i < n; i++) b[i] = y[i]; 
    
    free(y); matrix_free(&R); matrix_free(&Q); free(tau); return 0;
}
