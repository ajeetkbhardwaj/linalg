#include "lapack.h"
#include "blas.h"
#include "config.h"
#include <stdlib.h>
#include <string.h>

/* Solve Ax = b using LU factorization (wrapper) */
int solve_direct_lu(matrix_t *A, float *b, int nrhs) {
    if (!A || !A->data || !b) return -1;
    
    int n = A->rows;
    int *pivot = malloc(n * sizeof(int));
    if (!pivot) return -1;
    
    /* Factorize */
    int info = getrf(A, pivot);
    if (info != 0) {
        free(pivot);
        return info;
    }
    
    /* Solve */
    info = getrs(A, pivot, b, nrhs);
    free(pivot);
    return info;
}

/* Solve Ax = b using QR factorization (more stable for ill-conditioned) */
int solve_direct_qr(matrix_t *A, float *b, int nrhs) {
    if (!A || !A->data || !b) return -1;
    
    int m = A->rows, n = A->cols;
    
    /* Copy A for factorization */
    matrix_t A_copy = *A;
    A_copy.data = malloc(m * n * sizeof(float));
    if (!A_copy.data) return -1;
    memcpy(A_copy.data, A->data, m * n * sizeof(float));
    
    float *tau = malloc(n * sizeof(float));
    if (!tau) {
        free(A_copy.data);
        return -1;
    }
    
    /* QR factorization */
    int info = geqrf(&A_copy, NULL, tau);
    if (info != 0) {
        free(A_copy.data);
        free(tau);
        return info;
    }
    
    /* Apply Q^T to b */
    for (int k = 0; k < nrhs; k++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int i = j; i < m; i++) {
                sum += A_copy.data[i * n + j] * b[i * nrhs + k];
            }
            b[j * nrhs + k] = sum;
        }
    }
    
    /* Back-substitute with R */
    for (int k = 0; k < nrhs; k++) {
        for (int j = n - 1; j >= 0; j--) {
            b[j * nrhs + k] /= A_copy.data[j * n + j];
            for (int i = 0; i < j; i++) {
                b[i * nrhs + k] -= A_copy.data[i * n + j] * b[j * nrhs + k];
            }
        }
    }
    
    free(A_copy.data);
    free(tau);
    return 0;
}
