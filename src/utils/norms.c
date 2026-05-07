#include "../../include/linalg.h"
#include <math.h>

/* Calculate standard Matrix Norms */
float matrix_norm(const matrix_t *A, char norm_type) {
    if (!A || !A->data) return 0.0f;
    int m = A->rows, n = A->cols;
    float norm = 0.0f;

    if (norm_type == 'F' || norm_type == 'f') {
        /* Frobenius Norm: Square root of sum of absolute squares */
        for (int i = 0; i < m * n; i++) norm += A->data[i] * A->data[i];
        return sqrtf(norm);
    } else if (norm_type == '1') {
        /* 1-Norm: Maximum absolute column sum */
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int i = 0; i < m; i++) sum += fabsf(A->data[i * n + j]);
            if (sum > norm) norm = sum;
        }
        return norm;
    } else if (norm_type == 'I' || norm_type == 'i') {
        /* Infinity-Norm: Maximum absolute row sum */
        for (int i = 0; i < m; i++) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++) sum += fabsf(A->data[i * n + j]);
            if (sum > norm) norm = sum;
        }
        return norm;
    }
    
    return 0.0f;
}
