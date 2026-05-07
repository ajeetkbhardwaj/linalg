#include "m4linalg.h"
#include <math.h>
#include <dispatch/dispatch.h>

m4_error_t m4_potrf(m4_matrix_t *A) {
    if (!A || !A->data || A->rows != A->cols) {
        return M4_ERR_INVALID_ARGS;
    }

    int n = A->cols;
    int lda = n; /* Row-major stride */

    for (int j = 0; j < n; j++) {
        /* Compute diagonal element L[j, j] */
        float sum = 0.0f;
        for (int k = 0; k < j; k++) {
            float val = A->data[j * lda + k];
            sum += val * val;
        }

        float diag = A->data[j * lda + j] - sum;
        if (diag <= 0.0f) {
            return M4_ERR_SINGULAR; /* Matrix is not positive definite */
        }
        
        float l_jj = sqrtf(diag);
        A->data[j * lda + j] = l_jj;

        /* Compute off-diagonal elements L[i, j] for i > j */
        float inv_l_jj = 1.0f / l_jj;
        int row_count = n - (j + 1);
        
        if (row_count > 0) {
            if (row_count * j > 4096) {
                /* Parallelized update using Apple's Grand Central Dispatch */
                dispatch_apply(row_count, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t idx) {
                    int i = j + 1 + (int)idx;
                    float local_sum = 0.0f;
                    for (int k = 0; k < j; k++) {
                        local_sum += A->data[i * lda + k] * A->data[j * lda + k];
                    }
                    A->data[i * lda + j] = (A->data[i * lda + j] - local_sum) * inv_l_jj;
                    A->data[j * lda + i] = 0.0f;
                });
            } else {
                /* Serial fallback */
                for (int i = j + 1; i < n; i++) {
                    float local_sum = 0.0f;
                    for (int k = 0; k < j; k++) {
                        local_sum += A->data[i * lda + k] * A->data[j * lda + k];
                    }
                    A->data[i * lda + j] = (A->data[i * lda + j] - local_sum) * inv_l_jj;
                    A->data[j * lda + i] = 0.0f;
                }
            }
        }
    }

    return M4_SUCCESS;
}
