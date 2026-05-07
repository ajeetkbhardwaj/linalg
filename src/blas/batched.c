#include "../../include/blas.h"
#include "../../include/lapack.h"
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

linalg_error_t sgemm_batched(linalg_handle_t handle, char trans_a, char trans_b, int m, int n, int k,
                             float alpha, const float **A_array, const float **B_array, float beta, float **C_array, int batch_count) {
    (void)handle;
    if (!A_array || !B_array || !C_array || batch_count < 0) return LINALG_ERR_INVALID_VALUE;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < batch_count; i++) {
        sgemm(trans_a, trans_b, m, n, k, alpha, A_array[i], B_array[i], beta, C_array[i]);
    }
    return LINALG_SUCCESS;
}

linalg_error_t sgemm_strided_batched(linalg_handle_t handle, char trans_a, char trans_b, int m, int n, int k,
                                     float alpha, const float *A, long long stride_a, const float *B, long long stride_b, 
                                     float beta, float *C, long long stride_c, int batch_count) {
    (void)handle;
    if (!A || !B || !C || batch_count < 0) return LINALG_ERR_INVALID_VALUE;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < batch_count; i++) {
        sgemm(trans_a, trans_b, m, n, k, alpha, 
              A + i * stride_a, 
              B + i * stride_b, 
              beta, 
              C + i * stride_c);
    }
    return LINALG_SUCCESS;
}

linalg_error_t sgetrf_batched(linalg_handle_t handle, int n, float **A_array, int *pivot_array, int *info_array, int batch_count) {
    (void)handle;
    if (!A_array || !pivot_array || !info_array || batch_count < 0) return LINALG_ERR_INVALID_VALUE;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < batch_count; i++) {
        matrix_t mat = { A_array[i], n, n, false };
        info_array[i] = getrf(&mat, &pivot_array[i * n]);
    }
    return LINALG_SUCCESS;
}
