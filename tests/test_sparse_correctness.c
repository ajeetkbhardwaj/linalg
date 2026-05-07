#include "../include/sparse.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void assert_vec_eq(const float *a, const float *b, int n, const char *name) {
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    if (max_err < 1e-4) {
        printf("[PASS] %s (Max Error: %e)\n", name, max_err);
    } else {
        printf("[FAIL] %s (Max Error: %e)\n", name, max_err);
    }
}

int main() {
    printf("=== Sparse Formats Correctness Test ===\n");
    int n = 4;
    
    float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float y_expected[4] = {1.0f, 4.0f, 9.0f, 16.0f}; /* Simple diagonal multiply */
    
    /* 1. Coordinate (COO) Test */
    sparse_coo_t coo = sparse_coo_alloc(n, n, n);
    for(int i = 0; i < n; i++) {
        coo.row_idx[i] = i; coo.col_idx[i] = i; coo.values[i] = (float)(i + 1);
    }
    float y_coo[4] = {0};
    spmv_coo('N', 1.0f, &coo, x, 0.0f, y_coo);
    assert_vec_eq(y_coo, y_expected, n, "COO SpMV");
    sparse_coo_free(&coo);

    /* 2. Diagonal (DIA) Test */
    sparse_dia_t dia = sparse_dia_alloc(n, n, 1);
    dia.offsets[0] = 0;
    for(int i = 0; i < n; i++) dia.values[i] = (float)(i + 1);
    float y_dia[4] = {0};
    spmv_dia('N', 1.0f, &dia, x, 0.0f, y_dia);
    assert_vec_eq(y_dia, y_expected, n, "DIA SpMV");
    sparse_dia_free(&dia);
    
    /* 3. Block Sparse Row (BSR) Test (2x2 blocks) */
    sparse_bsr_t bsr = sparse_bsr_alloc(2, 2, 2, 2);
    bsr.row_ptr[0] = 0; bsr.row_ptr[1] = 1; bsr.row_ptr[2] = 2;
    bsr.col_idx[0] = 0; bsr.col_idx[1] = 1;
    
    /* Block 1 (top-left) */
    bsr.values[0] = 1.0f; bsr.values[1] = 0.0f; bsr.values[2] = 0.0f; bsr.values[3] = 2.0f;
    /* Block 2 (bottom-right) */
    bsr.values[4] = 3.0f; bsr.values[5] = 0.0f; bsr.values[6] = 0.0f; bsr.values[7] = 4.0f;
    
    float y_bsr[4] = {0};
    spmv_bsr('N', 1.0f, &bsr, x, 0.0f, y_bsr);
    assert_vec_eq(y_bsr, y_expected, n, "BSR SpMV");
    sparse_bsr_free(&bsr);

    return 0;
}