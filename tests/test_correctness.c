#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/m4linalg.h"
#include "../include/m4autodiff.h"

#define ASSERT_SUCCESS(err, msg) \
    if ((err) != M4_SUCCESS) { \
        printf("❌ FAIL: %s (Error Code: %d)\n", msg, err); \
        return 1; \
    } else { \
        printf("✅ PASS: %s\n", msg); \
    }

void fill_identity(m4_matrix_t *A) {
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            A->data[i * A->cols + j] = (i == j) ? 1.0f : 0.0f;
        }
    }
}

int test_lu_solve() {
    int n = 3;
    m4_matrix_t A = { .data = (float*)malloc(n * n * sizeof(float)), .rows = n, .cols = n };
    int ipiv[3];
    float b[3] = {1.0f, 0.0f, 1.0f};
    
    // Matrix: [2 -1 0; -1 2 -1; 0 -1 2]
    A.data[0] = 2; A.data[1] = -1; A.data[2] = 0;
    A.data[3] = -1; A.data[4] = 2; A.data[5] = -1;
    A.data[6] = 0; A.data[7] = -1; A.data[8] = 2;
    
    ASSERT_SUCCESS(m4_getrf(&A, ipiv), "LU Factorization");
    ASSERT_SUCCESS(m4_getrs(&A, ipiv, b), "LU Back-Substitution");
    
    // Expected solution for RHS [1, 0, 1] is [1.0, 1.0, 1.0]
    if (fabsf(b[0] - 1.0f) > 1e-4) return 1;
    
    free(A.data);
    return 0;
}

int test_cholesky() {
    int n = 2;
    m4_matrix_t A = { .data = (float*)malloc(n * n * sizeof(float)), .rows = n, .cols = n };
    // Matrix: [4 12; 12 45] (SPD)
    A.data[0] = 4; A.data[1] = 12; A.data[2] = 12; A.data[3] = 45;
    
    ASSERT_SUCCESS(m4_potrf(&A), "Cholesky Factorization");
    
    // Expected L matrix diagonal: [2, 3]
    if (fabsf(A.data[0] - 2.0f) > 1e-4 || fabsf(A.data[3] - 3.0f) > 1e-4) return 1;
    
    free(A.data);
    return 0;
}

int test_qr() {
    int n = 2;
    m4_matrix_t A = { .data = (float*)malloc(n * n * sizeof(float)), .rows = n, .cols = n };
    float tau[2];
    A.data[0] = 12; A.data[1] = -51; A.data[2] = 6; A.data[3] = 167;
    
    ASSERT_SUCCESS(m4_geqrf(&A, tau), "Householder QR Factorization");
    free(A.data);
    return 0;
}

int test_eigenvalue() {
    int n = 2;
    m4_matrix_t A = { .data = (float*)malloc(n * n * sizeof(float)), .rows = n, .cols = n };
    float w[2];
    A.data[0] = 2; A.data[1] = 1; A.data[2] = 1; A.data[3] = 2;
    
    ASSERT_SUCCESS(m4_syev(&A, w), "Jacobi Eigenvalue Solver");
    
    // Expected eigenvalues: 3 and 1
    if (fabsf(w[0] + w[1] - 4.0f) > 1e-4) return 1; // Sum of eigenvalues = Trace
    
    free(A.data);
    return 0;
}

int test_pcg() {
    int n = 2;
    m4_matrix_t A = { .data = (float*)malloc(n * n * sizeof(float)), .rows = n, .cols = n };
    float b[2] = {1, 2};
    float x[2] = {0, 0};
    A.data[0] = 4; A.data[1] = 1; A.data[2] = 1; A.data[3] = 3;
    
    m4_cg_workspace_t ws;
    m4_cg_query_workspace(n, &ws);
    
    int iters = 0;
    ASSERT_SUCCESS(m4_solve_pcg_jacobi(&A, b, x, 50, 1e-6f, &ws, &iters), "Jacobi PCG Solver");
    
    m4_cg_free_workspace(&ws);
    free(A.data);
    return 0;
}

int test_pgmres() {
    int n = 3;
    m4_matrix_t A = { .data = (float*)malloc(n * n * sizeof(float)), .rows = n, .cols = n };
    float b[3] = {1, 2, 3};
    float x[3] = {0, 0, 0};
    fill_identity(&A); // Identity matrix is trivially solvable
    
    m4_gmres_workspace_t ws;
    m4_gmres_query_workspace(n, 10, &ws);
    
    int iters = 0;
    ASSERT_SUCCESS(m4_solve_pgmres_ilu(&A, b, x, 10, 50, 1e-6f, &ws, 1e-4f, &iters), "ILU-Preconditioned GMRES");
    
    m4_gmres_free_workspace(&ws);
    free(A.data);
    return 0;
}

int test_mixed_precision() {
    int n = 2;
    m4_matrix_t A = { .data = (float*)malloc(n * n * sizeof(float)), .rows = n, .cols = n };
    double b[2] = {1.0, 2.0};
    double x[2] = {0.0, 0.0};
    A.data[0] = 4; A.data[1] = 1; A.data[2] = 1; A.data[3] = 3;
    
    int iters = 0;
    ASSERT_SUCCESS(m4_solve_mixed_precision_ir(&A, b, x, 10, 1e-12, &iters), "Mixed-Precision Iterative Refinement");
    
    free(A.data);
    return 0;
}

int test_batched_gemm() {
    int batch_count = 10;
    int n = 16;
    long long stride = n * n;
    float *A = (float*)malloc(batch_count * stride * sizeof(float));
    float *B = (float*)malloc(batch_count * stride * sizeof(float));
    float *C = (float*)calloc(batch_count * stride, sizeof(float));
    
    for(int i = 0; i < batch_count * stride; i++) {
        A[i] = 1.0f; 
        B[i] = 1.0f;
    }

    ASSERT_SUCCESS(m4_sgemm_batched('N', 'N', n, n, n, 1.0f, A, n, stride, B, n, stride, 0.0f, C, n, stride, batch_count), "Batched SGEMM (GCD Parallel)");
    
    free(A); free(B); free(C);
    return 0;
}

int test_ad_forward() {
    /* Test Forward-Mode AD: f(x) = exp(x) * sin(x) at x = 2.0 */
    m4_dual_t x = m4_dual_var(2.0f, 1.0f);
    m4_dual_t f = m4_dual_mul(m4_dual_exp(x), m4_dual_sin(x));
    
    float expected_val = expf(2.0f) * sinf(2.0f);
    float expected_deriv = expf(2.0f) * (sinf(2.0f) + cosf(2.0f));
    
    int err = (fabsf(f.real - expected_val) > 1e-4 || fabsf(f.dual - expected_deriv) > 1e-4) ? M4_ERR_INVALID_ARGS : M4_SUCCESS;
    ASSERT_SUCCESS(err, "Autodiff Forward-Mode (Duals)");
    return 0;
}

int test_ad_reverse_basic() {
    /* Test Reverse-Mode Tape: C = A @ B, Loss = sum(C). Tests MatMul JVP. */
    m4_ad_ctx_t *ctx = m4_ad_ctx_create(100, 1024 * 1024);
    float A_arr[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float B_arr[] = {5.0f, 6.0f, 7.0f, 8.0f};
    m4_matrix_t A_data = { .data = A_arr, .rows = 2, .cols = 2, .owns_data = false };
    m4_matrix_t B_data = { .data = B_arr, .rows = 2, .cols = 2, .owns_data = false };
    
    m4_ad_node_t *A = m4_ad_var(ctx, A_data, true);
    m4_ad_node_t *B = m4_ad_var(ctx, B_data, true);
    m4_ad_node_t *C = m4_ad_matmul(ctx, A, B);
    m4_ad_node_t *loss = m4_ad_sum(ctx, C);
    
    m4_ad_backward(ctx, loss);
    
    int err = M4_SUCCESS;
    /* dL/dA = dL/dC @ B^T = [[1, 1], [1, 1]] @ [[5, 7], [6, 8]] = [[11, 15], [11, 15]] */
    if (!A->grad || fabsf(A->grad->data.data[0] - 11.0f) > 1e-4) err = M4_ERR_INVALID_ARGS;
    if (!A->grad || fabsf(A->grad->data.data[1] - 15.0f) > 1e-4) err = M4_ERR_INVALID_ARGS;
    
    m4_ad_ctx_free(ctx);
    ASSERT_SUCCESS(err, "Autodiff Reverse-Mode (Computational Graph)");
    return 0;
}

int test_ad_implicit_solve() {
    /* Test Implicit Differentiation: Backpropagate through Ax = b */
    m4_ad_ctx_t *ctx = m4_ad_ctx_create(100, 1024 * 1024);
    float A_arr[] = {2.0f, 0.0f, 0.0f, 2.0f};
    float b_arr[] = {4.0f, 6.0f};
    m4_matrix_t A_data = { .data = A_arr, .rows = 2, .cols = 2, .owns_data = false };
    m4_matrix_t b_data = { .data = b_arr, .rows = 2, .cols = 1, .owns_data = false };
    
    m4_ad_node_t *A = m4_ad_var(ctx, A_data, true);
    m4_ad_node_t *b = m4_ad_var(ctx, b_data, true);
    
    m4_ad_node_t *x = m4_ad_solve(ctx, A, b); 
    m4_ad_node_t *loss = m4_ad_sum(ctx, x);   
    
    m4_ad_backward(ctx, loss);
    
    int err = M4_SUCCESS;
    /* Exact Gradients via Adjoint Method: dL/db = [0.5, 0.5]^T, dL/dA = [[-1.0, -1.5], [-1.0, -1.5]] */
    if (!b->grad || fabsf(b->grad->data.data[0] - 0.5f) > 1e-4) err = M4_ERR_INVALID_ARGS;
    if (!A->grad || fabsf(A->grad->data.data[0] - (-1.0f)) > 1e-4) err = M4_ERR_INVALID_ARGS;
    if (!A->grad || fabsf(A->grad->data.data[1] - (-1.5f)) > 1e-4) err = M4_ERR_INVALID_ARGS;
    
    m4_ad_ctx_free(ctx);
    ASSERT_SUCCESS(err, "Autodiff Implicit Differentiation (Ax = b)");
    return 0;
}

int main() {
    printf("==========================================\n");
    printf("🧪 M4LinAlg Correctness & Feature Testing \n");
    printf("==========================================\n\n");
    
    m4_autotune_init();
    
    int fails = 0;
    fails += test_lu_solve();
    fails += test_cholesky();
    fails += test_qr();
    fails += test_eigenvalue();
    fails += test_pcg();
    fails += test_pgmres();
    fails += test_mixed_precision();
    fails += test_batched_gemm();
    fails += test_ad_forward();
    fails += test_ad_reverse_basic();
    fails += test_ad_implicit_solve();
    
    printf("\n==========================================\n");
    if (fails == 0) {
        printf("🎉 ALL TESTS PASSED SUCCESSFULLY!\n");
    } else {
        printf("⚠️ %d TESTS FAILED. CHECK LOGS.\n", fails);
    }
    printf("==========================================\n");
    
    return fails;
}