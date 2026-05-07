#include "../include/sparse.h"
#include "../include/linalg.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("=== Preconditioner Convergence Test ===\n");
    int n = 100;
    int nnz = 3 * n - 2;
    sparse_csr_t A = sparse_csr_alloc(n, n, nnz);
    
    /* Create a 1D Poisson matrix (Tridiagonal: 2 on diag, -1 on off-diagonals) */
    int idx = 0;
    A.row_ptr[0] = 0;
    for (int i = 0; i < n; i++) {
        if (i > 0) { A.col_idx[idx] = i - 1; A.values[idx++] = -1.0f; }
        A.col_idx[idx] = i; A.values[idx++] = 2.0f;
        if (i < n - 1) { A.col_idx[idx] = i + 1; A.values[idx++] = -1.0f; }
        A.row_ptr[i+1] = idx;
    }
    
    float *b = malloc(n * sizeof(float));
    float *x_no = calloc(n, sizeof(float));
    float *x_jac = calloc(n, sizeof(float));
    float *x_ic0 = calloc(n, sizeof(float));
    for (int i = 0; i < n; i++) b[i] = 1.0f;
    
    /* Solve without preconditioning */
    int iter_no = solve_sparse_cg(&A, b, x_no, 1000, 1e-5f);
    
    /* Solve with Jacobi preconditioning */
    sparse_csr_t M_jac = jacobi_precond(&A);
    int iter_jac = solve_sparse_cg_precond(&A, b, x_jac, &M_jac, 1000, 1e-5f);
    
    /* Solve with Incomplete Cholesky (IC0) preconditioning */
    sparse_csr_t M_ic0 = ic0_precond(&A);
    int iter_ic0 = solve_sparse_cg_precond(&A, b, x_ic0, &M_ic0, 1000, 1e-5f);
    
    printf("CG Iterations (No Preconditioner): %d\n", iter_no);
    printf("CG Iterations (Jacobi):            %d\n", iter_jac);
    printf("CG Iterations (IC0):               %d\n", iter_ic0);
    
    if (iter_ic0 < iter_no) printf("[PASS] IC(0) successfully reduces iteration count.\n");
    else printf("[FAIL] IC(0) did not improve convergence.\n");
    
    sparse_csr_free(&A);
    sparse_csr_free(&M_jac);
    sparse_csr_free(&M_ic0);
    free(b); free(x_no); free(x_jac); free(x_ic0);
    
    return 0;
}