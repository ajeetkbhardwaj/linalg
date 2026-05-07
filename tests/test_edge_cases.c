#include "../include/linalg.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void test_singular_lu() {
    printf("--- Testing Singular Matrix LU Decomposition ---\n");
    matrix_t A = matrix_alloc(3, 3);
    /* Fill with 1s -> linearly dependent rows -> singular */
    for(int i = 0; i < 9; i++) A.data[i] = 1.0f; 
    
    int pivot[3];
    int info = sgetrf(&A, pivot);
    
    if (info > 0) {
        printf("[PASS] Singular matrix correctly detected at column %d.\n", info);
    } else {
        printf("[FAIL] Singular matrix was NOT detected! Info = %d\n", info);
    }
    matrix_free(&A);
}

void test_ill_conditioned() {
    printf("--- Testing Ill-Conditioned Matrix (Hilbert) Stability ---\n");
    int n = 5;
    matrix_t A = matrix_alloc(n, n);
    
    /* Hilbert Matrix: H_{i,j} = 1 / (i + j + 1) - known to be extremely ill-conditioned */
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            A.data[i * n + j] = 1.0f / (i + j + 1.0f);
        }
    }
    
    int pivot[5];
    int info = sgetrf(&A, pivot);
    
    printf("[%s] LU Factorization on 5x5 Hilbert Matrix completed (Info: %d).\n", 
           (info == 0) ? "PASS" : "WARN", info);
           
    matrix_free(&A);
}

int main() {
    test_singular_lu();
    test_ill_conditioned();
    return 0;
}