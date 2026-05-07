#include "../../include/sparse.h"
#include <stdlib.h>
#include <stdbool.h>

/* 
 * Basic Minimum Degree Ordering Algorithm for Sparse CSR Matrices.
 * Generates a row permutation array designed to minimize dense fill-in
 * during Sparse LU and Sparse Cholesky factorizations.
 */
int* sparse_minimum_degree_ordering(const sparse_csr_t *A) {
    if (!A || !A->values) return NULL;
    
    int n = A->rows;
    int *perm = malloc(n * sizeof(int));
    int *degree = malloc(n * sizeof(int));
    bool *eliminated = calloc(n, sizeof(bool));
    
    if (!perm || !degree || !eliminated) {
        free(perm); free(degree); free(eliminated);
        return NULL;
    }
    
    /* Phase 1: Initialize Degrees (degree equals edges attached to node in Graph) */
    for (int i = 0; i < n; i++) {
        degree[i] = A->row_ptr[i+1] - A->row_ptr[i];
    }
    
    /* Phase 2: Greedy Selection */
    for (int k = 0; k < n; k++) {
        int min_deg = n + 1;
        int min_node = -1;
        for (int i = 0; i < n; i++) {
            if (!eliminated[i] && degree[i] < min_deg) {
                min_deg = degree[i];
                min_node = i;
            }
        }
        
        if (min_node == -1) break; /* Failsafe */
        
        perm[k] = min_node;
        eliminated[min_node] = true;
        
        /* Phase 3: Approximate neighbor contraction degree reduction */
        for (int j = A->row_ptr[min_node]; j < A->row_ptr[min_node+1]; j++) {
            int neighbor = A->col_idx[j];
            if (!eliminated[neighbor] && degree[neighbor] > 0) degree[neighbor]--;
        }
    }
    
    free(degree); free(eliminated);
    return perm;
}

/* 
 * Symbolic Factorization: Computes the elimination tree and non-zero counts
 * for Sparse Cholesky factorization given a permutation mapping.
 */
int* sparse_symbolic_factorization(const sparse_csr_t *A, const int *perm, int *nnz_L) {
    if (!A || !perm || !nnz_L) return NULL;
    int n = A->rows;
    
    int *parent = malloc(n * sizeof(int));
    for (int i = 0; i < n; i++) parent[i] = -1;
    
    int *L_row_counts = calloc(n, sizeof(int));
    int *ancestor = malloc(n * sizeof(int));
    int *inv_perm = malloc(n * sizeof(int));
    
    for (int i = 0; i < n; i++) inv_perm[perm[i]] = i;
    
    for (int k = 0; k < n; k++) {
        ancestor[k] = -1;
        int node = perm[k];
        
        for (int p = A->row_ptr[node]; p < A->row_ptr[node+1]; p++) {
            int j = A->col_idx[p];
            int inv_j = inv_perm[j];
            
            if (inv_j < k) {
                int i = inv_j;
                while (i != -1 && i != k) {
                    int next = ancestor[i];
                    ancestor[i] = k;
                    if (next == -1) {
                        parent[i] = k;
                        L_row_counts[k]++;
                        break;
                    }
                    i = next;
                }
            }
        }
        L_row_counts[k]++; /* Diagonal element */
    }
    
    *nnz_L = 0;
    for (int i = 0; i < n; i++) *nnz_L += L_row_counts[i];
    
    free(ancestor); free(L_row_counts); free(inv_perm);
    return parent;
}
