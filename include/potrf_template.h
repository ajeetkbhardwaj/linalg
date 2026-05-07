/* src/lapack/potrf_template.h - Generic Cholesky Decomposition & Solve */
#ifndef SCALAR
#error "SCALAR must be defined"
#endif

#define CONCAT2(a, b) a ## b
#define CONCAT(a, b) CONCAT2(a, b)

#define POTRF_FUNC CONCAT(PREFIX, potrf)
#define POTRS_FUNC CONCAT(PREFIX, potrs)

int POTRF_FUNC(MATRIX_T *A) {
    if (!A || !A->data) return -1;
    int n = A->rows;
    if (A->cols != n) return -1;
    
    for (int j = 0; j < n; j++) {
        REAL_SCALAR sum_diag = 0;
        for (int k = 0; k < j; k++) {
            SCALAR val = A->data[j * n + k];
#if defined(COMPLEX_MATH)
            sum_diag += CREAL_MACRO(val)*CREAL_MACRO(val) + CIMAG_MACRO(val)*CIMAG_MACRO(val);
#else
            sum_diag += val * val;
#endif
        }
        
#if defined(COMPLEX_MATH)
        REAL_SCALAR diag = CREAL_MACRO(A->data[j * n + j]) - sum_diag;
#else
        REAL_SCALAR diag = A->data[j * n + j] - sum_diag;
#endif
        if (diag <= 0.0) return j + 1; /* Not SPD */
        
        REAL_SCALAR l_jj = SQRT_MACRO(diag);
        A->data[j * n + j] = l_jj;
        
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = j + 1; i < n; i++) {
            SCALAR sum = SCALAR_ZERO;
            for (int k = 0; k < j; k++) {
#if defined(COMPLEX_MATH)
                sum += A->data[i * n + k] * CONJ_MACRO(A->data[j * n + k]);
#else
                sum += A->data[i * n + k] * A->data[j * n + k];
#endif
            }
            A->data[i * n + j] = (A->data[i * n + j] - sum) / l_jj;
        }
    }
    return 0;
}

/* Solve L * L^H * x = b */
int POTRS_FUNC(const MATRIX_T *L, SCALAR *b) {
    if (!L || !L->data || !b) return -1;
    int n = L->rows;
    if (L->cols != n) return -1;
    
    /* Forward substitution: Solve L * y = b */
    for (int i = 0; i < n; i++) {
        SCALAR sum = b[i];
        for (int k = 0; k < i; k++) {
            sum -= L->data[i * n + k] * b[k];
        }
        b[i] = sum / L->data[i * n + i];
    }
    
    /* Backward substitution: Solve L^H * x = y */
    for (int i = n - 1; i >= 0; i--) {
        SCALAR sum = b[i];
        for (int k = i + 1; k < n; k++) {
#if defined(COMPLEX_MATH)
            sum -= CONJ_MACRO(L->data[k * n + i]) * b[k];
#else
            sum -= L->data[k * n + i] * b[k];
#endif
        }
        b[i] = sum / L->data[i * n + i];
    }
    return 0;
}

#undef POTRF_FUNC
#undef POTRS_FUNC
#undef CONCAT
#undef CONCAT2
