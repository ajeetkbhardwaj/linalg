/* src/blas/level2_template.h - Generic BLAS Level 2 Implementation */
#ifndef SCALAR
#error "SCALAR must be defined"
#endif

#define CONCAT2(a, b) a ## b
#define CONCAT(a, b) CONCAT2(a, b)

#define GEMV_FUNC CONCAT(PREFIX, gemv)
#define GER_FUNC  CONCAT(PREFIX, ger)

void GEMV_FUNC(char trans, int m, int n, SCALAR alpha, 
               const SCALAR *A, const SCALAR *x, SCALAR beta, SCALAR *y) {
    
    if (trans == 'N' || trans == 'n') {
        for (int i = 0; i < m; i++) {
            SCALAR sum = SCALAR_ZERO;
            for (int j = 0; j < n; j++) sum += A[i * n + j] * x[j];
            y[i] = (beta == SCALAR_ZERO ? SCALAR_ZERO : beta * y[i]) + alpha * sum;
        }
    } else {
        if (beta != SCALAR_ONE) {
            for (int i = 0; i < n; i++) y[i] = (beta == SCALAR_ZERO ? SCALAR_ZERO : beta * y[i]);
        }
        for (int i = 0; i < m; i++) {
            SCALAR temp = alpha * x[i];
            for (int j = 0; j < n; j++) y[j] += temp * A[i * n + j];
        }
    }
}

void GER_FUNC(int m, int n, SCALAR alpha, const SCALAR *x, const SCALAR *y, SCALAR *A) {
    if (alpha == SCALAR_ZERO) return;
    for (int i = 0; i < m; i++) {
        SCALAR temp = alpha * x[i];
        for (int j = 0; j < n; j++) A[i * n + j] += temp * y[j];
    }
}

#undef GEMV_FUNC
#undef GER_FUNC
#undef CONCAT
#undef CONCAT2

