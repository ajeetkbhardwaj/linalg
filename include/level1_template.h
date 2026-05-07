/* src/blas/level1_template.h - Generic BLAS Level 1 Implementation */
#ifndef SCALAR
#error "SCALAR must be defined"
#endif

#ifndef REAL_SCALAR
#define REAL_SCALAR SCALAR
#endif

#ifndef ABS_MACRO
#define ABS_MACRO(x) ((x) < 0 ? -(x) : (x))
#endif

#define CONCAT2(a, b) a ## b
#define CONCAT(a, b) CONCAT2(a, b)

#define AXPY_FUNC  CONCAT(PREFIX, axpy)
#define DOT_FUNC   CONCAT(PREFIX, dot)
#define NRM2_FUNC  CONCAT(PREFIX, nrm2)
#define SCAL_FUNC  CONCAT(PREFIX, scal)
#define COPY_FUNC  CONCAT(PREFIX, copy)
#define IAMAX_FUNC CONCAT(i, CONCAT(PREFIX, amax))

void AXPY_FUNC(int n, SCALAR alpha, const SCALAR *x, SCALAR *y) {
    if (alpha == SCALAR_ZERO) return;
    int i = 0;
#ifdef SIMD_ENABLED
    VEC_TYPE valpha = VEC_DUP(alpha);
    for (; i <= n - VEC_STEP * 2; i += VEC_STEP * 2) {
        VEC_TYPE vx1 = VEC_LOAD(&x[i]);
        VEC_TYPE vy1 = VEC_LOAD(&y[i]);
        VEC_TYPE vx2 = VEC_LOAD(&x[i + VEC_STEP]);
        VEC_TYPE vy2 = VEC_LOAD(&y[i + VEC_STEP]);
        
        vy1 = VEC_FMA(vy1, vx1, valpha);
        vy2 = VEC_FMA(vy2, vx2, valpha);
        
        VEC_STORE(&y[i], vy1);
        VEC_STORE(&y[i + VEC_STEP], vy2);
    }
#endif
    for (; i < n; i++) y[i] += alpha * x[i];
}

SCALAR DOT_FUNC(int n, const SCALAR *x, const SCALAR *y) {
    SCALAR sum = SCALAR_ZERO;
    int i = 0;
#ifdef SIMD_ENABLED
    VEC_TYPE vsum1 = VEC_DUP(SCALAR_ZERO);
    VEC_TYPE vsum2 = VEC_DUP(SCALAR_ZERO);
    for (; i <= n - VEC_STEP * 2; i += VEC_STEP * 2) {
        VEC_TYPE vx1 = VEC_LOAD(&x[i]);
        VEC_TYPE vy1 = VEC_LOAD(&y[i]);
        VEC_TYPE vx2 = VEC_LOAD(&x[i + VEC_STEP]);
        VEC_TYPE vy2 = VEC_LOAD(&y[i + VEC_STEP]);
        
        vsum1 = VEC_FMA(vsum1, vx1, vy1);
        vsum2 = VEC_FMA(vsum2, vx2, vy2);
    }
    VEC_TYPE vsum = VEC_ADD(vsum1, vsum2);
    sum = VEC_ADD_ACROSS(vsum);
#endif
    for (; i < n; i++) sum += x[i] * y[i];
    return sum;
}

REAL_SCALAR NRM2_FUNC(int n, const SCALAR *x) {
    REAL_SCALAR sum = 0;
    for (int i = 0; i < n; i++) {
#if defined(COMPLEX_MATH)
        REAL_SCALAR rx = CREAL_MACRO(x[i]);
        REAL_SCALAR ix = CIMAG_MACRO(x[i]);
        sum += rx * rx + ix * ix;
#else
        sum += x[i] * x[i];
#endif
    }
    return sqrt(sum);
}

void SCAL_FUNC(int n, SCALAR alpha, SCALAR *x) {
    if (alpha == SCALAR_ONE) return;
    int i = 0;
#ifdef SIMD_ENABLED
    VEC_TYPE valpha = VEC_DUP(alpha);
    for (; i <= n - VEC_STEP * 2; i += VEC_STEP * 2) {
        VEC_TYPE vx1 = VEC_LOAD(&x[i]);
        VEC_TYPE vx2 = VEC_LOAD(&x[i + VEC_STEP]);
        vx1 = VEC_MUL(vx1, valpha);
        vx2 = VEC_MUL(vx2, valpha);
        VEC_STORE(&x[i], vx1);
        VEC_STORE(&x[i + VEC_STEP], vx2);
    }
#endif
    for (; i < n; i++) x[i] *= alpha;
}

void COPY_FUNC(int n, const SCALAR *x, SCALAR *y) {
    for (int i = 0; i < n; i++) y[i] = x[i];
}

int IAMAX_FUNC(int n, const SCALAR *x, int incx) {
    if (n < 1 || incx <= 0) return 0;
    int max_idx = 0;
    REAL_SCALAR max_val = -1.0;
    for (int i = 0, idx = 0; i < n; i++, idx += incx) {
#if defined(COMPLEX_MATH)
        REAL_SCALAR val = ABS_MACRO(CREAL_MACRO(x[idx])) + ABS_MACRO(CIMAG_MACRO(x[idx]));
#else
        REAL_SCALAR val = ABS_MACRO(x[idx]);
#endif
        if (val > max_val) { max_val = val; max_idx = i; }
    }
    return max_idx;
}



