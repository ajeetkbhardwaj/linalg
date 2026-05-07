/* src/blas/level1.c */
#include "blas.h"
#include "config.h"
#include <complex.h>
#include <math.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

/* === Single Precision (float) === */
#define SCALAR float
#define REAL_SCALAR float
#define PREFIX s
#define SCALAR_ZERO 0.0f
#define SCALAR_ONE 1.0f
#define ABS_MACRO(x) fabsf(x)

#if defined(__aarch64__)
    #define SIMD_ENABLED
    #define VEC_TYPE float32x4_t
    #define VEC_LOAD(ptr) vld1q_f32(ptr)
    #define VEC_STORE(ptr, val) vst1q_f32(ptr, val)
    #define VEC_ADD(a, b) vaddq_f32(a, b)
    #define VEC_MUL(a, b) vmulq_f32(a, b)
    #define VEC_FMA(acc, a, b) vfmaq_f32(acc, a, b)
    #define VEC_DUP(val) vdupq_n_f32(val)
    #define VEC_ADD_ACROSS(vec) vaddvq_f32(vec)
    #define VEC_STEP 4
#endif

#include "../../include/level1_template.h"

#undef SIMD_ENABLED
#undef VEC_TYPE
#undef VEC_LOAD
#undef VEC_STORE
#undef VEC_ADD
#undef VEC_MUL
#undef VEC_FMA
#undef VEC_DUP
#undef VEC_ADD_ACROSS
#undef VEC_STEP
#undef SCALAR
#undef REAL_SCALAR
#undef PREFIX
#undef SCALAR_ZERO
#undef SCALAR_ONE
#undef ABS_MACRO

/* === Double Precision (double) === */
#define SCALAR double
#define REAL_SCALAR double
#define PREFIX d
#define SCALAR_ZERO 0.0
#define SCALAR_ONE 1.0
#define ABS_MACRO(x) fabs(x)

#if defined(__aarch64__)
    #define SIMD_ENABLED
    #define VEC_TYPE float64x2_t
    #define VEC_LOAD(ptr) vld1q_f64(ptr)
    #define VEC_STORE(ptr, val) vst1q_f64(ptr, val)
    #define VEC_ADD(a, b) vaddq_f64(a, b)
    #define VEC_MUL(a, b) vmulq_f64(a, b)
    #define VEC_FMA(acc, a, b) vfmaq_f64(acc, a, b)
    #define VEC_DUP(val) vdupq_n_f64(val)
    #define VEC_ADD_ACROSS(vec) vaddvq_f64(vec)
    #define VEC_STEP 2
#endif

#include "../../include/level1_template.h"

#undef SIMD_ENABLED
#undef VEC_TYPE
#undef VEC_LOAD
#undef VEC_STORE
#undef VEC_ADD
#undef VEC_MUL
#undef VEC_FMA
#undef VEC_DUP
#undef VEC_ADD_ACROSS
#undef VEC_STEP
#undef SCALAR
#undef REAL_SCALAR
#undef PREFIX
#undef SCALAR_ZERO
#undef SCALAR_ONE
#undef ABS_MACRO

/* === Complex Single Precision (float complex) === */
#define SCALAR float complex
#define REAL_SCALAR float
#define PREFIX c
#define SCALAR_ZERO (0.0f + 0.0f * I)
#define SCALAR_ONE (1.0f + 0.0f * I)
#define COMPLEX_MATH 1
#define ABS_MACRO(x) fabsf(x)
#define CREAL_MACRO(x) crealf(x)
#define CIMAG_MACRO(x) cimagf(x)
#include "../../include/level1_template.h"
#undef SCALAR
#undef REAL_SCALAR
#undef PREFIX
#undef SCALAR_ZERO
#undef SCALAR_ONE
#undef COMPLEX_MATH
#undef ABS_MACRO
#undef CREAL_MACRO
#undef CIMAG_MACRO

/* === Complex Double Precision (double complex) === */
#define SCALAR double complex
#define REAL_SCALAR double
#define PREFIX z
#define SCALAR_ZERO (0.0 + 0.0 * I)
#define SCALAR_ONE (1.0 + 0.0 * I)
#define COMPLEX_MATH 1
#define ABS_MACRO(x) fabs(x)
#define CREAL_MACRO(x) creal(x)
#define CIMAG_MACRO(x) cimag(x)
#include "../../include/level1_template.h"
