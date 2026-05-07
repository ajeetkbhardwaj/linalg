/* src/lapack/factorizations.c */
/* Touched to force recompilation of templates and spotrs */
#include "../../include/linalg.h"
#include "../../include/config.h"
#include <math.h>
#include <complex.h>

#undef getrf
#undef getrs
#undef potrf
#undef potrs
#undef geqrf

/* === Single Precision (float) === */
#define SCALAR float
#define REAL_SCALAR float
#define MATRIX_T matrix_t
#define PREFIX s
#define SCALAR_ZERO 0.0f
#define SCALAR_ONE 1.0f
#define ABS_MACRO(x) fabsf(x)
#define SQRT_MACRO(x) sqrtf(x)
#include "../../include/getrf_template.h"
#include "../../include/potrf_template.h"
/* Single precision QR is provided by qr.c */
#undef SCALAR
#undef REAL_SCALAR
#undef MATRIX_T
#undef PREFIX
#undef SCALAR_ZERO
#undef SCALAR_ONE
#undef ABS_MACRO
#undef SQRT_MACRO

/* === Double Precision (double) === */
#define SCALAR double
#define REAL_SCALAR double
#define MATRIX_T dmatrix_t
#define PREFIX d
#define SCALAR_ZERO 0.0
#define SCALAR_ONE 1.0
#define ABS_MACRO(x) fabs(x)
#define SQRT_MACRO(x) sqrt(x)
#include "../../include/getrf_template.h"
#include "../../include/potrf_template.h"
/* Double precision QR is provided by qr.c */
#undef SCALAR
#undef REAL_SCALAR
#undef MATRIX_T
#undef PREFIX
#undef SCALAR_ZERO
#undef SCALAR_ONE
#undef ABS_MACRO
#undef SQRT_MACRO

/* === Complex Single Precision (float complex) === */
#define SCALAR float complex
#define REAL_SCALAR float
#define MATRIX_T cmatrix_t
#define PREFIX c
#define SCALAR_ZERO (0.0f + 0.0f * I)
#define SCALAR_ONE (1.0f + 0.0f * I)
#define COMPLEX_MATH 1
/* Faster pivot evaluation using 1-norm instead of hypot */
#define ABS_MACRO(x) (fabsf(crealf(x)) + fabsf(cimagf(x)))
#define SQRT_MACRO(x) sqrtf(x)
#define CREAL_MACRO(x) crealf(x)
#define CIMAG_MACRO(x) cimagf(x)
#define CONJ_MACRO(x) conjf(x)
#include "../../include/getrf_template.h"
#include "../../include/potrf_template.h"
/* Complex precision QR is provided by qr.c */
/* ... Double complex definition omitted to save space, repeats pattern ... */
