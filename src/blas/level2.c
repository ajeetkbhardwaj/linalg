/* src/blas/level2.c */
#include "blas.h"
#include "config.h"

/* === Single Precision (float) === */
#define SCALAR float
#define PREFIX s
#define SCALAR_ZERO 0.0f
#define SCALAR_ONE 1.0f
#include "../../include/level2_template.h"
#undef SCALAR
#undef PREFIX
#undef SCALAR_ZERO
#undef SCALAR_ONE

/* === Double Precision (double) === */
#define SCALAR double
#define PREFIX d
#define SCALAR_ZERO 0.0
#define SCALAR_ONE 1.0
#include "../../include/level2_template.h"
#undef SCALAR
#undef PREFIX
#undef SCALAR_ZERO
#undef SCALAR_ONE

/* === Complex Single Precision (float complex) === */
#define SCALAR float complex
#define PREFIX c
#define SCALAR_ZERO (0.0f + 0.0f * I)
#define SCALAR_ONE (1.0f + 0.0f * I)
#include "../../include/level2_template.h"
#undef SCALAR
#undef PREFIX
#undef SCALAR_ZERO
#undef SCALAR_ONE

/* === Complex Double Precision (double complex) === */
#define SCALAR double complex
#define PREFIX z
#define SCALAR_ZERO (0.0 + 0.0 * I)
#define SCALAR_ONE (1.0 + 0.0 * I)
#include "../../include/level2_template.h"