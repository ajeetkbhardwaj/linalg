/* include/config.h - Hardware configuration */
#ifndef CONFIG_H
#define CONFIG_H

#include <stdint.h>
#include <stdbool.h>

/* === Hardware Specifications === */
#define L1I_CACHE_SIZE   (128 * 1024)
#define L1D_CACHE_SIZE   (128 * 1024)
#define L2_CACHE_SIZE    (16 * 1024 * 1024)

#define NEON_BITS        128
#define NEON_FLOATS      4
#define NEON_DOUBLES     2

#define AMX_M_TILE       16
#define AMX_N_TILE       16
#define AMX_K_TILE       32

/* === Cache Blocking Tuning === */
#define TILE_M           32
#define TILE_N           32
#define TILE_K           32
#define PREFETCH_DISTANCE 64

/* === Architecture Detection === */
#if defined(__APPLE__) && defined(__arm64__)
    #if defined(__ARM_FEATURE_MATRIX_MULTIPLY) || defined(__APPLE_AMX__)
        #define CURRENT_ARCH ARCH_AMX
    #elif defined(__ARM_NEON)
        #define CURRENT_ARCH ARCH_NEON
    #else
        #define CURRENT_ARCH ARCH_GENERIC
    #endif
#elif defined(__ARM_NEON)
    #define CURRENT_ARCH ARCH_NEON
#else
    #define CURRENT_ARCH ARCH_GENERIC
#endif

/* === Feature Macros === */
#ifdef __ARM_NEON
    #define HAS_NEON 1
    #include <arm_neon.h>
#else
    #define HAS_NEON 0
#endif

#if defined(__APPLE__) && defined(__arm64__) && defined(__apple_build_version__)
    #if __clang_major__ >= 17 && defined(__ARM_FEATURE_MATRIX_MULTIPLY)
        #define HAS_AMX 1
    #else
        #define HAS_AMX 0
    #endif
#else
    #define HAS_AMX 0
#endif

/* === Performance Hints === */
#define CACHE_LINE       64
#define ALIGN_CACHE      __attribute__((aligned(CACHE_LINE)))
#define PREFETCH_READ(ptr)   __builtin_prefetch((ptr), 0, 3)
#define PREFETCH_WRITE(ptr)  __builtin_prefetch((ptr), 1, 3)
#define LIKELY(x)   __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

/* === Math Configuration === */
#ifndef FAST_MATH
    #define FAST_MATH 1
#endif

#if FAST_MATH
    #define SQRT(x)   __builtin_sqrtf(x)
    #define RSQRT(x)  __builtin_rsqrtf(x)
    #define FMA(a,b,c) __builtin_fmaf(a, b, c)
#else
    #include <math.h>
    #define SQRT(x)   sqrtf(x)
    #define RSQRT(x)  (1.0f / sqrtf(x))
    #define FMA(a,b,c) ((a)*(b) + (c))
#endif

#define EPSILON    1.19209290e-07f

#endif /* CONFIG_H */
