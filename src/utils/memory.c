#include "linalg.h"
#include "config.h"
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

/* === High-Level Matrix/Vector Allocation === */

matrix_t matrix_alloc(int rows, int cols) {
    matrix_t m = {0};
    if (rows <= 0 || cols <= 0) return m;
    
    m.data = (float *)malloc(rows * cols * sizeof(float));
    if (!m.data) return m;
    
    m.rows = rows;
    m.cols = cols;
    m.owns_data = true;
    memset(m.data, 0, rows * cols * sizeof(float));
    return m;
}

void matrix_free(matrix_t *m) {
    if (!m) return;
    if (m->owns_data && m->data) {
        free(m->data);
        m->data = NULL;
    }
    m->rows = m->cols = 0;
}

vector_t vector_alloc(int size) {
    vector_t v = {0};
    if (size <= 0) return v;
    
    v.data = (float *)malloc(size * sizeof(float));
    if (!v.data) return v;
    
    v.size = size;
    v.owns_data = true;
    memset(v.data, 0, size * sizeof(float));
    return v;
}

void vector_free(vector_t *v) {
    if (!v) return;
    if (v->owns_data && v->data) {
        free(v->data);
        v->data = NULL;
    }
    v->size = 0;
}

/* === Low-Level Memory Management === */

void* aligned_alloc(size_t alignment, size_t size) {
    void *ptr = NULL;
#if defined(__APPLE__)
    if (posix_memalign(&ptr, alignment, size) != 0) return NULL;
#elif defined(_WIN32)
    ptr = _aligned_malloc(size, alignment);
#else
    ptr = aligned_alloc(alignment, size);
#endif
    if (ptr) {
        #ifdef __APPLE__
        madvise(ptr, size, MADV_RANDOM);
        #endif
    }
    return ptr;
}

void aligned_free(void *ptr) {
    if (!ptr) return;
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

void memset_nt(void *dst, int c, size_t n) {
    if (n < 4096) {
        memset(dst, c, n);
        return;
    }
    if (c != 0) {
        memset(dst, c, n);
        return;
    }
    float *f = (float*)dst;
    size_t i = 0;
    for (; i + 16 <= n / sizeof(float); i += 16) {
        #if HAS_NEON
        float32x4_t vval = vdupq_n_f32(0.0f);
        for (int j = 0; j < 4; j++) {
            vst1q_f32(&f[i + j * 4], vval);
        }
        #else
        for (int j = 0; j < 16; j++) f[i + j] = 0.0f;
        #endif
    }
    for (; i < n / sizeof(float); i++) f[i] = 0.0f;
}

void prefetch_range(const void *ptr, size_t size, int write) {
    const char *p = (const char*)ptr;
    size_t i = 0;
    while (i < size) {
        if (write) {
            __builtin_prefetch(p + i, 1, 1);
        } else {
            __builtin_prefetch(p + i, 0, 1);
        }
        i += CACHE_LINE;
    }
}

/* Architecture detection function */
arch_t detect_arch(void) {
    #if defined(__APPLE__) && defined(__arm64__)
        #if defined(__ARM_FEATURE_MATRIX_MULTIPLY) || defined(__APPLE_AMX__)
            return ARCH_AMX;
        #elif defined(__ARM_NEON)
            return ARCH_NEON;
        #else
            return ARCH_GENERIC;
        #endif
    #elif defined(__ARM_NEON)
        return ARCH_NEON;
    #else
        return ARCH_GENERIC;
    #endif
}

/* Set preferred architecture (for override/testing) */
static arch_t g_preferred_arch = ARCH_GENERIC;

void set_preferred_arch(arch_t arch) {
    g_preferred_arch = arch;
}
