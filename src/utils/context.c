#include "../../include/linalg.h"
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

struct linalg_context {
    arch_t arch;
    int num_threads;
};

linalg_error_t linalg_create(linalg_handle_t *handle) {
    if (!handle) return LINALG_ERR_INVALID_VALUE;
    
    *handle = (linalg_handle_t)malloc(sizeof(struct linalg_context));
    if (!*handle) return LINALG_ERR_MEMORY_ALLOCATION;
    
    (*handle)->arch = detect_arch();
    
#ifdef _OPENMP
    (*handle)->num_threads = omp_get_max_threads();
#else
    (*handle)->num_threads = 1;
#endif
    return LINALG_SUCCESS;
}

linalg_error_t linalg_destroy(linalg_handle_t handle) {
    if (handle) free(handle);
    return LINALG_SUCCESS;
}
