#include "../../include/tensor.h"
#include "../../include/blas.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

tensor_t tensor_alloc(int dims, const int *shape) {
    tensor_t T;
    T.dims = dims;
    T.shape = malloc(dims * sizeof(int));
    T.strides = malloc(dims * sizeof(int));
    T.total_size = 1;
    
    for (int i = dims - 1; i >= 0; i--) {
        T.shape[i] = shape[i];
        T.strides[i] = T.total_size;
        T.total_size *= shape[i];
    }
    T.data = calloc(T.total_size, sizeof(float));
    T.owns_data = true;
    return T;
}

void tensor_free(tensor_t *T) {
    if (T && T->owns_data) {
        free(T->data); free(T->shape); free(T->strides);
        T->owns_data = false;
    }
}

tensor_t tensor_reshape(tensor_t *T, int new_dims, const int *new_shape) {
    tensor_t res;
    res.data = T->data;
    res.dims = new_dims;
    res.shape = malloc(new_dims * sizeof(int));
    res.strides = malloc(new_dims * sizeof(int));
    res.total_size = 1;
    
    for (int i = new_dims - 1; i >= 0; i--) {
        res.shape[i] = new_shape[i];
        res.strides[i] = res.total_size;
        res.total_size *= new_shape[i];
    }
    res.owns_data = false; /* View purely mapped over existing memory */
    return res;
}

void tensor_add(const tensor_t *A, const tensor_t *B, tensor_t *C) {
    if (A->total_size == B->total_size && A->total_size == C->total_size) {
        for (int i = 0; i < A->total_size; i++) C->data[i] = A->data[i] + B->data[i];
    }
}

void tensor_mul(const tensor_t *A, const tensor_t *B, tensor_t *C) {
    if (A->total_size == B->total_size && A->total_size == C->total_size) {
        for (int i = 0; i < A->total_size; i++) C->data[i] = A->data[i] * B->data[i];
    }
}

void tensor_relu(tensor_t *T) {
    for (int i = 0; i < T->total_size; i++) {
        if (T->data[i] < 0.0f) T->data[i] = 0.0f;
    }
}

void tensor_softmax(tensor_t *T, int axis) {
    if (axis != T->dims - 1) return; /* Implement default trailing axis */
    int outer_size = T->total_size / T->shape[axis];
    int inner_size = T->shape[axis];
    
    for (int i = 0; i < outer_size; i++) {
        float max_val = T->data[i * inner_size];
        for (int j = 1; j < inner_size; j++) {
            if (T->data[i * inner_size + j] > max_val) max_val = T->data[i * inner_size + j];
        }
        float sum = 0.0f;
        for (int j = 0; j < inner_size; j++) {
            T->data[i * inner_size + j] = expf(T->data[i * inner_size + j] - max_val);
            sum += T->data[i * inner_size + j];
        }
        for (int j = 0; j < inner_size; j++) T->data[i * inner_size + j] /= sum;
    }
}

tensor_t tensor_reduce(const tensor_t *T, const int *axes, int n_axes, reduce_op_t op, bool keepdims) {
    (void)keepdims;
    /* Basic global reduction fallback if no specific axes provided */
    if (!axes || n_axes == 0) {
        int new_shape[] = {1};
        tensor_t res = tensor_alloc(1, new_shape);
        float val = (op == REDUCE_MIN) ? FLT_MAX : (op == REDUCE_MAX) ? -FLT_MAX : 0.0f;
        
        for (int i = 0; i < T->total_size; i++) {
            if (op == REDUCE_SUM || op == REDUCE_MEAN) val += T->data[i];
            else if (op == REDUCE_MAX && T->data[i] > val) val = T->data[i];
            else if (op == REDUCE_MIN && T->data[i] < val) val = T->data[i];
        }
        if (op == REDUCE_MEAN && T->total_size > 0) val /= T->total_size;
        res.data[0] = val;
        return res;
    }
    
    /* Advanced specific axis reduction omitted for brevity */
    return tensor_alloc(0, NULL);
}

void tensor_contract(const tensor_t *A, const tensor_t *B, tensor_t *C, 
                     const int *axes_a, const int *axes_b, int n_axes) {
    /* 
     * Standard 2D explicit matrix multiplication mapped to generalized contract footprint.
     * Generalized N-dimensional contraction is typically executed via:
     * 1. Permuting A so axes_a are contiguous at the end.
     * 2. Permuting B so axes_b are contiguous at the beginning.
     * 3. Matricizing (reshaping) A and B into 2D forms.
     * 4. Executing sgemm.
     * 5. Reshaping C back into the remaining N-dimensions.
     */
    if (A->dims == 2 && B->dims == 2 && n_axes == 1) {
        if (axes_a[0] == 1 && axes_b[0] == 0) {
            sgemm('N', 'N', A->shape[0], B->shape[1], A->shape[1],
                  1.0f, A->data, B->data, 0.0f, C->data);
        }
    }
}

tensor_t tensor_transpose(const tensor_t *T, const int *perm) {
    if (!T || !perm) return tensor_alloc(0, NULL);
    
    int *new_shape = malloc(T->dims * sizeof(int));
    for (int i = 0; i < T->dims; i++) new_shape[i] = T->shape[perm[i]];
    
    tensor_t res = tensor_alloc(T->dims, new_shape);
    free(new_shape);
    
    /* Map memory to permuted striding */
    for (int i = 0; i < T->total_size; i++) {
        int old_idx = i;
        int new_idx = 0;
        for (int d = 0; d < T->dims; d++) {
            int coord = old_idx / T->strides[d];
            old_idx %= T->strides[d];
            
            /* Find where dimension d went in the new shape */
            for (int k = 0; k < res.dims; k++) {
                if (perm[k] == d) { new_idx += coord * res.strides[k]; break; }
            }
        }
        res.data[new_idx] = T->data[i];
    }
    return res;
}
