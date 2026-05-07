#ifndef TENSOR_H
#define TENSOR_H

#include "linalg.h"

#ifdef __cplusplus
extern "C" {
#endif

/* === N-dimensional Tensor === */
typedef struct {
    float *data;           /* Flattened data in row-major order */
    int *shape;            /* Dimensions: shape[0] x shape[1] x ... */
    int *strides;          /* Strides for each dimension (computed) */
    int dims;              /* Number of dimensions */
    int total_size;        /* Product of shape[] */
    bool owns_data;        /* Whether to free data/shape/strides */
} tensor_t;

/* === Tensor Memory === */

/**
 * @brief Allocate tensor with given shape
 * @param dims Number of dimensions
 * @param shape Array of dimension sizes
 */
tensor_t tensor_alloc(int dims, const int *shape);

/**
 * @brief Free tensor resources
 */
void tensor_free(tensor_t *T);

/**
 * @brief Create view of tensor with new shape (no data copy)
 */
tensor_t tensor_reshape(tensor_t *T, int new_dims, const int *new_shape);

/* === Tensor Operations === */

/**
 * @brief Element-wise addition: C = A + B (broadcasting supported)
 */
void tensor_add(const tensor_t *A, const tensor_t *B, tensor_t *C);

/**
 * @brief Element-wise multiplication: C = A * B
 */
void tensor_mul(const tensor_t *A, const tensor_t *B, tensor_t *C);

/**
 * @brief Tensor contraction (generalized matrix multiply)
 * @param A, B Input tensors
 * @param C Output tensor
 * @param axes_a Axes of A to contract
 * @param axes_b Axes of B to contract
 * @param n_axes Number of contraction axes
 * 
 * Example: matrix multiply C[i,j] = sum_k A[i,k] * B[k,j]
 *   axes_a = {1}, axes_b = {0}, n_axes = 1
 */
void tensor_contract(const tensor_t *A, const tensor_t *B, 
                        tensor_t *C, const int *axes_a, 
                        const int *axes_b, int n_axes);

/**
 * @brief Transpose tensor axes
 * @param T Input tensor
 * @param perm Permutation array: new_axis[i] = old_axis[perm[i]]
 */
tensor_t tensor_transpose(const tensor_t *T, const int *perm);

/**
 * @brief Reduce tensor along axes (sum, mean, max, min)
 */
typedef enum { REDUCE_SUM, REDUCE_MEAN, REDUCE_MAX, REDUCE_MIN } reduce_op_t;

tensor_t tensor_reduce(const tensor_t *T, const int *axes, 
                                int n_axes, reduce_op_t op, bool keepdims);

/* === ML Helper Operations === */

/**
 * @brief Softmax along specified axis
 */
void tensor_softmax(tensor_t *T, int axis);

/**
 * @brief ReLU activation: T = max(0, T) in-place
 */
void tensor_relu(tensor_t *T);

#ifdef __cplusplus
}
#endif

#endif /* TENSOR_H */
