#ifndef M4AUTODIFF_H
#define M4AUTODIFF_H

#include "m4linalg.h"
#include <stdint.h>

/* ========================================================================= *
 * Forward-Mode Autodiff (Dual Numbers)
 * ========================================================================= */

typedef struct {
    float real;
    float dual;
} m4_dual_t;

m4_dual_t m4_dual_const(float val);
m4_dual_t m4_dual_var(float val, float derivative_seed);
m4_dual_t m4_dual_add(m4_dual_t a, m4_dual_t b);
m4_dual_t m4_dual_sub(m4_dual_t a, m4_dual_t b);
m4_dual_t m4_dual_mul(m4_dual_t a, m4_dual_t b);
m4_dual_t m4_dual_div(m4_dual_t a, m4_dual_t b);
m4_dual_t m4_dual_sin(m4_dual_t a);
m4_dual_t m4_dual_cos(m4_dual_t a);
m4_dual_t m4_dual_exp(m4_dual_t a);

/* Forward-Mode AD for Tensors (NEON Accelerated) */
typedef struct {
    m4_matrix_t real;
    m4_matrix_t dual;
} m4_dual_matrix_t;

m4_dual_matrix_t m4_dual_mat_alloc(int rows, int cols);
void m4_dual_mat_free(m4_dual_matrix_t *m);
m4_dual_matrix_t m4_dual_mat_add(m4_dual_matrix_t a, m4_dual_matrix_t b);
m4_dual_matrix_t m4_dual_mat_matmul(m4_dual_matrix_t a, m4_dual_matrix_t b);
m4_dual_matrix_t m4_dual_mat_relu(m4_dual_matrix_t a);

/* ========================================================================= *
 * Reverse-Mode Autodiff (Computational Graph / Tape)
 * ========================================================================= */

/* Supported Differentiable Operations */
typedef enum {
    M4_AD_OP_NONE,
    M4_AD_OP_ADD,
    M4_AD_OP_SUB,
    M4_AD_OP_MUL,
    M4_AD_OP_DIV,
    M4_AD_OP_MATMUL,
    M4_AD_OP_RELU,
    M4_AD_OP_EXP,
    M4_AD_OP_LOG,
    M4_AD_OP_SUM,
    M4_AD_OP_SOLVE,
    M4_AD_OP_TRANSPOSE,
    M4_AD_OP_SOLVE_CG,
    M4_AD_OP_SYEV
} m4_ad_op_type;

/* A Node in the Computational Graph */
typedef struct m4_ad_node {
    m4_matrix_t data;          /* Forward pass values */
    struct m4_ad_node *grad;   /* BACKWARD PASS GRADIENTS (Now a Node on the tape!) */
    bool requires_grad;        /* Does this node need gradients computed? */
    
    m4_ad_op_type op;          /* The operation that created this node */
    struct m4_ad_node *lhs;    /* Left operand */
    struct m4_ad_node *rhs;    /* Right operand */
    void *aux_data;            /* Auxiliary data (e.g., saved probs for Softmax) */
} m4_ad_node_t;

/* The "Tape" that records the sequence of operations for Backpropagation */
typedef struct {
    m4_ad_node_t **nodes;
    int capacity;
    int count;
    /* Memory Arena */
    uint8_t *arena;
    size_t arena_size;
    size_t arena_offset;
} m4_ad_ctx_t;

/* ========================================================================= *
 * Autodiff Context Management
 * ========================================================================= */
m4_ad_ctx_t* m4_ad_ctx_create(int capacity, size_t arena_size_bytes);
void m4_ad_ctx_free(m4_ad_ctx_t *ctx);

/* ========================================================================= *
 * Tensor Operations (Forward Pass)
 * ========================================================================= */

/* Register an input variable onto the tape */
m4_ad_node_t* m4_ad_var(m4_ad_ctx_t *ctx, m4_matrix_t data, bool requires_grad);

/* Differentiable Matrix Addition: C = A + B */
m4_ad_node_t* m4_ad_add(m4_ad_ctx_t *ctx, m4_ad_node_t *a, m4_ad_node_t *b);

/* Differentiable Matrix Subtraction: C = A - B */
m4_ad_node_t* m4_ad_sub(m4_ad_ctx_t *ctx, m4_ad_node_t *a, m4_ad_node_t *b);

/* Differentiable Element-wise Multiplication (with Broadcasting): C = A * B */
m4_ad_node_t* m4_ad_mul(m4_ad_ctx_t *ctx, m4_ad_node_t *a, m4_ad_node_t *b);

/* Differentiable Element-wise Division (with Broadcasting): C = A / B */
m4_ad_node_t* m4_ad_div(m4_ad_ctx_t *ctx, m4_ad_node_t *a, m4_ad_node_t *b);

/* Differentiable Matrix Multiplication: C = A @ B */
m4_ad_node_t* m4_ad_matmul(m4_ad_ctx_t *ctx, m4_ad_node_t *a, m4_ad_node_t *b);

/* Differentiable ReLU Activation: C = max(0, A) */
m4_ad_node_t* m4_ad_relu(m4_ad_ctx_t *ctx, m4_ad_node_t *a);

/* Differentiable Exponential: C = exp(A) */
m4_ad_node_t* m4_ad_exp(m4_ad_ctx_t *ctx, m4_ad_node_t *a);

/* Differentiable Natural Logarithm: C = log(A) */
m4_ad_node_t* m4_ad_log(m4_ad_ctx_t *ctx, m4_ad_node_t *a);

/* Differentiable Transpose: C = A^T */
m4_ad_node_t* m4_ad_transpose(m4_ad_ctx_t *ctx, m4_ad_node_t *a);

/* Deprecated: Use m4_ad_add natively as it now supports broadcasting */
m4_ad_node_t* m4_ad_add_bias(m4_ad_ctx_t *ctx, m4_ad_node_t *a, m4_ad_node_t *bias);

/* Differentiable Summation (Reduce to Scalar) */
m4_ad_node_t* m4_ad_sum(m4_ad_ctx_t *ctx, m4_ad_node_t *a);

/* Implicit Differentiation: Differentiable Linear Solver (Ax = b) */
m4_ad_node_t* m4_ad_solve(m4_ad_ctx_t *ctx, m4_ad_node_t *a, m4_ad_node_t *b);

/* Implicit Differentiation: Differentiable Iterative CG Solver (Ax = b for Symmetric A) */
m4_ad_node_t* m4_ad_solve_cg(m4_ad_ctx_t *ctx, m4_ad_node_t *a, m4_ad_node_t *b);

/* Implicit Differentiation: Differentiable Eigen-Decomposition (Symmetric A) */
m4_ad_node_t* m4_ad_syev(m4_ad_ctx_t *ctx, m4_ad_node_t *a);

/* ========================================================================= *
 * Backpropagation (Reverse Pass)
 * ========================================================================= */

void m4_ad_backward(m4_ad_ctx_t *ctx, m4_ad_node_t *loss);

/* Print the computational graph to the console */
void m4_ad_print_graph(m4_ad_node_t *node);

#endif /* M4AUTODIFF_H */
