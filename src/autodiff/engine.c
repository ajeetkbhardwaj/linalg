#include "m4autodiff.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

/* Forward-Mode AD Implementation */
m4_dual_t m4_dual_const(float val) { return (m4_dual_t){val, 0.0f}; }
m4_dual_t m4_dual_var(float val, float seed) { return (m4_dual_t){val, seed}; }
m4_dual_t m4_dual_add(m4_dual_t a, m4_dual_t b) { return (m4_dual_t){a.real + b.real, a.dual + b.dual}; }
m4_dual_t m4_dual_sub(m4_dual_t a, m4_dual_t b) { return (m4_dual_t){a.real - b.real, a.dual - b.dual}; }
m4_dual_t m4_dual_mul(m4_dual_t a, m4_dual_t b) { return (m4_dual_t){a.real * b.real, a.real * b.dual + a.dual * b.real}; }
m4_dual_t m4_dual_div(m4_dual_t a, m4_dual_t b) { 
    return (m4_dual_t){a.real / b.real, (a.dual * b.real - a.real * b.dual) / (b.real * b.real)}; 
}
m4_dual_t m4_dual_sin(m4_dual_t a) { return (m4_dual_t){sinf(a.real), a.dual * cosf(a.real)}; }
m4_dual_t m4_dual_cos(m4_dual_t a) { return (m4_dual_t){cosf(a.real), -a.dual * sinf(a.real)}; }
m4_dual_t m4_dual_exp(m4_dual_t a) { float e = expf(a.real); return (m4_dual_t){e, a.dual * e}; }

/* Forward-Mode AD Tensor Implementation */
m4_dual_matrix_t m4_dual_mat_alloc(int rows, int cols) {
    m4_dual_matrix_t m;
    m.real.rows = rows; m.real.cols = cols; m.real.owns_data = true;
    m.real.data = (float*)calloc(rows * cols, sizeof(float));
    m.dual.rows = rows; m.dual.cols = cols; m.dual.owns_data = true;
    m.dual.data = (float*)calloc(rows * cols, sizeof(float));
    return m;
}

void m4_dual_mat_free(m4_dual_matrix_t *m) {
    if (m->real.owns_data && m->real.data) free(m->real.data);
    if (m->dual.owns_data && m->dual.data) free(m->dual.data);
}

m4_dual_matrix_t m4_dual_mat_add(m4_dual_matrix_t a, m4_dual_matrix_t b) {
    m4_dual_matrix_t out = m4_dual_mat_alloc(a.real.rows, a.real.cols);
    for (int i = 0; i < out.real.rows * out.real.cols; i++) {
        out.real.data[i] = a.real.data[i] + b.real.data[i];
        out.dual.data[i] = a.dual.data[i] + b.dual.data[i];
    }
    return out;
}

m4_dual_matrix_t m4_dual_mat_matmul(m4_dual_matrix_t a, m4_dual_matrix_t b) {
    m4_dual_matrix_t out = m4_dual_mat_alloc(a.real.rows, b.real.cols);
    /* C.real = a.real @ b.real */
    m4_sgemm('N', 'N', a.real.rows, b.real.cols, a.real.cols, 1.0f, a.real.data, a.real.cols, b.real.data, b.real.cols, 0.0f, out.real.data, out.real.cols);
    
    /* C.dual = a.dual @ b.real + a.real @ b.dual (Product Rule!) */
    m4_sgemm('N', 'N', a.dual.rows, b.real.cols, a.dual.cols, 1.0f, a.dual.data, a.dual.cols, b.real.data, b.real.cols, 0.0f, out.dual.data, out.dual.cols);
    m4_sgemm('N', 'N', a.real.rows, b.dual.cols, a.real.cols, 1.0f, a.real.data, a.real.cols, b.dual.data, b.dual.cols, 1.0f, out.dual.data, out.dual.cols);
    
    return out;
}

m4_dual_matrix_t m4_dual_mat_relu(m4_dual_matrix_t a) {
    m4_dual_matrix_t out = m4_dual_mat_alloc(a.real.rows, a.real.cols);
    for (int i = 0; i < out.real.rows * out.real.cols; i++) {
        if (a.real.data[i] > 0.0f) {
            out.real.data[i] = a.real.data[i];
            out.dual.data[i] = a.dual.data[i]; /* Derivative of ReLU(x>0) is 1 * dx */
        } else {
            out.real.data[i] = 0.0f;
            out.dual.data[i] = 0.0f;           /* Derivative of ReLU(x<=0) is 0 * dx */
        }
    }
    return out;
}

/* Internal Helper: Bump Allocator */
static void* m4_ad_alloc(m4_ad_ctx_t *ctx, size_t size) {
    /* 64-byte alignment ensures optimal performance for Apple AMX and ARM NEON */
    size_t aligned_size = (size + 63) & ~63;
    if (ctx->arena_offset + aligned_size > ctx->arena_size) {
        fprintf(stderr, "FATAL: Autodiff Memory Arena Exhausted!\n");
        return NULL;
    }
    void *ptr = ctx->arena + ctx->arena_offset;
    ctx->arena_offset += aligned_size;
    memset(ptr, 0, size);
    return ptr;
}

/* Reverse-Mode AD Implementation */
/* Internal Helper: Allocate an empty matrix */
static m4_matrix_t m4_internal_matrix_zeros(m4_ad_ctx_t *ctx, int rows, int cols) {
    m4_matrix_t mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.owns_data = false; /* Data is now safely managed by the Arena */
    mat.data = (float*)m4_ad_alloc(ctx, rows * cols * sizeof(float));
    return mat;
}

/* Context Management */
m4_ad_ctx_t* m4_ad_ctx_create(int capacity, size_t arena_size_bytes) {
    m4_ad_ctx_t *ctx = (m4_ad_ctx_t*)malloc(sizeof(m4_ad_ctx_t));
    ctx->capacity = capacity;
    ctx->count = 0;
    ctx->nodes = (m4_ad_node_t**)malloc(capacity * sizeof(m4_ad_node_t*));
    ctx->arena_size = arena_size_bytes;
    ctx->arena_offset = 0;
    ctx->arena = (uint8_t*)malloc(arena_size_bytes);
    return ctx;
}

void m4_ad_ctx_free(m4_ad_ctx_t *ctx) {
    if (!ctx) return;
    for (int i = 0; i < ctx->count; i++) {
        m4_ad_node_t *node = ctx->nodes[i];
        if (node->data.owns_data) free(node->data.data);
        /* Node structs and aux_data are automatically swept with the arena */
    }
    free(ctx->arena);
    free(ctx->nodes);
    free(ctx);
}

/* Internal Helper: Push a new node to the tape */
static m4_ad_node_t* m4_ad_push_node(m4_ad_ctx_t *ctx, m4_matrix_t data, bool requires_grad, m4_ad_op_type op, m4_ad_node_t *lhs, m4_ad_node_t *rhs) {
    if (ctx->count >= ctx->capacity) return NULL; /* Out of capacity */
    
    m4_ad_node_t *node = (m4_ad_node_t*)m4_ad_alloc(ctx, sizeof(m4_ad_node_t));
    node->data = data;
    node->requires_grad = requires_grad;
    node->grad = NULL; /* Gradients start as NULL, built recursively during backward pass */
    node->op = op;
    node->lhs = lhs;
    node->rhs = rhs;
    node->aux_data = NULL;
    
    ctx->nodes[ctx->count++] = node;
    return node;
}

/* Create a variable */
m4_ad_node_t* m4_ad_var(m4_ad_ctx_t *ctx, m4_matrix_t data, bool requires_grad) {
    return m4_ad_push_node(ctx, data, requires_grad, M4_AD_OP_NONE, NULL, NULL);
}

/* Forward Ops */
m4_ad_node_t* m4_ad_add(m4_ad_ctx_t *ctx, m4_ad_node_t *a, m4_ad_node_t *b) {
    int ra = a->data.rows, ca = a->data.cols;
    int rb = b->data.rows, cb = b->data.cols;
    if ((ra != rb && ra != 1 && rb != 1) || (ca != cb && ca != 1 && cb != 1)) return NULL;
    
    int out_r = (ra > rb) ? ra : rb;
    int out_c = (ca > cb) ? ca : cb;
    m4_matrix_t out_data = m4_internal_matrix_zeros(ctx, out_r, out_c);
    
    for (int i = 0; i < out_r; i++) {
        for (int j = 0; j < out_c; j++) {
            float va = a->data.data[(ra == 1 ? 0 : i) * ca + (ca == 1 ? 0 : j)];
            float vb = b->data.data[(rb == 1 ? 0 : i) * cb + (cb == 1 ? 0 : j)];
            out_data.data[i * out_c + j] = va + vb;
        }
    }
    
    bool req_grad = a->requires_grad || b->requires_grad;
    return m4_ad_push_node(ctx, out_data, req_grad, M4_AD_OP_ADD, a, b);
}

m4_ad_node_t* m4_ad_sub(m4_ad_ctx_t *ctx, m4_ad_node_t *a, m4_ad_node_t *b) {
    int ra = a->data.rows, ca = a->data.cols;
    int rb = b->data.rows, cb = b->data.cols;
    if ((ra != rb && ra != 1 && rb != 1) || (ca != cb && ca != 1 && cb != 1)) return NULL;
    
    int out_r = (ra > rb) ? ra : rb;
    int out_c = (ca > cb) ? ca : cb;
    m4_matrix_t out_data = m4_internal_matrix_zeros(ctx, out_r, out_c);
    
    for (int i = 0; i < out_r; i++) {
        for (int j = 0; j < out_c; j++) {
            float va = a->data.data[(ra == 1 ? 0 : i) * ca + (ca == 1 ? 0 : j)];
            float vb = b->data.data[(rb == 1 ? 0 : i) * cb + (cb == 1 ? 0 : j)];
            out_data.data[i * out_c + j] = va - vb;
        }
    }
    
    bool req_grad = a->requires_grad || b->requires_grad;
    return m4_ad_push_node(ctx, out_data, req_grad, M4_AD_OP_SUB, a, b);
}

m4_ad_node_t* m4_ad_mul(m4_ad_ctx_t *ctx, m4_ad_node_t *a, m4_ad_node_t *b) {
    int ra = a->data.rows, ca = a->data.cols;
    int rb = b->data.rows, cb = b->data.cols;
    if ((ra != rb && ra != 1 && rb != 1) || (ca != cb && ca != 1 && cb != 1)) return NULL;
    
    int out_r = (ra > rb) ? ra : rb;
    int out_c = (ca > cb) ? ca : cb;
    m4_matrix_t out_data = m4_internal_matrix_zeros(ctx, out_r, out_c);
    
    for (int i = 0; i < out_r; i++) {
        for (int j = 0; j < out_c; j++) {
            float va = a->data.data[(ra == 1 ? 0 : i) * ca + (ca == 1 ? 0 : j)];
            float vb = b->data.data[(rb == 1 ? 0 : i) * cb + (cb == 1 ? 0 : j)];
            out_data.data[i * out_c + j] = va * vb;
        }
    }
    bool req_grad = a->requires_grad || b->requires_grad;
    return m4_ad_push_node(ctx, out_data, req_grad, M4_AD_OP_MUL, a, b);
}

m4_ad_node_t* m4_ad_div(m4_ad_ctx_t *ctx, m4_ad_node_t *a, m4_ad_node_t *b) {
    int ra = a->data.rows, ca = a->data.cols;
    int rb = b->data.rows, cb = b->data.cols;
    if ((ra != rb && ra != 1 && rb != 1) || (ca != cb && ca != 1 && cb != 1)) return NULL;
    
    int out_r = (ra > rb) ? ra : rb;
    int out_c = (ca > cb) ? ca : cb;
    m4_matrix_t out_data = m4_internal_matrix_zeros(ctx, out_r, out_c);
    
    for (int i = 0; i < out_r; i++) {
        for (int j = 0; j < out_c; j++) {
            float va = a->data.data[(ra == 1 ? 0 : i) * ca + (ca == 1 ? 0 : j)];
            float vb = b->data.data[(rb == 1 ? 0 : i) * cb + (cb == 1 ? 0 : j)];
            out_data.data[i * out_c + j] = va / (vb + 1e-8f);
        }
    }
    bool req_grad = a->requires_grad || b->requires_grad;
    return m4_ad_push_node(ctx, out_data, req_grad, M4_AD_OP_DIV, a, b);
}

m4_ad_node_t* m4_ad_matmul(m4_ad_ctx_t *ctx, m4_ad_node_t *a, m4_ad_node_t *b) {
    if (a->data.cols != b->data.rows) return NULL;
    
    m4_matrix_t out_data = m4_internal_matrix_zeros(ctx, a->data.rows, b->data.cols);
    
    /* Utilize our hardware-accelerated GEMM for the forward pass! */
    m4_sgemm('N', 'N', a->data.rows, b->data.cols, a->data.cols,
             1.0f, a->data.data, a->data.cols,
             b->data.data, b->data.cols,
             0.0f, out_data.data, out_data.cols);
             
    bool req_grad = a->requires_grad || b->requires_grad;
    return m4_ad_push_node(ctx, out_data, req_grad, M4_AD_OP_MATMUL, a, b);
}

m4_ad_node_t* m4_ad_relu(m4_ad_ctx_t *ctx, m4_ad_node_t *a) {
    m4_matrix_t out_data = m4_internal_matrix_zeros(ctx, a->data.rows, a->data.cols);
    for (int i = 0; i < out_data.rows * out_data.cols; i++) {
        out_data.data[i] = a->data.data[i] > 0.0f ? a->data.data[i] : 0.0f;
    }
    
    return m4_ad_push_node(ctx, out_data, a->requires_grad, M4_AD_OP_RELU, a, NULL);
}

m4_ad_node_t* m4_ad_exp(m4_ad_ctx_t *ctx, m4_ad_node_t *a) {
    m4_matrix_t out_data = m4_internal_matrix_zeros(ctx, a->data.rows, a->data.cols);
    for (int i = 0; i < out_data.rows * out_data.cols; i++) {
        out_data.data[i] = expf(a->data.data[i]);
    }
    return m4_ad_push_node(ctx, out_data, a->requires_grad, M4_AD_OP_EXP, a, NULL);
}

m4_ad_node_t* m4_ad_log(m4_ad_ctx_t *ctx, m4_ad_node_t *a) {
    m4_matrix_t out_data = m4_internal_matrix_zeros(ctx, a->data.rows, a->data.cols);
    for (int i = 0; i < out_data.rows * out_data.cols; i++) {
        out_data.data[i] = logf(a->data.data[i] + 1e-8f); /* Epsilon to prevent log(0) */
    }
    return m4_ad_push_node(ctx, out_data, a->requires_grad, M4_AD_OP_LOG, a, NULL);
}

m4_ad_node_t* m4_ad_add_bias(m4_ad_ctx_t *ctx, m4_ad_node_t *a, m4_ad_node_t *bias) {
    /* Forward to m4_ad_add which now natively supports broadcasting! */
    return m4_ad_add(ctx, a, bias);
}

m4_ad_node_t* m4_ad_transpose(m4_ad_ctx_t *ctx, m4_ad_node_t *a) {
    m4_matrix_t out_data = m4_internal_matrix_zeros(ctx, a->data.cols, a->data.rows);
    for(int i = 0; i < a->data.rows; i++) {
        for(int j = 0; j < a->data.cols; j++) {
            out_data.data[j * out_data.cols + i] = a->data.data[i * a->data.cols + j];
        }
    }
    return m4_ad_push_node(ctx, out_data, a->requires_grad, M4_AD_OP_TRANSPOSE, a, NULL);
}

m4_ad_node_t* m4_ad_sum(m4_ad_ctx_t *ctx, m4_ad_node_t *a) {
    m4_matrix_t out_data = m4_internal_matrix_zeros(ctx, 1, 1);
    float sum = 0.0f;
    for (int i = 0; i < a->data.rows * a->data.cols; i++) {
        sum += a->data.data[i];
    }
    out_data.data[0] = sum;
    return m4_ad_push_node(ctx, out_data, a->requires_grad, M4_AD_OP_SUM, a, NULL);
}

m4_ad_node_t* m4_ad_solve(m4_ad_ctx_t *ctx, m4_ad_node_t *A, m4_ad_node_t *b) {
    /* We require A to be square, and b to be a column vector */
    if (A->data.rows != A->data.cols || A->data.rows != b->data.rows || b->data.cols != 1) return NULL;
    
    int n = A->data.rows;
    m4_matrix_t A_copy = m4_internal_matrix_zeros(ctx, n, n);
    memcpy(A_copy.data, A->data.data, n * n * sizeof(float));
    
    m4_matrix_t x = m4_internal_matrix_zeros(ctx, n, 1);
    memcpy(x.data, b->data.data, n * sizeof(float));
    
    int *ipiv = (int*)m4_ad_alloc(ctx, n * sizeof(int));
    if (m4_getrf(&A_copy, ipiv) != M4_SUCCESS) {
        return NULL; /* Matrix was singular */
    }
    
    m4_getrs(&A_copy, ipiv, x.data);
    
    bool req_grad = A->requires_grad || b->requires_grad;
    /* Record ONLY the inputs and output to the tape. The LU operations are safely ignored! */
    return m4_ad_push_node(ctx, x, req_grad, M4_AD_OP_SOLVE, A, b);
}

m4_ad_node_t* m4_ad_solve_cg(m4_ad_ctx_t *ctx, m4_ad_node_t *A, m4_ad_node_t *b) {
    if (A->data.rows != A->data.cols || A->data.rows != b->data.rows || b->data.cols != 1) return NULL;
    
    int n = A->data.rows;
    m4_matrix_t x = m4_internal_matrix_zeros(ctx, n, 1);
    
    m4_cg_workspace_t ws;
    if (m4_cg_query_workspace(n, &ws) == M4_SUCCESS) {
        m4_solve_cg(&A->data, b->data.data, x.data, 1000, 1e-6f, &ws, NULL);
        m4_cg_free_workspace(&ws);
    }
    
    bool req_grad = A->requires_grad || b->requires_grad;
    return m4_ad_push_node(ctx, x, req_grad, M4_AD_OP_SOLVE_CG, A, b);
}

m4_ad_node_t* m4_ad_syev(m4_ad_ctx_t *ctx, m4_ad_node_t *a) {
    if (a->data.rows != a->data.cols) return NULL;
    int n = a->data.rows;
    
    m4_matrix_t A_copy = m4_internal_matrix_zeros(ctx, n, n);
    memcpy(A_copy.data, a->data.data, n * n * sizeof(float));
    
    m4_matrix_t *V = m4_ad_alloc(ctx, sizeof(m4_matrix_t));
    *V = m4_internal_matrix_zeros(ctx, n, n);
    m4_matrix_t w = m4_internal_matrix_zeros(ctx, n, 1);
    
    if (m4_syev_evec(&A_copy, w.data, V) != M4_SUCCESS) return NULL;
    
    m4_ad_node_t *node = m4_ad_push_node(ctx, w, a->requires_grad, M4_AD_OP_SYEV, a, NULL);
    node->aux_data = V; /* Save eigenvectors to the tape for analytical backprop! */
    return node;
}

/* Internal Helpers for Double-Tape Backward pass */
static m4_ad_node_t* m4_ad_step_mask(m4_ad_ctx_t *ctx, m4_ad_node_t *a) {
    m4_matrix_t out = m4_internal_matrix_zeros(ctx, a->data.rows, a->data.cols);
    for(int i = 0; i < out.rows * out.cols; i++) out.data[i] = a->data.data[i] > 0.0f ? 1.0f : 0.0f;
    return m4_ad_push_node(ctx, out, false, M4_AD_OP_NONE, NULL, NULL); 
}

static m4_ad_node_t* m4_ad_broadcast(m4_ad_ctx_t *ctx, m4_ad_node_t *a, int rows, int cols) {
    if (a->data.rows == rows && a->data.cols == cols) return a;
    m4_matrix_t zero_mat = m4_internal_matrix_zeros(ctx, rows, cols);
    m4_ad_node_t *zero_node = m4_ad_var(ctx, zero_mat, false);
    return m4_ad_add(ctx, zero_node, a); 
}

/* Backpropagation Engine */
void m4_ad_backward(m4_ad_ctx_t *ctx, m4_ad_node_t *loss) {
    /* Seed the gradient of the loss with a constant 1.0 matrix node */
    m4_matrix_t seed_data = m4_internal_matrix_zeros(ctx, loss->data.rows, loss->data.cols);
    for (int i = 0; i < seed_data.rows * seed_data.cols; i++) seed_data.data[i] = 1.0f;
    loss->grad = m4_ad_var(ctx, seed_data, false);
    
    /* Capture the original graph count so we only backpropagate through the forward nodes, 
       ignoring the newly generated gradient calculation nodes dynamically being added to the tape! */
    int original_count = ctx->count;
    for (int i = original_count - 1; i >= 0; i--) {
        m4_ad_node_t *node = ctx->nodes[i];
        
        if (!node->requires_grad || !node->grad) continue;
        
        if (node->op == M4_AD_OP_ADD) {
            if (node->lhs->requires_grad) node->lhs->grad = node->lhs->grad ? m4_ad_add(ctx, node->lhs->grad, node->grad) : node->grad;
            if (node->rhs->requires_grad) node->rhs->grad = node->rhs->grad ? m4_ad_add(ctx, node->rhs->grad, node->grad) : node->grad;
        } else if (node->op == M4_AD_OP_SUB) {
            if (node->lhs->requires_grad) node->lhs->grad = node->lhs->grad ? m4_ad_add(ctx, node->lhs->grad, node->grad) : node->grad;
            if (node->rhs->requires_grad) {
                m4_matrix_t zero_mat = m4_internal_matrix_zeros(ctx, node->grad->data.rows, node->grad->data.cols);
                m4_ad_node_t *zero_node = m4_ad_var(ctx, zero_mat, false);
                m4_ad_node_t *neg_grad = m4_ad_sub(ctx, zero_node, node->grad);
                node->rhs->grad = node->rhs->grad ? m4_ad_add(ctx, node->rhs->grad, neg_grad) : neg_grad;
            }
        } else if (node->op == M4_AD_OP_MUL) {
            if (node->lhs->requires_grad) {
                m4_ad_node_t *step = m4_ad_mul(ctx, node->grad, node->rhs);
                node->lhs->grad = node->lhs->grad ? m4_ad_add(ctx, node->lhs->grad, step) : step;
            }
            if (node->rhs->requires_grad) {
                m4_ad_node_t *step = m4_ad_mul(ctx, node->grad, node->lhs);
                node->rhs->grad = node->rhs->grad ? m4_ad_add(ctx, node->rhs->grad, step) : step;
            }
        } else if (node->op == M4_AD_OP_DIV) {
            if (node->lhs->requires_grad) {
                m4_ad_node_t *step = m4_ad_div(ctx, node->grad, node->rhs);
                node->lhs->grad = node->lhs->grad ? m4_ad_add(ctx, node->lhs->grad, step) : step;
            }
            if (node->rhs->requires_grad) {
                m4_ad_node_t *b_sq = m4_ad_mul(ctx, node->rhs, node->rhs);
                m4_ad_node_t *num = m4_ad_mul(ctx, node->grad, node->lhs);
                m4_ad_node_t *div = m4_ad_div(ctx, num, b_sq);
                
                m4_matrix_t zero_mat = m4_internal_matrix_zeros(ctx, div->data.rows, div->data.cols);
                m4_ad_node_t *zero_node = m4_ad_var(ctx, zero_mat, false);
                m4_ad_node_t *step = m4_ad_sub(ctx, zero_node, div);
                
                node->rhs->grad = node->rhs->grad ? m4_ad_add(ctx, node->rhs->grad, step) : step;
            }
        } else if (node->op == M4_AD_OP_MATMUL) {
            if (node->lhs->requires_grad) {
                m4_ad_node_t *b_T = m4_ad_transpose(ctx, node->rhs);
                m4_ad_node_t *step = m4_ad_matmul(ctx, node->grad, b_T);
                node->lhs->grad = node->lhs->grad ? m4_ad_add(ctx, node->lhs->grad, step) : step;
            }
            if (node->rhs->requires_grad) {
                m4_ad_node_t *a_T = m4_ad_transpose(ctx, node->lhs);
                m4_ad_node_t *step = m4_ad_matmul(ctx, a_T, node->grad);
                node->rhs->grad = node->rhs->grad ? m4_ad_add(ctx, node->rhs->grad, step) : step;
            }
        } else if (node->op == M4_AD_OP_RELU) {
            if (node->lhs->requires_grad) {
                m4_ad_node_t *mask = m4_ad_step_mask(ctx, node->lhs);
                m4_ad_node_t *step = m4_ad_mul(ctx, node->grad, mask);
                node->lhs->grad = node->lhs->grad ? m4_ad_add(ctx, node->lhs->grad, step) : step;
            }
        } else if (node->op == M4_AD_OP_EXP) {
            if (node->lhs->requires_grad) {
                m4_ad_node_t *step = m4_ad_mul(ctx, node->grad, node);
                node->lhs->grad = node->lhs->grad ? m4_ad_add(ctx, node->lhs->grad, step) : step;
            }
        } else if (node->op == M4_AD_OP_LOG) {
            if (node->lhs->requires_grad) {
                m4_ad_node_t *step = m4_ad_div(ctx, node->grad, node->lhs);
                node->lhs->grad = node->lhs->grad ? m4_ad_add(ctx, node->lhs->grad, step) : step;
            }
        } else if (node->op == M4_AD_OP_TRANSPOSE) {
            if (node->lhs->requires_grad) {
                m4_ad_node_t *step = m4_ad_transpose(ctx, node->grad);
                node->lhs->grad = node->lhs->grad ? m4_ad_add(ctx, node->lhs->grad, step) : step;
            }
        } else if (node->op == M4_AD_OP_SUM) {
            if (node->lhs->requires_grad) {
                m4_ad_node_t *step = m4_ad_broadcast(ctx, node->grad, node->lhs->data.rows, node->lhs->data.cols);
                node->lhs->grad = node->lhs->grad ? m4_ad_add(ctx, node->lhs->grad, step) : step;
            }
        } else if (node->op == M4_AD_OP_SOLVE) {
            if (node->rhs->requires_grad || node->lhs->requires_grad) {
                m4_ad_node_t *A_T = m4_ad_transpose(ctx, node->lhs);
                m4_ad_node_t *dL_db = m4_ad_solve(ctx, A_T, node->grad);
                
                if (node->rhs->requires_grad) {
                    node->rhs->grad = node->rhs->grad ? m4_ad_add(ctx, node->rhs->grad, dL_db) : dL_db;
                }
                if (node->lhs->requires_grad) {
                    m4_ad_node_t *x_T = m4_ad_transpose(ctx, node); // node is the output x
                    m4_ad_node_t *dL_dA_neg = m4_ad_matmul(ctx, dL_db, x_T);
                    
                    m4_matrix_t zero_mat = m4_internal_matrix_zeros(ctx, dL_dA_neg->data.rows, dL_dA_neg->data.cols);
                    m4_ad_node_t *zero_node = m4_ad_var(ctx, zero_mat, false);
                    m4_ad_node_t *dL_dA = m4_ad_sub(ctx, zero_node, dL_dA_neg);
                    
                    node->lhs->grad = node->lhs->grad ? m4_ad_add(ctx, node->lhs->grad, dL_dA) : dL_dA;
                }
            }
        } else if (node->op == M4_AD_OP_SOLVE_CG) {
            if (node->rhs->requires_grad || node->lhs->requires_grad) {
                /* CG assumes A is symmetric, so A^T = A. Thus dL_db = A^-1 dL_dx */
                m4_ad_node_t *dL_db = m4_ad_solve_cg(ctx, node->lhs, node->grad);
                
                if (node->rhs->requires_grad) {
                    node->rhs->grad = node->rhs->grad ? m4_ad_add(ctx, node->rhs->grad, dL_db) : dL_db;
                }
                if (node->lhs->requires_grad) {
                    m4_ad_node_t *x_T = m4_ad_transpose(ctx, node); 
                    m4_ad_node_t *dL_dA_neg = m4_ad_matmul(ctx, dL_db, x_T);
                    
                    m4_matrix_t zero_mat = m4_internal_matrix_zeros(ctx, dL_dA_neg->data.rows, dL_dA_neg->data.cols);
                    m4_ad_node_t *zero_node = m4_ad_var(ctx, zero_mat, false);
                    m4_ad_node_t *dL_dA = m4_ad_sub(ctx, zero_node, dL_dA_neg);
                    
                    node->lhs->grad = node->lhs->grad ? m4_ad_add(ctx, node->lhs->grad, dL_dA) : dL_dA;
                }
            }
        } else if (node->op == M4_AD_OP_SYEV) {
            if (node->lhs->requires_grad && node->aux_data) {
                m4_matrix_t *V = (m4_matrix_t*)node->aux_data;
                int n = V->rows;
                
                /* Construct diag(dL/dw) */
                m4_matrix_t dL_dw_diag = m4_internal_matrix_zeros(ctx, n, n);
                for (int j = 0; j < n; j++) dL_dw_diag.data[j * n + j] = node->grad->data.data[j];
                
                /* dL_dA = V * diag(dL_dw) * V^T */
                m4_ad_node_t *V_node = m4_ad_var(ctx, *V, false);
                m4_ad_node_t *dL_dw_diag_node = m4_ad_var(ctx, dL_dw_diag, false);
                m4_ad_node_t *V_T_node = m4_ad_transpose(ctx, V_node);
                
                m4_ad_node_t *step1 = m4_ad_matmul(ctx, V_node, dL_dw_diag_node);
                m4_ad_node_t *dL_dA = m4_ad_matmul(ctx, step1, V_T_node);
                
                node->lhs->grad = node->lhs->grad ? m4_ad_add(ctx, node->lhs->grad, dL_dA) : dL_dA;
            }
        }
    }
}

/* Graph Visualization */
static const char* m4_ad_op_name(m4_ad_op_type op) {
    switch(op) {
        case M4_AD_OP_NONE: return "Leaf (Variable/Constant)";
        case M4_AD_OP_ADD: return "Add";
        case M4_AD_OP_SUB: return "Subtract";
        case M4_AD_OP_MUL: return "Multiply";
        case M4_AD_OP_DIV: return "Divide";
        case M4_AD_OP_MATMUL: return "Matrix Multiply";
        case M4_AD_OP_RELU: return "ReLU";
        case M4_AD_OP_EXP: return "Exponential";
        case M4_AD_OP_LOG: return "Logarithm";
        case M4_AD_OP_SUM: return "Sum";
        case M4_AD_OP_SOLVE: return "Linear Solve (Implicit)";
        case M4_AD_OP_TRANSPOSE: return "Transpose";
        case M4_AD_OP_SOLVE_CG: return "Iterative Solve CG (Implicit)";
        case M4_AD_OP_SYEV: return "Eigen-Decomposition (Implicit)";
        default: return "Unknown";
    }
}

void m4_ad_print_graph(m4_ad_node_t *node) {
    if (!node) return;
    printf("-> [%s] shape: %dx%d (Requires Grad: %s)\n", 
           m4_ad_op_name(node->op), node->data.rows, node->data.cols,
           node->requires_grad ? "Yes" : "No");
    if (node->lhs) { printf("   LHS "); m4_ad_print_graph(node->lhs); }
    if (node->rhs) { printf("   RHS "); m4_ad_print_graph(node->rhs); }
}
