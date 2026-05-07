/* include/linalg.h - Main public API */
#ifndef LINALG_H
#define LINALG_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <complex.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* === Core Types === */
typedef struct {
    float *data;
    int rows, cols;
    bool owns_data;
} matrix_t;

typedef struct {
    float *data;
    int size;
    bool owns_data;
} vector_t;

/* Issue 1: Double precision support */
typedef struct {
    double *data;
    int rows, cols;
    bool owns_data;
} dmatrix_t;

/* Issue 2: Complex number support (Single Precision) */
typedef struct {
    float complex *data;
    int rows, cols;
    bool owns_data;
} cmatrix_t;

/* Issue 2: Complex number support (Double Precision) */
typedef struct {
    double complex *data;
    int rows, cols;
    bool owns_data;
} zmatrix_t;

/* === Architecture Enum === */
typedef enum {
    ARCH_GENERIC = 0,
    ARCH_NEON = 1,
    ARCH_AMX = 2,
} arch_t;

/* === Error Codes (Issue 14) === */
typedef enum {
    LINALG_SUCCESS = 0,
    LINALG_ERR_INVALID_VALUE = -1,
    LINALG_ERR_MEMORY_ALLOCATION = -2,
    LINALG_ERR_NOT_INITIALIZED = -3,
    LINALG_ERR_NOT_SUPPORTED = -4,
    LINALG_ERR_EXECUTION_FAILED = -5,
    LINALG_ERR_SINGULAR_MATRIX = -6,
    LINALG_ERR_CONVERGENCE = -7
} linalg_error_t;

/* === Library Context Handle (Issue 15) === */
typedef struct linalg_context *linalg_handle_t;

linalg_error_t linalg_create(linalg_handle_t *handle);
linalg_error_t linalg_destroy(linalg_handle_t handle);

/* === Memory Management === */
matrix_t matrix_alloc(int rows, int cols);
void matrix_free(matrix_t *m);
vector_t vector_alloc(int size);
void vector_free(vector_t *v);

/* === Matrix Norms (Issue 20) === */
float matrix_norm(const matrix_t *A, char norm_type);

/* === Workspace & Memory Reuse (Issue 13) === */
typedef struct {
    void *buffer;
    size_t size;
    bool is_active;
} workspace_t;

workspace_t workspace_alloc(size_t size_bytes);
void workspace_free(workspace_t *ws);
size_t query_workspace_gemm(int m, int n, int k);
size_t query_workspace_getrf(int m, int n);

/* === BLAS Level 1 === */
void saxpy(int n, float alpha, const float *x, float *y);
float sdot(int n, const float *x, const float *y);
float snrm2(int n, const float *x);
void sscal(int n, float alpha, float *x);
void scopy(int n, const float *x, float *y);
int isamax(int n, const float *x, int incx);

void daxpy(int n, double alpha, const double *x, double *y);
double ddot(int n, const double *x, const double *y);
double dnrm2(int n, const double *x);
void dscal(int n, double alpha, double *x);
void dcopy(int n, const double *x, double *y);
int idamax(int n, const double *x, int incx);

void caxpy(int n, float complex alpha, const float complex *x, float complex *y);
float complex cdot(int n, const float complex *x, const float complex *y);
float cnrm2(int n, const float complex *x);
void cscal(int n, float complex alpha, float complex *x);
void ccopy(int n, const float complex *x, float complex *y);
int icamax(int n, const float complex *x, int incx);

void zaxpy(int n, double complex alpha, const double complex *x, double complex *y);
double complex zdot(int n, const double complex *x, const double complex *y);
double znrm2(int n, const double complex *x);
void zscal(int n, double complex alpha, double complex *x);
void zcopy(int n, const double complex *x, double complex *y);
int izamax(int n, const double complex *x, int incx);

/* === BLAS Level 2 === */
void sgemv(char trans, int m, int n, float alpha, 
            const float *A, const float *x, float beta, float *y);
void sger(int m, int n, float alpha, const float *x, const float *y, float *A);

void dgemv(char trans, int m, int n, double alpha, 
            const double *A, const double *x, double beta, double *y);
void dger(int m, int n, double alpha, const double *x, const double *y, double *A);

void cgemv(char trans, int m, int n, float complex alpha, 
            const float complex *A, const float complex *x, float complex beta, float complex *y);
void cger(int m, int n, float complex alpha, const float complex *x, const float complex *y, float complex *A);

void zgemv(char trans, int m, int n, double complex alpha, 
            const double complex *A, const double complex *x, double complex beta, double complex *y);
void zger(int m, int n, double complex alpha, const double complex *x, const double complex *y, double complex *A);

/* === BLAS Level 3 === */
void sgemm(char trans_a, char trans_b, int m, int n, int k,
            float alpha, const float *A, const float *B, float beta, float *C);
void dgemm(char trans_a, char trans_b, int m, int n, int k,
            double alpha, const double *A, const double *B, double beta, double *C);

void cgemm(char trans_a, char trans_b, int m, int n, int k,
            float complex alpha, const float complex *A, const float complex *B, float complex beta, float complex *C);
void zgemm(char trans_a, char trans_b, int m, int n, int k,
            double complex alpha, const double complex *A, const double complex *B, double complex beta, double complex *C);

/* === Batched Operations (Issue 16) === */
linalg_error_t sgemm_batched(linalg_handle_t handle, char trans_a, char trans_b, int m, int n, int k,
                             float alpha, const float **A_array, const float **B_array, float beta, float **C_array, int batch_count);

linalg_error_t sgemm_strided_batched(linalg_handle_t handle, char trans_a, char trans_b, int m, int n, int k,
                                     float alpha, const float *A, long long stride_a, const float *B, long long stride_b, 
                                     float beta, float *C, long long stride_c, int batch_count);

linalg_error_t sgetrf_batched(linalg_handle_t handle, int n, float **A_array, int *pivot_array, int *info_array, int batch_count);

/* === LAPACK === */
int sgetrf(matrix_t *A, int *pivot);
int sgetrs(const matrix_t *LU, const int *pivot, float *b, int nrhs);
int spotrf(matrix_t *A);
int spotrs(const matrix_t *L, float *b);
int sgeqrf(matrix_t *A, matrix_t *Q, float *tau);

int dgetrf(dmatrix_t *A, int *pivot);
int dgetrs(const dmatrix_t *LU, const int *pivot, double *b, int nrhs);
int dpotrf(dmatrix_t *A);
int dpotrs(const dmatrix_t *L, double *b);
int dgeqrf(dmatrix_t *A, dmatrix_t *Q, double *tau);

int cgetrf(cmatrix_t *A, int *pivot);
int cgetrs(const cmatrix_t *LU, const int *pivot, float complex *b, int nrhs);
int cpotrf(cmatrix_t *A);
int cpotrs(const cmatrix_t *L, float complex *b);
int cgeqrf(cmatrix_t *A, cmatrix_t *Q, float complex *tau);

int zgetrf(zmatrix_t *A, int *pivot);
int zgetrs(const zmatrix_t *LU, const int *pivot, double complex *b, int nrhs);
int zpotrf(zmatrix_t *A);
int zpotrs(const zmatrix_t *L, double complex *b);
int zgeqrf(zmatrix_t *A, zmatrix_t *Q, double complex *tau);

/* Legacy aliases for backward compatibility */
#define getrf sgetrf
#define getrs sgetrs
#define potrf spotrf
#define potrs spotrs
#define geqrf sgeqrf

int gesvd_jacobi(const matrix_t *A, matrix_t *U, float *S, 
                    matrix_t *V, int max_iter, float tol);

/* === Eigenvalue Solvers (Issue 17) === */
float power_iteration(const matrix_t *A, float *eigvec, int max_iter, float tol);
int qr_algorithm(const matrix_t *A, float *eigvals, int max_iter, float tol);

/* === Solver Workspaces (Issues #1 & #6) === */
typedef struct {
    float *r;   // Residual vector
    float *p;   // Search direction vector
    float *Ap;  // Matrix-vector product buffer
    float *z;   // Preconditioner buffer
    int size;   // Allocated size
} cg_workspace_t;

linalg_error_t cg_workspace_alloc(cg_workspace_t *ws, int n);
void cg_workspace_free(cg_workspace_t *ws);

typedef struct {
    float *V;   // Krylov basis
    float *H;   // Hessenberg matrix
    float *cs;  // Givens cosines
    float *sn;  // Givens sines
    float *s;   // RHS for least squares
    float *r;   // Residual vector
    float *w;   // Temp vector for Arnoldi
    float *y;   // Temp vector for least squares solve
    int n;
    int restart;
} gmres_workspace_t;

linalg_error_t gmres_workspace_alloc(gmres_workspace_t *ws, int n, int restart);
void gmres_workspace_free(gmres_workspace_t *ws);

/* === Solvers === */
linalg_error_t solve_cg_ext(const matrix_t *A, const float *b, float *x, 
                            int max_iter, float tol, cg_workspace_t *ws, int *out_iters);
linalg_error_t solve_gmres_ext(const matrix_t *A, const float *b, float *x,
                               int restart, int max_iter, float tol, gmres_workspace_t *ws, int *out_iters);
int solve_direct_lu(matrix_t *A, float *b, int nrhs);
int solve_cg(const matrix_t *A, const float *b, float *x, 
                int max_iter, float tol);
int solve_gmres(const matrix_t *A, const float *b, float *x,
                    int restart, int max_iter, float tol);
linalg_error_t solve_mixed_precision_ir(const dmatrix_t *A, const double *b, double *x, 
                                        int max_iter, double tol, int *out_iters);

/* === Least Squares (Issue 18) === */
int solve_least_squares(const matrix_t *A, float *b);

/* === Architecture Dispatch === */
arch_t detect_arch(void);
void set_preferred_arch(arch_t arch);

#ifdef __cplusplus
}
#endif

#endif /* LINALG_H */
