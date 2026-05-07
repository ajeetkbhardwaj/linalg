/* include/blas.h - FIXED VERSION */
#ifndef BLAS_H
#define BLAS_H

#include "linalg.h"

#ifdef __cplusplus
extern "C" {
#endif

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

/* === BLAS Level 2 - MATCH linalg.h (8 params) === */
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

/* === BLAS Level 3 - MATCH linalg.h (10 params) === */
void sgemm(char trans_a, char trans_b, int m, int n, int k,
              float alpha, const float *A, const float *B, float beta, float *C);

void dgemm(char trans_a, char trans_b, int m, int n, int k,
              double alpha, const double *A, const double *B, double beta, double *C);

void cgemm(char trans_a, char trans_b, int m, int n, int k,
              float complex alpha, const float complex *A, const float complex *B, float complex beta, float complex *C);

void zgemm(char trans_a, char trans_b, int m, int n, int k,
              double complex alpha, const double complex *A, const double complex *B, double complex beta, double complex *C);

#ifdef __cplusplus
}
#endif

#endif /* BLAS_H */
