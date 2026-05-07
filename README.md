# Numerical Linear Algebra Library for MacBook Air M4

A library for large scale computation for linear algebra based algorithms in different applications.

[![Build](https://img.shields.io/badge/build-passing-brightgreen)](#)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon%20M4-orange)](#)

**m4linalg** is a high-performance numerical linear algebra library optimized for Apple's M4 architecture.

---

## 🌟 Project Status (As of Q2 2026)

This library is currently under active development. It provides a solid foundation with many implemented features, but also has known limitations that are being addressed.

### What's Working
- ✅ **BLAS/LAPACK-compatible API** for core operations (GEMM, LU, Cholesky).
- ✅ **NEON SIMD acceleration** for vector (BLAS L1) operations.
- ✅ **Sparse matrix support** (CSR/CSC formats) with matrix-vector products.
- ✅ **Iterative solvers** (CG, GMRES) for dense systems.
- ✅ **Professional build system** with dependency tracking & parallel compilation.

### Known Issues & Future Work
- ⚠️ **API Inconsistency**: The library is being consolidated to a consistent `m4_` function prefix. You may see other conventions in older examples.
- ⚠️ **Performance**: While NEON is used, further optimizations (AMX, multi-threading, cache-blocking) are in progress to close the gap with Apple's Accelerate framework.
- ⚠️ **Algorithm Stability**: The QR decomposition uses a method that can be unstable. This is slated for replacement.
- ⚠️ **Missing Features**: Preconditioners for solvers, eigenvalue solvers, and full tensor support are on the roadmap.

---

## 🚀 Quick Start

### Prerequisites

- MacBook Air/Pro with Apple M-series chip (M1-M4)
- macOS Sonoma or later
- Xcode 17+ with Clang

### Build

```bash
# Clone or create project structure
mkdir -p m4linalg/{src/{blas,lapack,solvers,sparse,kernels,utils},include,lib,obj,tests}
cd m4linalg

# Copy source files to respective directories

# Bootstrap build engine
clang -O3 build.c -o build

# Build library and tests
./build
```

### Run Tests & Benchmarks

```bash
# Correctness tests
./test_correctness

# Regression tests for factorizations
./test_regression 64

# Performance comparison vs Apple Accelerate
./bench 1024          # 1024×1024 matrix multiply
```

### Use in Your Project

```c
// In your C/C++ code:
#include "m4linalg.h"

// Create matrices
m4_matrix_t A = m4_matrix_alloc(100, 100);
m4_matrix_t B = m4_matrix_alloc(100, 100);
m4_matrix_t C = m4_matrix_alloc(100, 100);

// Matrix multiply: C = A * B
m4_sgemm('N', 'N', 100, 100, 100, 
         1.0f, A.data, A.cols, 
               B.data, B.cols, 
         0.0f, C.data, C.cols);

// Solve Ax = b via LU
float *b = malloc(100 * sizeof(float));
// ... fill b ...
m4_solve_direct_lu(&A, b, 1);  // b now contains solution x

// Cleanup
m4_matrix_free(&A); m4_matrix_free(&B); m4_matrix_free(&C);
free(b);
```

Compile your app:

```bash
clang -I./include -L./lib -lm4linalg -framework Accelerate your_app.c -o your_app
```

---

## 📦 API Reference

### Core Types

```c
typedef struct {
    float *data;      // Row-major: data[row*cols + col]
    int rows, cols;
    bool owns_data;   // Auto-free on matrix_free()
} m4_matrix_t;
```

### BLAS Level 1 (Vector)

| Function     | Description            |
| ------------ | ---------------------- |
| `m4_saxpy` | y = αx + y            |
| `m4_sdot`  | Dot product x·y       |
| `m4_snrm2` | Euclidean norm ‖x‖₂ |
| `m4_sscal` | x = αx                |
| `m4_scopy` | y = x                  |

### BLAS Level 2 (Matrix-Vector)

| Function     | Description              |
| ------------ | ------------------------ |
| `m4_sgemv` | y = α·op(A)·x + β·y |
| `m4_sger`  | A = α·x·yᵀ + A       |

### BLAS Level 3 (Matrix-Matrix)

| Function     | Description                  |
| ------------ | ---------------------------- |
| `m4_sgemm` | C = α·op(A)·op(B) + β·C |

### LAPACK Factorizations

| Function            | Description                    |
| ------------------- | ------------------------------ |
| `m4_getrf`        | LU decomposition with pivoting |
| `m4_geqrf`        | QR decomposition (Householder) |
| `m4_potrf`        | Cholesky (A = LLᵀ)            |
| `m4_gesvd_jacobi` | SVD via Jacobi iterations      |

### Solvers

| Function               | Description                      |
| ---------------------- | -------------------------------- |
| `m4_solve_direct_lu` | Direct solve via LU              |
| `m4_solve_cg`        | Conjugate Gradient (SPD systems) |
| `m4_solve_gmres`     | GMRES (non-symmetric systems)    |

---

## ⚙️ Architecture Optimization

### M4-Specific Flags

```bash
# In build.c CFLAGS:
-mcpu=apple-m4          # Target M4 instruction set
-ffast-math             # Aggressive FP optimizations
-MMD -MP                # Generate .d dependency files
```

### NEON Intrinsics

Vector operations automatically use Arm NEON when available:

```c
// In m4_neon.c:
float32x4_t vx = vld1q_f32(&x[i]);  // Load 4 floats
float32x4_t vy = vld1q_f32(&y[i]);
float32x4_t vsum = vaddq_f32(vx, vy);  // SIMD add
vst1q_f32(&res[i], vsum);  // Store result
```

---

## 📊 Performance

### Benchmark Results (M4, 1024×1024 SGEMM)

| Implementation     | GFLOPS | Notes               |
| ------------------ | ------ | ------------------- |
| m4linalg (NEON)    | ~85    | Cache-blocked, SIMD |
| Apple Accelerate   | ~120   | Hand-tuned by Apple |
| m4linalg (generic) | ~25    | Portable C fallback |

*Note: Performance varies with problem size. Current implementation is single-threaded. Future work on multi-threading and AMX optimization aims to close the performance gap with the Accelerate framework.*

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/neon-spmv`
3. Add tests for new functionality
4. Ensure all tests pass: `./test_correctness && ./test_regression`
5. Submit a pull request

### Coding Standards

- Use the `m4_` prefix for all new public functions.
- Document all public functions with Doxygen-style comments.
- Add NEON/AMX optimizations where beneficial (profile first!).
- Maintain backward compatibility with generic C fallbacks.

---

## 📜 License

MIT License - see LICENSE file for details.

---

> **m4linalg** - High-performance linear algebra, engineered for Apple Silicon.

