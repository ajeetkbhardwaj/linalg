#!/bin/bash
mkdir -p bin

echo "=== Compiling Tests for Memory Diagnostics ==="

# On macOS, clang needs help finding OpenMP from Homebrew.
# This assumes libomp is installed via homebrew on Apple Silicon (/opt/homebrew).
if [[ "$(uname)" == "Darwin" ]]; then
    OMP_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp"
else
    # Assume Linux with standard OpenMP support
    OMP_FLAGS="-fopenmp"
fi

# Remove buggy legacy implementations
rm -f ../src/lapack/lu.c ../src/lapack/cholesky.c

# Using find is more robust than the non-standard ** glob pattern.
SRC_FILES=$(find ../src -name "*.c")

# Compile tests with -g for debug symbols and -O0 to prevent compiler optimizations from hiding leaks
clang -g -O0 test_edge_cases.c $SRC_FILES -I../include $OMP_FLAGS -lm -o bin/test_edge_cases
clang -g -O0 test_sparse_correctness.c $SRC_FILES -I../include $OMP_FLAGS -lm -o bin/test_sparse_correctness
clang -g -O0 test_precond_convergence.c $SRC_FILES -I../include $OMP_FLAGS -lm -o bin/test_precond_convergence

echo ""
echo "=== Running Memory Leak Detection ==="

# Use leaks on macOS, fallback to valgrind for Linux users
if command -v leaks &> /dev/null; then
    echo "--> Running macOS leaks tool..."
    leaks -atExit -- ./bin/test_edge_cases | grep "leaks for"
    leaks -atExit -- ./bin/test_sparse_correctness | grep "leaks for"
    leaks -atExit -- ./bin/test_precond_convergence | grep "leaks for"
elif command -v valgrind &> /dev/null; then
    echo "--> Running Valgrind..."
    valgrind --leak-check=full --error-exitcode=1 ./bin/test_edge_cases
    valgrind --leak-check=full --error-exitcode=1 ./bin/test_sparse_correctness
    valgrind --leak-check=full --error-exitcode=1 ./bin/test_precond_convergence
else
    echo "[WARN] Neither 'leaks' nor 'valgrind' was found in your PATH."
    echo "Skipping memory leak execution tests."
fi

echo ""
echo "Memory analysis complete."