#!/usr/bin/env bash
set -euo pipefail

mkdir -p build

nvcc -O3 -std=c++17 cublas_sgemm_fp32.cu -lcublas -o build/cublas_sgemm_fp32
nvcc -O3 -std=c++17 cublas_sgemm_tf32.cu -lcublas -o build/cublas_sgemm_tf32
nvcc -O3 -std=c++17 cublaslt_matmul_tf32.cu -lcublas -lcublasLt -o build/cublaslt_matmul_tf32

echo "Baseline build complete."
