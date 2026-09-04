#include <cuda_runtime.h>
#include "cuda_kernel.cuh"

void cuda_gemm(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);
