#pragma once

#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#ifndef CEIL_DIV
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#endif

#ifndef OFFSET
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#endif

// SGPO_NAIVE_BASELINE: this file is deliberately unoptimized.
__global__ void gemm(
    int M,
    int N,
    int K,
    float alpha,
    float *A,
    float *B,
    float beta,
    float *C) {
    /* SHARED_DECL_BEGIN */
    /* No shared memory in the initial baseline. */
    /* SHARED_DECL_END */

    /* INDEX_MAPPING_BEGIN */
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    /* INDEX_MAPPING_END */

    /* REGISTER_DECL_BEGIN */
    float sum = 0.0f;
    /* REGISTER_DECL_END */

    /* GLOBAL_TO_SHARED_LOAD_BEGIN */
    /* Scalar baseline reads global memory directly. */
    /* GLOBAL_TO_SHARED_LOAD_END */

    /* SYNC_AFTER_LOAD_BEGIN */
    /* No synchronization is needed by the scalar baseline. */
    /* SYNC_AFTER_LOAD_END */

    /* MAIN_LOOP_BEGIN */
    if (row < M && col < N) {
        /* COMPUTE_INNER_BEGIN */
        for (int k = 0; k < K; ++k) {
            sum += A[OFFSET(row, k, K)] * B[OFFSET(k, col, N)];
        }
        /* COMPUTE_INNER_END */
    }
    /* NEXT_TILE_LOAD_BEGIN */
    /* No tiled preload in the scalar baseline. */
    /* NEXT_TILE_LOAD_END */
    /* MAIN_LOOP_END */

    /* STORE_BEGIN */
    if (row < M && col < N) {
        const int index = OFFSET(row, col, N);
        C[index] = alpha * sum + beta * C[index];
    }
    /* STORE_END */
}

void cuda_gemm(
    int M,
    int N,
    int K,
    float alpha,
    float *A,
    float *B,
    float beta,
    float *C) {
    /* LAUNCH_CONFIG_BEGIN */
    dim3 block(16, 16, 1);
    dim3 grid(CEIL_DIV(N, block.x), CEIL_DIV(M, block.y), 1);
    /* LAUNCH_CONFIG_END */

    gemm<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
