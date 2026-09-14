
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "kernel.h"

// cal offset from row col && ld , in row-major matrix, ld is the width of the matrix
#ifndef OFFSET
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#endif

// transfer float4
#ifndef FLOAT4
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#endif

#define checkCudaErrors(func)                                                    \
do {                                                                             \
    cudaError_t e = (func);                                                      \
    if (e != cudaSuccess) {                                                      \
        fprintf(stderr, "%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        fflush(stderr);                                                          \
        exit(EXIT_FAILURE);                                                      \
    }                                                                            \
} while (0)

#define checkCublasErrors(func)                                                  \
do {                                                                             \
    cublasStatus_t s = (func);                                                   \
    if (s != CUBLAS_STATUS_SUCCESS) {                                            \
        fprintf(stderr, "%s %d cuBLAS status: %d\n", __FILE__, __LINE__, (int)s); \
        fflush(stderr);                                                          \
        exit(EXIT_FAILURE);                                                      \
    }                                                                            \
} while (0)

#ifndef CEIL_DIV
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#endif

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("usage: ./main [M] [K] [N]\n");
        exit(0);
    }
    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = atoi(argv[3]);
    // int M=1024;
    // int N=1024;
    // int K=1024;

    size_t bytes_A = sizeof(float) * M * K;
    size_t bytes_B = sizeof(float) * K * N;
    size_t bytes_C = sizeof(float) * M * N;
    float* h_A = (float*)malloc(bytes_A);
    float* h_B = (float*)malloc(bytes_B);
    float* h_C = (float*)calloc((size_t)M * (size_t)N, sizeof(float));
    float* h_C1 = (float*)calloc((size_t)M * (size_t)N, sizeof(float));
    if (!h_A || !h_B || !h_C || !h_C1) {
        fprintf(stderr, "host allocation failed\n");
        exit(EXIT_FAILURE);
    }

    float* d_A;
    float* d_B;
    float* d_C;

    checkCudaErrors(cudaMalloc(&d_A, bytes_A));
    checkCudaErrors(cudaMalloc(&d_B, bytes_B));
    checkCudaErrors(cudaMalloc(&d_C, bytes_C));
    double msecPerMatrixMul[2] = {0, 0};
    double gigaFlops[2] = {0, 0};
    double flopsPerMatrixMul = 2.0 * M * N * K;


    // generate A
    for( int i = 0; i < M * K; i++ ){
        h_A[i] = i / 13;
    }

    // generate B
    for( int i = 0; i < K * N; i++ ) {
        h_B[i] = i % 13;
    }

    checkCudaErrors(cudaMemcpy( d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy( d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    float msecTotal = 0;
    int nIter = 50;
    int warmupIter = 5;

    checkCudaErrors(cudaMemcpy( d_C, h_C, bytes_C, cudaMemcpyHostToDevice));
    for (int run = 0; run < warmupIter; run++) {
        float alpha = 1.0f;
        float beta = 0.0f;
        cuda_gemm(M, N, K, alpha, d_A, d_B, beta, d_C);
        checkCudaErrors(cudaGetLastError());
    }
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0 ; run < nIter; run ++ ) {
        float alpha = 1.0f;
        float beta = 0.0f;
        cuda_gemm(M, N, K, alpha, d_A, d_B, beta, d_C);
        checkCudaErrors(cudaGetLastError());
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
    checkCudaErrors(cudaMemcpy( h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    msecPerMatrixMul[0] = msecTotal / nIter;
    gigaFlops[0] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[0] / 1000.0f);
    printf( "My gemm Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops[0],
        msecPerMatrixMul[0],
        flopsPerMatrixMul);

    // cublas
    cublasHandle_t blas_handle = nullptr;
    checkCublasErrors(cublasCreate(&blas_handle));
    checkCublasErrors(cublasSetMathMode(blas_handle, CUBLAS_PEDANTIC_MATH));
    float alpha = 1.0;
    float beta = 0;
    checkCudaErrors(cudaMemset(d_C, 0, bytes_C));
    for (int run = 0; run < warmupIter; run++) {
        checkCublasErrors(cublasSgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha,
            d_B, N, d_A, K, &beta, d_C, N
        ));
    }
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemset(d_C, 0, bytes_C));
    checkCudaErrors(cudaEventRecord(start));
    for (int run = 0 ; run < nIter; run ++ ) {
        checkCublasErrors(cublasSgemm(blas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha,
            d_B, N, d_A, K, &beta, d_C, N
        ));
    }
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

    checkCudaErrors(cudaMemcpy( h_C1, d_C, bytes_C, cudaMemcpyDeviceToHost));

    msecPerMatrixMul[1] = msecTotal / nIter;
    gigaFlops[1] = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul[1] / 1000.0f);
    printf( "CuBlas Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n",
        gigaFlops[1],
        msecPerMatrixMul[1],
        flopsPerMatrixMul);

    checkCublasErrors(cublasDestroy(blas_handle));
    
    double eps = 1.e-6;  // machine zero
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        double abs_err = fabs(h_C[i] - h_C1[i]);
        double dot_length = K;
        double abs_val = fabs(h_C[i]);
        double rel_err = abs_err / abs_val / dot_length;
        if (rel_err > eps) {
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n",
                    i, h_C[i], h_C1[i], eps);
            correct = false;
            break;
        }
    }

    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    printf("ratio= %f\n", gigaFlops[0] / gigaFlops[1]);
    
    // Free Memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C1);
}
