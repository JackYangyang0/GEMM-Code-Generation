#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                         \
        if (err__ != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,     \
                         __LINE__, cudaGetErrorString(err__));             \
            std::exit(1);                                                   \
        }                                                                  \
    } while (0)

#define CHECK_CUBLAS(call)                                                 \
    do {                                                                   \
        cublasStatus_t st__ = (call);                                       \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                \
            std::fprintf(stderr, "cuBLAS error at %s:%d: status=%d\n",      \
                         __FILE__, __LINE__, static_cast<int>(st__));       \
            std::exit(2);                                                   \
        }                                                                  \
    } while (0)

struct GemmArgs {
    int M = 512;
    int K = 512;
    int N = 512;
    int warmup = 10;
    int iters = 100;
};

inline GemmArgs parse_args(int argc, char **argv) {
    GemmArgs args;
    if (argc >= 4) {
        args.M = std::atoi(argv[1]);
        args.K = std::atoi(argv[2]);
        args.N = std::atoi(argv[3]);
    }
    if (argc >= 5) {
        args.iters = std::atoi(argv[4]);
    }
    if (argc >= 6) {
        args.warmup = std::atoi(argv[5]);
    }
    if (args.M <= 0 || args.K <= 0 || args.N <= 0 || args.iters <= 0 || args.warmup < 0) {
        std::fprintf(stderr, "usage: %s M K N [iters] [warmup]\n", argv[0]);
        std::exit(3);
    }
    return args;
}

inline void init_inputs(float *A, float *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M * K; ++i) {
        A[i] = static_cast<float>((i % 17) - 8) / 17.0f;
    }
    for (int i = 0; i < K * N; ++i) {
        B[i] = static_cast<float>((i % 13) - 6) / 13.0f;
    }
    for (int i = 0; i < M * N; ++i) {
        C[i] = 0.0f;
    }
}

inline double benchmark_ms(void (*fn)(void *), void *ctx, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) {
        fn(ctx);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        fn(ctx);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return static_cast<double>(ms) / static_cast<double>(iters);
}

inline double max_abs_error_sampled(
    const float *A,
    const float *B,
    const float *C,
    int M,
    int K,
    int N,
    int row_step = 17,
    int col_step = 19) {
    double max_err = 0.0;
    for (int row = 0; row < M; row += row_step) {
        for (int col = 0; col < N; col += col_step) {
            double ref = 0.0;
            for (int k = 0; k < K; ++k) {
                ref += static_cast<double>(A[row * K + k]) * static_cast<double>(B[k * N + col]);
            }
            double err = std::fabs(static_cast<double>(C[row * N + col]) - ref);
            if (err > max_err) {
                max_err = err;
            }
        }
    }
    return max_err;
}

inline void print_baseline_metric(
    const char *name,
    const char *math_mode,
    int M,
    int K,
    int N,
    double latency_ms,
    double max_abs_error) {
    const double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    const double gflops = (flops * 1.0e-9) / (latency_ms / 1000.0);
    std::printf(
        "SGPO_BASELINE name=%s math_mode=%s M=%d K=%d N=%d latency_ms=%.9f gflops=%.9f max_abs_error=%.9e\n",
        name, math_mode, M, K, N, latency_ms, gflops, max_abs_error);
}
