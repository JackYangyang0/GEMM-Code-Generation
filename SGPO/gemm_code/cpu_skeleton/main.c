#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

#include "kernel.h"

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

static double now_ms(void) {
#ifdef _WIN32
    static LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    if (frequency.QuadPart == 0) {
        QueryPerformanceFrequency(&frequency);
    }
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)frequency.QuadPart;
#elif defined(CLOCK_MONOTONIC)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
#else
    return (double)clock() * 1000.0 / (double)CLOCKS_PER_SEC;
#endif
}

static void reference_gemm(int M, int N, int K, const float *A, const float *B, float *C) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += A[OFFSET(m, k, K)] * B[OFFSET(k, n, N)];
            }
            C[OFFSET(m, n, N)] = acc;
        }
    }
}

int main(int argc, char **argv) {
    if (argc != 4) {
        printf("usage: ./gemm_cpu [M] [K] [N]\n");
        return 0;
    }

    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = atoi(argv[3]);
    size_t bytes_A = sizeof(float) * (size_t)M * (size_t)K;
    size_t bytes_B = sizeof(float) * (size_t)K * (size_t)N;
    size_t bytes_C = sizeof(float) * (size_t)M * (size_t)N;

    float *A = (float *)malloc(bytes_A);
    float *B = (float *)malloc(bytes_B);
    float *C = (float *)malloc(bytes_C);
    float *C_ref = (float *)malloc(bytes_C);
    if (!A || !B || !C || !C_ref) {
        printf("Error: host allocation failed\n");
        free(A);
        free(B);
        free(C);
        free(C_ref);
        return 1;
    }

    for (int i = 0; i < M * K; ++i) {
        A[i] = (float)(i % 17) / 17.0f;
    }
    for (int i = 0; i < K * N; ++i) {
        B[i] = (float)(i % 13) / 13.0f;
    }
    memset(C, 0, bytes_C);
    memset(C_ref, 0, bytes_C);

    int nIter = 10;
    const char *bench_iters_env = getenv("SGPO_CPU_BENCH_ITERS");
    if (bench_iters_env && atoi(bench_iters_env) > 0) {
        nIter = atoi(bench_iters_env);
    }
    double start = now_ms();
    for (int run = 0; run < nIter; ++run) {
        memset(C, 0, bytes_C);
        cpu_gemm(M, N, K, 1.0f, A, B, 0.0f, C);
    }
    double elapsed_ms = now_ms() - start;
    double latency_ms = elapsed_ms / (double)nIter;
    if (latency_ms <= 0.0) {
        latency_ms = 1.0e-6;
    }
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = (flops * 1.0e-9) / (latency_ms / 1000.0);

    reference_gemm(M, N, K, A, B, C_ref);

    double max_abs_error = 0.0;
    int correct = 1;
    for (int i = 0; i < M * N; ++i) {
        double abs_err = fabs((double)C[i] - (double)C_ref[i]);
        if (abs_err > max_abs_error) {
            max_abs_error = abs_err;
        }
        if (abs_err > 1.0e-3) {
            correct = 0;
            printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > 1.000000E-03\n", i, C[i], C_ref[i]);
            break;
        }
    }

    printf("SGPO CPU GEMM Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,\n", gflops, latency_ms, flops);
    printf("%s\n", correct ? "Result= PASS" : "Result= FAIL");
    printf("SGPO_METRIC backend=cpu correctness=%s latency_ms=%.6f gflops=%.6f max_abs_error=%.9f\n",
           correct ? "pass" : "fail", latency_ms, gflops, max_abs_error);

    free(A);
    free(B);
    free(C);
    free(C_ref);
    return correct ? 0 : 2;
}
