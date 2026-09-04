#include <cublas_v2.h>

#include "baseline_common.cuh"

struct CublasSgemmContext {
    cublasHandle_t handle;
    int M;
    int K;
    int N;
    const float *A;
    const float *B;
    float *C;
    float alpha = 1.0f;
    float beta = 0.0f;
};

static void run_cublas_sgemm_row_major(void *opaque) {
    CublasSgemmContext *ctx = static_cast<CublasSgemmContext *>(opaque);
    CHECK_CUBLAS(cublasSgemm(
        ctx->handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        ctx->N,
        ctx->M,
        ctx->K,
        &ctx->alpha,
        ctx->B,
        ctx->N,
        ctx->A,
        ctx->K,
        &ctx->beta,
        ctx->C,
        ctx->N));
}

int main(int argc, char **argv) {
    GemmArgs args = parse_args(argc, argv);
    const size_t bytes_A = sizeof(float) * static_cast<size_t>(args.M) * static_cast<size_t>(args.K);
    const size_t bytes_B = sizeof(float) * static_cast<size_t>(args.K) * static_cast<size_t>(args.N);
    const size_t bytes_C = sizeof(float) * static_cast<size_t>(args.M) * static_cast<size_t>(args.N);

    float *h_A = static_cast<float *>(std::malloc(bytes_A));
    float *h_B = static_cast<float *>(std::malloc(bytes_B));
    float *h_C = static_cast<float *>(std::malloc(bytes_C));
    if (!h_A || !h_B || !h_C) {
        return 4;
    }
    init_inputs(h_A, h_B, h_C, args.M, args.K, args.N);

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    CHECK_CUDA(cudaMalloc(&d_A, bytes_A));
    CHECK_CUDA(cudaMalloc(&d_B, bytes_B));
    CHECK_CUDA(cudaMalloc(&d_C, bytes_C));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C, 0, bytes_C));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
#if defined(CUBLAS_TF32_TENSOR_OP_MATH)
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
#else
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
#endif

    CublasSgemmContext ctx{handle, args.M, args.K, args.N, d_A, d_B, d_C};
    double latency_ms = benchmark_ms(run_cublas_sgemm_row_major, &ctx, args.warmup, args.iters);
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
    double max_abs_error = max_abs_error_sampled(h_A, h_B, h_C, args.M, args.K, args.N);
    print_baseline_metric("cublas_sgemm_tf32", "CUBLAS_TF32_TENSOR_OP_MATH", args.M, args.K, args.N, latency_ms, max_abs_error);

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    std::free(h_A);
    std::free(h_B);
    std::free(h_C);
    return 0;
}
