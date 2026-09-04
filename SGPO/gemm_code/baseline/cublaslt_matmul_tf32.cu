#include <cublasLt.h>
#include <cublas_v2.h>

#include <vector>

#include "baseline_common.cuh"

struct CublasLtContext {
    cublasLtHandle_t handle;
    cublasLtMatmulDesc_t operation;
    cublasLtMatrixLayout_t a_layout;
    cublasLtMatrixLayout_t b_layout;
    cublasLtMatrixLayout_t c_layout;
    cublasLtMatrixLayout_t d_layout;
    cublasLtMatmulAlgo_t algo;
    void *workspace;
    size_t workspace_size;
    int M;
    int K;
    int N;
    const float *A;
    const float *B;
    float *C;
    float alpha = 1.0f;
    float beta = 0.0f;
};

static void check_cublaslt(cublasStatus_t status, const char *file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLASLt error at %s:%d: status=%d\n", file, line, static_cast<int>(status));
        std::exit(2);
    }
}

#define CHECK_CUBLASLT(call) check_cublaslt((call), __FILE__, __LINE__)

static void run_cublaslt_matmul(void *opaque) {
    CublasLtContext *ctx = static_cast<CublasLtContext *>(opaque);
    CHECK_CUBLASLT(cublasLtMatmul(
        ctx->handle,
        ctx->operation,
        &ctx->alpha,
        ctx->A,
        ctx->a_layout,
        ctx->B,
        ctx->b_layout,
        &ctx->beta,
        ctx->C,
        ctx->c_layout,
        ctx->C,
        ctx->d_layout,
        &ctx->algo,
        ctx->workspace,
        ctx->workspace_size,
        0));
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

    cublasLtHandle_t lt_handle;
    CHECK_CUBLASLT(cublasLtCreate(&lt_handle));

    cublasLtMatmulDesc_t operation;
    cublasLtMatrixLayout_t a_layout;
    cublasLtMatrixLayout_t b_layout;
    cublasLtMatrixLayout_t c_layout;
    cublasLtMatrixLayout_t d_layout;
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F));
    cublasOperation_t trans = CUBLAS_OP_N;
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)));

    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&a_layout, CUDA_R_32F, args.M, args.K, args.K));
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&b_layout, CUDA_R_32F, args.K, args.N, args.N));
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&c_layout, CUDA_R_32F, args.M, args.N, args.N));
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&d_layout, CUDA_R_32F, args.M, args.N, args.N));
    cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(a_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(b_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(c_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(d_layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, sizeof(row_order)));

    const size_t workspace_size = 16 * 1024 * 1024;
    void *workspace = nullptr;
    CHECK_CUDA(cudaMalloc(&workspace, workspace_size));

    cublasLtMatmulPreference_t preference;
    CHECK_CUBLASLT(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUBLASLT(cublasLtMatmulPreferenceSetAttribute(
        preference,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size,
        sizeof(workspace_size)));

    cublasLtMatmulHeuristicResult_t heuristic;
    int returned = 0;
    CHECK_CUBLASLT(cublasLtMatmulAlgoGetHeuristic(
        lt_handle,
        operation,
        a_layout,
        b_layout,
        c_layout,
        d_layout,
        preference,
        1,
        &heuristic,
        &returned));
    if (returned == 0) {
        std::fprintf(stderr, "No cuBLASLt matmul heuristic was returned.\n");
        return 5;
    }

    CublasLtContext ctx{
        lt_handle,
        operation,
        a_layout,
        b_layout,
        c_layout,
        d_layout,
        heuristic.algo,
        workspace,
        workspace_size,
        args.M,
        args.K,
        args.N,
        d_A,
        d_B,
        d_C};

    double latency_ms = benchmark_ms(run_cublaslt_matmul, &ctx, args.warmup, args.iters);
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));
    double max_abs_error = max_abs_error_sampled(h_A, h_B, h_C, args.M, args.K, args.N);
    print_baseline_metric("cublaslt_matmul_tf32", "CUBLAS_COMPUTE_32F_FAST_TF32", args.M, args.K, args.N, latency_ms, max_abs_error);

    CHECK_CUBLASLT(cublasLtMatmulPreferenceDestroy(preference));
    CHECK_CUDA(cudaFree(workspace));
    CHECK_CUBLASLT(cublasLtMatrixLayoutDestroy(a_layout));
    CHECK_CUBLASLT(cublasLtMatrixLayoutDestroy(b_layout));
    CHECK_CUBLASLT(cublasLtMatrixLayoutDestroy(c_layout));
    CHECK_CUBLASLT(cublasLtMatrixLayoutDestroy(d_layout));
    CHECK_CUBLASLT(cublasLtMatmulDescDestroy(operation));
    CHECK_CUBLASLT(cublasLtDestroy(lt_handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    std::free(h_A);
    std::free(h_B);
    std::free(h_C);
    return 0;
}
