#pragma once

#include <cuda_runtime.h>

#ifndef CEIL_DIV
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#endif

#ifndef OFFSET
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#endif

#ifndef FLOAT4
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#endif

template <
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    int WMITER,
    int WNITER,
    int TM,
    int TN>
__global__ void gemm(
    int M,
    int N,
    int K,
    float alpha,
    float *A,
    float *B,
    float beta,
    float *C) {
    /*
     * SHARED_DECL_BEGIN
     */
    __shared__ float As[2][BK][BM];
    __shared__ float Bs[2][BK][BN];
    /*
     * SHARED_DECL_END
     */

    const int tid = threadIdx.x;
    const int wid = tid / 32;
    const int lane = tid % 32;
    const int thread_num = BM * BN / WM / WN * 32;
    const int tile_m0 = blockIdx.y * BM;
    const int tile_n0 = blockIdx.x * BN;

    /*
     * INDEX_MAPPING_BEGIN
     */
    const int load_a_smem_m = tid / (BK / 4);
    const int load_a_smem_k = (tid % (BK / 4)) * 4;
    const int load_b_smem_k = tid / (BN / 4);
    const int load_b_smem_n = (tid % (BN / 4)) * 4;
    const int hightA = thread_num / (BK / 4);
    const int hightB = thread_num / (BN / 4);
    const int Wrow = wid / (BN / WN);
    const int Wcol = wid % (BN / WN);
    const int Trow = lane / (WNITER / TN);
    const int Tcol = lane % (WNITER / TN);
    /*
     * INDEX_MAPPING_END
     */

    /*
     * REGISTER_DECL_BEGIN
     */
    float results[WM / WMITER * TM][WN / WNITER * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};
    /*
     * REGISTER_DECL_END
     */

    const int A_offset = tile_m0 * K;
    const int B_offset = tile_n0;
    const int C_offset = tile_m0 * N + tile_n0;
    const int k_tiles = K / BK;

    /*
     * GLOBAL_TO_SHARED_LOAD_BEGIN
     */
    for (int loadOffset = 0; loadOffset < BM; loadOffset += hightA) {
        float4 tmp = FLOAT4(A[OFFSET(load_a_smem_m + loadOffset, load_a_smem_k, K) + A_offset]);
        As[0][load_a_smem_k][load_a_smem_m + loadOffset] = tmp.x;
        As[0][load_a_smem_k + 1][load_a_smem_m + loadOffset] = tmp.y;
        As[0][load_a_smem_k + 2][load_a_smem_m + loadOffset] = tmp.z;
        As[0][load_a_smem_k + 3][load_a_smem_m + loadOffset] = tmp.w;
    }
    for (int loadOffset = 0; loadOffset < BK; loadOffset += hightB) {
        FLOAT4(Bs[0][load_b_smem_k + loadOffset][load_b_smem_n]) =
            FLOAT4(B[OFFSET(load_b_smem_k + loadOffset, load_b_smem_n, N) + B_offset]);
    }
    /*
     * GLOBAL_TO_SHARED_LOAD_END
     */

    /*
     * MAIN_LOOP_BEGIN
     */
    for (int bkIdx = 1; bkIdx < k_tiles; ++bkIdx) {
        /*
         * SYNC_AFTER_LOAD_BEGIN
         */
        __syncthreads();
        /*
         * SYNC_AFTER_LOAD_END
         */
        const int comp_flag = (bkIdx - 1) & 1;
        const int mem_flag = bkIdx & 1;

        /*
         * COMPUTE_INNER_BEGIN
         */
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int wm = 0; wm < WM / WMITER; ++wm) {
                #pragma unroll
                for (int wn = 0; wn < WN / WNITER; ++wn) {
                    #pragma unroll
                    for (int i = 0; i < TM; ++i) {
                        regM[i] = As[comp_flag][k][Wrow * WM + wm * WMITER + Trow * TM + i];
                    }
                    #pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        regN[j] = Bs[comp_flag][k][Wcol * WN + wn * WNITER + Tcol * TN + j];
                    }
                    #pragma unroll
                    for (int i = 0; i < TM; ++i) {
                        #pragma unroll
                        for (int j = 0; j < TN; ++j) {
                            results[wm * TM + i][wn * TN + j] += regM[i] * regN[j];
                        }
                    }
                }
            }
        }
        /*
         * COMPUTE_INNER_END
         */

        /*
         * NEXT_TILE_LOAD_BEGIN
         */
        for (int loadOffset = 0; loadOffset < BM; loadOffset += hightA) {
            float4 tmp = FLOAT4(A[OFFSET(load_a_smem_m + loadOffset, load_a_smem_k + bkIdx * BK, K) + A_offset]);
            As[mem_flag][load_a_smem_k][load_a_smem_m + loadOffset] = tmp.x;
            As[mem_flag][load_a_smem_k + 1][load_a_smem_m + loadOffset] = tmp.y;
            As[mem_flag][load_a_smem_k + 2][load_a_smem_m + loadOffset] = tmp.z;
            As[mem_flag][load_a_smem_k + 3][load_a_smem_m + loadOffset] = tmp.w;
        }
        for (int loadOffset = 0; loadOffset < BK; loadOffset += hightB) {
            FLOAT4(Bs[mem_flag][load_b_smem_k + loadOffset][load_b_smem_n]) =
                FLOAT4(B[OFFSET(load_b_smem_k + loadOffset + bkIdx * BK, load_b_smem_n, N) + B_offset]);
        }
        /*
         * NEXT_TILE_LOAD_END
         */
    }

    __syncthreads();
    const int comp_flag = (k_tiles - 1) & 1;
    #pragma unroll
    for (int k = 0; k < BK; ++k) {
        #pragma unroll
        for (int wm = 0; wm < WM / WMITER; ++wm) {
            #pragma unroll
            for (int wn = 0; wn < WN / WNITER; ++wn) {
                #pragma unroll
                for (int i = 0; i < TM; ++i) {
                    regM[i] = As[comp_flag][k][Wrow * WM + wm * WMITER + Trow * TM + i];
                }
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    regN[j] = Bs[comp_flag][k][Wcol * WN + wn * WNITER + Tcol * TN + j];
                }
                #pragma unroll
                for (int i = 0; i < TM; ++i) {
                    #pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        results[wm * TM + i][wn * TN + j] += regM[i] * regN[j];
                    }
                }
            }
        }
    }
    /*
     * MAIN_LOOP_END
     */

    /*
     * STORE_BEGIN
     */
    #pragma unroll
    for (int wm = 0; wm < WM / WMITER; ++wm) {
        #pragma unroll
        for (int wn = 0; wn < WN / WNITER; ++wn) {
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                #pragma unroll
                for (int n = 0; n < TN; ++n) {
                    const int c_index = OFFSET(
                        Wrow * WM + wm * WMITER + Trow * TM + m,
                        Wcol * WN + wn * WNITER + Tcol * TN + n,
                        N) + C_offset;
                    C[c_index] = alpha * results[m + wm * TM][n + wn * TN] + beta * C[c_index];
                }
            }
        }
    }
    /*
     * STORE_END
     */
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
    /*
     * LAUNCH_CONFIG_BEGIN
     */
    static const int BM = 64;
    static const int BN = 64;
    static const int BK = 16;
    static const int WM = 32;
    static const int WN = 32;
    static const int WMITER = 16;
    static const int WNITER = 32;
    static const int TM = 4;
    static const int TN = 4;
    /*
     * LAUNCH_CONFIG_END
     */

    dim3 threadsPerBlock((BM * BN) / (WM * WN) * 32);
    dim3 blocksPerGrid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    gemm<BM, BN, BK, WM, WN, WMITER, WNITER, TM, TN>
        <<<blocksPerGrid, threadsPerBlock>>>(M, N, K, alpha, A, B, beta, C);
}
