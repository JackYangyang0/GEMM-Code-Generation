from __future__ import annotations

import shutil
import json
import math
from pathlib import Path
from typing import Any

from SGPO.verification.gemm_semantic_checker import check_gemm_semantic_obligations


DEFAULT_REPAIR_FILES = ("main.cpp", "kernel.h", "cuda_kernel.cuh")
DEFAULT_CPU_REPAIR_FILES = ("main.c", "kernel.h", "cpu_kernel.c")


def generate_semantic_repair_candidate(
    source_chain_dir: Path,
    output_chain_dir: Path,
    diagnosis: dict[str, Any] | None = None,
    ir: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a deterministic repaired chain candidate for semantic-repair tests.

    This is intentionally local and deterministic so unit tests do not depend on
    an external LLM. It mirrors what the LLM repair step should produce from the
    structured semantic defects: a complete, correct GEMM kernel body with
    cooperative shared loads, full K-tile dataflow, and TM/TN store coverage.
    """

    if target_backend(ir or {}) == "cpu":
        return generate_cpu_semantic_repair_candidate(source_chain_dir, output_chain_dir, diagnosis, ir)

    output_chain_dir.mkdir(parents=True, exist_ok=True)
    for relative_path in DEFAULT_REPAIR_FILES:
        source = source_chain_dir / relative_path
        target = output_chain_dir / relative_path
        if source.exists() and relative_path != "cuda_kernel.cuh":
            if source.resolve() != target.resolve():
                shutil.copyfile(source, target)

    launch = normalize_launch_config_for_repair(extract_launch_config_from_ir(ir or {}))
    if should_emit_throughput_kernel(ir or {}):
        kernel_content = build_throughput_cuda_kernel_cuh(normalize_throughput_launch_config(launch))
    else:
        kernel_content = build_repaired_cuda_kernel_cuh(launch)
    (output_chain_dir / "cuda_kernel.cuh").write_text(kernel_content, encoding="utf-8")
    ensure_kernel_header_includes_cuda_kernel(output_chain_dir / "kernel.h")

    semantic_report = check_gemm_semantic_obligations(output_chain_dir, ir or {})
    result = {
        "status": "repaired" if semantic_report.get("accepted") else "repair_generated_with_remaining_defects",
        "source_chain_dir": str(source_chain_dir),
        "output_chain_dir": str(output_chain_dir),
        "diagnosis_defect_count": (diagnosis or {}).get("defect_count", 0),
        "diagnosis": diagnosis or {},
        "semantic_obligations": semantic_report,
        "files": list(DEFAULT_REPAIR_FILES),
    }
    (output_chain_dir / "semantic_repair_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def generate_cpu_semantic_repair_candidate(
    source_chain_dir: Path,
    output_chain_dir: Path,
    diagnosis: dict[str, Any] | None = None,
    ir: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a deterministic CPU C repair that preserves GEMM semantics."""

    output_chain_dir.mkdir(parents=True, exist_ok=True)
    for relative_path in DEFAULT_CPU_REPAIR_FILES:
        source = source_chain_dir / relative_path
        target = output_chain_dir / relative_path
        if source.exists() and relative_path != "cpu_kernel.c":
            if source.resolve() != target.resolve():
                shutil.copyfile(source, target)
            if relative_path == "main.c":
                ensure_cpu_benchmark_iterations(target)

    repaired_kernel = build_repaired_cpu_kernel_c(ir or {})
    (output_chain_dir / "cpu_kernel.c").write_text(repaired_kernel, encoding="utf-8")
    result = {
        "status": "repaired",
        "backend": "cpu",
        "source_chain_dir": str(source_chain_dir),
        "output_chain_dir": str(output_chain_dir),
        "diagnosis_defect_count": (diagnosis or {}).get("defect_count", 0),
        "diagnosis": diagnosis or {},
        "files": list(DEFAULT_CPU_REPAIR_FILES),
        "repair_notes": [
            "accumulate over the complete K domain before storing C",
            "cover every output element in each L1 tile with register-tile tail guards",
            "index A as A[(m + rm) * K + k] and B as B[k * N + (n + rn)]",
            "apply alpha and beta exactly once at final store",
            "avoid SIMD lane mapping bugs by keeping K as the scalar reduction domain unless a verified N-lane micro-kernel is generated",
            "avoid OpenMP packed-buffer races by not sharing mutable packed panels across parallel workers",
        ],
    }
    (output_chain_dir / "semantic_repair_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def build_repaired_cpu_kernel_c(ir: dict[str, Any]) -> str:
    strategy_ids = set(ir.get("strategy", {}).get("applied_strategy_ids", []) or [])
    strategy_ids.update(
        item.get("strategy_id")
        for item in ir.get("strategy", {}).get("applied_micro_strategies", []) or []
        if isinstance(item, dict)
    )
    joined = "\n".join(str(item) for item in strategy_ids)
    if "AVX512" in joined:
        if cpu_supports_avx512(ir):
            return build_repaired_cpu_avx512_kernel_c(ir)
        if cpu_supports_avx2(ir):
            return build_repaired_cpu_avx2_kernel_c(ir)
    if "AVX2" in joined:
        return build_repaired_cpu_avx2_kernel_c(ir)

    cpu_tiling = ir.get("cpu_tiling", {}) or {}
    l2_m = safe_positive_int(cpu_tiling.get("l2_block_m"), 128)
    l2_n = safe_positive_int(cpu_tiling.get("l2_block_n"), 128)
    l2_k = safe_positive_int(cpu_tiling.get("l2_block_k"), 128)
    l1_m = safe_positive_int(cpu_tiling.get("l1_block_m"), min(32, l2_m))
    l1_n = safe_positive_int(cpu_tiling.get("l1_block_n"), min(32, l2_n))
    l1_k = safe_positive_int(cpu_tiling.get("l1_block_k"), min(64, l2_k))
    reg_m = min(safe_positive_int(cpu_tiling.get("register_m"), 4), 8)
    reg_n = min(safe_positive_int(cpu_tiling.get("register_n"), 4), 8)
    return f"""#include "kernel.h"

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

void cpu_gemm(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {{
    /* SGPO_CPU_PATCH_KERNEL_BEGIN */
    const int L2_BLOCK_M = {l2_m};
    const int L2_BLOCK_N = {l2_n};
    const int L2_BLOCK_K = {l2_k};
    const int L1_BLOCK_M = {l1_m};
    const int L1_BLOCK_N = {l1_n};
    const int L1_BLOCK_K = {l1_k};
    const int RM = {reg_m};
    const int RN = {reg_n};

    for (int m0 = 0; m0 < M; m0 += L2_BLOCK_M) {{
        const int m_l2_end = (m0 + L2_BLOCK_M < M) ? (m0 + L2_BLOCK_M) : M;
        for (int n0 = 0; n0 < N; n0 += L2_BLOCK_N) {{
            const int n_l2_end = (n0 + L2_BLOCK_N < N) ? (n0 + L2_BLOCK_N) : N;
            for (int mi = m0; mi < m_l2_end; mi += L1_BLOCK_M) {{
                const int m_l1_end = (mi + L1_BLOCK_M < m_l2_end) ? (mi + L1_BLOCK_M) : m_l2_end;
                for (int nj = n0; nj < n_l2_end; nj += L1_BLOCK_N) {{
                    const int n_l1_end = (nj + L1_BLOCK_N < n_l2_end) ? (nj + L1_BLOCK_N) : n_l2_end;

                    for (int m = mi; m < m_l1_end; m += RM) {{
                        const int rm_count = (m + RM < m_l1_end) ? RM : (m_l1_end - m);
                        for (int n = nj; n < n_l1_end; n += RN) {{
                            const int rn_count = (n + RN < n_l1_end) ? RN : (n_l1_end - n);
                            float acc[8][8] = {{0.0f}};

                            for (int k0 = 0; k0 < K; k0 += L2_BLOCK_K) {{
                                const int k_l2_end = (k0 + L2_BLOCK_K < K) ? (k0 + L2_BLOCK_K) : K;
                                for (int kk = k0; kk < k_l2_end; kk += L1_BLOCK_K) {{
                                    const int k_l1_end = (kk + L1_BLOCK_K < k_l2_end) ? (kk + L1_BLOCK_K) : k_l2_end;
                                    for (int k = kk; k < k_l1_end; ++k) {{
                                        for (int rm = 0; rm < rm_count; ++rm) {{
                                            const float a_val = A[OFFSET(m + rm, k, K)];
                                            for (int rn = 0; rn < rn_count; ++rn) {{
                                                acc[rm][rn] += a_val * B[OFFSET(k, n + rn, N)];
                                            }}
                                        }}
                                    }}
                                }}
                            }}

                            for (int rm = 0; rm < rm_count; ++rm) {{
                                for (int rn = 0; rn < rn_count; ++rn) {{
                                    const int c_idx = OFFSET(m + rm, n + rn, N);
                                    C[c_idx] = alpha * acc[rm][rn] + beta * C[c_idx];
                                }}
                            }}
                        }}
                    }}
                }}
            }}
        }}
    }}
    /* SGPO_CPU_PATCH_KERNEL_END */
}}
"""


def ensure_cpu_benchmark_iterations(main_path: Path) -> None:
    if not main_path.exists():
        return
    content = main_path.read_text(encoding="utf-8")
    if "SGPO_CPU_BENCH_ITERS" in content:
        return
    content = content.replace(
        "    int nIter = 3;\n",
        (
            "    int nIter = 10;\n"
            "    const char *bench_iters_env = getenv(\"SGPO_CPU_BENCH_ITERS\");\n"
            "    if (bench_iters_env && atoi(bench_iters_env) > 0) {\n"
            "        nIter = atoi(bench_iters_env);\n"
            "    }\n"
        ),
    )
    main_path.write_text(content, encoding="utf-8")


def build_repaired_cpu_avx512_kernel_c(ir: dict[str, Any]) -> str:
    cpu_tiling = ir.get("cpu_tiling", {}) or {}
    l2_m = safe_positive_int(cpu_tiling.get("l2_block_m"), 128)
    l2_n = safe_positive_int(cpu_tiling.get("l2_block_n"), 128)
    l2_k = safe_positive_int(cpu_tiling.get("l2_block_k"), 128)
    return f"""#include "kernel.h"
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

void cpu_gemm(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {{
    /* SGPO_CPU_PATCH_KERNEL_BEGIN */
    const int L2_BLOCK_M = {l2_m};
    const int L2_BLOCK_N = {l2_n};
    const int L2_BLOCK_K = {l2_k};
    const int MR = 4;
    const int NR = 16;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int m0 = 0; m0 < M; m0 += L2_BLOCK_M) {{
        for (int n0 = 0; n0 < N; n0 += L2_BLOCK_N) {{
            const int m_end = (m0 + L2_BLOCK_M < M) ? (m0 + L2_BLOCK_M) : M;
            const int n_end = (n0 + L2_BLOCK_N < N) ? (n0 + L2_BLOCK_N) : N;

            for (int m = m0; m < m_end; m += MR) {{
                const int full_m = (m + MR <= m_end);
                for (int n = n0; n < n_end; n += NR) {{
                    const int full_n = (n + NR <= n_end);

                    if (full_m && full_n) {{
                        __m512 acc0 = _mm512_setzero_ps();
                        __m512 acc1 = _mm512_setzero_ps();
                        __m512 acc2 = _mm512_setzero_ps();
                        __m512 acc3 = _mm512_setzero_ps();

                        for (int k0 = 0; k0 < K; k0 += L2_BLOCK_K) {{
                            const int k_end = (k0 + L2_BLOCK_K < K) ? (k0 + L2_BLOCK_K) : K;
                            const int kc = k_end - k0;
                            float *a_panel = (float*)malloc((size_t)MR * (size_t)kc * sizeof(float));
                            float *b_panel = (float*)malloc((size_t)kc * (size_t)NR * sizeof(float));
                            if (!a_panel || !b_panel) {{
                                free(a_panel);
                                free(b_panel);
                                for (int k = k0; k < k_end; ++k) {{
                                    __m512 b = _mm512_loadu_ps(&B[OFFSET(k, n, N)]);
                                    acc0 = _mm512_fmadd_ps(_mm512_broadcastss_ps(_mm_load_ss(&A[OFFSET(m + 0, k, K)])), b, acc0);
                                    acc1 = _mm512_fmadd_ps(_mm512_broadcastss_ps(_mm_load_ss(&A[OFFSET(m + 1, k, K)])), b, acc1);
                                    acc2 = _mm512_fmadd_ps(_mm512_broadcastss_ps(_mm_load_ss(&A[OFFSET(m + 2, k, K)])), b, acc2);
                                    acc3 = _mm512_fmadd_ps(_mm512_broadcastss_ps(_mm_load_ss(&A[OFFSET(m + 3, k, K)])), b, acc3);
                                }}
                                continue;
                            }}

                            for (int kk = 0; kk < kc; ++kk) {{
                                a_panel[0 * kc + kk] = A[OFFSET(m + 0, k0 + kk, K)];
                                a_panel[1 * kc + kk] = A[OFFSET(m + 1, k0 + kk, K)];
                                a_panel[2 * kc + kk] = A[OFFSET(m + 2, k0 + kk, K)];
                                a_panel[3 * kc + kk] = A[OFFSET(m + 3, k0 + kk, K)];
                                memcpy(&b_panel[kk * NR], &B[OFFSET(k0 + kk, n, N)], NR * sizeof(float));
                            }}

                            for (int kk = 0; kk < kc; ++kk) {{
                                __m512 b = _mm512_loadu_ps(&b_panel[kk * NR]);
                                acc0 = _mm512_fmadd_ps(_mm512_broadcastss_ps(_mm_load_ss(&a_panel[0 * kc + kk])), b, acc0);
                                acc1 = _mm512_fmadd_ps(_mm512_broadcastss_ps(_mm_load_ss(&a_panel[1 * kc + kk])), b, acc1);
                                acc2 = _mm512_fmadd_ps(_mm512_broadcastss_ps(_mm_load_ss(&a_panel[2 * kc + kk])), b, acc2);
                                acc3 = _mm512_fmadd_ps(_mm512_broadcastss_ps(_mm_load_ss(&a_panel[3 * kc + kk])), b, acc3);
                            }}

                            free(a_panel);
                            free(b_panel);
                        }}

                        const __m512 alpha_v = _mm512_set1_ps(alpha);
                        if (beta == 0.0f) {{
                            _mm512_storeu_ps(&C[OFFSET(m + 0, n, N)], _mm512_mul_ps(alpha_v, acc0));
                            _mm512_storeu_ps(&C[OFFSET(m + 1, n, N)], _mm512_mul_ps(alpha_v, acc1));
                            _mm512_storeu_ps(&C[OFFSET(m + 2, n, N)], _mm512_mul_ps(alpha_v, acc2));
                            _mm512_storeu_ps(&C[OFFSET(m + 3, n, N)], _mm512_mul_ps(alpha_v, acc3));
                        }} else {{
                            const __m512 beta_v = _mm512_set1_ps(beta);
                            __m512 c0 = _mm512_loadu_ps(&C[OFFSET(m + 0, n, N)]);
                            __m512 c1 = _mm512_loadu_ps(&C[OFFSET(m + 1, n, N)]);
                            __m512 c2 = _mm512_loadu_ps(&C[OFFSET(m + 2, n, N)]);
                            __m512 c3 = _mm512_loadu_ps(&C[OFFSET(m + 3, n, N)]);
                            _mm512_storeu_ps(&C[OFFSET(m + 0, n, N)], _mm512_fmadd_ps(beta_v, c0, _mm512_mul_ps(alpha_v, acc0)));
                            _mm512_storeu_ps(&C[OFFSET(m + 1, n, N)], _mm512_fmadd_ps(beta_v, c1, _mm512_mul_ps(alpha_v, acc1)));
                            _mm512_storeu_ps(&C[OFFSET(m + 2, n, N)], _mm512_fmadd_ps(beta_v, c2, _mm512_mul_ps(alpha_v, acc2)));
                            _mm512_storeu_ps(&C[OFFSET(m + 3, n, N)], _mm512_fmadd_ps(beta_v, c3, _mm512_mul_ps(alpha_v, acc3)));
                        }}
                    }} else {{
                        const int rm_count = (m + MR <= m_end) ? MR : (m_end - m);
                        const int rn_count = (n + NR <= n_end) ? NR : (n_end - n);
                        float acc[4][16] = {{0.0f}};
                        for (int k = 0; k < K; ++k) {{
                            for (int rm = 0; rm < rm_count; ++rm) {{
                                const float a_val = A[OFFSET(m + rm, k, K)];
                                for (int rn = 0; rn < rn_count; ++rn) {{
                                    acc[rm][rn] += a_val * B[OFFSET(k, n + rn, N)];
                                }}
                            }}
                        }}
                        for (int rm = 0; rm < rm_count; ++rm) {{
                            for (int rn = 0; rn < rn_count; ++rn) {{
                                const int c_idx = OFFSET(m + rm, n + rn, N);
                                C[c_idx] = alpha * acc[rm][rn] + beta * C[c_idx];
                            }}
                        }}
                    }}
                }}
            }}
        }}
    }}
    /* SGPO_CPU_PATCH_KERNEL_END */
}}
"""


def build_repaired_cpu_avx2_kernel_c(ir: dict[str, Any]) -> str:
    strategy_ids = set(ir.get("strategy", {}).get("applied_strategy_ids", []) or [])
    strategy_ids.update(
        item.get("strategy_id")
        for item in ir.get("strategy", {}).get("applied_micro_strategies", []) or []
        if isinstance(item, dict)
    )
    joined = "\n".join(str(item) for item in strategy_ids)
    microkernel = ir.get("cpu_microkernel", {}) or {}
    mr = safe_positive_int(microkernel.get("mr"), 4)
    if "6x16" in joined or mr >= 6:
        return build_repaired_cpu_avx2_generated_kernel_c(ir, 6, 16)
    if "4x8" in joined:
        return build_repaired_cpu_avx2_generated_kernel_c(ir, 4, 8)
    return build_repaired_cpu_avx2_generated_kernel_c(ir, 4, 16)


def build_repaired_cpu_avx2_generated_kernel_c(ir: dict[str, Any], mr: int, nr: int) -> str:
    cpu_tiling = ir.get("cpu_tiling", {}) or {}
    l2_m = safe_positive_int(cpu_tiling.get("l2_block_m"), 128)
    l2_n = safe_positive_int(cpu_tiling.get("l2_block_n"), 128)
    l2_k = safe_positive_int(cpu_tiling.get("l2_block_k"), 128)
    vecs = nr // 8
    lines = [
        '#include "kernel.h"',
        "#include <immintrin.h>",
        "#ifdef _OPENMP",
        "#include <omp.h>",
        "#endif",
        "",
        "#define OFFSET(row, col, ld) ((row) * (ld) + (col))",
        "",
        "void cpu_gemm(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {",
        "    /* SGPO_CPU_PATCH_KERNEL_BEGIN */",
        "    const float * restrict A_data = A;",
        "    const float * restrict B_data = B;",
        "    float * restrict C_data = C;",
        "    const int lda = K;",
        "    const int ldb = N;",
        "    const int ldc = N;",
        f"    const int L2_BLOCK_M = {l2_m};",
        f"    const int L2_BLOCK_N = {l2_n};",
        f"    const int L2_BLOCK_K = {l2_k};",
        f"    const int MR = {mr};",
        f"    const int NR = {nr};",
        "",
        "    #pragma omp parallel for collapse(2) schedule(static)",
        "    for (int n0 = 0; n0 < N; n0 += L2_BLOCK_N) {",
        "        for (int m0 = 0; m0 < M; m0 += L2_BLOCK_M) {",
        "            const int n_end = (n0 + L2_BLOCK_N < N) ? (n0 + L2_BLOCK_N) : N;",
        "            const int m_end = (m0 + L2_BLOCK_M < M) ? (m0 + L2_BLOCK_M) : M;",
        "            for (int m = m0; m < m_end; m += MR) {",
        "                const int full_m = (m + MR <= m_end);",
        "                for (int n = n0; n < n_end; n += NR) {",
        "                    const int full_n = (n + NR <= n_end);",
        "                    if (full_m && full_n) {",
    ]
    for r in range(mr):
        for v in range(vecs):
            lines.append(f"                        __m256 acc{r}{v} = _mm256_setzero_ps();")
    lines.extend(
        [
            "                        for (int k0 = 0; k0 < K; k0 += L2_BLOCK_K) {",
            "                            const int k_end = (k0 + L2_BLOCK_K < K) ? (k0 + L2_BLOCK_K) : K;",
            "                            for (int k = k0; k < k_end; ++k) {",
        ]
    )
    for v in range(vecs):
        lines.append(f"                                __m256 b{v} = _mm256_loadu_ps(&B_data[k * ldb + n + {v * 8}]);")
    for r in range(mr):
        lines.append(f"                                __m256 a{r} = _mm256_broadcast_ss(&A_data[(m + {r}) * lda + k]);")
        for v in range(vecs):
            lines.append(f"                                acc{r}{v} = _mm256_fmadd_ps(a{r}, b{v}, acc{r}{v});")
    lines.extend(
        [
            "                            }",
            "                        }",
            "                        const __m256 alpha_v = _mm256_set1_ps(alpha);",
            "                        if (beta == 0.0f) {",
        ]
    )
    for r in range(mr):
        for v in range(vecs):
            lines.append(
                f"                            _mm256_storeu_ps(&C_data[(m + {r}) * ldc + n + {v * 8}], _mm256_mul_ps(alpha_v, acc{r}{v}));"
            )
    lines.extend(
        [
            "                        } else {",
            "                            const __m256 beta_v = _mm256_set1_ps(beta);",
        ]
    )
    for r in range(mr):
        for v in range(vecs):
            lines.append(f"                            __m256 c{r}{v} = _mm256_loadu_ps(&C_data[(m + {r}) * ldc + n + {v * 8}]);")
            lines.append(
                f"                            _mm256_storeu_ps(&C_data[(m + {r}) * ldc + n + {v * 8}], _mm256_fmadd_ps(beta_v, c{r}{v}, _mm256_mul_ps(alpha_v, acc{r}{v})));"
            )
    lines.extend(
        [
            "                        }",
            "                    } else {",
            "                        const int rm_count = (m + MR <= m_end) ? MR : (m_end - m);",
            "                        const int rn_count = (n + NR <= n_end) ? NR : (n_end - n);",
            "                        float acc[8][32] = {0.0f};",
            "                        for (int k0 = 0; k0 < K; k0 += L2_BLOCK_K) {",
            "                            const int k_end = (k0 + L2_BLOCK_K < K) ? (k0 + L2_BLOCK_K) : K;",
            "                            for (int k = k0; k < k_end; ++k) {",
            "                                for (int rm = 0; rm < rm_count; ++rm) {",
            "                                    const float a_val = A_data[(m + rm) * lda + k];",
            "                                    for (int rn = 0; rn < rn_count; ++rn) {",
            "                                        acc[rm][rn] += a_val * B_data[k * ldb + n + rn];",
            "                                    }",
            "                                }",
            "                            }",
            "                        }",
            "                        for (int rm = 0; rm < rm_count; ++rm) {",
            "                            for (int rn = 0; rn < rn_count; ++rn) {",
            "                                const int c_idx = (m + rm) * ldc + n + rn;",
            "                                C_data[c_idx] = alpha * acc[rm][rn] + beta * C_data[c_idx];",
            "                            }",
            "                        }",
            "                    }",
            "                }",
            "            }",
            "        }",
            "    }",
            "    /* SGPO_CPU_PATCH_KERNEL_END */",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def build_repaired_cpu_avx2_4x16_kernel_c(ir: dict[str, Any]) -> str:
    cpu_tiling = ir.get("cpu_tiling", {}) or {}
    l2_m = safe_positive_int(cpu_tiling.get("l2_block_m"), 128)
    l2_n = safe_positive_int(cpu_tiling.get("l2_block_n"), 128)
    l2_k = safe_positive_int(cpu_tiling.get("l2_block_k"), 128)
    return f"""#include "kernel.h"
#include <immintrin.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

void cpu_gemm(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {{
    /* SGPO_CPU_PATCH_KERNEL_BEGIN */
    const int L2_BLOCK_M = {l2_m};
    const int L2_BLOCK_N = {l2_n};
    const int L2_BLOCK_K = {l2_k};
    const int MR = 4;
    const int NR = 16;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int n0 = 0; n0 < N; n0 += L2_BLOCK_N) {{
        const int n_end = (n0 + L2_BLOCK_N < N) ? (n0 + L2_BLOCK_N) : N;
        for (int m0 = 0; m0 < M; m0 += L2_BLOCK_M) {{
            const int m_end = (m0 + L2_BLOCK_M < M) ? (m0 + L2_BLOCK_M) : M;

            for (int m = m0; m < m_end; m += MR) {{
                const int full_m = (m + MR <= m_end);
                for (int n = n0; n < n_end; n += NR) {{
                    const int full_n = (n + NR <= n_end);

                    if (full_m && full_n) {{
                        __m256 acc00 = _mm256_setzero_ps();
                        __m256 acc01 = _mm256_setzero_ps();
                        __m256 acc10 = _mm256_setzero_ps();
                        __m256 acc11 = _mm256_setzero_ps();
                        __m256 acc20 = _mm256_setzero_ps();
                        __m256 acc21 = _mm256_setzero_ps();
                        __m256 acc30 = _mm256_setzero_ps();
                        __m256 acc31 = _mm256_setzero_ps();

                        for (int k0 = 0; k0 < K; k0 += L2_BLOCK_K) {{
                            const int k_end = (k0 + L2_BLOCK_K < K) ? (k0 + L2_BLOCK_K) : K;
                            for (int k = k0; k < k_end; ++k) {{
                                const float *a_panel = &A[OFFSET(m, k, K)];
                                const float *b_panel = &B[OFFSET(k, n, N)];
                                __m256 b0 = _mm256_loadu_ps(&b_panel[0]);
                                __m256 b1 = _mm256_loadu_ps(&b_panel[8]);
                                __m256 a0 = _mm256_broadcast_ss(&a_panel[0 * K]);
                                __m256 a1 = _mm256_broadcast_ss(&a_panel[1 * K]);
                                __m256 a2 = _mm256_broadcast_ss(&a_panel[2 * K]);
                                __m256 a3 = _mm256_broadcast_ss(&a_panel[3 * K]);
                                acc00 = _mm256_fmadd_ps(a0, b0, acc00);
                                acc01 = _mm256_fmadd_ps(a0, b1, acc01);
                                acc10 = _mm256_fmadd_ps(a1, b0, acc10);
                                acc11 = _mm256_fmadd_ps(a1, b1, acc11);
                                acc20 = _mm256_fmadd_ps(a2, b0, acc20);
                                acc21 = _mm256_fmadd_ps(a2, b1, acc21);
                                acc30 = _mm256_fmadd_ps(a3, b0, acc30);
                                acc31 = _mm256_fmadd_ps(a3, b1, acc31);
                            }}
                        }}

                        const __m256 alpha_v = _mm256_set1_ps(alpha);
                        if (beta == 0.0f) {{
                            _mm256_storeu_ps(&C[OFFSET(m + 0, n + 0, N)], _mm256_mul_ps(alpha_v, acc00));
                            _mm256_storeu_ps(&C[OFFSET(m + 0, n + 8, N)], _mm256_mul_ps(alpha_v, acc01));
                            _mm256_storeu_ps(&C[OFFSET(m + 1, n + 0, N)], _mm256_mul_ps(alpha_v, acc10));
                            _mm256_storeu_ps(&C[OFFSET(m + 1, n + 8, N)], _mm256_mul_ps(alpha_v, acc11));
                            _mm256_storeu_ps(&C[OFFSET(m + 2, n + 0, N)], _mm256_mul_ps(alpha_v, acc20));
                            _mm256_storeu_ps(&C[OFFSET(m + 2, n + 8, N)], _mm256_mul_ps(alpha_v, acc21));
                            _mm256_storeu_ps(&C[OFFSET(m + 3, n + 0, N)], _mm256_mul_ps(alpha_v, acc30));
                            _mm256_storeu_ps(&C[OFFSET(m + 3, n + 8, N)], _mm256_mul_ps(alpha_v, acc31));
                        }} else {{
                            const __m256 beta_v = _mm256_set1_ps(beta);
                            __m256 c00 = _mm256_loadu_ps(&C[OFFSET(m + 0, n + 0, N)]);
                            __m256 c01 = _mm256_loadu_ps(&C[OFFSET(m + 0, n + 8, N)]);
                            __m256 c10 = _mm256_loadu_ps(&C[OFFSET(m + 1, n + 0, N)]);
                            __m256 c11 = _mm256_loadu_ps(&C[OFFSET(m + 1, n + 8, N)]);
                            __m256 c20 = _mm256_loadu_ps(&C[OFFSET(m + 2, n + 0, N)]);
                            __m256 c21 = _mm256_loadu_ps(&C[OFFSET(m + 2, n + 8, N)]);
                            __m256 c30 = _mm256_loadu_ps(&C[OFFSET(m + 3, n + 0, N)]);
                            __m256 c31 = _mm256_loadu_ps(&C[OFFSET(m + 3, n + 8, N)]);
                            _mm256_storeu_ps(&C[OFFSET(m + 0, n + 0, N)], _mm256_fmadd_ps(beta_v, c00, _mm256_mul_ps(alpha_v, acc00)));
                            _mm256_storeu_ps(&C[OFFSET(m + 0, n + 8, N)], _mm256_fmadd_ps(beta_v, c01, _mm256_mul_ps(alpha_v, acc01)));
                            _mm256_storeu_ps(&C[OFFSET(m + 1, n + 0, N)], _mm256_fmadd_ps(beta_v, c10, _mm256_mul_ps(alpha_v, acc10)));
                            _mm256_storeu_ps(&C[OFFSET(m + 1, n + 8, N)], _mm256_fmadd_ps(beta_v, c11, _mm256_mul_ps(alpha_v, acc11)));
                            _mm256_storeu_ps(&C[OFFSET(m + 2, n + 0, N)], _mm256_fmadd_ps(beta_v, c20, _mm256_mul_ps(alpha_v, acc20)));
                            _mm256_storeu_ps(&C[OFFSET(m + 2, n + 8, N)], _mm256_fmadd_ps(beta_v, c21, _mm256_mul_ps(alpha_v, acc21)));
                            _mm256_storeu_ps(&C[OFFSET(m + 3, n + 0, N)], _mm256_fmadd_ps(beta_v, c30, _mm256_mul_ps(alpha_v, acc30)));
                            _mm256_storeu_ps(&C[OFFSET(m + 3, n + 8, N)], _mm256_fmadd_ps(beta_v, c31, _mm256_mul_ps(alpha_v, acc31)));
                        }}
                    }} else {{
                        const int rm_count = (m + MR <= m_end) ? MR : (m_end - m);
                        const int rn_count = (n + NR <= n_end) ? NR : (n_end - n);
                        float acc[4][16] = {{0.0f}};
                        for (int k0 = 0; k0 < K; k0 += L2_BLOCK_K) {{
                            const int k_end = (k0 + L2_BLOCK_K < K) ? (k0 + L2_BLOCK_K) : K;
                            for (int k = k0; k < k_end; ++k) {{
                                for (int rm = 0; rm < rm_count; ++rm) {{
                                    const float a_val = A[OFFSET(m + rm, k, K)];
                                    for (int rn = 0; rn < rn_count; ++rn) {{
                                        acc[rm][rn] += a_val * B[OFFSET(k, n + rn, N)];
                                    }}
                                }}
                            }}
                        }}
                        for (int rm = 0; rm < rm_count; ++rm) {{
                            for (int rn = 0; rn < rn_count; ++rn) {{
                                const int c_idx = OFFSET(m + rm, n + rn, N);
                                C[c_idx] = alpha * acc[rm][rn] + beta * C[c_idx];
                            }}
                        }}
                    }}
                }}
            }}
        }}
    }}
    /* SGPO_CPU_PATCH_KERNEL_END */
}}
"""


def cpu_supports_avx512(ir: dict[str, Any]) -> bool:
    isa = ((ir.get("hardware", {}) or {}).get("cpu_isa", {}) or {})
    return isa.get("avx512f") is True


def cpu_supports_avx2(ir: dict[str, Any]) -> bool:
    isa = ((ir.get("hardware", {}) or {}).get("cpu_isa", {}) or {})
    return isa.get("avx2") is not False and isa.get("fma") is not False


def safe_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def target_backend(ir: dict[str, Any]) -> str:
    target = ir.get("target", {}) or {}
    backend = target.get("backend") or target.get("device") or target.get("language")
    if isinstance(backend, str) and backend.lower() in {"cpu", "c"}:
        return "cpu"
    return "cuda"


def should_emit_throughput_kernel(ir: dict[str, Any]) -> bool:
    strategy_ids = set(ir.get("strategy", {}).get("applied_strategy_ids", []) or [])
    strategy_ids.update(
        item.get("strategy_id")
        for item in ir.get("strategy", {}).get("applied_micro_strategies", []) or []
        if isinstance(item, dict)
    )
    vectorization = ir.get("vectorization", {}) or {}
    return (
        bool(ir.get("pipeline", {}).get("double_buffering"))
        or any("DoubleBuffer" in strategy_id for strategy_id in strategy_ids)
        or any("float4" in strategy_id for strategy_id in strategy_ids)
        or any("WarpThreadTile" in strategy_id for strategy_id in strategy_ids)
        or int((vectorization.get("A") or {}).get("vector_width") or 1) >= 4
        or int((vectorization.get("B") or {}).get("vector_width") or 1) >= 4
    )


def extract_launch_config_from_ir(ir: dict[str, Any]) -> dict[str, int]:
    tiling = ir.get("tiling", {}) or {}
    warp_tile = tiling.get("warp_tile", {}) or {}
    return {
        "BM": int(tiling.get("block_m") or 128),
        "BN": int(tiling.get("block_n") or 128),
        "BK": int(tiling.get("block_k") or 8),
        "WM": int(warp_tile.get("warp_m") or 32),
        "WN": int(warp_tile.get("warp_n") or 32),
        "WMITER": int(warp_tile.get("warp_m_iter") or tiling.get("warp_m_iter") or 4),
        "WNITER": int(warp_tile.get("warp_n_iter") or tiling.get("warp_n_iter") or 4),
        "TM": int(tiling.get("thread_m") or 2),
        "TN": int(tiling.get("thread_n") or 2),
    }


def normalize_launch_config_for_repair(config: dict[str, int]) -> dict[str, int]:
    """Keep one logical output tile per CUDA thread for the deterministic repair.

    The repair kernel intentionally avoids a per-output group loop so shared
    loads happen once per K tile. If the selected TM/TN would require more than
    1024 groups per block, conservatively enlarge TN/TM until the block fits.
    """

    result = dict(config)
    while output_group_count(result) > 1024:
        if result["TN"] <= result["TM"]:
            result["TN"] *= 2
        else:
            result["TM"] *= 2
    return result


def output_group_count(config: dict[str, int]) -> int:
    groups_m = math.ceil(config["BM"] / config["TM"])
    groups_n = math.ceil(config["BN"] / config["TN"])
    return groups_m * groups_n


def normalize_throughput_launch_config(config: dict[str, int]) -> dict[str, int]:
    result = dict(config)
    tm = result["TM"]
    tn = result["TN"]
    wm = result["WM"]
    wn = result["WN"]
    for wniter in sorted(divisors(wn), reverse=True):
        if wniter % tn != 0:
            continue
        wmiter = 32 * tm * tn // wniter
        if wmiter > 0 and wmiter <= wm and wmiter % tm == 0 and wm % wmiter == 0:
            result["WMITER"] = wmiter
            result["WNITER"] = wniter
            return result
    result["WMITER"] = min(wm, 16)
    result["WNITER"] = min(wn, 32)
    return result


def divisors(value: int) -> list[int]:
    return [item for item in range(1, value + 1) if value % item == 0]


def ensure_kernel_header_includes_cuda_kernel(header_path: Path) -> None:
    content = header_path.read_text(encoding="utf-8") if header_path.exists() else ""
    if '#include "cuda_kernel.cuh"' in content:
        return
    if not content.strip():
        content = (
            "#pragma once\n"
            "#include <cuda_runtime.h>\n"
            "#include \"cuda_kernel.cuh\"\n"
            "\n"
            "void cuda_gemm(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C);\n"
        )
    else:
        lines = content.splitlines()
        insert_at = 1 if lines and lines[0].strip() == "#pragma once" else 0
        lines.insert(insert_at, '#include "cuda_kernel.cuh"')
        content = "\n".join(lines) + "\n"
    header_path.write_text(content, encoding="utf-8")


def build_repaired_cuda_kernel_cuh(config: dict[str, int]) -> str:
    return f"""#pragma once

#include <cuda_runtime.h>

#ifndef CEIL_DIV
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
#endif

#ifndef OFFSET
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
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
    float *C) {{
    const int tid = threadIdx.x;
    const int thread_count = blockDim.x;
    const int tile_m0 = blockIdx.y * BM;
    const int tile_n0 = blockIdx.x * BN;

    /*
     * INDEX_MAPPING_BEGIN
     */
    const int tile_cols = BN / TN;
    const int tile_rows = BM / TM;
    const int groups_per_tile = tile_rows * tile_cols;
    const int group = tid;
    const int tile_m = group / tile_cols;
    const int tile_n = group % tile_cols;
    const int local_m_base = tile_m * TM;
    const int local_n_base = tile_n * TN;
    const bool group_valid = group < groups_per_tile;
    /*
     * INDEX_MAPPING_END
     */

    /*
     * SHARED_DECL_BEGIN
     */
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    /*
     * SHARED_DECL_END
     */

    /*
     * REGISTER_DECL_BEGIN
     */
    float acc[TM][TN];
    for (int tm = 0; tm < TM; ++tm) {{
        for (int tn = 0; tn < TN; ++tn) {{
            acc[tm][tn] = 0.0f;
        }}
    }}
    /*
     * REGISTER_DECL_END
     */

    /*
     * MAIN_LOOP_BEGIN
     */
    for (int k = 0; k < K; k += BK) {{
        /*
         * GLOBAL_TO_SHARED_LOAD_BEGIN
         */
        for (int load_idx = tid; load_idx < BM * BK; load_idx += thread_count) {{
            const int local_m = load_idx / BK;
            const int local_k = load_idx % BK;
            const int global_m = tile_m0 + local_m;
            const int global_k = k + local_k;
            As[local_m][local_k] =
                (global_m < M && global_k < K) ? A[OFFSET(global_m, global_k, K)] : 0.0f;
        }}
        for (int load_idx = tid; load_idx < BK * BN; load_idx += thread_count) {{
            const int local_k = load_idx / BN;
            const int local_n = load_idx % BN;
            const int global_k = k + local_k;
            const int global_n = tile_n0 + local_n;
            Bs[local_k][local_n] =
                (global_k < K && global_n < N) ? B[OFFSET(global_k, global_n, N)] : 0.0f;
        }}
        /*
         * GLOBAL_TO_SHARED_LOAD_END
         */

        /*
         * SYNC_AFTER_LOAD_BEGIN
         */
        __syncthreads();
        /*
         * SYNC_AFTER_LOAD_END
         */

        /*
         * COMPUTE_INNER_BEGIN
         */
        if (group_valid) {{
            for (int bk = 0; bk < BK; ++bk) {{
                for (int tm = 0; tm < TM; ++tm) {{
                    const int local_m = local_m_base + tm;
                    const float a_val = (local_m < BM) ? As[local_m][bk] : 0.0f;
                    for (int tn = 0; tn < TN; ++tn) {{
                        const int local_n = local_n_base + tn;
                        const float b_val = (local_n < BN) ? Bs[bk][local_n] : 0.0f;
                        acc[tm][tn] += a_val * b_val;
                    }}
                }}
            }}
        }}
        /*
         * COMPUTE_INNER_END
         */
        __syncthreads();
    }}
    /*
     * MAIN_LOOP_END
     */

    /*
     * STORE_BEGIN
     */
    if (group_valid) {{
        for (int tm = 0; tm < TM; ++tm) {{
            const int global_m = tile_m0 + local_m_base + tm;
            for (int tn = 0; tn < TN; ++tn) {{
                const int global_n = tile_n0 + local_n_base + tn;
                if (global_m < M && global_n < N && local_m_base + tm < BM && local_n_base + tn < BN) {{
                    C[OFFSET(global_m, global_n, N)] =
                        alpha * acc[tm][tn] + beta * C[OFFSET(global_m, global_n, N)];
                }}
            }}
        }}
    }}
    /*
     * STORE_END
     */
}}

void cuda_gemm(
    int M,
    int N,
    int K,
    float alpha,
    float *A,
    float *B,
    float beta,
    float *C) {{
    /*
     * LAUNCH_CONFIG_BEGIN
     */
    static const int BM = {config["BM"]};
    static const int BN = {config["BN"]};
    static const int BK = {config["BK"]};
    static const int WM = {config["WM"]};
    static const int WN = {config["WN"]};
    static const int WMITER = {config["WMITER"]};
    static const int WNITER = {config["WNITER"]};
    static const int TM = {config["TM"]};
    static const int TN = {config["TN"]};
    /*
     * LAUNCH_CONFIG_END
     */

    static const int threads_per_block = ((BM / TM) * (BN / TN));
    dim3 threadsPerBlock(threads_per_block);
    dim3 blocksPerGrid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));

    gemm<BM, BN, BK, WM, WN, WMITER, WNITER, TM, TN>
        <<<blocksPerGrid, threadsPerBlock>>>(M, N, K, alpha, A, B, beta, C);
}}
"""


def build_throughput_cuda_kernel_cuh(config: dict[str, int]) -> str:
    return f"""#pragma once

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
    float *C) {{
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
    float results[WM / WMITER * TM][WN / WNITER * TN] = {{0.0f}};
    float regM[TM] = {{0.0f}};
    float regN[TN] = {{0.0f}};
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
    for (int loadIdx = tid * 4; loadIdx < BM * BK; loadIdx += thread_num * 4) {{
        const int load_m = loadIdx / BK;
        const int load_k = loadIdx % BK;
        const float4 tmp = FLOAT4(A[OFFSET(load_m, load_k, K) + A_offset]);
        As[0][load_k][load_m] = tmp.x;
        As[0][load_k + 1][load_m] = tmp.y;
        As[0][load_k + 2][load_m] = tmp.z;
        As[0][load_k + 3][load_m] = tmp.w;
    }}
    for (int loadIdx = tid * 4; loadIdx < BK * BN; loadIdx += thread_num * 4) {{
        const int load_k = loadIdx / BN;
        const int load_n = loadIdx % BN;
        FLOAT4(Bs[0][load_k][load_n]) = FLOAT4(B[OFFSET(load_k, load_n, N) + B_offset]);
    }}
    /*
     * GLOBAL_TO_SHARED_LOAD_END
     */

    /*
     * MAIN_LOOP_BEGIN
     */
    for (int bkIdx = 1; bkIdx < k_tiles; ++bkIdx) {{
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
        for (int k = 0; k < BK; ++k) {{
            #pragma unroll
            for (int wm = 0; wm < WM / WMITER; ++wm) {{
                #pragma unroll
                for (int wn = 0; wn < WN / WNITER; ++wn) {{
                    #pragma unroll
                    for (int i = 0; i < TM; ++i) {{
                        regM[i] = As[comp_flag][k][Wrow * WM + wm * WMITER + Trow * TM + i];
                    }}
                    #pragma unroll
                    for (int j = 0; j < TN; ++j) {{
                        regN[j] = Bs[comp_flag][k][Wcol * WN + wn * WNITER + Tcol * TN + j];
                    }}
                    #pragma unroll
                    for (int i = 0; i < TM; ++i) {{
                        #pragma unroll
                        for (int j = 0; j < TN; ++j) {{
                            results[wm * TM + i][wn * TN + j] += regM[i] * regN[j];
                        }}
                    }}
                }}
            }}
        }}
        /*
         * COMPUTE_INNER_END
         */

        /*
         * GLOBAL_TO_SHARED_LOAD_BEGIN
         */
        for (int loadIdx = tid * 4; loadIdx < BM * BK; loadIdx += thread_num * 4) {{
            const int load_m = loadIdx / BK;
            const int load_k = loadIdx % BK;
            const float4 tmp = FLOAT4(A[OFFSET(load_m, load_k + bkIdx * BK, K) + A_offset]);
            As[mem_flag][load_k][load_m] = tmp.x;
            As[mem_flag][load_k + 1][load_m] = tmp.y;
            As[mem_flag][load_k + 2][load_m] = tmp.z;
            As[mem_flag][load_k + 3][load_m] = tmp.w;
        }}
        for (int loadIdx = tid * 4; loadIdx < BK * BN; loadIdx += thread_num * 4) {{
            const int load_k = loadIdx / BN;
            const int load_n = loadIdx % BN;
            FLOAT4(Bs[mem_flag][load_k][load_n]) =
                FLOAT4(B[OFFSET(load_k + bkIdx * BK, load_n, N) + B_offset]);
        }}
        /*
         * GLOBAL_TO_SHARED_LOAD_END
         */
    }}

    __syncthreads();
    const int comp_flag = (k_tiles - 1) & 1;
    #pragma unroll
    for (int k = 0; k < BK; ++k) {{
        #pragma unroll
        for (int wm = 0; wm < WM / WMITER; ++wm) {{
            #pragma unroll
            for (int wn = 0; wn < WN / WNITER; ++wn) {{
                #pragma unroll
                for (int i = 0; i < TM; ++i) {{
                    regM[i] = As[comp_flag][k][Wrow * WM + wm * WMITER + Trow * TM + i];
                }}
                #pragma unroll
                for (int j = 0; j < TN; ++j) {{
                    regN[j] = Bs[comp_flag][k][Wcol * WN + wn * WNITER + Tcol * TN + j];
                }}
                #pragma unroll
                for (int i = 0; i < TM; ++i) {{
                    #pragma unroll
                    for (int j = 0; j < TN; ++j) {{
                        results[wm * TM + i][wn * TN + j] += regM[i] * regN[j];
                    }}
                }}
            }}
        }}
    }}
    /*
     * MAIN_LOOP_END
     */

    /*
     * STORE_BEGIN
     */
    #pragma unroll
    for (int wm = 0; wm < WM / WMITER; ++wm) {{
        #pragma unroll
        for (int wn = 0; wn < WN / WNITER; ++wn) {{
            #pragma unroll
            for (int m = 0; m < TM; ++m) {{
                #pragma unroll
                for (int n = 0; n < TN; ++n) {{
                    const int c_index = OFFSET(
                        Wrow * WM + wm * WMITER + Trow * TM + m,
                        Wcol * WN + wn * WNITER + Tcol * TN + n,
                        N) + C_offset;
                    C[c_index] = alpha * results[m + wm * TM][n + wn * TN] + beta * C[c_index];
                }}
            }}
        }}
    }}
    /*
     * STORE_END
     */
}}

void cuda_gemm(
    int M,
    int N,
    int K,
    float alpha,
    float *A,
    float *B,
    float beta,
    float *C) {{
    /*
     * LAUNCH_CONFIG_BEGIN
     */
    static const int BM = {config["BM"]};
    static const int BN = {config["BN"]};
    static const int BK = {config["BK"]};
    static const int WM = {config["WM"]};
    static const int WN = {config["WN"]};
    static const int WMITER = {config["WMITER"]};
    static const int WNITER = {config["WNITER"]};
    static const int TM = {config["TM"]};
    static const int TN = {config["TN"]};
    /*
     * LAUNCH_CONFIG_END
     */

    dim3 threadsPerBlock((BM * BN) / (WM * WN) * 32);
    dim3 blocksPerGrid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    gemm<BM, BN, BK, WM, WN, WMITER, WNITER, TM, TN>
        <<<blocksPerGrid, threadsPerBlock>>>(M, N, K, alpha, A, B, beta, C);
}}
"""
