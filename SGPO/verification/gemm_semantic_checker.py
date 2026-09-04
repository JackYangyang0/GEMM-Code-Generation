from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def check_gemm_semantic_obligations(chain_dir: Path, ir: dict[str, Any] | None = None) -> dict[str, Any]:
    kernel_path = chain_dir / "cuda_kernel.cuh"
    if not kernel_path.exists():
        return {
            "accepted": False,
            "results": [
                make_result(
                    "GEMM_KERNEL_FILE_EXISTS",
                    "fail",
                    "cuda_kernel.cuh must exist before semantic GEMM checks can run.",
                    "Compile.MissingKernelSource",
                    "restore cuda_kernel.cuh or regenerate the chain code bundle",
                )
            ],
        }

    content = kernel_path.read_text(encoding="utf-8")
    host_path = chain_dir / "main.cpp"
    host_content = host_path.read_text(encoding="utf-8") if host_path.exists() else ""
    code = strip_cpp_comments(content)
    host_code = strip_cpp_comments(host_content)
    launch_config = extract_launch_config(code)
    tm = int_value(launch_config.get("TM") or get_path(ir or {}, "tiling.thread_m"))
    tn = int_value(launch_config.get("TN") or get_path(ir or {}, "tiling.thread_n"))
    bm = int_value(launch_config.get("BM") or get_path(ir or {}, "tiling.block_m"))
    bn = int_value(launch_config.get("BN") or get_path(ir or {}, "tiling.block_n"))
    bk = int_value(launch_config.get("BK") or get_path(ir or {}, "tiling.block_k"))

    results = [
        check_cuda_compile_hazard_patterns(code, launch_config),
        check_no_scalar_fallback_mixed_with_optimized_regions(content, code),
        check_warp_lane_fragment_coverage(code, launch_config, ir or {}),
        check_thread_tile_mapping_covers_block(code, bm, bn, tm, tn),
        check_thread_tile_store_coverage(content, code, tm, tn),
        check_register_accumulator_coverage(code, tm, tn),
        check_k_loop_updates_A_and_B_tiles(code),
        check_fast_k_loop_divisibility(code, launch_config, ir or {}),
        check_vectorized_fast_path_guards(code, host_code, launch_config, ir or {}),
        check_A_register_initialization(code, bk),
        check_shared_memory_is_cooperatively_loaded(code, bm, bn, bk),
        check_shared_load_not_nested_under_output_group_loop(code),
        check_shared_memory_is_used_by_compute(code),
    ]

    return {
        "accepted": all(item["status"] == "pass" for item in results),
        "launch_config": {key: value for key, value in launch_config.items() if key in {"BM", "BN", "BK", "TM", "TN", "WM", "WN", "WMITER", "WNITER"}},
        "results": results,
    }


def check_no_scalar_fallback_mixed_with_optimized_regions(content: str, code: str) -> dict[str, Any]:
    shared_pos = content.find("SHARED_DECL_BEGIN")
    fallback_patterns = [
        r"\bsgpo_acc\b",
        r"for\s*\([^)]*\bsgpo_k\b[^)]*<\s*K",
        r"for\s*\([^)]*\bsgpo_tile_idx\b[^)]*<\s*BM\s*\*\s*BN",
    ]
    if shared_pos >= 0:
        pre_shared = strip_cpp_comments(content[:shared_pos])
        has_fallback = any(re.search(pattern, pre_shared) for pattern in fallback_patterns)
    else:
        has_fallback = any(re.search(pattern, code) for pattern in fallback_patterns) and "__shared__" in code
    if has_fallback:
        return make_result(
            "GEMM_NO_SCALAR_FALLBACK_MIXED_WITH_OPTIMIZED_KERNEL",
            "fail",
            "optimized chain code must not keep the scalar fallback path before the tiled/shared-memory path.",
            "GEMM.Semantic.FallbackPathOverlapsOptimizedPath",
            "remove the scalar fallback body once real tiled compute/store code is generated",
        )
    return make_result(
        "GEMM_NO_SCALAR_FALLBACK_MIXED_WITH_OPTIMIZED_KERNEL",
        "pass",
        "no scalar fallback path is mixed into the optimized kernel body.",
    )


def check_thread_tile_store_coverage(content: str, code: str, tm: int | None, tn: int | None) -> dict[str, Any]:
    if not tm or not tn or tm * tn <= 1:
        return make_result(
            "GEMM_THREAD_TILE_STORE_COVERS_TM_TN",
            "pass",
            "TM/TN are scalar or unavailable; store coverage check is deferred.",
        )
    store_region = region_between_markers(content, "STORE_BEGIN", "STORE_END") or tail_from_last_c_store(code)
    store_code = strip_cpp_comments(store_region)
    c_store_count = len(re.findall(r"\bC\s*\[[^\]]+\]\s*=", store_code))
    has_tm_loop = has_loop_bound(store_code, "TM") or has_loop_bound(store_code, str(tm))
    has_tn_loop = has_loop_bound(store_code, "TN") or has_loop_bound(store_code, str(tn))
    has_2d_acc_store = bool(re.search(r"acc\s*\[[^\]]+\]\s*\[[^\]]+\]", store_code))
    has_results_store = bool(re.search(r"results\s*\[[^\]]+\]\s*\[[^\]]+\]", store_code))
    if c_store_count >= tm * tn or (has_tm_loop and has_tn_loop and (has_2d_acc_store or has_results_store)):
        return make_result(
            "GEMM_THREAD_TILE_STORE_COVERS_TM_TN",
            "pass",
            f"store region covers TM*TN outputs for TM={tm}, TN={tn}.",
        )
    return make_result(
        "GEMM_THREAD_TILE_STORE_COVERS_TM_TN",
        "fail",
        f"TM={tm}, TN={tn}, but the store region does not cover all TM*TN C elements per thread.",
        "GEMM.Semantic.ThreadTileStoreIncomplete",
        "add nested TM/TN store loops and write every accumulator element to its corresponding C coordinate",
        {
            "tm": tm,
            "tn": tn,
            "c_store_count": c_store_count,
            "has_tm_loop": has_tm_loop,
            "has_tn_loop": has_tn_loop,
            "has_results_store": has_results_store,
        },
    )


def check_thread_tile_mapping_covers_block(code: str, bm: int | None, bn: int | None, tm: int | None, tn: int | None) -> dict[str, Any]:
    if not bm or not bn or not tm or not tn or tm * tn <= 1:
        return make_result(
            "GEMM_THREAD_TILE_MAPPING_COVERS_BLOCK",
            "pass",
            "TM/TN are scalar or unavailable; mapping coverage check is deferred.",
        )
    has_bad_linear_mapping = bool(
        re.search(r"\bbase_element\s*=\s*group\s*\*\s*TM\s*\*\s*TN\s*;", code)
        and re.search(r"\blocal_m_base\s*=\s*base_element\s*/\s*BN\s*;", code)
        and re.search(r"\blocal_n_base\s*=\s*base_element\s*%\s*BN\s*;", code)
    )
    has_tile_cols = bool(re.search(r"\b(?:tile_cols|groups_n)\s*=\s*BN\s*/\s*TN\s*;", code))
    has_2d_group_mapping = bool(
        (
            re.search(r"\blocal_m_base\s*=\s*\(?\s*group\s*/\s*(?:tile_cols|groups_n|\(BN\s*/\s*TN\))\s*\)?\s*\*\s*TM\s*;", code)
            or (
                re.search(r"\btile_m\s*=\s*group\s*/\s*(?:tile_cols|groups_n|\(BN\s*/\s*TN\))\s*;", code)
                and re.search(r"\blocal_m_base\s*=\s*tile_m\s*\*\s*TM\s*;", code)
            )
        )
        and (
            re.search(r"\blocal_n_base\s*=\s*\(?\s*group\s*%\s*(?:tile_cols|groups_n|\(BN\s*/\s*TN\))\s*\)?\s*\*\s*TN\s*;", code)
            or (
                re.search(r"\btile_n\s*=\s*group\s*%\s*(?:tile_cols|groups_n|\(BN\s*/\s*TN\))\s*;", code)
                and re.search(r"\blocal_n_base\s*=\s*tile_n\s*\*\s*TN\s*;", code)
            )
        )
    )
    if has_2d_group_mapping or (has_tile_cols and not has_bad_linear_mapping):
        return make_result(
            "GEMM_THREAD_TILE_MAPPING_COVERS_BLOCK",
            "pass",
            "thread-tile group mapping uses 2D tile coordinates and can cover BM x BN without holes.",
        )
    has_warp_lane_mapping = all(token in code for token in ["Wrow", "Wcol", "Trow", "Tcol", "wid", "lane"])
    if has_warp_lane_mapping:
        return make_result(
            "GEMM_THREAD_TILE_MAPPING_COVERS_BLOCK",
            "pass",
            "warp/lane mapping is used; detailed coverage is checked by GEMM_WARP_LANE_FRAGMENT_COVERS_WARP_TILE.",
        )
    return make_result(
        "GEMM_THREAD_TILE_MAPPING_COVERS_BLOCK",
        "fail",
        "thread-tile group mapping can leave holes in C because consecutive groups jump by TM*TN in linear element space.",
        "GEMM.Semantic.ThreadTileMappingHasHoles",
        "map group to 2D tile coordinates: tile_m = group / (BN / TN), tile_n = group % (BN / TN), local_m_base = tile_m * TM, local_n_base = tile_n * TN",
        {
            "bm": bm,
            "bn": bn,
            "tm": tm,
            "tn": tn,
            "has_bad_linear_mapping": has_bad_linear_mapping,
            "has_2d_group_mapping": has_2d_group_mapping,
        },
    )


def check_warp_lane_fragment_coverage(code: str, launch_config: dict[str, int], ir: dict[str, Any]) -> dict[str, Any]:
    if "wid" not in code or "lane" not in code or "Trow" not in code or "Tcol" not in code:
        return make_result(
            "GEMM_WARP_LANE_FRAGMENT_COVERS_WARP_TILE",
            "pass",
            "warp/lane fragment mapping is not used.",
        )
    wm = int_value(launch_config.get("WM") or get_path(ir, "tiling.warp_tile.warp_m"))
    wn = int_value(launch_config.get("WN") or get_path(ir, "tiling.warp_tile.warp_n"))
    wmiter = int_value(launch_config.get("WMITER") or get_path(ir, "tiling.warp_tile.warp_m_iter"))
    wniter = int_value(launch_config.get("WNITER") or get_path(ir, "tiling.warp_tile.warp_n_iter"))
    tm = int_value(launch_config.get("TM") or get_path(ir, "tiling.thread_m"))
    tn = int_value(launch_config.get("TN") or get_path(ir, "tiling.thread_n"))
    warp_size = int_value(get_path(ir, "hardware.warp_size")) or 32
    if not all([wm, wn, wmiter, wniter, tm, tn]):
        return make_result(
            "GEMM_WARP_LANE_FRAGMENT_COVERS_WARP_TILE",
            "pass",
            "warp/lane fragment parameters are unavailable; check deferred.",
        )

    fragment_lane_count = (wmiter // tm) * (wniter // tn) if tm and tn else None
    divides_tile = wm % wmiter == 0 and wn % wniter == 0 if wmiter and wniter else False
    has_expected_trow = bool(re.search(r"\bTrow\s*=\s*lane\s*/\s*\(?\s*WNITER\s*/\s*TN\s*\)?", code))
    has_expected_tcol = bool(re.search(r"\bTcol\s*=\s*lane\s*%\s*\(?\s*WNITER\s*/\s*TN\s*\)?", code))
    if fragment_lane_count == warp_size and divides_tile and has_expected_trow and has_expected_tcol:
        return make_result(
            "GEMM_WARP_LANE_FRAGMENT_COVERS_WARP_TILE",
            "pass",
            "warp/lane fragment mapping covers each warp tile without holes.",
        )
    return make_result(
        "GEMM_WARP_LANE_FRAGMENT_COVERS_WARP_TILE",
        "fail",
        "warp/lane fragment mapping can leave holes because WMITER/WNITER/TM/TN do not map exactly one warp of lanes.",
        "GEMM.Semantic.WarpLaneFragmentCoverageIncomplete",
        "choose WMITER/WNITER so (WMITER / TM) * (WNITER / TN) == warp_size and WM % WMITER == 0 and WN % WNITER == 0; for 4x4 thread tiles prefer WMITER=16 and WNITER=32",
        {
            "wm": wm,
            "wn": wn,
            "wmiter": wmiter,
            "wniter": wniter,
            "tm": tm,
            "tn": tn,
            "warp_size": warp_size,
            "fragment_lane_count": fragment_lane_count,
            "divides_tile": divides_tile,
            "has_expected_trow": has_expected_trow,
            "has_expected_tcol": has_expected_tcol,
        },
    )


def check_register_accumulator_coverage(code: str, tm: int | None, tn: int | None) -> dict[str, Any]:
    if not tm or not tn or tm * tn <= 1:
        return make_result("GEMM_REGISTER_ACCUMULATOR_COVERS_TM_TN", "pass", "scalar accumulator coverage is sufficient.")
    has_2d_acc = bool(re.search(r"\bfloat\s+acc\s*\[\s*TM\s*\]\s*\[\s*TN\s*\]", code))
    has_init_loops = has_loop_bound(code, "TM") and has_loop_bound(code, "TN") and bool(re.search(r"acc\s*\[[^\]]+\]\s*\[[^\]]+\]\s*=\s*0\.0f", code))
    has_compute_loops = has_loop_bound(code, "TM") and has_loop_bound(code, "TN") and bool(re.search(r"acc\s*\[[^\]]+\]\s*\[[^\]]+\]\s*\+=", code))
    if has_2d_acc and has_init_loops and has_compute_loops:
        return make_result("GEMM_REGISTER_ACCUMULATOR_COVERS_TM_TN", "pass", "2D accumulator is initialized and accumulated over TM/TN.")
    has_results_tile = bool(re.search(r"\bfloat\s+results\s*\[[^\]]+\]\s*\[[^\]]+\]\s*=\s*\{\s*0\.0f\s*\}", code))
    has_results_compute = bool(re.search(r"\bresults\s*\[[^\]]+\]\s*\[[^\]]+\]\s*\+=", code))
    has_register_fragments = "regM" in code and "regN" in code
    if has_results_tile and has_results_compute and has_register_fragments:
        return make_result(
            "GEMM_REGISTER_ACCUMULATOR_COVERS_TM_TN",
            "pass",
            "warp/lane results tile is initialized and accumulated through regM/regN fragments.",
        )
    return make_result(
        "GEMM_REGISTER_ACCUMULATOR_COVERS_TM_TN",
        "fail",
        "register accumulator shape exists, but compute does not cover every TM/TN accumulator element.",
        "GEMM.Semantic.RegisterTileComputeIncomplete",
        "compute all acc[tm][tn] values inside nested TM/TN loops or explicit unrolled equivalents",
        {"has_2d_acc": has_2d_acc, "has_init_loops": has_init_loops, "has_compute_loops": has_compute_loops},
    )


def check_k_loop_updates_A_and_B_tiles(code: str) -> dict[str, Any]:
    k_loop = extract_first_loop_body(code, r"for\s*\(\s*int\s+k\s*=\s*0\s*;\s*k\s*<\s*K\s*;\s*k\s*\+=\s*BK\s*\)")
    if not k_loop and re.search(r"for\s*\(\s*int\s+bkIdx\s*=\s*1\s*;\s*bkIdx\s*<\s*k_tiles\s*;", code):
        has_ping_pong_load = bool(
            "comp_flag" in code
            and "mem_flag" in code
            and re.search(r"\bbkIdx\s*\*\s*BK", code)
            and re.search(r"\bAs\s*\[\s*mem_flag\s*\]", code)
            and re.search(r"\bBs\s*\[\s*mem_flag\s*\]", code)
        )
        if has_ping_pong_load:
            return make_result(
                "GEMM_K_LOOP_UPDATES_A_AND_B_TILES",
                "pass",
                "double-buffered bkIdx loop updates both A and B tile data.",
            )
    if not k_loop:
        return make_result(
            "GEMM_K_LOOP_UPDATES_A_AND_B_TILES",
            "fail",
            "optimized GEMM must have a K loop advancing by BK.",
            "GEMM.Semantic.KLoopMissing",
            "add a for (int k = 0; k < K; k += BK) loop around shared loads and compute",
        )
    k_dependent_load = bool(
        re.search(r"\bglobal_k\s*=\s*k\s*\+", k_loop)
        or re.search(r"OFFSET\s*\([^)]*\b(?:k|global_k|local_k|k_inner)\b", k_loop)
    )
    updates_a = bool(re.search(r"\b(?:A_reg|As0|As1|As|shared_A|sA)\b[^=;]*=", k_loop) and k_dependent_load)
    updates_b = bool(re.search(r"\b(?:Bs0|Bs1|Bs|shared_B|sB)\b[^=;]*=", k_loop) and k_dependent_load)
    if updates_a and updates_b:
        return make_result("GEMM_K_LOOP_UPDATES_A_AND_B_TILES", "pass", "K loop updates both A and B tile data.")
    return make_result(
        "GEMM_K_LOOP_UPDATES_A_AND_B_TILES",
        "fail",
        "K loop does not reload both A and B tiles for each BK slice.",
        "GEMM.Semantic.KLoopDataflowIncomplete",
        "move cooperative A/B tile loads into the K loop and index global memory with k + local_k",
        {"updates_a": updates_a, "updates_b": updates_b},
    )


def check_fast_k_loop_divisibility(code: str, launch_config: dict[str, int], ir: dict[str, Any]) -> dict[str, Any]:
    uses_floor_k_tiles = bool(re.search(r"\bk_tiles\s*=\s*K\s*/\s*BK\s*;", code))
    if not uses_floor_k_tiles:
        return make_result(
            "GEMM_FAST_K_LOOP_HAS_DIVISIBILITY_PROOF",
            "pass",
            "K loop does not use floor-divided k_tiles.",
        )
    k = int_value(get_path(ir, "problem.K"))
    bk = int_value(launch_config.get("BK") or get_path(ir, "tiling.block_k"))
    static_divisible = bool(k and bk and k % bk == 0)
    assumes_static = get_path(ir, "safety.boundary_policy") in {"static_divisible_no_guard", "StaticDivisibleNoGuard"}
    if static_divisible or assumes_static or "CEIL_DIV(K, BK)" in code:
        return make_result(
            "GEMM_FAST_K_LOOP_HAS_DIVISIBILITY_PROOF",
            "pass",
            "floor-divided k_tiles is backed by static divisibility or an explicit fast-path policy.",
        )
    return make_result(
        "GEMM_FAST_K_LOOP_HAS_DIVISIBILITY_PROOF",
        "fail",
        "K / BK fast path can skip tail K elements when K is not divisible by BK.",
        "GEMM.Semantic.KTileTailDropped",
        "use CEIL_DIV(K, BK) with guarded loads, or require and record static_divisible_no_guard only when K % BK == 0",
        {"K": k, "BK": bk, "uses_floor_k_tiles": uses_floor_k_tiles, "static_divisible": static_divisible},
    )


def check_vectorized_fast_path_guards(code: str, host_code: str, launch_config: dict[str, int], ir: dict[str, Any]) -> dict[str, Any]:
    uses_float4 = "FLOAT4" in code or "float4" in code
    has_no_guards = "if (" not in code
    if not uses_float4 or not has_no_guards:
        return make_result(
            "GEMM_VECTORIZED_FAST_PATH_HAS_PROOF",
            "pass",
            "vectorized loads/stores either are not used or have runtime guards.",
        )
    m = int_value(get_path(ir, "problem.M"))
    n = int_value(get_path(ir, "problem.N"))
    k = int_value(get_path(ir, "problem.K"))
    bm = int_value(launch_config.get("BM") or get_path(ir, "tiling.block_m"))
    bn = int_value(launch_config.get("BN") or get_path(ir, "tiling.block_n"))
    bk = int_value(launch_config.get("BK") or get_path(ir, "tiling.block_k"))
    static_shape_ok = bool(m and n and k and bm and bn and bk and m % bm == 0 and n % bn == 0 and k % bk == 0 and k % 4 == 0 and n % 4 == 0)
    cuda_malloc_backed = all(token in host_code for token in ["cudaMalloc(&d_A", "cudaMalloc(&d_B", "cudaMalloc(&d_C"])
    alignment_proven = bool(
        get_path(ir, "vectorization.A.alignment_proven")
        and get_path(ir, "vectorization.B.alignment_proven")
        and (get_path(ir, "vectorization.C.alignment_proven") or "C[" in code)
    )
    static_policy = get_path(ir, "safety.boundary_policy") in {"static_divisible_no_guard", "StaticDivisibleNoGuard"}
    if static_shape_ok and (alignment_proven or static_policy or cuda_malloc_backed):
        return make_result(
            "GEMM_VECTORIZED_FAST_PATH_HAS_PROOF",
            "pass",
            "unguarded float4 fast path has static shape divisibility and alignment proof/policy.",
        )
    return make_result(
        "GEMM_VECTORIZED_FAST_PATH_HAS_PROOF",
        "fail",
        "unguarded float4 fast path needs static divisibility and alignment proof; otherwise it can read/write invalid elements.",
        "GEMM.Semantic.VectorizedFastPathWithoutProof",
        "add boundary/alignment guards, add a scalar tail path, or require static_divisible_no_guard with A/B/C alignment_proven and M/N/K divisibility by tile/vector width",
        {
            "M": m,
            "N": n,
            "K": k,
            "BM": bm,
            "BN": bn,
            "BK": bk,
            "static_shape_ok": static_shape_ok,
            "alignment_proven": alignment_proven,
            "cuda_malloc_backed": cuda_malloc_backed,
            "static_policy": static_policy,
        },
    )


def check_A_register_initialization(code: str, bk: int | None) -> dict[str, Any]:
    if "A_reg" not in code:
        return make_result("GEMM_A_REGISTER_FULLY_INITIALIZED", "pass", "no A_reg array is used.")
    uses_indexed_a_reg = bool(re.search(r"\bA_reg\s*\[\s*bk\s*\]", code))
    has_full_init_loop = bool(
        re.search(r"for\s*\([^)]*<\s*BK[^)]*\)\s*\{[^{}]*A_reg\s*\[[^\]]+\]\s*=", code, flags=re.DOTALL)
    )
    has_value_init = bool(re.search(r"\bfloat\s+A_reg\s*\[\s*BK\s*\]\s*=\s*\{\s*\}", code))
    if not uses_indexed_a_reg or has_full_init_loop or has_value_init:
        return make_result("GEMM_A_REGISTER_FULLY_INITIALIZED", "pass", "A_reg is fully initialized before indexed use.")
    return make_result(
        "GEMM_A_REGISTER_FULLY_INITIALIZED",
        "fail",
        "A_reg[bk] is consumed across BK, but the code does not initialize every A_reg element.",
        "GEMM.Semantic.UninitializedRegisterTile",
        "initialize every A_reg[bk] in a BK loop before compute, or replace A_reg use with shared A tile loads",
        {"bk": bk, "uses_indexed_a_reg": uses_indexed_a_reg, "has_full_init_loop": has_full_init_loop},
    )


def check_shared_memory_is_cooperatively_loaded(code: str, bm: int | None, bn: int | None, bk: int | None) -> dict[str, Any]:
    if "__shared__" not in code:
        return make_result("GEMM_SHARED_MEMORY_COOPERATIVE_LOAD", "pass", "shared memory is not used.")
    shared_load_code = region_from_first_shared_store(code)
    has_a_load = bool(re.search(r"\b(?:As0|As1|As|shared_A|sA)\s*\[[^\]]+\]", shared_load_code) and re.search(r"\bA\s*\[", shared_load_code))
    has_b_load = bool(re.search(r"\b(?:Bs0|Bs1|Bs|shared_B|sB)\s*\[[^\]]+\]", shared_load_code) and re.search(r"\bB\s*\[", shared_load_code))
    has_thread_strided_loop = bool(
        re.search(r"for\s*\([^)]*=\s*(?:tid|sgpo_linear_tid)[^;]*;[^;]*<[^;]*;[^)]*\+=\s*(?:blockDim\.x|sgpo_thread_count|thread_count)", code)
    )
    suspicious_per_thread_whole_tile = bool(re.search(r"for\s*\([^)]*<\s*BN[^)]*\)\s*\{[^{}]*for\s*\([^)]*<\s*BK", code, flags=re.DOTALL))
    has_vector_thread_mapped_load = bool(
        "FLOAT4" in code
        and "load_a_smem_m" in code
        and "load_a_smem_k" in code
        and "load_b_smem_k" in code
        and "load_b_smem_n" in code
        and "As[" in code
        and "Bs[" in code
        and "A[OFFSET" in code
        and "B[OFFSET" in code
    )
    if has_a_load and has_b_load and (has_thread_strided_loop or has_vector_thread_mapped_load) and not suspicious_per_thread_whole_tile:
        return make_result("GEMM_SHARED_MEMORY_COOPERATIVE_LOAD", "pass", "A/B shared tiles are cooperatively loaded.")
    return make_result(
        "GEMM_SHARED_MEMORY_COOPERATIVE_LOAD",
        "fail",
        "shared-memory GEMM must cooperatively load both A and B tiles instead of repeating whole-tile loads per thread.",
        "GEMM.Semantic.SharedLoadNotCooperative",
        "use a thread-strided loop over BM*BK and BK*BN so all threads cooperatively populate shared A/B tiles",
        {
            "bm": bm,
            "bn": bn,
            "bk": bk,
            "has_a_load": has_a_load,
            "has_b_load": has_b_load,
            "has_thread_strided_loop": has_thread_strided_loop,
            "has_vector_thread_mapped_load": has_vector_thread_mapped_load,
            "suspicious_per_thread_whole_tile": suspicious_per_thread_whole_tile,
        },
    )


def check_shared_load_not_nested_under_output_group_loop(code: str) -> dict[str, Any]:
    if "__shared__" not in code:
        return make_result("GEMM_SHARED_LOAD_OUTSIDE_OUTPUT_GROUP_LOOP", "pass", "shared memory is not used.")
    group_loop = extract_first_loop_body(code, r"for\s*\(\s*int\s+group\s*=\s*tid\s*;")
    if not group_loop:
        return make_result("GEMM_SHARED_LOAD_OUTSIDE_OUTPUT_GROUP_LOOP", "pass", "no output group loop is present.")
    has_shared_load_inside_group = bool(
        re.search(r"for\s*\(\s*int\s+load_idx\b", group_loop)
        and re.search(r"\b(?:As0|As1|As|Bs0|Bs1|Bs|shared_A|shared_B|sA|sB)\s*\[[^\]]+\]\s*=", group_loop)
    )
    has_sync_inside_group = "__syncthreads()" in group_loop
    if not has_shared_load_inside_group and not has_sync_inside_group:
        return make_result(
            "GEMM_SHARED_LOAD_OUTSIDE_OUTPUT_GROUP_LOOP",
            "pass",
            "shared loads and block synchronization are not nested under the per-output group loop.",
        )
    return make_result(
        "GEMM_SHARED_LOAD_OUTSIDE_OUTPUT_GROUP_LOOP",
        "fail",
        "shared tile loads or __syncthreads are nested under the per-output group loop, causing repeated tile loads and excessive synchronization.",
        "GEMM.Performance.RepeatedSharedLoadPerOutputGroup",
        "move cooperative shared A/B tile loads and __syncthreads outside the per-output group loop; keep K-tile load once per block, then compute each thread's assigned output groups from that shared tile",
        {
            "has_shared_load_inside_group": has_shared_load_inside_group,
            "has_sync_inside_group": has_sync_inside_group,
        },
    )


def check_shared_memory_is_used_by_compute(code: str) -> dict[str, Any]:
    if "__shared__" not in code:
        return make_result("GEMM_SHARED_MEMORY_USED_IN_COMPUTE", "pass", "shared memory is not used.")
    compute_regions = "\n".join(re.findall(r"COMPUTE_INNER_BEGIN(.*?)COMPUTE_INNER_END", code, flags=re.DOTALL))
    if not compute_regions:
        compute_regions = code
    uses_a_shared = bool(re.search(r"\b(?:As0|As1|As|shared_A|sA)\s*\[", compute_regions))
    uses_b_shared = bool(re.search(r"\b(?:Bs0|Bs1|Bs|shared_B|sB)\s*\[", compute_regions))
    if uses_a_shared and uses_b_shared:
        return make_result("GEMM_SHARED_MEMORY_USED_IN_COMPUTE", "pass", "compute consumes both shared A and shared B tiles.")
    return make_result(
        "GEMM_SHARED_MEMORY_USED_IN_COMPUTE",
        "fail",
        "shared A/B buffers are declared, but compute does not consume both shared tiles.",
        "GEMM.Semantic.SharedTileNotUsedInCompute",
        "compute acc[tm][tn] from shared A and shared B values loaded for the current K tile",
        {"uses_a_shared": uses_a_shared, "uses_b_shared": uses_b_shared},
    )


def check_cuda_compile_hazard_patterns(code: str, launch_config: dict[str, int]) -> dict[str, Any]:
    hazards: list[dict[str, Any]] = []
    single_scope_names = [
        "tid",
        "lane",
        "Wrow",
        "Wcol",
        "tile_m0",
        "tile_n0",
        "lane_m",
        "lane_n",
        "warp_m",
        "warp_n",
    ]
    for name in single_scope_names:
        declarations = re.findall(r"\b(?:const\s+)?int\s+" + re.escape(name) + r"\b", code)
        if len(declarations) > 1:
            hazards.append({"kind": "redeclaration", "symbol": name, "count": len(declarations)})

    wm = int_value(launch_config.get("WM"))
    wn = int_value(launch_config.get("WN"))
    wmiter = int_value(launch_config.get("WMITER"))
    wniter = int_value(launch_config.get("WNITER"))
    tm = int_value(launch_config.get("TM"))
    tn = int_value(launch_config.get("TN"))
    uses_warp_fragment_extents = "WM / WMITER" in code or "WN / WNITER" in code
    if uses_warp_fragment_extents and all(value for value in [wm, wn, wmiter, wniter, tm, tn]):
        if wmiter > wm or wniter > wn or wm % wmiter != 0 or wn % wniter != 0:
            hazards.append(
                {
                    "kind": "invalid_warp_fragment_extent",
                    "WM": wm,
                    "WN": wn,
                    "WMITER": wmiter,
                    "WNITER": wniter,
                }
            )
        elif (wmiter // tm) * (wniter // tn) != 32:
            hazards.append(
                {
                    "kind": "warp_fragment_does_not_cover_32_lanes",
                    "WMITER": wmiter,
                    "WNITER": wniter,
                    "TM": tm,
                    "TN": tn,
                    "lane_slots": (wmiter // tm) * (wniter // tn),
                }
            )

    for alias in ["As_ptr", "Bs_ptr", "As_use", "Bs_use"]:
        if re.search(r"\bfloat\s+" + alias + r"\s*=\s*(?:As|Bs)\s*\[", code) and re.search(r"\b" + alias + r"\s*\[", code):
            hazards.append({"kind": "shared_memory_rank_mismatch", "symbol": alias})

    required_declarations = {
        "A_reg": r"\b(?:float|auto)\s+A_reg\b",
        "B_reg": r"\b(?:float|auto)\s+B_reg\b",
        "regM": r"\bfloat\s+regM\s*\[",
        "regN": r"\bfloat\s+regN\s*\[",
        "tmp": r"\b(?:float4|auto)\s+tmp\b",
        "local_m_prefetch": r"\b(?:const\s+)?int\s+local_m_prefetch\b",
        "local_n_prefetch": r"\b(?:const\s+)?int\s+local_n_prefetch\b",
        "local_k_prefetch": r"\b(?:const\s+)?int\s+local_k_prefetch\b",
        "bkIdx": r"\b(?:for\s*\([^)]*\bbkIdx\b|(?:const\s+)?int\s+bkIdx\b)",
        "j": r"\bfor\s*\([^)]*(?:int\s+)?j\b",
    }
    for symbol, declaration_pattern in required_declarations.items():
        if re.search(r"\b" + symbol + r"\b", code) and not re.search(declaration_pattern, code):
            hazards.append({"kind": "undeclared_symbol", "symbol": symbol})

    if not hazards:
        return make_result("GEMM_CUDA_COMPILE_HAZARDS", "pass", "no known CUDA compile-hazard pattern was detected.")
    return make_result(
        "GEMM_CUDA_COMPILE_HAZARDS",
        "fail",
        "generated CUDA code contains known patterns that frequently become nvcc compile errors.",
        "Compile.CUDAStaticHazard",
        "regenerate or repair the kernel using canonical scoped index variables, valid warp fragment extents, declared register symbols, and shared-memory aliases whose rank matches their uses",
        {"hazards": hazards},
    )


def make_result(
    check_id: str,
    status: str,
    message: str,
    failure_type: str | None = None,
    repair_action: str | None = None,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "id": check_id,
        "status": status,
        "message": message,
        "failure_type": failure_type,
        "repair_action": repair_action,
        "detail": detail or {},
    }
    return {key: value for key, value in result.items() if value is not None}


def extract_launch_config(code: str) -> dict[str, int]:
    config = {}
    for name, value in re.findall(r"static\s+const\s+int\s+(BM|BN|BK|WM|WN|WMITER|WNITER|TM|TN)\s*=\s*(\d+)\s*;", code):
        config[name] = int(value)
    return config


def strip_cpp_comments(content: str) -> str:
    without_block_comments = re.sub(r"/\*.*?\*/", lambda match: "\n" * match.group(0).count("\n"), content, flags=re.DOTALL)
    return re.sub(r"//.*", "", without_block_comments)


def region_between_markers(content: str, begin: str, end: str) -> str:
    start = content.find(begin)
    finish = content.find(end, start + len(begin)) if start >= 0 else -1
    if start < 0 or finish < 0:
        return ""
    return content[start + len(begin):finish]


def tail_from_last_c_store(code: str) -> str:
    matches = list(re.finditer(r"\bC\s*\[[^\]]+\]\s*=", code))
    if not matches:
        return ""
    return code[max(0, matches[-1].start() - 500):matches[-1].end() + 500]


def has_loop_bound(code: str, bound: str) -> bool:
    return bool(re.search(r"for\s*\([^)]*<\s*" + re.escape(bound) + r"\b", code))


def extract_first_loop_body(code: str, loop_pattern: str) -> str:
    match = re.search(loop_pattern, code)
    if not match:
        return ""
    brace = code.find("{", match.end())
    if brace < 0:
        return ""
    depth = 0
    for index in range(brace, len(code)):
        char = code[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return code[brace + 1:index]
    return ""


def region_from_first_shared_store(code: str) -> str:
    match = re.search(r"\b(?:As0|As1|As|Bs0|Bs1|Bs|shared_A|shared_B|sA|sB)\s*\[[^\]]+\]", code)
    if not match:
        return ""
    return code[match.start():match.start() + 5000]


def get_path(data: dict[str, Any], path: str, default: Any = None) -> Any:
    node: Any = data
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def int_value(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
