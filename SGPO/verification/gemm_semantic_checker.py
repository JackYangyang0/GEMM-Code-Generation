from __future__ import annotations

import re
from pathlib import Path
from typing import Any


IDENTIFIER = r"[A-Za-z_]\w*"


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
        check_loop_index_scope(code),
        check_shared_array_pointer_layout(code, launch_config),
        check_shared_tile_indices_are_local(code),
        check_first_k_tile_is_accumulated(code),
        check_cta_swizzle_preserves_grid_extent(code),
        check_store_bounds_use_global_coordinates(content, code),
        check_no_scalar_fallback_mixed_with_optimized_regions(content, code),
        check_warp_lane_fragment_coverage(code, launch_config, ir or {}),
        check_thread_tile_mapping_covers_block(code, bm, bn, tm, tn),
        check_thread_tile_store_coverage(content, code, tm, tn),
        check_register_accumulator_coverage(code, tm, tn),
        check_k_loop_updates_A_and_B_tiles(code),
        check_fast_k_loop_divisibility(code, launch_config, ir or {}),
        check_vectorized_fast_path_guards(code, host_code, launch_config, ir or {}),
        check_A_register_initialization(code, bk),
        check_vector_shared_load_mapping_bounds(code, launch_config),
        check_shared_memory_is_cooperatively_loaded(code, bm, bn, bk),
        check_shared_load_not_nested_under_output_group_loop(code),
        check_shared_memory_is_used_by_compute(code),
    ]

    return {
        "accepted": all(item["status"] == "pass" for item in results),
        "launch_config": {key: value for key, value in launch_config.items() if key in {"BM", "BN", "BK", "TM", "TN", "WM", "WN", "WMITER", "WNITER"}},
        "results": results,
    }


def check_loop_index_scope(code: str) -> dict[str, Any]:
    for name in ("bkIdx", "k_iter", "k4"):
        declaration = re.search(rf"\b(?:int|const\s+int)\s+{name}\b", code)
        use = re.search(rf"\b{name}\b", code)
        if use and (not declaration or use.start() < declaration.start()):
            return make_result(
                "GEMM_LOOP_INDEX_SCOPE_VALID", "fail",
                f"{name} is referenced before its declaration or outside its valid loop scope.",
                "Compile.LoopIndexOutOfScope",
                f"move the {name}-dependent expression inside the declaring loop, or remove {name} from initialization code",
                {"symbol": name, "first_use_offset": use.start(), "declaration_offset": declaration.start() if declaration else None},
            )
    return make_result("GEMM_LOOP_INDEX_SCOPE_VALID", "pass", "K-loop indices are not used before declaration.")


def check_shared_array_pointer_layout(code: str, launch_config: dict[str, int]) -> dict[str, Any]:
    declarations = {
        name: (first, second)
        for name, first, second in re.findall(
            r"__shared__\s+float\s+(As0|As1|Bs0|Bs1)\s*\[\s*(\w+)\s*\]\s*\[\s*(\w+)\s*\]", code
        )
    }
    pointer_widths = {
        name: width
        for name, width in re.findall(r"float\s*\(\s*\*(As_ptr|As_final|Bs_ptr|Bs_final)\s*\)\s*\[\s*(\w+)\s*\]", code)
    }
    mismatches = []
    for pointer, array in re.findall(r"\b(As_ptr|As_final|Bs_ptr|Bs_final)\s*=\s*(As0|As1|Bs0|Bs1)\s*;", code):
        if pointer in pointer_widths and array in declarations and declarations[array][1] != pointer_widths[pointer]:
            mismatches.append({
                "pointer": pointer, "pointer_inner_extent": pointer_widths[pointer],
                "array": array, "array_inner_extent": declarations[array][1],
            })
    if mismatches:
        return make_result(
            "GEMM_SHARED_ARRAY_POINTER_LAYOUT_MATCHES", "fail",
            "shared-memory arrays decay to pointer types with incompatible inner extents.",
            "Compile.SharedArrayPointerTypeMismatch",
            "make the shared array rank/order match its pointer view and compute indexing (for As_ptr[k][m], declare As[BK][BM])",
            {"mismatches": mismatches, "launch_config": launch_config},
        )
    return make_result("GEMM_SHARED_ARRAY_POINTER_LAYOUT_MATCHES", "pass", "shared arrays and their pointer views have matching inner extents.")


def check_shared_tile_indices_are_local(code: str) -> dict[str, Any]:
    pattern = re.compile(
        r"\b(As0|As1|Bs0|Bs1)\s*\[\s*([^\]]*\bbkIdx\b[^\]]*|[^\]]*\*\s*BK[^\]]*)\s*\]"
        r"|\b(As0|As1|Bs0|Bs1)\s*\[[^\]]+\]\s*\[\s*([^\]]*\bbkIdx\b[^\]]*|[^\]]*\*\s*BK[^\]]*)\s*\]"
    )
    bad = [match.group(0) for match in pattern.finditer(code)]
    if bad:
        return make_result(
            "GEMM_SHARED_TILE_INDICES_ARE_LOCAL", "fail",
            "global K-tile offsets are used as shared-memory indices and can exceed the BK-sized buffer.",
            "GEMM.Semantic.SharedMemoryTileIndexOutOfBounds",
            "apply (bkIdx + 1) * BK only to the global-memory address; index shared A/B with local k4..k4+3",
            {"expressions": bad[:8]},
        )
    return make_result("GEMM_SHARED_TILE_INDICES_ARE_LOCAL", "pass", "shared-memory tile indices remain local to the allocated tile.")


def check_first_k_tile_is_accumulated(code: str) -> dict[str, Any]:
    for body in extract_if_bodies(code, r"bkIdx\s*==\s*0"):
        initializes_fragment = bool(re.search(r"\breg[MN]\s*\[", body))
        accumulates = bool(re.search(r"\b(?:results|acc)\s*\[[^\]]+\](?:\s*\[[^\]]+\])?\s*\+=", body))
        if initializes_fragment and not accumulates:
            return make_result(
                "GEMM_FIRST_K_TILE_IS_ACCUMULATED", "fail",
                "the bkIdx == 0 branch initializes a register fragment but performs no multiply-accumulate, dropping the first K tile.",
                "GEMM.Semantic.FirstKTileDropped",
                "run the same BK-wide multiply-accumulate for tile zero, using the prefetched A values only as a load substitution",
            )
    return make_result("GEMM_FIRST_K_TILE_IS_ACCUMULATED", "pass", "no control-flow pattern that drops the first K tile was detected.")


def check_cta_swizzle_preserves_grid_extent(code: str) -> dict[str, Any]:
    changes_grid_extent = bool(
        re.search(r"grid_x_orig\s*=\s*CEIL_DIV\s*\(\s*N\s*,\s*BN\s*\)", code)
        and re.search(r"\bgrid_x\s*=\s*[^;]*(?:grid_x_orig\s*&|grid_x_orig\s*<<|grid_x_orig\s*>>)[^;]*;", code)
        and re.search(r"dim3\s+blocksPerGrid\s*\(\s*grid_x\s*,", code)
    )
    if changes_grid_extent:
        return make_result(
            "GEMM_CTA_SWIZZLE_PRESERVES_GRID_EXTENT", "fail",
            "CTA swizzle transforms gridDim.x itself, changing the number of launched CTAs.",
            "GEMM.Semantic.CTASwizzleChangesGridExtent",
            "launch CEIL_DIV(N, BN) by CEIL_DIV(M, BM) CTAs and apply a bijective mapping to blockIdx inside the kernel",
        )
    return make_result("GEMM_CTA_SWIZZLE_PRESERVES_GRID_EXTENT", "pass", "CTA scheduling does not alter the required grid extent.")


def check_store_bounds_use_global_coordinates(content: str, code: str) -> dict[str, Any]:
    store = strip_cpp_comments(region_between_markers(content, "STORE_BEGIN", "STORE_END"))
    if not store:
        return make_result("GEMM_STORE_BOUNDS_USE_GLOBAL_COORDINATES", "pass", "store region is unavailable; check deferred.")
    has_block_offsets = "C_offset" in store or bool(re.search(r"\b(?:tile_m0|blockIdx\.y\s*\*\s*BM)\b", store))
    dimension_guards = [item for item in extract_if_conditions(store) if re.search(r"<\s*M\b", item) and re.search(r"<\s*N\b", item)]
    guard_is_global = any(re.search(r"tile_m0|tile_n0|blockIdx|global_[mn]\b", item) for item in dimension_guards)
    if has_block_offsets and dimension_guards and not guard_is_global:
        return make_result(
            "GEMM_STORE_BOUNDS_USE_GLOBAL_COORDINATES", "fail",
            "C address includes the block offset, but its M/N bounds check uses only block-local coordinates.",
            "GEMM.Semantic.OutputBoundaryGuardUsesLocalCoordinates",
            "form global_m = tile_m0 + local_m and global_n = tile_n0 + local_n; use them for both the bounds guard and C index",
        )
    return make_result("GEMM_STORE_BOUNDS_USE_GLOBAL_COORDINATES", "pass", "output bounds are global or no conflicting local-only guard was detected.")


def extract_if_bodies(code: str, condition_pattern: str) -> list[str]:
    bodies: list[str] = []
    for match in re.finditer(r"if\s*\(\s*" + condition_pattern + r"\s*\)\s*\{", code):
        brace = code.find("{", match.start())
        depth = 0
        for index in range(brace, len(code)):
            if code[index] == "{":
                depth += 1
            elif code[index] == "}":
                depth -= 1
                if depth == 0:
                    bodies.append(code[brace + 1:index])
                    break
    return bodies


def extract_if_conditions(code: str) -> list[str]:
    conditions: list[str] = []
    for match in re.finditer(r"\bif\s*\(", code):
        start = code.find("(", match.start())
        depth = 0
        for index in range(start, len(code)):
            if code[index] == "(":
                depth += 1
            elif code[index] == ")":
                depth -= 1
                if depth == 0:
                    conditions.append(code[start + 1:index])
                    break
    return conditions


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
    if re.search(r"\blane_cols\s*=\s*WNITER\s*/\s*TN\s*;", code):
        has_expected_trow |= bool(re.search(r"\bTrow\s*=\s*lane\s*/\s*lane_cols\s*;", code))
        has_expected_tcol |= bool(re.search(
            r"\bTcol\s*=\s*(?:lane\s*%\s*lane_cols|lane\s*-\s*Trow\s*\*\s*lane_cols)\s*;", code))
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
    accumulator_candidates = find_2d_float_accumulator_candidates(code)
    valid_candidates = [
        item
        for item in accumulator_candidates
        if item["initialized"] and item["accumulated"]
    ]
    if valid_candidates:
        return make_result(
            "GEMM_REGISTER_ACCUMULATOR_COVERS_TM_TN",
            "pass",
            "a 2D register accumulator tile is initialized and accumulated over the thread tile.",
            detail={"accumulators": valid_candidates[:4]},
        )
    return make_result(
        "GEMM_REGISTER_ACCUMULATOR_COVERS_TM_TN",
        "fail",
        "register accumulator shape exists, but compute does not cover every TM/TN accumulator element.",
        "GEMM.Semantic.RegisterTileComputeIncomplete",
        "compute all acc[tm][tn] values inside nested TM/TN loops or explicit unrolled equivalents",
        {"accumulators": accumulator_candidates[:8], "has_tm_loop": has_loop_bound(code, "TM"), "has_tn_loop": has_loop_bound(code, "TN")},
    )


def check_k_loop_updates_A_and_B_tiles(code: str) -> dict[str, Any]:
    k_tile_loops = extract_k_tile_loop_bodies(code)
    shared_names = shared_array_names(code)
    for loop in k_tile_loops:
        body = loop["body"]
        loop_var = loop["var"]
        k_dependent_load = code_segment_has_k_tile_dependency(body, loop_var)
        updates_a = shared_assignment_from_global(body, shared_names, "A") and k_dependent_load
        updates_b = shared_assignment_from_global(body, shared_names, "B") and k_dependent_load
        direct_global_compute = bool(
            not shared_names
            and re.search(r"\bA\s*\[", body)
            and re.search(r"\bB\s*\[", body)
            and re.search(r"\b(?:acc|results|\w+)\s*\[[^\]]+\](?:\s*\[[^\]]+\])?\s*\+=", body)
        )
        if (updates_a and updates_b) or direct_global_compute:
            return make_result(
                "GEMM_K_LOOP_UPDATES_A_AND_B_TILES",
                "pass",
                "K-tile loop updates the A/B data used by compute.",
                detail={"loop_var": loop_var, "updates_a": updates_a, "updates_b": updates_b},
            )

    if not k_tile_loops and re.search(r"for\s*\(\s*int\s+bkIdx\s*=\s*1\s*;\s*bkIdx\s*<\s*k_tiles\s*;", code):
        has_ping_pong_load = bool(
            "comp_flag" in code
            and "mem_flag" in code
            and re.search(r"\bbkIdx\s*\*\s*BK", code)
            and shared_assignment_from_global(code, shared_names, "A")
            and shared_assignment_from_global(code, shared_names, "B")
        )
        if has_ping_pong_load:
            return make_result(
                "GEMM_K_LOOP_UPDATES_A_AND_B_TILES",
                "pass",
                "double-buffered bkIdx loop updates both A and B tile data.",
            )
    if not k_tile_loops:
        return make_result(
            "GEMM_K_LOOP_UPDATES_A_AND_B_TILES",
            "fail",
            "optimized GEMM must have a K loop advancing by BK.",
            "GEMM.Semantic.KLoopMissing",
            "add a K-tile loop advancing by BK around shared loads and compute",
        )
    return make_result(
        "GEMM_K_LOOP_UPDATES_A_AND_B_TILES",
        "fail",
        "K loop does not reload both A and B tiles for each BK slice.",
        "GEMM.Semantic.KLoopDataflowIncomplete",
        "move cooperative A/B tile loads into the K loop and index global memory with k + local_k",
        {
            "k_tile_loops": [
                {
                    "var": loop["var"],
                    "updates_a": shared_assignment_from_global(loop["body"], shared_names, "A"),
                    "updates_b": shared_assignment_from_global(loop["body"], shared_names, "B"),
                    "k_dependent": code_segment_has_k_tile_dependency(loop["body"], loop["var"]),
                }
                for loop in k_tile_loops[:4]
            ]
        },
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
    shared_names = shared_array_names(code)
    shared_load_code = region_from_first_shared_store(code)
    has_a_load = shared_assignment_from_global(shared_load_code or code, shared_names, "A")
    has_b_load = shared_assignment_from_global(shared_load_code or code, shared_names, "B")
    has_thread_strided_loop = has_thread_cooperative_load_pattern(code)
    suspicious_per_thread_whole_tile = bool(re.search(r"for\s*\([^)]*<\s*BN[^)]*\)\s*\{[^{}]*for\s*\([^)]*<\s*BK", code, flags=re.DOTALL))
    has_vector_thread_mapped_load = bool(
        "FLOAT4" in code
        and has_a_load
        and has_b_load
        and has_thread_derived_index(code)
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
            "shared_arrays": shared_names,
        },
    )


def check_vector_shared_load_mapping_bounds(code: str, launch_config: dict[str, int]) -> dict[str, Any]:
    if "__shared__" not in code or "FLOAT4" not in code:
        return make_result(
            "GEMM_VECTOR_SHARED_LOAD_MAPPING_BOUNDED",
            "pass",
            "vectorized shared-memory loading is not used.",
        )
    required = ("BM", "BN", "BK", "WM", "WN")
    if any(int_value(launch_config.get(name)) is None for name in required):
        return make_result(
            "GEMM_VECTOR_SHARED_LOAD_MAPPING_BOUNDED",
            "pass",
            "launch constants are incomplete; vector load bound check is deferred.",
        )

    bm, bn, bk, wm, wn = (int(launch_config[name]) for name in required)
    threads = (bm // wm) * (bn // wn) * 32 if wm > 0 and wn > 0 else 0
    hazards = []
    legacy_a_names = re.findall(
        r"\b(load_a\w*m)\s*=\s*(?:tid|threadIdx\.x)\s*/\s*\(\s*BK\s*/\s*4\s*\)", code
    )
    legacy_b_names = re.findall(
        r"\b(load_b\w*k)\s*=\s*(?:tid|threadIdx\.x)\s*/\s*\(\s*BN\s*/\s*4\s*\)", code
    )
    a_guarded = bool(legacy_a_names) and all(has_upper_bound_guard(code, name, "BM") for name in legacy_a_names)
    b_guarded = bool(legacy_b_names) and all(has_upper_bound_guard(code, name, "BK") for name in legacy_b_names)
    if legacy_a_names and not a_guarded and threads > bm * (bk // 4):
        hazards.append({"operand": "A", "threads": threads, "vector_tasks": bm * (bk // 4)})
    if legacy_b_names and not b_guarded and threads > bk * (bn // 4):
        hazards.append({"operand": "B", "threads": threads, "vector_tasks": bk * (bn // 4)})
    if not hazards:
        return make_result(
            "GEMM_VECTOR_SHARED_LOAD_MAPPING_BOUNDED",
            "pass",
            "vectorized shared-memory load indices are bounded by tile task counts.",
        )
    return make_result(
        "GEMM_VECTOR_SHARED_LOAD_MAPPING_BOUNDED",
        "fail",
        "thread-derived vector load coordinates can exceed the shared-memory tile when the block has more threads than vector load tasks.",
        "GEMM.Semantic.SharedLoadThreadMappingOutOfBounds",
        "use a flattened loop such as for (loadIdx = tid * 4; loadIdx < tile_elements; loadIdx += blockDim.x * 4), then derive local row/column coordinates from loadIdx",
        {"hazards": hazards, "launch_config": {name: launch_config[name] for name in required}},
    )


def has_upper_bound_guard(code: str, variable: str, upper_bound: str) -> bool:
    return bool(
        re.search(
            rf"\bif\s*\([^)]*\b{re.escape(variable)}\b\s*<\s*{re.escape(upper_bound)}\b[^)]*\)",
            code,
        )
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
    shared_names = shared_array_names(code)
    uses_a_shared = bool(re.search(shared_access_regex(shared_names_matching(shared_names, "A")), compute_regions))
    uses_b_shared = bool(re.search(shared_access_regex(shared_names_matching(shared_names, "B")), compute_regions))
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
    match = re.search(shared_access_regex(shared_array_names(code)), code)
    if not match:
        return ""
    return code[match.start():match.start() + 5000]


def find_2d_float_accumulator_candidates(code: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    pattern = re.compile(
        rf"\bfloat\s+({IDENTIFIER})\s*(\[[^\]]+\]\s*\[[^\]]+\])\s*(=\s*\{{[^;]*\}})?\s*;"
    )
    for match in pattern.finditer(code):
        line_start = code.rfind("\n", 0, match.start()) + 1
        line_prefix = code[line_start:match.start()]
        if "__shared__" in line_prefix:
            continue
        name = match.group(1)
        escaped = re.escape(name)
        initialized = bool(match.group(3)) or bool(
            re.search(
                rf"\b{escaped}\s*\[[^\]]+\]\s*\[[^\]]+\]\s*=\s*(?:0|0\.0f|0\.0|0\.f)\b",
                code,
            )
        )
        accumulated = bool(re.search(rf"\b{escaped}\s*\[[^\]]+\]\s*\[[^\]]+\]\s*\+=", code))
        stored = bool(
            re.search(
                rf"\bC\s*\[[^\]]+\]\s*=[^;]*\b{escaped}\s*\[[^\]]+\]\s*\[[^\]]+\]",
                code,
                flags=re.DOTALL,
            )
        )
        candidates.append(
            {
                "name": name,
                "shape": match.group(2),
                "initialized": initialized,
                "accumulated": accumulated,
                "stored": stored,
                "declaration": match.group(0),
            }
        )
    return candidates


def shared_array_names(code: str) -> list[str]:
    names = set(
        re.findall(
            rf"__shared__\s+(?:__align__\s*\([^)]*\)\s*)?(?:float|float4|half|__half|double)\s+({IDENTIFIER})\s*\[",
            code,
        )
    )
    for fallback in ("As0", "As1", "Bs0", "Bs1", "As", "Bs", "shared_A", "shared_B", "sA", "sB"):
        if re.search(rf"\b{re.escape(fallback)}\s*\[", code):
            names.add(fallback)
    return sorted(names)


def shared_names_matching(names: list[str], tensor: str) -> list[str]:
    target = tensor.upper()
    matched: list[str] = []
    for name in names:
        compact = name.replace("_", "").upper()
        if target == "A" and ("A" in compact or compact.endswith("AS") or compact.startswith("SA")):
            matched.append(name)
        elif target == "B" and ("B" in compact or compact.endswith("BS") or compact.startswith("SB")):
            matched.append(name)
    return matched or names


def shared_access_regex(names: list[str]) -> str:
    if not names:
        return r"a^"
    alternatives = "|".join(re.escape(name) for name in names)
    return rf"\b(?:{alternatives})\s*(?:\[[^\]]+\])+"


def shared_assignment_from_global(code: str, shared_names: list[str], global_name: str) -> bool:
    matching_names = shared_names_matching(shared_names, global_name)
    shared_access = shared_access_regex(matching_names)
    global_access = rf"\b{re.escape(global_name)}\s*\["
    shared_lhs = rf"(?:FLOAT4\s*\(\s*)?{shared_access}(?:\s*\))?\s*="
    if re.search(shared_lhs + rf"[^;]*{global_access}", code, flags=re.DOTALL):
        return True
    temp_loads = re.findall(
        rf"\b(?:float4|float|auto)\s+({IDENTIFIER})\s*=\s*(?:FLOAT4\s*\(\s*)?[^;]*{global_access}[^;]*;",
        code,
        flags=re.DOTALL,
    )
    for temp_name in temp_loads:
        if re.search(shared_lhs + rf"[^;]*\b{re.escape(temp_name)}\b(?:\.[xyzw])?", code, flags=re.DOTALL):
            return True
    return False


def has_thread_derived_index(code: str) -> bool:
    return bool(re.search(r"\b(?:tid|threadIdx\.x|sgpo_linear_tid|lane)\b", code))


def has_thread_cooperative_load_pattern(code: str) -> bool:
    thread_symbol = r"(?:tid|threadIdx\.x|sgpo_linear_tid|lane)"
    thread_count = r"(?:blockDim\.x|sgpo_thread_count|thread_count|threads_per_block)"
    load_idx_loop = re.search(
        rf"for\s*\([^)]*=\s*{thread_symbol}[^;]*;[^;]*;[^)]*\+=\s*{thread_count}\s*\)",
        code,
    )
    thread_offset = re.search(
        rf"(?:\+|\-|\*|/|%)\s*{thread_symbol}\b|\b{thread_symbol}\s*(?:\+|\-|\*|/|%)",
        code,
    )
    if load_idx_loop or thread_offset and re.search(rf"for\s*\([^)]*;[^;]*<[^;]*;[^)]*\+=", code):
        return True

    derived_vars = thread_derived_index_names(code)
    for match in re.finditer(rf"for\s*\([^)]*;[^;]*<[^;]*;[^)]*\+=\s*(?:{thread_count}|{IDENTIFIER})\s*\)", code):
        body = extract_loop_body_from_match(code, match)
        if body and any(re.search(rf"\b{re.escape(name)}\b", body) for name in derived_vars):
            return True
    return False


def thread_derived_index_names(code: str) -> set[str]:
    names: set[str] = set()
    thread_symbol = r"(?:tid|threadIdx\.x|sgpo_linear_tid|lane)"
    for name, expression in re.findall(rf"\b(?:const\s+)?int\s+({IDENTIFIER})\s*=\s*([^;]*{thread_symbol}[^;]*);", code):
        names.add(name)
    return names


def code_segment_has_k_tile_dependency(segment: str, loop_var: str) -> bool:
    var = re.escape(loop_var)
    return bool(
        re.search(rf"\bglobal_k\s*=\s*[^;]*\b{var}\b", segment)
        or re.search(rf"\b{var}\b\s*(?:\+|\-|\*|/|%)\s*(?:BK|{IDENTIFIER})", segment)
        or re.search(rf"(?:BK|{IDENTIFIER})\s*(?:\+|\-|\*|/|%)\s*\b{var}\b", segment)
        or re.search(rf"OFFSET\s*\([^;]*\b{var}\b", segment, flags=re.DOTALL)
        or re.search(rf"\b[AB]\s*\[[^;]*\b{var}\b", segment, flags=re.DOTALL)
    )


def extract_k_tile_loop_bodies(code: str) -> list[dict[str, str]]:
    loops: list[dict[str, str]] = []
    seen_offsets: set[int] = set()
    patterns = [
        rf"for\s*\(\s*(?:int|long|size_t)\s+(?P<var>{IDENTIFIER})\s*=\s*0\s*;\s*(?P=var)\s*<\s*K\s*;\s*(?:(?:\+\+\s*(?P=var))|(?:(?P=var)\s*\+\+)|(?:(?P=var)\s*\+=\s*BK)|(?:(?P=var)\s*=\s*(?P=var)\s*\+\s*BK))\s*\)",
        rf"for\s*\(\s*(?:int|long|size_t)\s+(?P<var>{IDENTIFIER})\s*=\s*[01]\s*;\s*(?P=var)\s*<\s*k_tiles\s*;[^)]*\)",
        rf"for\s*\(\s*(?:int|long|size_t)\s+(?P<var>{IDENTIFIER})\s*=\s*0\s*;\s*(?P=var)\s*<\s*CEIL_DIV\s*\(\s*K\s*,\s*BK\s*\)\s*;[^)]*\)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, code):
            if match.start() in seen_offsets:
                continue
            body = extract_loop_body_from_match(code, match)
            if body:
                seen_offsets.add(match.start())
                loops.append({"var": match.group("var"), "body": body, "header": match.group(0)})
    return loops


def extract_loop_body_from_match(code: str, match: re.Match[str]) -> str:
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
