"""Deterministic consistency fallback for tiled CUDA GEMM memory access."""
from __future__ import annotations

import re
from pathlib import Path


REGIONS = ("SHARED_DECL", "INDEX_MAPPING", "GLOBAL_TO_SHARED_LOAD", "MAIN_LOOP", "STORE")


def enforce_memory_access_plan(code_dir: Path) -> dict:
    path = code_dir / "cuda_kernel.cuh"
    if not path.exists():
        return {"status": "not_run", "reason": "cuda_kernel.cuh is missing"}
    source = path.read_text(encoding="utf-8")
    defects = memory_access_consistency_defects(source)
    if not defects:
        return {"status": "pass", "materialized": False, "defects": []}

    updated = source
    replacements = safe_scalar_regions()
    for region in REGIONS:
        updated = replace_region(updated, region, replacements[region])
    updated = updated.replace("const int k_tiles = K / BK;", "const int k_tiles = CEIL_DIV(K, BK);")
    remaining = memory_access_consistency_defects(updated)
    if remaining:
        return {"status": "fail", "materialized": False, "defects": remaining}
    path.write_text(updated, encoding="utf-8")
    return {
        "status": "pass",
        "materialized": True,
        "mode": "deterministic_safe_scalar_fallback",
        "trigger_defects": defects,
        "modified_regions": list(REGIONS),
    }


def memory_access_consistency_defects(source: str) -> list[dict]:
    defects = []
    a_km = bool(re.search(r"As\s*\[\s*2\s*\]\s*\[\s*BK", source))
    a_mk = bool(re.search(r"As\s*\[\s*2\s*\]\s*\[\s*BM", source))
    b_kn = bool(re.search(r"Bs\s*\[\s*2\s*\]\s*\[\s*BK", source))
    b_nk = bool(re.search(r"Bs\s*\[\s*2\s*\]\s*\[\s*BN", source))
    if a_mk and re.search(r"As\s*\[[^]]+\]\s*\[\s*(?:k|load_a_smem_k)", source):
        defects.append({"id": "A_SHARED_LAYOUT_MIXED", "message": "As is declared [M][K] but used as [K][M]."})
    if a_km and re.search(r"As\s*\[[^]]+\]\s*\[\s*load_a_smem_m", source):
        defects.append({"id": "A_SHARED_LAYOUT_MIXED", "message": "As is declared [K][M] but used as [M][K]."})
    if b_nk and re.search(r"Bs\s*\[[^]]+\]\s*\[\s*(?:k|load_b_smem_k)", source):
        defects.append({"id": "B_SHARED_LAYOUT_MIXED", "message": "Bs is declared [N][K] but used as [K][N]."})
    if b_kn and re.search(r"Bs\s*\[[^]]+\]\s*\[\s*(?:load_b_smem_n|bs_idx)", source):
        defects.append({"id": "B_SHARED_LAYOUT_MIXED", "message": "Bs is declared [K][N] but used as [N][K]."})

    first = anchor_body(source, "GLOBAL_TO_SHARED_LOAD")
    later = anchor_body(source, "NEXT_TILE_LOAD")
    if ("FLOAT4(" in first) != ("FLOAT4(" in later):
        defects.append({"id": "TILE_LOAD_VECTOR_POLICY_MISMATCH",
                        "message": "Initial and subsequent tiles use different vector policies."})
    if "FLOAT4(" in later and not vector_mapping_is_proven(source):
        defects.append({"id": "UNPROVEN_FLOAT4_TILE_LOAD",
                        "message": "A subsequent-tile float4 address is not structurally proven 16-byte aligned."})
    return unique_defects(defects)


def vector_mapping_is_proven(source: str) -> bool:
    return bool(
        re.search(r"load_a_smem_k\s*=\s*\([^;]*%\s*\(BK\s*/\s*4\)\)\s*\*\s*4", source)
        and re.search(r"load_b_smem_n\s*=\s*\([^;]*%\s*\(BN\s*/\s*4\)\)\s*\*\s*4", source)
    )


def safe_scalar_regions() -> dict[str, list[str]]:
    load = [
        "for (int linear = tid; linear < BM * BK; linear += thread_num) {",
        "    const int smem_m = linear / BK;",
        "    const int smem_k = linear - smem_m * BK;",
        "    const int global_m = tile_m0 + smem_m;",
        "    As[0][smem_k][smem_m] = (global_m < M && smem_k < K) ? A[OFFSET(global_m, smem_k, K)] : 0.0f;",
        "}",
        "for (int linear = tid; linear < BK * BN; linear += thread_num) {",
        "    const int smem_k = linear / BN;",
        "    const int smem_n = linear - smem_k * BN;",
        "    const int global_n = tile_n0 + smem_n;",
        "    Bs[0][smem_k][smem_n] = (smem_k < K && global_n < N) ? B[OFFSET(smem_k, global_n, N)] : 0.0f;",
        "}",
    ]
    compute = compute_lines("comp_flag", "valid_k")
    next_load = [line.replace("As[0]", "As[mem_flag]").replace("Bs[0]", "Bs[mem_flag]")
                 .replace("smem_k < K", "global_k < K")
                 .replace("A[OFFSET(global_m, smem_k, K)]", "A[OFFSET(global_m, global_k, K)]")
                 .replace("B[OFFSET(smem_k, global_n, N)]", "B[OFFSET(global_k, global_n, N)]")
                 for line in load]
    # Define the global K coordinate immediately after each local coordinate.
    next_load.insert(3, "    const int global_k = bkIdx * BK + smem_k;")
    second_loop = next(i for i, line in enumerate(next_load) if line.startswith("for (int linear") and i > 0)
    next_load.insert(second_loop + 3, "    const int global_k = bkIdx * BK + smem_k;")
    main = [
        "for (int bkIdx = 1; bkIdx < k_tiles; ++bkIdx) {",
        "    __syncthreads();",
        "    const int comp_flag = (bkIdx - 1) & 1;",
        "    const int mem_flag = bkIdx & 1;",
        "    const int valid_k = min(BK, K - (bkIdx - 1) * BK);",
        *["    " + line for line in compute],
        "    /* NEXT_TILE_LOAD_BEGIN */",
        *["    " + line for line in next_load],
        "    /* NEXT_TILE_LOAD_END */",
        "}",
        "__syncthreads();",
        "const int comp_flag = (k_tiles - 1) & 1;",
        "const int valid_k = K - (k_tiles - 1) * BK;",
        *compute,
    ]
    return {
        "SHARED_DECL": ["__shared__ float As[2][BK][BM];", "__shared__ float Bs[2][BK][BN];"],
        "INDEX_MAPPING": [
            "const int warp_tiles_n = BN / WN;", "const int Wrow = wid / warp_tiles_n;",
            "const int Wcol = wid - Wrow * warp_tiles_n;", "const int lane_cols = WNITER / TN;",
            "const int Trow = lane / lane_cols;", "const int Tcol = lane - Trow * lane_cols;",
        ],
        "GLOBAL_TO_SHARED_LOAD": load,
        "MAIN_LOOP": main,
        "STORE": [
            "#pragma unroll", "for (int wm = 0; wm < WM / WMITER; ++wm) {",
            "    #pragma unroll", "    for (int wn = 0; wn < WN / WNITER; ++wn) {",
            "        #pragma unroll", "        for (int m = 0; m < TM; ++m) {",
            "            #pragma unroll", "            for (int n = 0; n < TN; ++n) {",
            "                const int global_m = tile_m0 + Wrow * WM + wm * WMITER + Trow * TM + m;",
            "                const int global_n = tile_n0 + Wcol * WN + wn * WNITER + Tcol * TN + n;",
            "                if (global_m < M && global_n < N) {",
            "                    const int c_index = OFFSET(global_m, global_n, N);",
            "                    C[c_index] = alpha * results[wm * TM + m][wn * TN + n] + beta * C[c_index];",
            "                }", "            }", "        }", "    }", "}",
        ],
    }


def compute_lines(buffer: str, limit: str) -> list[str]:
    return [
        "#pragma unroll", f"for (int k = 0; k < {limit}; ++k) {{",
        "    #pragma unroll", "    for (int wm = 0; wm < WM / WMITER; ++wm) {",
        "        #pragma unroll", "        for (int wn = 0; wn < WN / WNITER; ++wn) {",
        "            #pragma unroll", "            for (int i = 0; i < TM; ++i)",
        f"                regM[i] = As[{buffer}][k][Wrow * WM + wm * WMITER + Trow * TM + i];",
        "            #pragma unroll", "            for (int j = 0; j < TN; ++j)",
        f"                regN[j] = Bs[{buffer}][k][Wcol * WN + wn * WNITER + Tcol * TN + j];",
        "            #pragma unroll", "            for (int i = 0; i < TM; ++i) {",
        "                #pragma unroll", "                for (int j = 0; j < TN; ++j)",
        "                    results[wm * TM + i][wn * TN + j] += regM[i] * regN[j];",
        "            }", "        }", "    }", "}",
    ]


def anchor_body(source: str, region: str) -> str:
    match = re.search(rf"{region}_BEGIN.*?\*/(?P<body>.*?)/\*.*?{region}_END", source, re.S)
    return match.group("body") if match else ""


def replace_region(source: str, region: str, lines: list[str]) -> str:
    begin_marker = source.find(f"{region}_BEGIN")
    end_marker = source.find(f"{region}_END", begin_marker + 1)
    if begin_marker < 0 or end_marker < 0:
        raise ValueError(f"Missing anchor region: {region}")
    begin_open = source.rfind("/*", 0, begin_marker)
    begin_close = source.find("*/", begin_marker) + 2
    end_open = source.rfind("/*", begin_close, end_marker)
    end_close = source.find("*/", end_marker) + 2
    if min(begin_open, begin_close, end_open, end_close) < 0:
        raise ValueError(f"Malformed anchor comments: {region}")
    line_start = source.rfind("\n", 0, begin_open) + 1
    prefix = source[line_start:begin_open]
    body = "\n".join(prefix + line if line else "" for line in lines)
    return source[:begin_close] + "\n" + body + "\n" + source[end_open:end_close] + source[end_close:]


def unique_defects(items: list[dict]) -> list[dict]:
    return list({item["id"]: item for item in items}.values())
