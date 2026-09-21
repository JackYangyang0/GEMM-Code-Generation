"""Deterministic consistency fallback for tiled CUDA GEMM memory access."""
from __future__ import annotations

import re
from pathlib import Path


REGIONS = (
    "SHARED_DECL",
    "INDEX_MAPPING",
    "REGISTER_DECL",
    "GLOBAL_TO_SHARED_LOAD",
    "MAIN_LOOP",
    "STORE",
)


def enforce_memory_access_plan(code_dir: Path, *, allow_scalar_fallback: bool = False) -> dict:
    """Inspect without mutation by default; scalar materialization is opt-in.

    The opt-in is for standalone baseline construction, not strategy repair:
    scalar materialization does not preserve a candidate's optimization contract.
    """
    path = code_dir / "cuda_kernel.cuh"
    if not path.exists():
        return {"status": "not_run", "reason": "cuda_kernel.cuh is missing"}
    source = path.read_text(encoding="utf-8")
    defects = memory_access_consistency_defects(source)
    if not defects:
        return {
            "status": "pass",
            "checked": True,
            "materialized": False,
            "mode": "existing_source_verified",
            "defects": [],
        }

    if not allow_scalar_fallback:
        return {
            "status": "fail",
            "checked": True,
            "materialized": False,
            "mode": "diagnostic_only_preserve_strategies",
            "defects": defects,
            "reason": "Memory access defects require strategy-preserving repair; source was not changed.",
            "repair_action": "Repair the reported indexing/layout defects while retaining vector widths, buffer stages and selected strategies.",
        }

    updated = source
    replacements = safe_scalar_regions()
    try:
        for region in REGIONS:
            updated = replace_region(updated, region, replacements[region])
    except ValueError as exc:
        return {
            "status": "fail",
            "checked": True,
            "materialized": False,
            "reason": str(exc),
            "defects": defects,
        }
    updated = updated.replace("const int k_tiles = K / BK;", "const int k_tiles = CEIL_DIV(K, BK);")
    remaining = memory_access_consistency_defects(updated)
    if remaining:
        return {"status": "fail", "materialized": False, "defects": remaining}
    path.write_text(updated, encoding="utf-8")
    return {
        "status": "pass",
        "checked": True,
        "materialized": True,
        "mode": "deterministic_safe_scalar_fallback",
        "trigger_defects": defects,
        "modified_regions": list(REGIONS),
    }


def memory_access_consistency_defects(source: str) -> list[dict]:
    defects = []
    defects.extend(anchor_contract_defects(source))
    defects.extend(shared_array_rank_defects(source))
    defects.extend(duplicate_block_offset_defects(source))
    defects.extend(register_extent_defects(source))
    defects.extend(accumulator_layout_defects(source))
    defects.extend(k_tile_coverage_defects(source))
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
    from SGPO.verification.optimization_preservation import vector_load_widths
    if "NEXT_TILE_LOAD_BEGIN" in source and vector_load_widths(first) != vector_load_widths(later):
        defects.append({"id": "TILE_LOAD_VECTOR_POLICY_MISMATCH",
                        "message": "Initial and subsequent tiles use different vector policies."})
    if "FLOAT4(" in later and not vector_mapping_is_proven(source):
        defects.append({"id": "UNPROVEN_FLOAT4_TILE_LOAD",
                        "message": "A subsequent-tile float4 address is not structurally proven 16-byte aligned."})
    if re.search(r'float4\s+(\w+)\s*=\s*FLOAT4\(A\[gm\s*\*\s*K\s*\+\s*gk\]\);\s*'
                 r'As\[[^]]+\]\[local_k\]\[local_m\]\s*=\s*\1\.x;\s*'
                 r'As\[[^]]+\]\[local_k\]\[local_m\s*\+\s*1\]\s*=\s*\1\.y;', source):
        defects.append({"id": "A_FLOAT4_SCATTER_AXIS",
                        "message": "Row-major A float4 contains consecutive K elements; [stage][K][M] shared storage must scatter along K, not M. Repair vector index mapping and both tile loads together."})
    return unique_defects(defects)


REQUIRED_ANCHORS = (
    "SHARED_DECL",
    "INDEX_MAPPING",
    "REGISTER_DECL",
    "GLOBAL_TO_SHARED_LOAD",
    "MAIN_LOOP",
    "STORE",
    "LAUNCH_CONFIG",
)


def anchor_contract_defects(source: str) -> list[dict]:
    defects = []
    for region in REQUIRED_ANCHORS:
        begin_count = len(re.findall(rf"\b{region}_BEGIN\b", source))
        end_count = len(re.findall(rf"\b{region}_END\b", source))
        if begin_count != 1 or end_count != 1:
            defects.append({
                "id": f"ANCHOR_CONTRACT_{region}",
                "message": f"{region} requires exactly one BEGIN/END pair; "
                           f"found begin={begin_count}, end={end_count}.",
            })
    return defects


def shared_array_rank_defects(source: str) -> list[dict]:
    defects = []
    declarations = {}
    for match in re.finditer(r"__shared__\s+float\s+(As|Bs)\s*((?:\[[^\]]+\])+)", source):
        declarations[match.group(1)] = len(re.findall(r"\[", match.group(2)))
    for name, rank in declarations.items():
        max_use_rank = 0
        for match in re.finditer(rf"\b{name}\s*((?:\[[^\]]+\])+)", source):
            max_use_rank = max(max_use_rank, len(re.findall(r"\[", match.group(1))))
        if max_use_rank > rank:
            defects.append({
                "id": f"{name.upper()}_SHARED_RANK_MISMATCH",
                "message": f"{name} is declared with rank {rank} but accessed with rank {max_use_rank}.",
            })
    return defects


def duplicate_block_offset_defects(source: str) -> list[dict]:
    defects = []
    assignments = {
        name: expression
        for name, expression in re.findall(
            r"(?:const\s+)?int\s+([A-Za-z_]\w*)\s*=\s*([^;]+);", source
        )
    }
    for operand, tile_name, offset_name in (
        ("A", "tile_m0", "A_offset"),
        ("B", "tile_n0", "B_offset"),
    ):
        if not re.search(rf"\b{offset_name}\s*=\s*[^;]*\b{tile_name}\b", source):
            continue
        for match in re.finditer(rf"\b{operand}\s*\[([^\]]+)\]", source):
            expression = match.group(1)
            if not re.search(rf"\b{offset_name}\b", expression):
                continue
            names = re.findall(r"\b[A-Za-z_]\w*\b", expression)
            already_global = tile_name in names or any(
                tile_name in assignments.get(name, "") for name in names
            )
            if already_global:
                defects.append({
                    "id": f"{operand}_BLOCK_OFFSET_APPLIED_TWICE",
                    "message": f"{operand} address combines {offset_name} with an index that already contains {tile_name}.",
                })
                break

    for match in re.finditer(r"\bA\s*\[([^\]]+)\]", source):
        if re.search(r"\btile_n0\b", match.group(1)):
            defects.append({
                "id": "A_K_INDEX_USES_N_TILE_OFFSET",
                "message": "A's K coordinate incorrectly contains tile_n0.",
            })
            break
    for match in re.finditer(r"\bB\s*\[([^\]]+)\]", source):
        expression = match.group(1)
        first_term = expression.split("*", 1)[0]
        if re.search(r"\btile_n0\b", first_term):
            defects.append({
                "id": "B_K_INDEX_USES_N_TILE_OFFSET",
                "message": "B's K coordinate incorrectly contains tile_n0.",
            })
            break
    return defects


def register_extent_defects(source: str) -> list[dict]:
    defects = []
    patterns = (
        ("regM", "TM", r"\bregM\s*\[\s*wm\s*\*\s*TM"),
        ("regN", "TN", r"\bregN\s*\[\s*wn\s*\*\s*TN"),
        ("regM_arr", "WMITER * TM", r"\bregM_arr\s*\[\s*k\s*\*\s*TM"),
        ("regN_arr", "WNITER * TN", r"\bregN_arr\s*\[\s*k\s*\*\s*TN"),
    )
    for name, declared_extent, indexed_pattern in patterns:
        extent_pattern = re.escape(declared_extent).replace(r"\ ", r"\s*")
        declaration = re.search(
            rf"\bfloat\s+{name}\s*\[\s*{extent_pattern}\s*\]",
            source,
        )
        if declaration and re.search(indexed_pattern, source):
            defects.append({
                "id": f"REGISTER_EXTENT_MISMATCH_{name.upper()}",
                "message": f"{name}[{declared_extent}] is indexed by a wider loop coordinate.",
            })
    return defects


def accumulator_layout_defects(source: str) -> list[dict]:
    defects = []
    for name in ("results", "accum"):
        accesses = [re.sub(r"\s+", "", value) for value in re.findall(rf"\b{name}\s*\[([^\]]+)\]", source)]
        if not accesses:
            continue
        has_full_wn_stride = any(re.search(r"\*WN(?:\+|$)", value) for value in accesses)
        has_packed_wn_stride = any("WN/WNITER*TN" in value or "WN_FLAT" in value for value in accesses)
        if has_full_wn_stride and has_packed_wn_stride:
            defects.append({
                "id": f"ACCUMULATOR_LAYOUT_MISMATCH_{name.upper()}",
                "message": f"{name} uses both full-WN and packed per-thread strides.",
            })
    return defects


def k_tile_coverage_defects(source: str) -> list[dict]:
    from SGPO.verification.targeted_cuda_repair import remove_duplicate_first_tile
    _, repeated = remove_duplicate_first_tile(source)
    if repeated:
        return [{"id": "DUPLICATE_FIRST_K_TILE_REDUCTION",
                 "message": "The identical tile-zero reduction runs before and inside the ping-pong loop; remove the redundant prologue, not the synchronized loop computation."}]
    main = anchor_body(source, "MAIN_LOOP")
    loop = re.search(r"for\s*\([^;]*bkIdx\s*=\s*1\s*;", main)
    if not loop:
        return []
    prefix = main[:loop.start()]
    accumulator_pattern = r"(?:results|accum|acc)\s*(?:\[[^\]]+\])+\s*\+="
    if re.search(accumulator_pattern, prefix):
        return []
    loop_body = main[loop.end():]
    first_accumulate = re.search(accumulator_pattern, loop_body)
    first_shared_overwrite = re.search(r"\b(?:As|Bs)\s*(?:\[[^\]]+\])+\s*=", loop_body)
    # Loading tile 1 into the other buffer does not overwrite tile 0. Recognize
    # this ping-pong form before applying the single-buffer overwrite heuristic.
    if first_accumulate:
        before = loop_body[:first_accumulate.start()]
        writes = re.findall(r'\b(?:As|Bs)\s*\[([^\]]+)\](?:\s*\[[^\]]+\])+\s*=', before)
        opposite = (re.search(r'const\s+int\s+comp_flag\s*=\s*\(bkIdx\s*-\s*1\)\s*&\s*1\s*;', before)
                    and re.search(r'const\s+int\s+mem_flag\s*=\s*bkIdx\s*&\s*1\s*;', before))
        immutable = all(len(re.findall(r'\b' + name + r'\s*=(?!=)', before)) == 1
                        and not re.search(r'\b' + name + r'\s*(?:\+=|-=|\+\+|--)', before)
                        for name in ('comp_flag', 'mem_flag'))
        reads = all(re.search(r'=\s*' + name + r'\s*\[comp_flag\]', before) for name in ('As', 'Bs'))
        drain = re.search(r'comp_flag\s*=\s*\(k_tiles\s*-\s*1\)\s*&\s*1', loop_body)
        if opposite and immutable and reads and writes and all(w.strip() == 'mem_flag' for w in writes) and drain:
            return []
    if first_accumulate and (not first_shared_overwrite or first_accumulate.start() < first_shared_overwrite.start()):
        return []
    return [{
        "id": "FIRST_K_TILE_NOT_ACCUMULATED",
        "message": "The K-tile loop starts at one but no accumulator update computes tile zero before it.",
    }]


def vector_mapping_is_proven(source: str) -> bool:
    # Recognize the coupled local-vector -> global-coordinate -> actual load
    # form, rather than requiring the legacy load_a_smem_k variable names.
    later = anchor_body(source, "NEXT_TILE_LOAD")
    # Expand immutable tile-base aliases only when there is no shadow declaration.
    # This accepts equivalent expressions without accepting shifted vector loads.
    for alias in re.findall(r"\bconst\s+int\s+(\w+)\s*=\s*bkIdx\s*\*\s*BK\s*;", later):
        if len(re.findall(r"\bint\s+" + re.escape(alias) + r"\b", later)) == 1:
            later = re.sub(r"\b" + re.escape(alias) + r"\s*\+", "bkIdx * BK +", later)
    later = re.sub(r"\bconst\s+int\b", "int", later)
    compact = re.sub(r"\s+", "", later)
    direct_a = (r"int(?P<ak>\w+)=\(\w+%\(BK/4\)\)\*4;"
                r"(?:const)?float4\w+=FLOAT4\(A\[\(tile_m0\+\w+\)\*K\+bkIdx\*BK\+(?P=ak)\]\);")
    direct_b = (r"int(?P<bn>\w+)=\(\w+%\(BN/4\)\)\*4;"
                r"(?:const)?float4\w+=FLOAT4\(B\[\(bkIdx\*BK\+\w+\)\*N\+tile_n0\+(?P=bn)\]\);")
    if re.search(direct_a, compact) and re.search(direct_b, compact) and len(re.findall(r"FLOAT4\([AB]\[", compact)) == 2:
        return True
    # Common next-tile vector loop with an immutable, unshifted K base.
    base = re.search(r"int(\w+)=\(bkIdx\+1\)\*BK;", compact)
    if base:
        name = base.group(1)
        original_const = re.search(r"\bconst\s+int\s+" + re.escape(name) + r"\s*=", anchor_body(source, "NEXT_TILE_LOAD"))
        unique = len(re.findall(r"\bint\s+" + re.escape(name) + r"\b", later)) == 1
        a = (r"int(?P<ak>\w+)=\(\w+%\(BK/4\)\)\*4;"
             r"(?:const)?float4\w+=FLOAT4\(A\[\(tile_m0\+\w+\)\*K\+" + re.escape(name) + r"\+(?P=ak)\]\);")
        b = (r"int(?P<bn>\w+)=\(\w+%\(BN/4\)\)\*4;"
             r"(?:const)?float4\w+=FLOAT4\(B\[\(" + re.escape(name) + r"\+\w+\)\*N\+tile_n0\+(?P=bn)\]\);")
        if original_const and unique and re.search(a, compact) and re.search(b, compact) and len(re.findall(r"FLOAT4\([AB]\[", compact)) == 2:
            return True
    a_pattern = (r"int(?P<ak>\w+)=\(\w+%\(BK/4\)\)\*4;"
                 r"int(?P<gm>\w+)=tile_m0\+\w+;"
                 r"int(?P<gk>\w+)=bkIdx\*BK\+(?P=ak);"
                 r"if\([^{};]+\)\{float4\w+=FLOAT4\(A\[OFFSET\((?P=gm),(?P=gk),K\)\]\);")
    b_pattern = (r"int(?P<bn>\w+)=\(\w+%\(BN/4\)\)\*4;"
                 r"int(?P<gk>\w+)=bkIdx\*BK\+\w+;"
                 r"int(?P<gn>\w+)=tile_n0\+(?P=bn);"
                 r"if\([^{};]+\)\{float4\w+=FLOAT4\(B\[OFFSET\((?P=gk),(?P=gn),N\)\]\);")
    if (re.search(a_pattern, compact) and re.search(b_pattern, compact)
            and len(re.findall(r"FLOAT4\([AB]\[", compact)) == 2):
        return True
    return bool(
        re.search(r"load_a_(?:smem|thread)_k\s*=\s*\([^;]*%\s*\(BK\s*/\s*4\)\)\s*\*\s*4", source)
        and re.search(r"(?:load_b_smem_n|load_b_thread_n|b_col)\s*=\s*\([^;]*%\s*\(BN\s*/\s*4\)\)\s*\*\s*4", source)
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
        "REGISTER_DECL": [
            "float results[WM / WMITER * TM][WN / WNITER * TN] = {0.0f};",
            "float regM[TM] = {0.0f};",
            "float regN[TN] = {0.0f};",
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
    comments = list(re.finditer(r"/\*[\s\S]*?\*/|//[^\n]*", source))
    begin = next((c for c in comments if f"{region}_BEGIN" in c[0]), None)
    if begin is None:
        return ""
    end = next((c for c in comments if c.start() >= begin.end() and f"{region}_END" in c[0]), None)
    return source[begin.end():end.start()] if end else ""


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
