"""Narrow repairs for diagnosed tiled float4 load schedules; preserve compute."""
import re
from SGPO.verification.gemm_semantic_checker import extract_launch_config


def repair_cooperative_loads(source, ir):
    config = extract_launch_config(source)
    p = ir.get("problem", {})
    required = ("BM", "BN", "BK", "WM", "WN")
    if any(not isinstance(config.get(k), int) or config[k] <= 0 for k in required):
        return source, []
    bm, bn, bk, wm, wn = (config[k] for k in required)
    if bm % wm or bn % wn or bk % 4 or bn % 4:
        return source, []
    threads = bm // wm * (bn // wn) * 32
    if not all(isinstance(p.get(k), int) and p[k] > 0 and p[k] % tile == 0
               for k, tile in (("M", bm), ("N", bn), ("K", bk))):
        return source, []
    if p.get("layout_A") != "row_major" or p.get("trans_A") is not False:
        return source, []
    if p.get("layout_B", "row_major") != "row_major" or p.get("trans_B", False):
        return source, []
    # Only handle these known unguarded, full-tile schedules. Other forms go to LLM.
    if "const int tid = threadIdx.x;" not in source:
        return source, []
    launch = re.sub(r"\s+", "", source)
    if "dim3threadsPerBlock((BM*BN)/(WM*WN)*32);" not in launch:
        return source, []
    changes = []
    # A one-vector-per-thread guard cannot cover a larger tile. Keep the
    # existing address, scatter, and boundary code; only distribute all vectors.
    for operand, count in (("a", bm * bk // 4), ("b", bk * bn // 4)):
        count_name = operand + "_vec_count"
        expression = "BM * (BK / 4)" if operand == "a" else "BK * (BN / 4)"
        declaration = f"const int {count_name} = {expression};"
        pattern = rf"int v = tid;\s*if \(v < {count_name}\) \{{"
        if count > threads and declaration in source and "const int num_threads = blockDim.x;" in source:
            source, replaced = re.subn(pattern,
                f"for (int v = tid; v < {count_name}; v += num_threads) {{", source)
            if replaced:
                changes.append({"id": "VECTOR_TILE_COVERAGE_LOOP", "operand": operand,
                                "sites": replaced, "vectors": count, "threads": threads})
    scalar_schedule = {
        "const int load_a_smem_m = tid % BM;": "const int load_a_smem_m = tid / (BK / 4);",
        "const int load_a_smem_k = (tid / BM) % BK;": "const int load_a_smem_k = (tid % (BK / 4)) * 4;",
        "const int load_b_smem_n = tid % BN;": "const int load_b_smem_n = (tid % (BN / 4)) * 4;",
        "const int load_b_smem_k = (tid / BN) % BK;": "const int load_b_smem_k = tid / (BN / 4);",
        "CEIL_DIV(BM * BK, thread_num)": "CEIL_DIV(BM * BK / 4, thread_num)",
        "CEIL_DIV(BN * BK, thread_num)": "CEIL_DIV(BN * BK / 4, thread_num)",
        "loadOffset * (thread_num / BK)": "loadOffset * (thread_num / (BK / 4))",
        "loadOffset * (thread_num / BN)": "loadOffset * (thread_num / (BN / 4))",
    }
    if (all(s in source for s in scalar_schedule) and threads % (bk // 4) == 0
            and threads % (bn // 4) == 0 and "FLOAT4(A[OFFSET(gm, gk, K)])" in source
            and "FLOAT4(B[OFFSET(gk, gn, N)])" in source
            and "As[mem_flag][a_k + 3][a_m] = tmp.w;" in source
            and "Bs[mem_flag][b_k][b_n + 3] = bv.w;" in source):
        for old, new in scalar_schedule.items():
            source = source.replace(old, new)
        changes.append({"id": "SCALAR_SCHEDULE_USED_FOR_FLOAT4", "threads": threads})
    replacements = {
        "const int load_a_steps = (BM * BK) / (BK * 4);": f"const int load_a_steps = (BM * BK / 4) / {threads};",
        "const int load_b_steps = (BK * BN) / (BN * 4);": f"const int load_b_steps = (BK * BN / 4) / {threads};",
        "load_a_smem_m + step * (BK / 4)": f"load_a_smem_m + step * ({threads} / (BK / 4))",
        "load_b_smem_k + step * (BN / 4)": f"load_b_smem_k + step * ({threads} / (BN / 4))",
    }
    step_signature = all(text in source for text in replacements)
    if step_signature and bm * bk // 4 % threads == 0 and bk * bn // 4 % threads == 0:
        for old, new in replacements.items():
            source = source.replace(old, new)
        changes.append({"id": "COOPERATIVE_LOAD_STEP_COUNT_AND_STRIDE",
                        "threads": threads, "a_vectors": bm*bk//4, "b_vectors": bk*bn//4})

    if (bm * bk // 4 == threads and
        "const int load_a_thread_k = tid / (BM / 4);" in source and
        "const int load_a_thread_m = (tid % (BM / 4)) * 4;" in source and
        re.search(r"As\[(?:0|mem_flag)\]\[(?:a_k|load_a_thread_k) \+ 3\]\[a_m\] = tmp.w;", source)):
        source = source.replace("const int load_a_thread_k = tid / (BM / 4);",
                                "const int load_a_thread_k = (tid % (BK / 4)) * 4;")
        source = source.replace("const int load_a_thread_m = (tid % (BM / 4)) * 4;",
                                "const int load_a_thread_m = tid / (BK / 4);")
        changes.append({"id": "A_FLOAT4_K_GROUP_MAPPING"})

    # Recognize complete leaf blocks, not arbitrary code between load anchors.
    pattern = re.compile(r"\{\s*int b_k = (?P<k>[^;]+);\s*int b_n = (?P<n>[^;]+);"
                         r"(?P<body>[^{}]+?)\}")
    def fix_b(match):
        k, n, body = match['k'].strip(), match['n'].strip(), match['body']
        offset = None
        if (k, n) in (("load_b_smem_k", "tile_n0 + load_b_smem_n"),
                       ("load_b_thread_k", "load_b_thread_n")):
            offset = "0"
        elif (k, n) in (("bkIdx * BK + load_b_smem_k", "tile_n0 + load_b_smem_n"),
                         ("load_b_thread_k + bkIdx * BK", "load_b_thread_n")):
            offset = "bkIdx * BK"
        if offset is None:
            return match[0]
        stage = "0" if offset == "0" else "mem_flag"
        compact = re.sub(r"\s+", "", body)
        direct = f"FLOAT4(Bs[{stage}][{'b_k' if stage == '0' else 'load_b_thread_k'}][b_n])=FLOAT4(B[B_base+b_k*N+b_n]);"
        components = "float4tmp=FLOAT4(B[b_k*N+b_n]);" + "".join(
            f"Bs[{stage}][load_b_smem_k][load_b_smem_n{('+'+str(u)) if u else ''}]=tmp.{c};"
            for u, c in enumerate("xyzw"))
        if compact not in (direct, components):
            return match[0]
        if compact == components and bk * bn // 4 <= threads:
            return match[0]
        changes.append({"id": "B_FLOAT4_COMPLETE_COVERAGE_AND_PADDED_SCATTER",
                        "stage": stage, "required_vectors": bk*bn//4,
                        "previous_vectors": threads})
        return f'''{{
            for (int b_vec = tid; b_vec < BK * (BN / 4); b_vec += {threads}) {{
                const int b_row = b_vec / (BN / 4);
                const int b_col = (b_vec % (BN / 4)) * 4;
                const int b_global_k = ({offset}) + b_row;
                const int b_global_n = tile_n0 + b_col;
                const float4 b_value = FLOAT4(B[b_global_k * N + b_global_n]);
                Bs[{stage}][b_row][b_col] = b_value.x;
                Bs[{stage}][b_row][b_col + 1] = b_value.y;
                Bs[{stage}][b_row][b_col + 2] = b_value.z;
                Bs[{stage}][b_row][b_col + 3] = b_value.w;
            }}
        }}'''
    source = pattern.sub(fix_b, source)
    return source, changes


def diagnose_cooperative_loads(source, ir):
    _, changes = repair_cooperative_loads(source, ir)
    return {"status": "defects_found" if changes else "no_supported_defect_detected",
            "scope": "recognized full-tile cooperative float4 schedules only; not a complete memory proof",
            "defects": changes}
