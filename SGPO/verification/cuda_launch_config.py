"""Materialize host-side shared-memory attributes without changing the GEMM body."""
from __future__ import annotations

import re

from SGPO.generate_ir.gpu_resources import get, proposed_pipeline_bytes, pipeline_stages, shared_memory_limit


BEGIN = "// SGPO_SHARED_LAUNCH_BEGIN"
END = "// SGPO_SHARED_LAUNCH_END"


def configure_shared_launch(source: str, ir: dict) -> str:
    dynamic = get(ir, "memory.shared_memory_allocation") == "dynamic"
    carveout = get(ir, "memory.shared_memory_carveout_percent")
    if not dynamic and carveout is None:
        return source
    source = re.sub(re.escape(BEGIN) + r"[\s\S]*?" + re.escape(END) + r"\n?", "", source)
    # Only the standard wrapper is supported; ambiguity is reported instead of guessing.
    wrapper = re.search(r"\bvoid\s+cuda_gemm\s*\([^;{}]*\)\s*\{", source)
    if wrapper is None:
        raise ValueError("Shared-memory configuration requires the standard cuda_gemm wrapper")
    launches = list(re.finditer(r"(?P<kernel>\b[A-Za-z_]\w*(?:<[^;{}]+?>)?)\s*<<<(?P<config>[^>]+)>>>", source[wrapper.end():]))
    if len(launches) != 1:
        raise ValueError("Shared-memory configuration requires exactly one GEMM launch in cuda_gemm")
    match = launches[0]
    kernel = match.group("kernel")
    config = match.group("config").split(",")
    if len(config) not in (2, 3, 4) or any("(" in part or ")" in part for part in config):
        raise ValueError("Use named grid/block/stream variables for shared-memory launch materialization")
    statements = [BEGIN, "{"]
    attributes = []
    if dynamic:
        code = re.sub(r"//[^\n]*|/\*[\s\S]*?\*/", " ", source)
        if not re.search(r"extern\s+__shared__\s+", code):
            raise ValueError("Dynamic shared policy requires a real extern __shared__ allocation")
        size = proposed_pipeline_bytes(ir, pipeline_stages(ir))
        if size <= 0 or (shared_memory_limit(ir) and size > shared_memory_limit(ir)):
            raise ValueError(f"Invalid dynamic shared allocation: {size} bytes")
        if len(config) == 2:
            config.append(str(size))
        else:
            config[2] = str(size)
        if get(ir, "memory.shared_memory_optin") is True:
            attributes.append(("cudaFuncAttributeMaxDynamicSharedMemorySize", size))
    if carveout is not None:
        if type(carveout) is not int or not 0 <= carveout <= 100:
            raise ValueError("Shared-memory carveout must be an integer percentage in [0, 100]")
        attributes.append(("cudaFuncAttributePreferredSharedMemoryCarveout", carveout))
    for index, (attribute, value) in enumerate(attributes):
        statements += [f"  cudaError_t sgpo_attr_{index} = cudaFuncSetAttribute({kernel}, {attribute}, {value});",
                       f"  if (sgpo_attr_{index} != cudaSuccess) {{",
                       f'    fprintf(stderr, "CUDA error: %s\\n", cudaGetErrorString(sgpo_attr_{index}));',
                       "    return;", "  }"]
    statements += ["}", END]
    start, end = wrapper.end() + match.start(), wrapper.end() + match.end()
    result = source[:start] + "\n".join(statements) + "\n" + kernel + "<<<" + ",".join(config) + ">>>" + source[end:]
    if "#include <stdio.h>" not in result:
        result = "#include <stdio.h>\n" + result
    return result
