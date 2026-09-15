from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from SGPO.utils.common_utils import load_json, save_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IR = ROOT / "data" / "IRs" / "ir_patch" / "optir.verified.json"
DEFAULT_POSTCHECK = ROOT / "results" / "check" / "post_check_result.json"
DEFAULT_OUTPUT = ROOT / "results" / "defect" / "defect_diagnosis.json"
DEFAULT_OUTPUT_IR = ROOT / "data" / "IRs" / "ir_patch" / "optir.diagnosed.json"


RUNTIME_ERROR_RULES = [
    {
        "patterns": ["compute-sanitizer racecheck failed", "compute-sanitizer synccheck failed"],
        "defect_type": "Pipeline.AsyncSynchronizationViolation",
        "related_fields": ["pipeline.stage_count", "pipeline.async_copy", "synchronization.sync_policy"],
        "repair_action": "inspect the sanitizer report; correct commit/wait ordering, uniform CTA synchronization and stage-buffer reuse, then rerun; do not repair by merely disabling the checker",
    },
    {
        "patterns": ["returncode=-11", "returncode=139", "segmentation fault"],
        "defect_type": "Runtime.HostSegmentationFault",
        "related_fields": [
            "verification.runtime_safety.returncode",
            "memory.global_load_A",
            "memory.global_load_B",
            "memory.global_store_C",
            "mapping.thread_to_output",
        ],
        "repair_action": "repair generated kernel indexing and boundary guards, then rerun with CUDA launch/error checks enabled",
    },
    {
        "patterns": ["cublas status"],
        "defect_type": "Runtime.CUBLASFailure",
        "related_fields": [
            "verification.runtime_safety.host_error",
            "hardware.cuda_driver_version",
            "hardware.cuda_runtime_version",
            "baseline.cublas",
        ],
        "repair_action": "check cuBLAS initialization and call status; if CUDA driver/runtime is invalid, fix the environment before evaluating generated kernels",
    },
    {
        "patterns": ["unsupported display driver", "driver/library version mismatch", "cuda driver version is insufficient"],
        "defect_type": "Runtime.CUDADriverRuntimeMismatch",
        "related_fields": [
            "hardware.cuda_driver_version",
            "hardware.cuda_runtime_version",
            "build.resolved_arch",
            "verification.runtime_safety.host_error",
        ],
        "repair_action": "use a container CUDA runtime compatible with the host NVIDIA driver, or rebuild/run in an environment with matching driver/runtime libraries",
    },
    {
        "patterns": ["illegal instruction"],
        "defect_type": "CPU.Runtime.UnsupportedISA",
        "related_fields": [
            "hardware.cpu_isa",
            "cpu_vectorization.isa",
            "cpu_compiler.vector_isa",
            "strategy.applied_strategy_ids",
        ],
        "repair_action": "replace unsupported SIMD strategy with a hardware-supported ISA such as AVX2/FMA, or remove ISA-specific intrinsics",
    },
    {
        "patterns": ["access violation"],
        "defect_type": "CPU.Runtime.MemoryAccessViolation",
        "related_fields": [
            "problem.boundary_guard",
            "cpu_tiling",
            "cpu_memory.pack_a",
            "cpu_memory.pack_b",
        ],
        "repair_action": "repair CPU indexing, tile-tail guards, and packed-buffer bounds before rerunning",
    },
    {
        "patterns": ["misaligned address"],
        "defect_type": "Vectorization.AlignmentViolation",
        "related_fields": [
            "vectorization.A.vector_width",
            "vectorization.A.alignment_guard",
            "vectorization.A.alignment_proven",
            "vectorization.B.vector_width",
            "vectorization.B.alignment_guard",
            "vectorization.B.alignment_proven",
            "vectorization.C.vector_width",
            "vectorization.C.alignment_guard",
            "vectorization.C.alignment_proven",
        ],
        "repair_action": "inspect every float4 access in GLOBAL_TO_SHARED_LOAD, NEXT_TILE_LOAD and STORE; "
                         "repair vector-aware INDEX_MAPPING and strides, add actual alignment guards with scalar fallback "
                         "or reduce vector width in BOTH initial and subsequent tile loads; "
                         "keep all shared-memory writes and compute reads consistent with SHARED_DECL dimensions",
    },
    {
        "patterns": ["illegal memory access", "out of bounds"],
        "defect_type": "Memory.OutOfBounds",
        "related_fields": [
            "problem.boundary_guard",
            "memory.global_load_A.boundary_guard",
            "memory.global_load_B.boundary_guard",
            "memory.global_store_C.boundary_guard",
            "mapping.grid_x",
            "mapping.grid_y",
        ],
        "repair_action": "restore boundary guards and verify index mapping",
    },
    {
        "patterns": ["invalid configuration argument"],
        "defect_type": "Resource.InvalidLaunchConfiguration",
        "related_fields": [
            "mapping.block_dim_x",
            "mapping.block_dim_y",
            "mapping.block_dim_z",
            "mapping.threads_per_block",
            "hardware.max_threads_per_block",
        ],
        "repair_action": "reduce launch dimensions or increase thread tile size",
    },
    {
        "patterns": ["out of memory"],
        "defect_type": "Resource.OutOfMemory",
        "related_fields": [
            "hardware.global_memory_bytes",
            "resource.shared_memory.total_bytes",
            "memory.shared_A.shape",
            "memory.shared_B.shape",
        ],
        "repair_action": "reduce tile size or shared-memory allocation",
    },
]


CHECK_FAILURE_RULES = {
    "Pipeline.AsyncCopyProtocolMissing": {
        "related_fields": ["pipeline.async_copy", "pipeline.stage_count", "resource.shared_memory"],
        "repair_action": "materialize cp.async or an equivalent CUDA async-copy API with commit/wait; preserve alignment, tail zero-fill, CTA-wide synchronization before consumption and before buffer reuse; handle short K and drain outstanding copies, or fall back to fewer stages/synchronous buffering",
    },
    "Resource.ThreadBlockOverflow": {
        "related_fields": [
            "tiling.block_m",
            "tiling.block_n",
            "tiling.thread_m",
            "tiling.thread_n",
            "mapping.threads_per_block",
            "hardware.max_threads_per_block",
        ],
        "repair_action": "increase thread_m/thread_n or reduce block_m/block_n so threads_per_block fits hardware",
    },
    "Resource.SharedMemoryOverflow": {
        "related_fields": [
            "resource.shared_memory.total_bytes",
            "hardware.max_shared_memory_per_block_bytes",
            "memory.shared_A.shape",
            "memory.shared_B.shape",
            "tiling.block_m",
            "tiling.block_n",
            "tiling.block_k",
        ],
        "repair_action": "reduce block tile size, reduce block_k, or remove shared-memory buffering",
    },
    "StrategyCompliance.TilingNotApplied": {
        "related_fields": [
            "tiling.enabled",
            "tiling.block_m",
            "tiling.block_n",
            "tiling.block_k",
        ],
        "repair_action": "set tiling.enabled and fill block_m/block_n/block_k",
    },
    "Memory.OutOfBounds": {
        "related_fields": [
            "memory.global_load_A.boundary_guard",
            "memory.global_load_B.boundary_guard",
            "memory.global_store_C.boundary_guard",
            "problem.boundary_guard",
        ],
        "repair_action": "add or restore boundary guards for global load/store",
    },
    "Vectorization.AlignmentViolation": {
        "related_fields": [
            "vectorization.A.vector_width",
            "vectorization.A.alignment_guard",
            "vectorization.A.alignment_proven",
            "vectorization.B.vector_width",
            "vectorization.B.alignment_guard",
            "vectorization.B.alignment_proven",
            "vectorization.C.vector_width",
            "vectorization.C.alignment_guard",
            "vectorization.C.alignment_proven",
        ],
        "repair_action": "add alignment guard/proof or reduce vector_width",
    },
    "Vectorization.TailHandlingError": {
        "related_fields": [
            "vectorization.A.tail_handling",
            "vectorization.B.tail_handling",
            "vectorization.C.tail_handling",
            "problem.M",
            "problem.N",
            "problem.K",
        ],
        "repair_action": "add vector tail handling or fall back to scalar path for tails",
    },
    "CPU.Semantic.SIMDLaneMappingViolation": {
        "related_fields": [
            "cpu_microkernel.family",
            "cpu_microkernel.mr",
            "cpu_microkernel.nr",
            "cpu_vectorization.policy",
            "cpu_vectorization.isa",
            "cpu_memory.pack_b_layout",
        ],
        "repair_action": "rewrite CPU SIMD micro-kernel so vector lanes map to contiguous N outputs, scalar A is broadcast, and K remains the reduction loop",
    },
    "CPU.Parallel.SharedPackedBufferRace": {
        "related_fields": [
            "cpu_parallel.policy",
            "cpu_memory.pack_a",
            "cpu_memory.pack_b",
            "cpu_memory.pack_layout",
        ],
        "repair_action": "make packed A/B panel buffers thread-private, allocate them inside the OpenMP tile body, or allocate one buffer per worker",
    },
    "CPU.StrategyImplementation.AVX512Missing": {
        "related_fields": [
            "cpu_vectorization.isa",
            "cpu_microkernel.family",
            "cpu_microkernel.mr",
            "cpu_microkernel.nr",
        ],
        "repair_action": "materialize the selected AVX512 strategy with _mm512_broadcastss_ps, _mm512_loadu_ps, _mm512_fmadd_ps, and _mm512_storeu_ps in the compute micro-kernel",
    },
    "CPU.StrategyImplementation.AVX2Missing": {
        "related_fields": [
            "cpu_vectorization.isa",
            "cpu_microkernel.family",
            "cpu_microkernel.mr",
            "cpu_microkernel.nr",
        ],
        "repair_action": "materialize the selected AVX2 strategy with _mm256_broadcast_ss, _mm256_loadu_ps, _mm256_fmadd_ps, and _mm256_storeu_ps in the compute micro-kernel",
    },
    "CPU.StrategyImplementation.OpenMPMissing": {
        "related_fields": [
            "cpu_parallel.policy",
            "cpu_parallel.num_threads",
            "cpu_loop_schedule.parallel_loop",
        ],
        "repair_action": "add a real #pragma omp parallel or parallel for over independent output tiles without overlapping C writes",
    },
    "CPU.StrategyImplementation.PackingMissingOrUnused": {
        "related_fields": [
            "cpu_memory.pack_a",
            "cpu_memory.pack_b",
            "cpu_memory.pack_layout",
            "cpu_packing.panel_layout",
        ],
        "repair_action": "allocate and fill packed A/B panels, then make the micro-kernel consume the packed buffers instead of reading raw A/B directly",
    },
    "CPU.StrategyImplementation.PanelDriverMissing": {
        "related_fields": [
            "cpu_loop_schedule.macro_kernel",
            "cpu_memory.pack_a",
            "cpu_memory.pack_b",
            "cpu_tiling.l2_block_m",
            "cpu_tiling.l2_block_n",
            "cpu_tiling.l2_block_k",
        ],
        "repair_action": "emit an OpenBLAS-style macro-kernel with N-panel, K-panel, M-panel loops and packed panels feeding an MR x NR micro-kernel",
    },
    "CPU.StrategyImplementation.MicroKernelMissing": {
        "related_fields": [
            "cpu_microkernel.family",
            "cpu_microkernel.mr",
            "cpu_microkernel.nr",
            "cpu_vectorization.policy",
        ],
        "repair_action": "replace plain scalar loops with an explicit MR x NR register-blocked micro-kernel using the selected SIMD or pragma-SIMD compute policy",
    },
    "StrategyCompliance.UnknownCondition": {
        "related_fields": [
            "strategy.postconditions",
            "strategy.missing_postconditions",
            "verification.strategy_compliance",
        ],
        "repair_action": "replace natural-language postcondition with explicit IR field or add a dedicated checker",
    },
}


def diagnose_defects(ir: dict[str, Any], post_check: dict[str, Any] | None = None) -> dict[str, Any]:
    defects = []
    strategy_id = get_related_strategy(ir)

    defects.extend(diagnose_post_check(post_check or ir.get("post_check"), strategy_id))
    defects.extend(diagnose_runtime(ir, strategy_id))
    defects.extend(diagnose_compile(ir, strategy_id))
    defects.extend(diagnose_code_semantics(ir, strategy_id))
    defects.extend(diagnose_correctness(ir, strategy_id))
    for report in (ir.get("code_verification", {}).get("strategy_reports", []) or []):
        defects.extend(diagnose_post_check(report, report.get("strategy_id") or strategy_id))

    unique_defects = deduplicate_defects(defects)
    return {
        "stage": "Defect Diagnosis",
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "related_strategy": strategy_id,
        "defect_count": len(unique_defects),
        "defects": unique_defects,
    }


def diagnose_post_check(post_check: dict[str, Any] | None, strategy_id: str | None) -> list[dict[str, Any]]:
    if not post_check:
        return []
    defects = []
    for item in post_check.get("results", []) or []:
        if item.get("status") not in {"fail", "unknown"}:
            continue
        failure_type = item.get("failure_type") or infer_failure_type_from_check_id(item.get("id", ""))
        rule = CHECK_FAILURE_RULES.get(failure_type, {})
        defects.append(
            make_defect(
                defect_type=failure_type or "StrategyCompliance.CheckFailed",
                related_strategy=strategy_id,
                related_fields=rule.get("related_fields", []),
                repair_action=rule.get("repair_action", "inspect failed IR check and repair affected fields"),
                evidence={
                    "source": "post_check",
                    "check_id": item.get("id"),
                    "status": item.get("status"),
                    "message": item.get("message"),
                    "detail": item.get("detail"),
                },
            )
        )
    return defects


def diagnose_runtime(ir: dict[str, Any], strategy_id: str | None) -> list[dict[str, Any]]:
    runtime = ir.get("verification", {}).get("runtime_safety", {})
    cuda_error = runtime.get("cuda_error")
    host_error = runtime.get("host_error")
    runtime_error = cuda_error or host_error
    returncode = runtime.get("returncode")
    if not runtime_error:
        if runtime.get("status") != "fail" and returncode in (None, 0):
            return []
        defect_type = "Runtime.HostSegmentationFault" if returncode in {-11, 139} else "Runtime.ProcessCrashed"
        return [
            make_defect(
                defect_type=defect_type,
                related_strategy=strategy_id,
                related_fields=[
                    "verification.runtime_safety.returncode",
                    "memory.global_load_A",
                    "memory.global_load_B",
                    "memory.global_store_C",
                    "mapping.thread_to_output",
                    "tiling.block_m",
                    "tiling.block_n",
                    "tiling.block_k",
                ],
                repair_action=(
                    "repair generated kernel indexing and boundary guards; add post-kernel cudaGetLastError/cudaDeviceSynchronize "
                    "so CUDA-side illegal access is captured before the host process crashes"
                ),
                evidence={
                    "source": "runtime_safety",
                    "cuda_error": cuda_error,
                    "host_error": host_error,
                    "runtime_safety": runtime,
                    "returncode": returncode,
                },
            )
        ]
    lowered = runtime_error.lower()
    for rule in RUNTIME_ERROR_RULES:
        if any(pattern in lowered for pattern in rule["patterns"]):
            return [
                make_defect(
                    defect_type=rule["defect_type"],
                    related_strategy=strategy_id,
                    related_fields=rule["related_fields"],
                    repair_action=rule["repair_action"],
                    evidence={
                        "source": "runtime_safety",
                        "cuda_error": cuda_error,
                        "host_error": host_error,
                        "runtime_safety": runtime,
                    },
                )
            ]
    return [
        make_defect(
            defect_type="Runtime.ExecutionError",
            related_strategy=strategy_id,
            related_fields=[],
            repair_action="inspect runtime error and map to a specific repair rule",
            evidence={"source": "runtime_safety", "cuda_error": cuda_error, "host_error": host_error},
        )
    ]


def diagnose_compile(ir: dict[str, Any], strategy_id: str | None) -> list[dict[str, Any]]:
    compile_node = ir.get("verification", {}).get("compile", {})
    if compile_node.get("status") != "fail":
        return []
    message = compile_node.get("error_message") or ""
    lowered = message.lower()
    if "threads_per_block must not exceed" in lowered or "too many resources requested" in lowered:
        defect_type = "Resource.ThreadBlockOverflow"
        rule = CHECK_FAILURE_RULES[defect_type]
    elif "has already been declared" in lowered or "already been declared in the current scope" in lowered:
        defect_type = "Compile.CUDAScopeRedeclaration"
        rule = {
            "related_fields": ["patch_generation.code_patch", "code.region_scope", "mapping.index_variables"],
            "repair_action": "remove duplicate local declarations or rename variables; each skeleton region must define tid/lane/tile indices only once in a single scope",
        }
    elif "pointer-to-object type" in lowered and "type \"float\"" in lowered:
        defect_type = "Compile.SharedMemoryRankMismatch"
        rule = {
            "related_fields": ["memory.shared_A.shape", "memory.shared_B.shape", "code.shared_memory_indexing"],
            "repair_action": "make shared-memory declaration rank match all uses; do not bind a scalar element such as As[x][y] to a pointer-like variable and index it again",
        }
    elif "size of an array must be greater than zero" in lowered or "variable-sized object may not be initialized" in lowered:
        defect_type = "Compile.InvalidStaticArrayShape"
        rule = {
            "related_fields": ["tiling.warp_tile.warp_m_iter", "tiling.warp_tile.warp_n_iter", "tiling.thread_m", "tiling.thread_n", "register.accumulator"],
            "repair_action": "choose WMITER/WNITER so WM/WMITER and WN/WNITER are positive integers, and use compile-time constant accumulator extents",
        }
    elif "identifier" in lowered and ("is undefined" in lowered or "was not declared" in lowered):
        defect_type = "Compile.UndeclaredKernelSymbol"
        rule = {
            "related_fields": ["patch_generation.code_patch", "code.region_scope", "compute.k_loop", "register.A", "register.B"],
            "repair_action": "define the symbol in the same scope before use, or replace it with the canonical shared-memory/register variable used by the current kernel template",
        }
    elif "identifier" in lowered or "was not declared" in lowered:
        defect_type = "SyntaxOrAPI.UndeclaredIdentifier"
        rule = {
            "related_fields": ["patch_generation.code_patch"],
            "repair_action": "fix generated code symbols or include required header/API",
        }
    else:
        defect_type = "Compile.CUDACompilationError"
        rule = {
            "related_fields": ["patch_generation.code_patch"],
            "repair_action": "repair generated CUDA/C++ patch using compiler diagnostics",
        }
    return [
        make_defect(
            defect_type=defect_type,
            related_strategy=strategy_id,
            related_fields=rule["related_fields"],
            repair_action=rule["repair_action"],
            evidence={
                "source": "compile",
                "status": compile_node.get("status"),
                "error_message": message,
                "returncode": compile_node.get("returncode"),
            },
        )
    ]


def diagnose_correctness(ir: dict[str, Any], strategy_id: str | None) -> list[dict[str, Any]]:
    correctness = ir.get("verification", {}).get("correctness", {})
    if correctness.get("status") != "fail":
        return []
    return [
        make_defect(
            defect_type="Semantic.CorrectnessMismatch",
            related_strategy=strategy_id,
            related_fields=[
                "problem.indexing_rule",
                "mapping.thread_to_output",
                "tiling.block_m",
                "tiling.block_n",
                "tiling.block_k",
            ],
            repair_action="check index mapping, reduction loop coverage, accumulator initialization, and C store mapping",
            evidence={
                "source": "correctness",
                "max_abs_error": correctness.get("max_abs_error"),
                "max_rel_error": correctness.get("max_rel_error"),
                "error_message": correctness.get("error_message"),
            },
        )
    ]


def diagnose_code_semantics(ir: dict[str, Any], strategy_id: str | None) -> list[dict[str, Any]]:
    semantic = (
        ir.get("code_completeness", {})
        .get("semantic_obligations", {})
    )
    defects = []
    defects.extend(diagnose_code_completeness_results(ir.get("code_completeness", {}), strategy_id))
    for item in semantic.get("results", []) or []:
        if item.get("status") != "fail":
            continue
        defect_type = item.get("failure_type") or "GEMM.Semantic.ObligationViolation"
        defects.append(
            make_defect(
                defect_type=defect_type,
                related_strategy=strategy_id,
                related_fields=related_fields_for_semantic_defect(defect_type),
                repair_action=item.get("repair_action") or "repair generated GEMM code to satisfy the failed semantic obligation",
                evidence={
                    "source": "code_semantic_obligation",
                    "check_id": item.get("id"),
                    "message": item.get("message"),
                    "detail": item.get("detail"),
                    "launch_config": semantic.get("launch_config"),
                },
            )
        )
    return defects


def diagnose_code_completeness_results(code_completeness: dict[str, Any], strategy_id: str | None) -> list[dict[str, Any]]:
    defects = []
    for item in code_completeness.get("results", []) or []:
        if item.get("status") != "fail":
            continue
        defect_type = item.get("failure_type")
        if not defect_type:
            continue
        rule = CHECK_FAILURE_RULES.get(defect_type, {})
        defects.append(
            make_defect(
                defect_type=defect_type,
                related_strategy=strategy_id,
                related_fields=rule.get("related_fields", related_fields_for_semantic_defect(defect_type)),
                repair_action=item.get("repair_action") or rule.get("repair_action", "repair failed code completeness obligation"),
                evidence={
                    "source": "code_completeness",
                    "check_id": item.get("id"),
                    "message": item.get("message"),
                    "detail": item.get("detail"),
                },
            )
        )
    return defects


def related_fields_for_semantic_defect(defect_type: str) -> list[str]:
    if defect_type in {"Compile.CUDAStaticHazard", "Compile.CUDAScopeRedeclaration"}:
        return ["patch_generation.code_patch", "code.region_scope", "mapping.index_variables"]
    if defect_type == "Compile.SharedMemoryRankMismatch":
        return ["memory.shared_A.shape", "memory.shared_B.shape", "code.shared_memory_indexing"]
    if defect_type == "Compile.InvalidStaticArrayShape":
        return [
            "tiling.warp_tile.warp_m",
            "tiling.warp_tile.warp_n",
            "tiling.warp_tile.warp_m_iter",
            "tiling.warp_tile.warp_n_iter",
            "tiling.thread_m",
            "tiling.thread_n",
            "register.accumulator",
        ]
    if defect_type == "Compile.UndeclaredKernelSymbol":
        return ["patch_generation.code_patch", "code.region_scope", "compute.k_loop", "register.A", "register.B"]
    if defect_type == "GEMM.Semantic.ThreadTileMappingHasHoles":
        return ["tiling.thread_m", "tiling.thread_n", "tiling.block_n", "mapping.thread_to_output", "mapping.group_to_tile"]
    if defect_type == "GEMM.Semantic.WarpLaneFragmentCoverageIncomplete":
        return [
            "tiling.warp_tile.warp_m",
            "tiling.warp_tile.warp_n",
            "tiling.warp_tile.warp_m_iter",
            "tiling.warp_tile.warp_n_iter",
            "tiling.thread_m",
            "tiling.thread_n",
            "hardware.warp_size",
            "mapping.lane_layout",
        ]
    if defect_type == "GEMM.Semantic.ThreadTileStoreIncomplete":
        return ["tiling.thread_m", "tiling.thread_n", "mapping.thread_to_output", "memory.global_store_C"]
    if defect_type == "GEMM.Semantic.RegisterTileComputeIncomplete":
        return ["tiling.thread_m", "tiling.thread_n", "register.accumulator", "compute.inner_loop"]
    if defect_type == "GEMM.Semantic.KLoopDataflowIncomplete":
        return ["tiling.block_k", "memory.global_load_A", "memory.global_load_B", "compute.k_loop"]
    if defect_type == "GEMM.Semantic.KTileTailDropped":
        return ["problem.K", "tiling.block_k", "safety.boundary_policy", "memory.global_load_A.boundary_guard", "memory.global_load_B.boundary_guard"]
    if defect_type == "GEMM.Semantic.VectorizedFastPathWithoutProof":
        return [
            "problem.M",
            "problem.N",
            "problem.K",
            "tiling.block_m",
            "tiling.block_n",
            "tiling.block_k",
            "vectorization.A.alignment_proven",
            "vectorization.B.alignment_proven",
            "vectorization.C.alignment_proven",
            "safety.boundary_policy",
        ]
    if defect_type == "GEMM.Semantic.UninitializedRegisterTile":
        return ["tiling.block_k", "register.A", "compute.inner_loop"]
    if defect_type == "GEMM.Semantic.SharedLoadNotCooperative":
        return ["memory.shared_A", "memory.shared_B", "mapping.threads_per_block", "tiling.block_m", "tiling.block_n", "tiling.block_k"]
    if defect_type == "GEMM.Performance.RepeatedSharedLoadPerOutputGroup":
        return ["memory.shared_A", "memory.shared_B", "compute.k_loop", "mapping.thread_to_output", "synchronization.syncthreads"]
    if defect_type == "GEMM.Semantic.SharedTileNotUsedInCompute":
        return ["memory.shared_A", "memory.shared_B", "compute.inner_loop"]
    if defect_type == "GEMM.Semantic.FallbackPathOverlapsOptimizedPath":
        return ["patch_generation.code_patch", "compute.kernel_body"]
    if defect_type == "CPU.Semantic.SIMDLaneMappingViolation":
        return [
            "cpu_microkernel.family",
            "cpu_microkernel.mr",
            "cpu_microkernel.nr",
            "cpu_vectorization.policy",
            "cpu_vectorization.isa",
            "cpu_memory.pack_b_layout",
        ]
    if defect_type == "CPU.Parallel.SharedPackedBufferRace":
        return ["cpu_parallel.policy", "cpu_memory.pack_a", "cpu_memory.pack_b", "cpu_memory.pack_layout"]
    return ["patch_generation.code_patch", "code_completeness.semantic_obligations"]


def make_defect(
    defect_type: str,
    related_strategy: str | None,
    related_fields: list[str],
    repair_action: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "defect_type": defect_type,
        "related_strategy": related_strategy,
        "related_fields": related_fields,
        "repair_action": repair_action,
        "evidence": evidence,
    }


def deduplicate_defects(defects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for defect in defects:
        key = (
            defect.get("defect_type"),
            defect.get("related_strategy"),
            tuple(defect.get("related_fields", [])),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(defect)
    return result


def infer_failure_type_from_check_id(check_id: str) -> str | None:
    if check_id.startswith("POSTCONDITION:"):
        return "StrategyCompliance.PostconditionViolation"
    return None


def get_related_strategy(ir: dict[str, Any]) -> str | None:
    strategy = ir.get("strategy", {})
    return (
        strategy.get("current_strategy_id")
        or ir.get("patch_generation", {}).get("strategy_id")
        or ir.get("verification", {}).get("strategy_id")
    )


def attach_diagnosis(ir: dict[str, Any], diagnosis: dict[str, Any]) -> dict[str, Any]:
    next_ir = copy.deepcopy(ir)
    next_ir["defect_diagnosis"] = diagnosis
    return next_ir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map SGPO verification failures to structured defects.")
    parser.add_argument("--ir", type=Path, default=DEFAULT_IR)
    parser.add_argument("--postcheck", type=Path, default=DEFAULT_POSTCHECK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-ir", type=Path, default=DEFAULT_OUTPUT_IR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ir = load_json(args.ir)
    post_check = load_json(args.postcheck) if args.postcheck and args.postcheck.exists() else None
    diagnosis = diagnose_defects(ir, post_check)
    save_json(args.output, diagnosis)
    save_json(args.output_ir, attach_diagnosis(ir, diagnosis))
    print(json.dumps(diagnosis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
