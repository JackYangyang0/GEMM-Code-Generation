from __future__ import annotations

import argparse
import json
import copy
import hashlib
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any

PACKAGE_PARENT = Path(__file__).resolve().parents[1]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from SGPO.diagnosis.defect_diagnosis import attach_diagnosis, diagnose_defects
from SGPO.generate_ir.ir_checker import (
    check_after_codegen, check_before_codegen, check_code_verification,
    get_precondition_predicates, get_patch_ir_postcondition_predicates, get_code_verification_constraints,
)
from SGPO.generate_ir.ir_extraction import build_extracted_ir
from SGPO.generate_ir.strategy_filter import filter_strategies_by_preconditions
from SGPO.generate_ir.strategy_index_filter import (
    filter_strategy_index,
    find_missing_requirements,
    infer_stage_order_from_index,
)
from SGPO.generate_ir.strategy_library_merge import load_merged_strategy_documents
from SGPO.generate_ir.stage_controller import (
    StageController, synthesize_ir_updates, derive_fields,
    choose_warp_iteration_extents, warp_iteration_is_valid,
)
from SGPO.generate_ir.gpu_resources import shared_memory_usage, shared_memory_limit, proposed_pipeline_bytes
from SGPO.generate_ir.tiling_planner import (
    build_joint_tiling_candidates,
    make_diverse_tiling_pool,
    tiling_plan_for_ir,
)
from SGPO.generate_ir.gpu_architecture import (
    build_gpu_architecture_profile,
    estimate_tiling_execution,
)
from SGPO.llm.openai_client import OpenAICompatibleClient
from SGPO.llm.patch_generator import (
    build_patch_ir,
    generate_patch_with_llm,
    load_code_context,
    load_code_region_context,
    load_strategy,
)
from SGPO.llm.concrete_code_generator import (
    apply_generated_code_files,
    generate_code_files_from_patch_with_llm,
)
from SGPO.llm.c_code_generator import (
    CPU_CODE_FILES,
    apply_cpu_c_code_files,
    generate_cpu_c_code_from_patch_with_llm,
)
from SGPO.llm.strategy_selector import (
    get_joint_tiling_selection_from_llm,
    get_micro_strategy_from_llm,
    get_unlock_batch_plan_from_llm,
)
from SGPO.utils.common_utils import load_config, load_json, save_json
from SGPO.utils.execution import (
    chain_client, chain_namespace, configure_execution, defer_latest,
    execution_config, parallel_chain_map,
)
from SGPO.verification.build_run_verifier import (
    DEFAULT_BENCHMARK_RUNS,
    DEFAULT_BENCHMARK_WARMUP_RUNS,
    DEFAULT_VCVARS64,
    compile_gemm,
    update_compile_result,
    verify_build_and_run,
)
from SGPO.verification.c_build_run_verifier import verify_cpu_build_and_run
from SGPO.verification.c_build_run_verifier import compile_cpu_gemm, update_cpu_compile_result
from SGPO.verification.code_ast_extractor import extract_code_ast
from SGPO.verification.gemm_semantic_checker import check_gemm_semantic_obligations
from SGPO.verification.gemm_semantic_repair import (
    build_throughput_cuda_kernel_cuh,
    extract_launch_config_from_ir,
    generate_semantic_repair_candidate,
    normalize_launch_config_for_repair,
    normalize_throughput_launch_config,
)
from SGPO.verification.strategy_realization_oracle import verify_strategy_realization
from SGPO.verification.patch_apply_verify import mark_patch_apply_failed, summarize_verification


ROOT = Path(__file__).resolve().parent
DEFAULT_DESCRIPTION = (
    # "I need to generate a high-performance CUDA GEMM implementation for "
    # "row-major fp32 NN GEMM. Matrix Size: (M:4096 N:4096 K:4096)."
    "I need to generate a high-performance CUDA GEMM implementation for the row master fp32 NN GEMM. "
    "matrix size: (M: 512 N: 512 K: 512)."
)
DEFAULT_IR_PATCH_DIR = ROOT / "data" / "IRs" / "ir_patch"
DEFAULT_IR_OUTPUT = DEFAULT_IR_PATCH_DIR / "optir.extracted.json"
DEFAULT_TEMPLATE = ROOT / "data" / "IRs" / "optir.json"
DEFAULT_STRATEGY_INDEX = ROOT / "data" / "lib" / "strategy_index.json"
DEFAULT_STRATEGY_LIBRARY = ROOT / "data" / "lib" / "strategy_library.json"
DEFAULT_DEPENDENCY_GRAPH = ROOT / "data" / "graph" / "dependency_graph.json"
DEFAULT_CPU_STRATEGY_INDEX = ROOT / "data" / "lib" / "cpu_strategy_index.json"
DEFAULT_CPU_STRATEGY_LIBRARY = ROOT / "data" / "lib" / "cpu_strategy_library.json"
DEFAULT_CPU_DEPENDENCY_GRAPH = ROOT / "data" / "graph" / "cpu_dependency_graph.json"
DEFAULT_FILTERED_INDEX_OUTPUT = ROOT / "data" / "lib" / "filter" / "strategy_index.filtered.json"
DEFAULT_CHECK_DIR = ROOT / "results" / "check"
DEFAULT_PATCH_DIR = ROOT / "results" / "patch"
DEFAULT_CODE_DIR = ROOT / "results" / "code"
DEFAULT_DEFECT_DIR = ROOT / "results" / "defect"
DEFAULT_CHAIN_DIR = ROOT / "results" / "chain"
DEFAULT_SELECTED_STRATEGY_OUTPUT = DEFAULT_CHECK_DIR / "selected_strategy.json"
DEFAULT_PRECHECK_OUTPUT = DEFAULT_CHECK_DIR / "pre_check_result.json"
DEFAULT_PATCH_OUTPUT = DEFAULT_PATCH_DIR / "generated_patch.json"
DEFAULT_PATCH_IR_OUTPUT = DEFAULT_IR_PATCH_DIR / "optir.patch.json"
DEFAULT_POSTCHECK_OUTPUT = DEFAULT_CHECK_DIR / "post_check_result.json"
DEFAULT_DIAGNOSIS_OUTPUT = DEFAULT_DEFECT_DIR / "defect_diagnosis.json"
DEFAULT_PATCHED_CODE_OUTPUT = DEFAULT_CODE_DIR / "patched_code_files.json"
DEFAULT_CODE_AST_OUTPUT = DEFAULT_CODE_DIR / "code_ast.json"
DEFAULT_FINAL_CODE_OUTPUT = DEFAULT_CODE_DIR / "final_code.json"
DEFAULT_FINAL_CODE_BUNDLE_OUTPUT = DEFAULT_CODE_DIR / "final_code_bundle.txt"
DEFAULT_EVOLUTION_OUTPUT = DEFAULT_CHECK_DIR / "evolution_summary.json"
DEFAULT_TOP_RESULTS_OUTPUT = DEFAULT_CHECK_DIR / "top_3_terminal_results.json"
DEFAULT_PERFORMANCE_UNLOCK_OUTPUT = DEFAULT_CHECK_DIR / "performance_unlock_summary.json"
DEFAULT_UNLOCK_BATCH_PLAN_OUTPUT = DEFAULT_CHECK_DIR / "unlock_batch_plan.json"
DEFAULT_UNLOCK_BATCH_RESULT_OUTPUT = DEFAULT_CHECK_DIR / "unlock_batch_result.json"
DEFAULT_UNLOCK_BATCH_PATCH_OUTPUT = DEFAULT_PATCH_DIR / "unlock_patch.json"
DEFAULT_UNLOCK_BATCH_DEFECT_OUTPUT = DEFAULT_DEFECT_DIR / "unlock_defect.json"
DEFAULT_VERIFIED_IR_OUTPUT = DEFAULT_IR_PATCH_DIR / "optir.verified.json"
DEFAULT_CODE_ROOT = ROOT / "gemm_code" / "skeleton"
DEFAULT_CPU_CODE_ROOT = ROOT / "gemm_code" / "cpu_skeleton"
DEFAULT_GENERATED_CODE_ROOT = ROOT / "gemm_code" / "code"
DEFAULT_GENERATED_CPU_CODE_ROOT = ROOT / "gemm_code" / "cpu_code"
DEFAULT_CODE_FILES = [
    "main.cpp",
    "cuda_kernel.cuh",
    "kernel.h",
]
DEFAULT_LLM_CONTEXT_FILES = [
    "kernel.h",
    "cuda_kernel.cuh",
]
DEFAULT_AST_CODE_FILES = [
    "cuda_kernel.cuh",
]
DEFAULT_PROMPT = ROOT / "llm" / "prompts" / "get_strategy_prompt.txt"
DEFAULT_MICRO_STRATEGY_PROMPT = ROOT / "llm" / "prompts" / "get_micro_strategy_prompt.txt"
DEFAULT_PATCH_PROMPT = ROOT / "llm" / "prompts" / "generate_patch_prompt.txt"
DEFAULT_PATCH_TO_CODE_PROMPT = ROOT / "llm" / "prompts" / "generate_concrete_code_prompt.txt"
DEFAULT_CPU_PATCH_PROMPT = ROOT / "llm" / "prompts" / "generate_cpu_c_patch_prompt.txt"
DEFAULT_CPU_PATCH_TO_CODE_PROMPT = ROOT / "llm" / "prompts" / "generate_cpu_c_code_prompt.txt"
DEFAULT_CONFIG = ROOT / "conf.yaml"
DEFAULT_MAX_REPAIR_ATTEMPTS = 3
CORE_CONSTRUCTION_COUPLED_PROFILE = "core_construction_coupled"
DEFAULT_STRATEGY_PROFILE = CORE_CONSTRUCTION_COUPLED_PROFILE
DEFAULT_UNLOCKED_PERFORMANCE_PROFILE = "throughput_exploration"
DEFAULT_SELECTION_MODE = "architecture_aware_tiling_funnel"
DEFAULT_MAX_FRONTIER_STATES_PER_STAGE = 3
DEFAULT_SINGLE_PATH_MODE = False
DEFAULT_TOP_K_STRATEGIES_PER_SUBPHASE = 3
DEFAULT_TOP_K_FINAL_RESULTS = 3
DEFAULT_TILING_TOP_K_PER_LEVEL = 3
DEFAULT_TILING_RESOURCE_POOL_SIZE = 32
DEFAULT_TILING_LLM_TOP_N = 3
DEFAULT_PHASE1_FRONTIER_SIZE = 6
DEFAULT_COMPILE_SHORTLIST_SIZE = 6
DEFAULT_CORE_CONSTRUCTION_FRONTIER_STATES = DEFAULT_PHASE1_FRONTIER_SIZE
DEFAULT_FIRST_PERFORMANCE_PRUNE_STAGE = "Tiling"
DEFAULT_PRESERVE_TILING_LINEAGES = True
DEFAULT_TOP_K_PER_TILING_LINEAGE = 1
DEFAULT_LAZY_FALLBACK_EXECUTION = False
DEFAULT_CLEAN_GENERATED_OUTPUTS = True
DEFAULT_COMPILE_EACH_STRATEGY_STEP = False
DEFAULT_COMPILE_EACH_STAGE = False
DEFAULT_STAGE_BENCHMARK_RUNS = 1
DEFAULT_STAGE_BENCHMARK_WARMUP_RUNS = 0
DEFAULT_STAGE_MMR_LAMBDA = 0.6
DEFAULT_STAGE_PERFORMANCE_FLOOR_RATIO = 0.8
STAGE_MMR_LAMBDA = DEFAULT_STAGE_MMR_LAMBDA
STAGE_PERFORMANCE_FLOOR_RATIO = DEFAULT_STAGE_PERFORMANCE_FLOOR_RATIO
FEEDBACK_SEARCH_LIMITS = {
    "max_rounds": 3,
    "experiments_per_round": 3,
    "max_rollback_depth": 2,
    "max_total_compilations": 12,
    "no_improvement_patience": 2,
}
DEFAULT_VERIFY_INITIAL_SKELETON = True
DEFAULT_STEP_COMPILE_TIMEOUT_SECONDS = 180
MAX_ARTIFACT_STEM_LENGTH = 140
STABLE_BASELINE_STAGES = ["Tiling", "Layout", "MappingReordering", "Vectorization", "Epilogue"]
CORE_CONSTRUCTION_COUPLED_STAGES = ["Tiling", "Layout", "MappingReordering", "Vectorization", "Pipeline", "Epilogue"]
THROUGHPUT_EXPLORATION_STAGES = ["Tiling", "Layout", "MappingReordering", "Vectorization", "Pipeline", "Epilogue"]
PERFORMANCE_UNLOCK_SUCCESS_STATUSES = {"pass"}
PHASE2_GENERAL = "phase2_unlock_general"
PHASE2_LARGE_MATRIX = "phase2_unlock_large_matrix"
EXCLUDED_DEFAULT_PHASE = "excluded_default"
PHASE1_CORE_COUPLED = "phase1_core_coupled"
DEFAULT_UNLOCK_BATCH_REPAIR_ATTEMPTS = 3
DETERMINISTIC_MATERIALIZATION_MODE = "deterministic"
LLM_PATCH_MATERIALIZATION_MODE = "llm_patch"
DETERMINISTIC_STRATEGY_PREFIXES = (
    "Tiling.BlockTile.",
    "Tiling.WarpTile.",
    "Tiling.ThreadTile.",
    "Mapping.LaneLayout.",
)
DETERMINISTIC_STRATEGY_IDS = {
    "Tiling.WarpTile.WMxWN",
    "Tiling.WarpTile.ParametricWMxWN",
    "Mapping.Warp.BasicWidLane",
    "Mapping.WarpThreadTile.WarpLaneFragmentBasic",
    "Mapping.WarpThreadTile.WarpLaneFragment2D",
    "Mapping.Warp.OutputFragment2D",
    "Mapping.LaneLayout.2D_8x4",
    "Layout.SharedMemory.AB.Basic",
    "Layout.RegisterTile.C",
    "Register.AccumulatorLayout.2DArray",
    "Safety.BoundaryPolicy.GeneralGuarded",
    "Safety.BoundaryPolicy.StaticDivisibleNoGuard",
    "Safety.AssumeDivisibleAligned",
    "Vectorization.AlignmentGuard",
    "Pipeline.NoAsyncCopy.V1",
    "Epilogue.AlphaBeta.General",
    "Epilogue.BetaZero.FastPath",
    "Epilogue.BetaOne.FastPath",
    "Epilogue.StoreC.CoalescedScalar",
    "Vectorization.StoreC.SafeScalar",
    "Mapping.WarpStore.CoalescedC",
}
REQUIRED_CONSTRUCTION_SUBPHASES = {
    "Tiling.BlockTileSelection",
    "Tiling.WarpTileSelection",
    "Tiling.ThreadTileSelection",
    "Tiling.MappingDerivation",
    "Layout.SharedMemoryDeclaration",
    "Layout.RegisterAccumulatorLayout",
}
THROUGHPUT_PROFILE_PREFERRED_IDS = {
    "Tiling.BlockTileSelection": [
        "Tiling.BlockTile.128x128x16",
        "Tiling.BlockTile.128x64x16",
        "Tiling.BlockTile.64x128x16",
    ],
    "Tiling.MappingDerivation": [
        "Mapping.WarpThreadTile.WarpLaneFragment2D",
        "Mapping.WarpThreadTile.WarpLaneFragmentBasic",
        "Mapping.Warp.OutputFragment2D",
    ],
    "Layout.SharedMemoryTransform": [
        "Layout.SharedMemory.TransposeA",
        "Layout.SharedMemory.PaddingAB.Plus1",
        "Layout.SharedMemory.PaddingB.Plus1",
    ],
    "Reordering.CooperativeLoadMapping": [
        "Reordering.CooperativeVectorLoadAB.float4",
        "Reordering.WarpCooperativeLoadAB.float4",
        "Layout.SharedMemory.VectorizedStoreA.float4",
        "Layout.SharedMemory.VectorizedStoreB.float4",
    ],
    "Reordering.ComputeSchedule": [
        "Reordering.KLoop.UnrollBK",
        "Reordering.KLoop.PragmaUnroll",
        "Reordering.KLoop.Unroll8",
    ],
    "Vectorization.AlignmentPolicy": [
        "Safety.AssumeDivisibleAligned",
        "Safety.BoundaryPolicy.StaticDivisibleNoGuard",
        "Vectorization.AlignmentGuard",
    ],
    "Vectorization.LoadVectorization": [
        "Vectorization.GlobalLoadAB.float4",
        "Reordering.CooperativeVectorLoadAB.float4",
        "Reordering.WarpCooperativeLoadAB.float4",
    ],
    "Vectorization.StoreVectorization": [
        "Epilogue.StoreC.Vectorized.float4",
        "Vectorization.StoreC.float4",
        "Vectorization.StoreC.GuardedVectorStore",
    ],
    "Pipeline.BufferingSelection": [
        "Pipeline.DoubleBuffer.SharedAB",
        "Pipeline.WarpAwareDoubleBuffer.SharedAB",
        "Pipeline.DoubleBuffer.SharedAB.V1Enabled",
    ],
    "Pipeline.PrefetchSelection": [
        "Pipeline.WarpRegisterPrefetchAB",
        "Memory.Prefetch.GlobalToRegisterA",
        "Memory.Prefetch.GlobalToRegisterB",
    ],
    "Pipeline.LargeMatrixScheduling": [
        "Memory.L2Reuse.CTASwizzleGroupedN",
        "Mapping.CTASwizzle.GroupedN",
        "Scheduling.WaveQuantization.SMResidentBlocks",
        "Scheduling.PersistentCTA.StaticTileLoop",
        "Memory.L2Reuse.CTASwizzleGroupedM",
        "Mapping.CTASwizzle.GroupedM",
        "Reduction.StreamK.WorkDecomposition",
    ],
    "Epilogue.StorePolicy": [
        "Epilogue.StoreC.Vectorized.float4",
        "Vectorization.StoreC.float4",
        "Epilogue.StoreC.CoalescedScalar",
    ],
    "CPUTiling.L2BlockSelection": [
        "CPU.Tiling.L2Block.256x128x128",
        "CPU.Tiling.L2Block.192x128x128",
        "CPU.Tiling.L2Block.128x128x128",
    ],
    "CPUTiling.L1BlockSelection": [
        "CPU.Tiling.L1Block.32x64x64",
        "CPU.Tiling.L1Block.32x32x64",
        "CPU.Tiling.L1Block.16x32x64",
    ],
    "CPUTiling.RegisterBlockSelection": [
        "CPU.Tiling.RegisterBlock.4x8",
        "CPU.Tiling.RegisterBlock.4x4",
        "CPU.Tiling.RegisterBlock.2x4",
    ],
    "CPUPacking.PanelPackingPolicy": [
        "CPU.Packing.PackAB.MRxKC_KCxNR",
        "CPU.Memory.PackAB.PanelMajor",
        "CPU.Packing.PackB.KCxNR",
    ],
    "CPUPacking.PackBufferOwnership": [
        "CPU.Memory.PackBuffer.ThreadPrivate",
        "CPU.Memory.PackBuffer.TilePrivate",
    ],
    "CPUPacking.PrefetchPolicy": [
        "CPU.Memory.Prefetch.ABPanel",
        "CPU.Memory.Prefetch.BPanel",
        "CPU.Memory.Prefetch.Disabled",
    ],
    "CPUMicroKernel.KernelShapeSelection": [
        "CPU.MicroKernel.AVX512.FMA.8x16",
        "CPU.MicroKernel.AVX2.FMA.6x16",
        "CPU.MicroKernel.AVX2.FMA.4x8",
    ],
    "CPUVectorization.SIMDPolicy": [
        "CPU.Vectorization.AVX512.FMA.Explicit",
        "CPU.Vectorization.AVX2.FMA.Explicit",
        "CPU.Vectorization.PragmaSIMD",
    ],
    "CPULoopSchedule.MacroKernelDriverSelection": [
        "CPU.MacroKernel.OpenBLASStyle.PanelDriver",
    ],
    "CPULoopSchedule.LoopOrderSelection": [
        "CPU.LoopOrder.OpenBLASPanelMajor",
        "CPU.LoopOrder.PackedPanelMajor",
        "CPU.LoopOrder.IKJ",
    ],
    "CPULoopSchedule.UnrollSelection": [
        "CPU.KLoop.Unroll8",
        "CPU.KLoop.Unroll4",
        "CPU.KLoop.NoUnroll",
    ],
    "CPUParallelization.ThreadPolicy": [
        "CPU.Parallel.OpenMP.Collapse2",
        "CPU.Threading.OpenMP.TilePartition",
        "CPU.Parallel.OpenMP.RowBlock",
    ],
    "CPUCompiler.FlagPolicy": [
        "CPU.Compiler.NativeO3OpenMP",
        "CPU.Compiler.NativeO3",
        "CPU.Compiler.MSVC.AVX2OpenMP",
    ],
    "CPUEpilogue.StorePolicy": [
        "CPU.TailKernel.FullTileFastPathScalarCleanup",
        "CPU.Epilogue.Store.BetaZeroFastPath",
        "CPU.Epilogue.Store.AlphaBeta",
    ],
}
STABLE_BASELINE_DENY_PREFIXES = (
    "Pipeline.",
    "Memory.Prefetch.",
    "Vectorization.GlobalLoad",
    "Vectorization.StoreC.float",
    "Vectorization.StoreC.GuardedVectorStore",
    "Vectorization.StoreC.AlignedNoGuard",
    "Reordering.CooperativeVectorLoad",
    "Reordering.WarpCooperativeLoad",
    "Layout.SharedMemory.VectorizedStore",
    "Epilogue.StoreC.Vectorized",
    "Compiler.",
    "Scheduling.",
    "Tuning.",
)
STABLE_BASELINE_DENY_IDS = {
    "Pipeline.WarpRegisterPrefetchAB",
    "Pipeline.DoubleBuffer.SharedAB",
    "Pipeline.DoubleBuffer.SharedAB.V1Enabled",
    "Pipeline.WarpAwareDoubleBuffer.SharedAB",
    "Pipeline.SoftwarePrefetch.RegisterA",
    "Safety.AssumeDivisibleAligned",
    "Safety.BoundaryPolicy.StaticDivisibleNoGuard",
    "Epilogue.Fusion.Bias",
    "Epilogue.Fusion.GELU",
    "Epilogue.Fusion.ReLU",
}


def main() -> None:
    args = parse_args()
    if args.resume_terminal:
        from SGPO.verification.resume_terminal import resume_terminal
        resume_terminal(args)
        return
    description = DEFAULT_DESCRIPTION
    user_question = DEFAULT_DESCRIPTION

    template = load_json(Path(DEFAULT_TEMPLATE))
    current_ir = build_extracted_ir(template, description)
    if args.matrix_size:
        if min(args.matrix_size) <= 0:
            raise ValueError("--matrix-size requires positive M N K")
        current_ir["problem"].update(zip(("M", "N", "K"), args.matrix_size))
        description = f"{description}\nTarget matrix dimensions (override): M={args.matrix_size[0]}, N={args.matrix_size[1]}, K={args.matrix_size[2]}."
        user_question = description
    save_json(Path(DEFAULT_IR_OUTPUT), current_ir)

    raw_strategy_index, strategy_library = load_strategy_documents(current_ir)
    dependency_graph = load_dependency_graph(current_ir)
    strategy_profile = infer_strategy_profile(current_ir, raw_strategy_index)
    current_ir.setdefault("strategy", {})["profile"] = strategy_profile
    stage_order = infer_app_stage_order(current_ir, dependency_graph, raw_strategy_index, strategy_library)
    code_root = backend_code_root(current_ir)
    code_files = backend_code_files(current_ir)

    config = load_config(Path(DEFAULT_CONFIG))
    from SGPO.specialization.shape_specialization import resolve_config, load_existing_seed, specialize_shapes
    specialization_config = resolve_config(config, args.shape_reuse, args.reuse_family)
    existing_seed = None
    if target_backend(current_ir) == "cuda" and specialization_config.get("enabled") and specialization_config["seed_source"] == "existing":
        existing_seed = load_existing_seed(specialization_config["seed_dir"], current_ir)
    configure_execution(config.get("execution"))
    print(f"SGPO execution: chain_workers={execution_config().chain_workers}, "
          f"llm_max_concurrency={execution_config().llm_max_concurrency}, "
          f"compile_workers={execution_config().compile_workers}, gpu_verification_workers=1")
    if existing_seed is None and should_clean_generated_outputs(args, config):
        clean_generated_outputs(current_ir)
        save_json(Path(DEFAULT_IR_OUTPUT), current_ir)
    build_platform = resolve_build_platform(args, config)
    compile_each_step = should_compile_each_strategy_step(args, config)
    compile_each_stage = should_compile_each_stage(args, config)
    per_step_compile_timeout = step_compile_timeout_seconds(config)
    per_stage_benchmark_runs = stage_benchmark_runs(config)
    per_stage_benchmark_warmup_runs = stage_benchmark_warmup_runs(config)
    terminal_benchmark_runs = benchmark_runs(args, config)
    terminal_benchmark_warmup_runs = benchmark_warmup_runs(args, config)
    if existing_seed is not None:
        report = specialize_shapes(
            existing_seed, strategy_library, ROOT / "gemm_code" / "shape_specialization",
            specialization_config, build_platform, terminal_benchmark_runs,
            terminal_benchmark_warmup_runs, client=OpenAICompatibleClient(config['llm']))
        save_json(DEFAULT_CHECK_DIR / "shape_specialization.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    restore_initial_cuda_skeleton(code_root, current_ir)
    if should_verify_initial_skeleton(args, config):
        initial_skeleton_ir = verify_initial_skeleton_before_search(
            current_ir=current_ir,
            code_root=code_root,
            build_platform=build_platform,
            timeout_seconds=per_step_compile_timeout,
            benchmark_runs=per_stage_benchmark_runs,
            benchmark_warmup_runs=per_stage_benchmark_warmup_runs,
        )
        if not oracle_verification_passed(initial_skeleton_ir):
            raise RuntimeError(
                "Initial GEMM skeleton failed compile/run verification. "
                f"See {DEFAULT_CHECK_DIR / 'initial_skeleton_verification.json'}."
            )
    client = OpenAICompatibleClient(config["llm"])
    search_config = config.get("search", {}) or {}
    performance_config = config.get("performance_unlock", {}) or {}
    configure_search_policy(search_config, config.get("feedback_search", {}) or {})
    max_frontier_states = 1 if DEFAULT_SINGLE_PATH_MODE else int(
        search_config.get("max_frontier_states_per_stage", DEFAULT_MAX_FRONTIER_STATES_PER_STAGE) or 0
    )
    tiling_top_k_per_level = int(
        search_config.get("tiling_top_k_per_level", DEFAULT_TILING_TOP_K_PER_LEVEL) or 0
    )
    tiling_resource_pool_size = int(
        search_config.get("tiling_resource_pool_size", DEFAULT_TILING_RESOURCE_POOL_SIZE) or 0
    )
    tiling_llm_top_n = int(
        search_config.get("tiling_llm_top_n", DEFAULT_TILING_LLM_TOP_N) or 0
    )
    compile_shortlist_size = int(
        search_config.get("compile_shortlist_size", DEFAULT_COMPILE_SHORTLIST_SIZE) or 0
    )
    core_construction_frontier_states = int(
        search_config.get("core_construction_frontier_states", DEFAULT_CORE_CONSTRUCTION_FRONTIER_STATES) or 0
    )
    first_performance_prune_stage = str(
        search_config.get("first_performance_prune_stage", DEFAULT_FIRST_PERFORMANCE_PRUNE_STAGE)
    )
    preserve_tiling_lineages = bool(
        search_config.get("preserve_tiling_lineages", DEFAULT_PRESERVE_TILING_LINEAGES)
    )
    top_k_per_tiling_lineage = int(
        search_config.get("top_k_per_tiling_lineage", DEFAULT_TOP_K_PER_TILING_LINEAGE) or 0
    )
    lazy_fallback_execution = bool(
        search_config.get("lazy_fallback_execution", DEFAULT_LAZY_FALLBACK_EXECUTION)
    )
    auto_unlock_performance = bool(performance_config.get("enabled", False)) and target_backend(current_ir) != "cpu"
    unlocked_profile = performance_config.get("profile", DEFAULT_UNLOCKED_PERFORMANCE_PROFILE)
    best_overall_candidate = None
    best_partial_candidate = None

    initial_source_snapshot = snapshot_source_files(code_root, code_files)
    frontier = [
        make_frontier_state(
            state_id="root",
            current_ir=current_ir,
            source_snapshot=initial_source_snapshot,
            history={
                "applied_strategy_ids": [],
                "applied_micro_strategies": [],
                "failed_strategy_counts": {},
                "completed_subphases": [],
                "events": [],
            },
            path=[],
            path_code=[],
        )
    ]
    stage_summaries = []
    terminal_fallback_states = []
    for stage in stage_order:
        if not frontier:
            stage_summaries.append(
                {
                    "stage": stage,
                    "selection_mode": DEFAULT_SELECTION_MODE,
                    "input_frontier_count": 0,
                    "output_frontier_count": 0,
                    "accepted_candidate_count": 0,
                    "status": "skipped",
                    "reason": "frontier is empty; previous stage produced no accepted chain state",
                }
            )
            continue
        stage_frontier_limit = frontier_limit_for_stage(
            stage=stage,
            backend=target_backend(current_ir),
            normal_limit=max_frontier_states,
            core_limit=core_construction_frontier_states,
            first_performance_prune_stage=first_performance_prune_stage,
        )
        # Stage survivors are a global budget, not a budget per BlockTile root.
        use_tiling_lineage_beam = False
        stage_result = run_exhaustive_stage(
            stage=stage,
            frontier=frontier,
            raw_strategy_index=raw_strategy_index,
            dependency_graph=dependency_graph,
            strategy_library=strategy_library,
            client=client,
            user_question=user_question,
            profile=strategy_profile,
            max_frontier_states=stage_frontier_limit,
            exhaust_pending=False,
            build_platform=build_platform,
            compile_each_step=compile_each_step,
            compile_each_stage=compile_each_stage,
            step_compile_timeout_seconds=per_step_compile_timeout,
            stage_benchmark_runs=per_stage_benchmark_runs,
            stage_benchmark_warmup_runs=per_stage_benchmark_warmup_runs,
            tiling_top_k_per_level=tiling_top_k_per_level,
            tiling_resource_pool_size=tiling_resource_pool_size,
            tiling_llm_top_n=tiling_llm_top_n,
            preserve_tiling_lineages=use_tiling_lineage_beam,
            top_k_per_tiling_lineage=top_k_per_tiling_lineage,
            lazy_fallback_execution=lazy_fallback_execution,
        )
        stage_summaries.append(stage_result["summary"])
        terminal_fallback_states.extend(stage_result.get("terminal_states", []))
        best_partial_candidate = choose_better_partial_candidate(
            best_partial_candidate,
            choose_best_candidate(stage_result.get("accepted_candidates", [])),
        )
        frontier = stage_result["frontier"]
        stage_result["summary"]["post_stage_performance_prune"] = {
            "enabled": False,
            "reason": "Performance ranking is deferred until Phase 1 terminal verification.",
        }

    terminal_input_states = build_phase1_compile_shortlist(
        frontier,
        terminal_fallback_states,
        compile_shortlist_size,
    )
    baseline_terminal_verification = verify_terminal_chains(
        frontier=terminal_input_states,
        strategy_library=strategy_library,
        build_platform=build_platform,
        benchmark_runs=terminal_benchmark_runs,
        benchmark_warmup_runs=terminal_benchmark_warmup_runs,
        client=client,
    )
    terminal_verification = baseline_terminal_verification
    performance_phase = None
    if auto_unlock_performance and terminal_verification_has_correct_chain(baseline_terminal_verification):
        performance_phase = run_unlocked_performance_phase(
            baseline_terminal_verification=baseline_terminal_verification,
            raw_strategy_index=raw_strategy_index,
            dependency_graph=dependency_graph,
            strategy_library=strategy_library,
            client=client,
            user_question=user_question,
            profile=unlocked_profile,
            build_platform=build_platform,
            compile_each_step=compile_each_step,
            step_compile_timeout_seconds=per_step_compile_timeout,
            benchmark_runs=terminal_benchmark_runs,
            benchmark_warmup_runs=terminal_benchmark_warmup_runs,
        )
        save_json(Path(DEFAULT_PERFORMANCE_UNLOCK_OUTPUT), summarize_performance_phase(performance_phase))
        terminal_verification = merge_terminal_verifications(
            baseline_terminal_verification,
            performance_phase["terminal_verification"],
            phases=["stable_baseline", "performance_unlock"],
        )
        if is_better_candidate(terminal_verification.get("best_candidate"), baseline_terminal_verification.get("best_candidate")):
            frontier = performance_phase["frontier"]
        stage_summaries.extend(performance_phase["stage_summaries"])
    best_overall_candidate = terminal_verification.get("best_candidate")
    top_terminal_results = terminal_verification.get("top_terminal_results", [])
    save_json(Path(DEFAULT_TOP_RESULTS_OUTPUT), {
        "selection_mode": DEFAULT_SELECTION_MODE,
        "top_k_strategies_per_subphase": DEFAULT_TOP_K_STRATEGIES_PER_SUBPHASE,
        "max_frontier_states_per_stage": max_frontier_states,
        "top_k_final_results": DEFAULT_TOP_K_FINAL_RESULTS,
        "results": top_terminal_results,
    })
    stage_summaries.append(terminal_verification["summary"])

    transfer_config = specialization_config
    if target_backend(current_ir) != "cpu" and transfer_config.get("enabled", False):
        print("Shape specialization: independent Top-1 Tile tuning per target shape", flush=True)
        try:
            transfer_report = specialize_shapes(
                best_overall_candidate, strategy_library,
                ROOT / "gemm_code" / "shape_specialization",
                transfer_config, build_platform, terminal_benchmark_runs,
                terminal_benchmark_warmup_runs, client=client,
            )
        except Exception as exc:
            transfer_report = {"stage": "ShapeSpecialization", "status": "failed", "error": str(exc)}
        save_json(DEFAULT_CHECK_DIR / "shape_specialization.json", transfer_report)
        stage_summaries.append(transfer_report)
        from SGPO.specialization.shape_specialization import current_shape_candidates
        specialized_candidates = current_shape_candidates(transfer_report, best_overall_candidate, current_ir)
        if specialized_candidates:
            terminal_verification = merge_terminal_verifications(terminal_verification, {
                "verified_candidates": specialized_candidates,
                "summary": {"phase": "shape_specialization", "terminal_chain_count": len(specialized_candidates)},
            })
            best_overall_candidate = terminal_verification.get("best_candidate")
            top_terminal_results = terminal_verification.get("top_terminal_results", [])
            save_json(Path(DEFAULT_TOP_RESULTS_OUTPUT), {
                "selection_mode": DEFAULT_SELECTION_MODE,
                "top_k_final_results": DEFAULT_TOP_K_FINAL_RESULTS,
                "results": top_terminal_results,
            })

    best_state = terminal_verification.get("best_state") or (select_best_frontier_state(frontier) if frontier else None)
    if best_overall_candidate:
        final_ir = best_overall_candidate["verified_ir"]
        final_source_snapshot = best_overall_candidate["source_snapshot"]
        final_history = best_overall_candidate.get("history")
    elif best_state:
        final_ir = best_state["current_ir"]
        final_source_snapshot = best_state["source_snapshot"]
        final_history = best_state["history"]
    elif best_partial_candidate:
        final_ir = mark_partial_frontier(
            best_partial_candidate["verified_ir"],
            stage_summaries,
            best_partial_candidate,
        )
        final_source_snapshot = best_partial_candidate["source_snapshot"]
        final_history = best_partial_candidate.get("history")
    else:
        final_ir = mark_no_terminal_frontier(current_ir, stage_summaries)
        final_source_snapshot = initial_source_snapshot
        final_history = {
            "applied_strategy_ids": [],
            "failed_strategy_counts": {},
            "events": [
                {
                    "stage": "Search",
                    "status": "blocked",
                    "reason": "No strategy chain completed all required stage verification steps.",
                }
            ],
        }
    final_history = final_history or {"applied_strategy_ids": [], "failed_strategy_counts": {}, "events": []}
    final_ir.setdefault("strategy", {})["applied_strategy_ids"] = final_history.get("applied_strategy_ids", [])
    final_ir.setdefault("strategy", {})["failed_strategy_counts"] = final_history.get("failed_strategy_counts", {})
    final_ir.setdefault("strategy", {})["history"] = final_history.get("events", [])
    final_ir.setdefault("strategy", {})["final_selected_strategy_id"] = (
        best_overall_candidate.get("strategy_id") if best_overall_candidate else None
    )
    final_ir.setdefault("strategy", {})["final_selected_code_dir"] = (
        (best_overall_candidate or best_partial_candidate or {}).get("candidate_code_dir")
    )
    save_json(Path(DEFAULT_VERIFIED_IR_OUTPUT), final_ir)

    final_code = collect_final_code_from_snapshot(final_source_snapshot, final_ir, final_history, stage_summaries)
    save_json(Path(DEFAULT_FINAL_CODE_OUTPUT), final_code)
    write_final_code_bundle(Path(DEFAULT_FINAL_CODE_BUNDLE_OUTPUT), final_code)
    if best_overall_candidate and terminal_chain_is_accepted(best_overall_candidate["verified_ir"]):
        materialize_final_generated_files(code_root, final_source_snapshot, final_ir)
    save_json(Path(DEFAULT_EVOLUTION_OUTPUT), {
        "selection_mode": DEFAULT_SELECTION_MODE,
        "tiling_search_funnel": {
            "resource_pool_size": tiling_resource_pool_size,
            "llm_top_n": tiling_llm_top_n,
            "phase1_frontier_size": max_frontier_states,
            "compile_shortlist_size": compile_shortlist_size,
            "final_benchmark_top_k": DEFAULT_TOP_K_FINAL_RESULTS,
        },
        "stage_order": stage_order,
        "strategy_profile": strategy_profile,
        "performance_unlock": {
            "enabled": auto_unlock_performance,
            "build_platform": build_platform,
            "compile_each_strategy_step": compile_each_step,
            "compile_each_stage": compile_each_stage,
            "lazy_fallback_execution": lazy_fallback_execution,
            "step_compile_timeout_seconds": per_step_compile_timeout,
            "triggered": performance_phase is not None,
            "profile": unlocked_profile if performance_phase is not None else None,
        },
        "frontier_state_count": len(frontier),
        "terminal_chain_count": terminal_verification["summary"]["terminal_chain_count"],
        "verified_terminal_chain_count": terminal_verification["summary"]["verified_terminal_chain_count"],
        "generated_code_root": str(Path(DEFAULT_GENERATED_CODE_ROOT)),
        "final_skeleton_dir": str(code_root),
        "best_overall_strategy_id": best_overall_candidate.get("strategy_id") if best_overall_candidate else None,
        "best_overall_code_dir": best_overall_candidate.get("candidate_code_dir") if best_overall_candidate else None,
        "best_overall_gflops": candidate_gflops(best_overall_candidate),
        "best_partial_strategy_id": best_partial_candidate.get("strategy_id") if best_partial_candidate else None,
        "best_partial_code_dir": best_partial_candidate.get("candidate_code_dir") if best_partial_candidate else None,
        "applied_strategy_ids": final_history.get("applied_strategy_ids", []),
        "failed_strategy_counts": final_history.get("failed_strategy_counts", {}),
        "stage_summaries": stage_summaries,
        "final_code_output": str(Path(DEFAULT_FINAL_CODE_OUTPUT)),
        "final_code_bundle_output": str(Path(DEFAULT_FINAL_CODE_BUNDLE_OUTPUT)),
        "top_results_output": str(Path(DEFAULT_TOP_RESULTS_OUTPUT)),
        "performance_unlock_output": str(Path(DEFAULT_PERFORMANCE_UNLOCK_OUTPUT)) if performance_phase is not None else None,
        "top_3_terminal_results": top_terminal_results,
        "verified_ir_output": str(Path(DEFAULT_VERIFIED_IR_OUTPUT)),
    })

    print(
        json.dumps(
            {
                "ir_output": str(Path(DEFAULT_IR_OUTPUT)),
                "verified_ir_output": str(Path(DEFAULT_VERIFIED_IR_OUTPUT)),
                "final_code_output": str(Path(DEFAULT_FINAL_CODE_OUTPUT)),
                "final_code_bundle_output": str(Path(DEFAULT_FINAL_CODE_BUNDLE_OUTPUT)),
                "top_results_output": str(Path(DEFAULT_TOP_RESULTS_OUTPUT)),
                "performance_unlock_output": str(Path(DEFAULT_PERFORMANCE_UNLOCK_OUTPUT))
                if performance_phase is not None
                else None,
                "evolution_output": str(Path(DEFAULT_EVOLUTION_OUTPUT)),
                "selection_mode": DEFAULT_SELECTION_MODE,
                "tiling_search_funnel": {
                    "resource_pool_size": tiling_resource_pool_size,
                    "llm_top_n": tiling_llm_top_n,
                    "phase1_frontier_size": max_frontier_states,
                    "compile_shortlist_size": compile_shortlist_size,
                    "final_benchmark_top_k": DEFAULT_TOP_K_FINAL_RESULTS,
                },
                "top_k_strategies_per_subphase": DEFAULT_TOP_K_STRATEGIES_PER_SUBPHASE,
                "top_k_final_results": DEFAULT_TOP_K_FINAL_RESULTS,
                "benchmark_runs": terminal_benchmark_runs,
                "benchmark_warmup_runs": terminal_benchmark_warmup_runs,
                "strategy_profile": strategy_profile,
                "performance_unlock": {
                    "enabled": auto_unlock_performance,
                    "triggered": performance_phase is not None,
                    "profile": unlocked_profile if performance_phase is not None else None,
                },
                "frontier_state_count": len(frontier),
                "terminal_chain_count": terminal_verification["summary"]["terminal_chain_count"],
                "verified_terminal_chain_count": terminal_verification["summary"]["verified_terminal_chain_count"],
                "generated_code_root": str(Path(DEFAULT_GENERATED_CODE_ROOT)),
                "final_skeleton_dir": str(code_root),
                "best_overall_strategy_id": best_overall_candidate.get("strategy_id") if best_overall_candidate else None,
                "best_overall_code_dir": best_overall_candidate.get("candidate_code_dir") if best_overall_candidate else None,
                "best_overall_gflops": candidate_gflops(best_overall_candidate),
                "top_3_terminal_results": summarize_terminal_results_for_console(top_terminal_results),
                "applied_strategy_ids": final_history.get("applied_strategy_ids", []),
                "failed_strategy_counts": final_history.get("failed_strategy_counts", {}),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def publish_shape_reuse_result(result, code_root, registry):
    from SGPO.specialization.shape_search import archive
    candidates = result.get("candidates", [])
    top = build_top_terminal_results(candidates, 3)
    save_json(Path(DEFAULT_TOP_RESULTS_OUTPUT), {"selection_mode": "deterministic_shape_search", "results": top})
    save_json(Path(DEFAULT_EVOLUTION_OUTPUT), {"selection_mode": "deterministic_shape_search",
        "target_reached": not result["fallback_required"], "reason": result.get("reason"),
        "rounds": result.get("rounds", []), "top_3_terminal_results": top})
    if candidates:
        best = max(candidates, key=lambda c: candidate_gflops(c) or 0)
        ir = best["verified_ir"]
        final = collect_final_code_from_snapshot(best["source_snapshot"], ir, best["history"], [])
        save_json(Path(DEFAULT_FINAL_CODE_OUTPUT), final)
        write_final_code_bundle(Path(DEFAULT_FINAL_CODE_BUNDLE_OUTPUT), final)
        save_json(Path(DEFAULT_VERIFIED_IR_OUTPUT), ir)
        materialize_final_generated_files(code_root, best["source_snapshot"], ir)
        for candidate in candidates[:3]:
            archive(candidate, registry)
    else:
        save_json(Path(DEFAULT_FINAL_CODE_OUTPUT), {"status": "no_verified_specialization",
            "reason": result.get("reason"), "files": []})
    print(json.dumps({"shape_reuse_reason": result.get("reason"), "top_results": top}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SGPO GEMM strategy-guided optimization.")
    parser.add_argument('--resume-terminal', action='store_true',
                        help='Resume current terminal chains with real LLM repair and publish verified results; do not regenerate or clean outputs.')
    parser.add_argument("--matrix-size", nargs=3, type=int, metavar=("M", "N", "K"))
    parser.add_argument("--shape-reuse", choices=("auto", "off", "only"), default="auto",
                        help="Compatibility switch: auto uses shape_specialization config; off disables it; only requires an existing seed.")
    parser.add_argument("--reuse-family", type=Path,
                        help="Use this existing CUDA directory as the shape_specialization seed; skip full generation and output cleanup.")
    parser.add_argument(
        "--build-platform",
        choices=["windows", "linux"],
        default="windows",
        help="Compilation/runtime platform for generated code. Defaults to build.platform in conf.yaml, then windows.",
    )
    parser.add_argument(
        "--keep-generated-cache",
        action="store_true",
        help="Keep previous generated code/results instead of clearing SGPO output directories at startup.",
    )
    parser.add_argument(
        "--skip-step-compile",
        action="store_true",
        help="Do not compile each generated strategy step before advancing to the next step.",
    )
    parser.add_argument(
        "--skip-stage-compile",
        action="store_true",
        help="Do not compile/run at stage completion before advancing to the next stage.",
    )
    parser.add_argument(
        "--skip-initial-skeleton-check",
        action="store_true",
        help="Do not compile/run the initial skeleton before strategy search.",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=None,
        help="Measured executable runs per terminal candidate. Defaults to build.benchmark_runs, then 10.",
    )
    parser.add_argument(
        "--benchmark-warmup-runs",
        type=int,
        default=None,
        help="Warmup executable runs before measured runs. Defaults to build.benchmark_warmup_runs, then 2.",
    )
    return parser.parse_args()


def should_clean_generated_outputs(args: argparse.Namespace, config: dict[str, Any]) -> bool:
    if getattr(args, "keep_generated_cache", False):
        return False
    run_config = config.get("run", {}) or {}
    if "clean_generated_outputs" in run_config:
        return bool(run_config.get("clean_generated_outputs"))
    return DEFAULT_CLEAN_GENERATED_OUTPUTS


def should_compile_each_strategy_step(args: argparse.Namespace, config: dict[str, Any]) -> bool:
    if getattr(args, "skip_step_compile", False):
        return False
    build_config = config.get("build", {}) or {}
    if "compile_each_strategy_step" in build_config:
        return bool(build_config.get("compile_each_strategy_step"))
    return DEFAULT_COMPILE_EACH_STRATEGY_STEP


def should_compile_each_stage(args: argparse.Namespace, config: dict[str, Any]) -> bool:
    if getattr(args, "skip_stage_compile", False):
        return False
    build_config = config.get("build", {}) or {}
    if "compile_each_stage" in build_config:
        return bool(build_config.get("compile_each_stage"))
    return DEFAULT_COMPILE_EACH_STAGE


def stage_benchmark_runs(config: dict[str, Any]) -> int:
    build_config = config.get("build", {}) or {}
    return max(1, int(build_config.get("stage_benchmark_runs", DEFAULT_STAGE_BENCHMARK_RUNS)))


def stage_benchmark_warmup_runs(config: dict[str, Any]) -> int:
    build_config = config.get("build", {}) or {}
    return max(0, int(build_config.get("stage_benchmark_warmup_runs", DEFAULT_STAGE_BENCHMARK_WARMUP_RUNS)))


def configure_search_policy(search_config: dict[str, Any], feedback_config: dict[str, Any]) -> None:
    global STAGE_MMR_LAMBDA, STAGE_PERFORMANCE_FLOOR_RATIO
    selection = search_config.get("stage_survivor_selection", {}) or {}
    STAGE_MMR_LAMBDA = min(1.0, max(0.0, float(selection.get("mmr_lambda", DEFAULT_STAGE_MMR_LAMBDA))))
    STAGE_PERFORMANCE_FLOOR_RATIO = min(
        1.0,
        max(0.0, float(selection.get("performance_floor_ratio", DEFAULT_STAGE_PERFORMANCE_FLOOR_RATIO))),
    )
    for key, default in list(FEEDBACK_SEARCH_LIMITS.items()):
        FEEDBACK_SEARCH_LIMITS[key] = max(1, int(feedback_config.get(key, default)))


def should_verify_initial_skeleton(args: argparse.Namespace, config: dict[str, Any]) -> bool:
    if getattr(args, "skip_initial_skeleton_check", False):
        return False
    build_config = config.get("build", {}) or {}
    if "verify_initial_skeleton" in build_config:
        return bool(build_config.get("verify_initial_skeleton"))
    return DEFAULT_VERIFY_INITIAL_SKELETON


def step_compile_timeout_seconds(config: dict[str, Any]) -> int:
    build_config = config.get("build", {}) or {}
    return int(build_config.get("step_compile_timeout_seconds", DEFAULT_STEP_COMPILE_TIMEOUT_SECONDS))


def benchmark_runs(args: argparse.Namespace, config: dict[str, Any]) -> int:
    if getattr(args, "benchmark_runs", None) is not None:
        return max(1, int(args.benchmark_runs))
    build_config = config.get("build", {}) or {}
    return max(1, int(build_config.get("benchmark_runs", DEFAULT_BENCHMARK_RUNS)))


def benchmark_warmup_runs(args: argparse.Namespace, config: dict[str, Any]) -> int:
    if getattr(args, "benchmark_warmup_runs", None) is not None:
        return max(0, int(args.benchmark_warmup_runs))
    build_config = config.get("build", {}) or {}
    return max(0, int(build_config.get("benchmark_warmup_runs", DEFAULT_BENCHMARK_WARMUP_RUNS)))


def clean_generated_outputs(ir: dict[str, Any] | None = None) -> None:
    for path in generated_output_dirs(ir):
        safe_remove_generated_output_dir(path)


def generated_output_dirs(ir: dict[str, Any] | None = None) -> list[Path]:
    return [
        backend_generated_code_root(ir or {}),
        Path(DEFAULT_GENERATED_CODE_ROOT),
        Path(DEFAULT_GENERATED_CPU_CODE_ROOT),
        Path(DEFAULT_CHECK_DIR),
        Path(DEFAULT_PATCH_DIR),
        Path(DEFAULT_CODE_DIR),
        Path(DEFAULT_DEFECT_DIR),
        Path(DEFAULT_CHAIN_DIR),
        Path(DEFAULT_IR_PATCH_DIR),
    ]


def safe_remove_generated_output_dir(path: Path) -> None:
    resolved_root = ROOT.resolve()
    resolved = path.resolve()
    if resolved == resolved_root or resolved_root not in [resolved, *resolved.parents]:
        raise ValueError(f"Refusing to clean path outside SGPO project: {path}")
    if not path.exists():
        return
    shutil.rmtree(path)


def resolve_build_platform(args: argparse.Namespace, config: dict[str, Any]) -> str:
    configured = (config.get("build", {}) or {}).get("platform")
    value = args.build_platform or configured or "windows"
    value = str(value).strip().lower()
    return "linux" if value in {"linux", "unix", "posix"} else "windows"


def restore_initial_cuda_skeleton(code_root: Path, ir: dict[str, Any]) -> None:
    if target_backend(ir) != "cuda":
        return
    template = ROOT / "gemm_code" / "baseline_template" / "cuda_kernel.cuh"
    restore_source_files(code_root, {"cuda_kernel.cuh": template.read_text(encoding="utf-8")})


def verify_initial_skeleton_before_search(
    current_ir: dict[str, Any],
    code_root: Path,
    build_platform: str,
    timeout_seconds: int,
    benchmark_runs: int,
    benchmark_warmup_runs: int,
) -> dict[str, Any]:
    if target_backend(current_ir) == "cpu":
        verified_ir = verify_cpu_build_and_run(
            ir=current_ir,
            source_dir=code_root,
            build_dir=code_root / "build",
            build_platform=build_platform,
            timeout_seconds=timeout_seconds,
            benchmark_runs=benchmark_runs,
            benchmark_warmup_runs=benchmark_warmup_runs,
        )
    else:
        verified_ir = verify_build_and_run(
            ir=current_ir,
            source_dir=code_root,
            build_dir=code_root / "build",
            build_platform=build_platform,
            timeout_seconds=timeout_seconds,
            benchmark_runs=benchmark_runs,
            benchmark_warmup_runs=benchmark_warmup_runs,
        )
    verified_ir.setdefault("verification", {})["summary"] = summarize_verification(verified_ir)
    save_json(DEFAULT_CHECK_DIR / "initial_skeleton_verification.json", verified_ir)
    return verified_ir


def verify_stage_completion_build_run(
    ir: dict[str, Any],
    source_snapshot: dict[str, str],
    stage: str,
    chain_path: list[str],
    chain_code: list[int],
    build_platform: str,
    timeout_seconds: int,
    benchmark_runs: int = DEFAULT_STAGE_BENCHMARK_RUNS,
    benchmark_warmup_runs: int = DEFAULT_STAGE_BENCHMARK_WARMUP_RUNS,
) -> dict[str, Any]:
    stage_code_dir = chain_code_path(chain_path or [stage], chain_code, ir)
    restore_source_files(stage_code_dir, source_snapshot)
    build_dir = stage_code_dir / f"build_stage_{safe_name(stage)}"
    if target_backend(ir) == "cpu":
        verified_ir = verify_cpu_build_and_run(
            ir=ir,
            source_dir=stage_code_dir,
            build_dir=build_dir,
            build_platform=build_platform,
            timeout_seconds=timeout_seconds,
            benchmark_runs=benchmark_runs,
            benchmark_warmup_runs=benchmark_warmup_runs,
        )
    else:
        verified_ir = verify_build_and_run(
            ir=ir,
            source_dir=stage_code_dir,
            build_dir=build_dir,
            build_platform=build_platform,
            timeout_seconds=timeout_seconds,
            benchmark_runs=benchmark_runs,
            benchmark_warmup_runs=benchmark_warmup_runs,
        )
    verified_ir.setdefault("verification", {})["summary"] = summarize_verification(verified_ir)
    verified_ir.setdefault("verification", {}).setdefault("stage_gates", {})[stage] = {
        "status": "pass" if oracle_verification_passed(verified_ir) else "fail",
        "build_dir": str(build_dir),
        "benchmark_runs": benchmark_runs,
        "benchmark_warmup_runs": benchmark_warmup_runs,
    }
    return verified_ir


def attach_stage_compile_run_report(stage_report: dict[str, Any], verified_ir: dict[str, Any]) -> dict[str, Any]:
    report = copy.deepcopy(stage_report)
    verification = verified_ir.get("verification", {}) or {}
    performance = verified_ir.get("performance", {}) or {}
    compile_status = (verification.get("compile") or {}).get("status")
    correctness_status = (verification.get("correctness") or {}).get("status")
    runtime_status = (verification.get("runtime_safety") or {}).get("status")
    accepted = oracle_verification_passed(verified_ir)
    report["stage_compile_run"] = {
        "enabled": True,
        "accepted": accepted,
        "compile_status": compile_status,
        "correctness_status": correctness_status,
        "runtime_safety_status": runtime_status,
        "cuda_error": (verification.get("runtime_safety") or {}).get("cuda_error"),
        "latency_ms": performance.get("latency_ms"),
        "gflops": performance.get("gflops"),
        "gflops_mean": performance.get("gflops_mean"),
        "benchmark_runs": performance.get("benchmark_runs"),
    }
    if not accepted:
        report["accepted"] = False
        report.setdefault("results", []).append(
            {
                "id": "STAGE_COMPILE_RUN_GATE",
                "status": "fail",
                "message": "stage completed IR checks, but compile/correctness/runtime stage gate failed",
            }
        )
    return report


def verify_strategy_step_compile_gate(
    ir: dict[str, Any],
    candidate_code_dir: Path,
    build_platform: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    next_ir = copy.deepcopy(ir)
    build_dir = candidate_code_dir / "build_step"
    build_dir.mkdir(parents=True, exist_ok=True)
    if target_backend(next_ir) == "cpu":
        exe_name = "sgpo_step_gemm_cpu" if build_platform == "linux" else "sgpo_step_gemm_cpu.exe"
        compile_result = compile_cpu_gemm(
            source_dir=candidate_code_dir,
            exe_path=build_dir / exe_name,
            timeout_seconds=timeout_seconds,
            ir=next_ir,
            build_platform=build_platform,
        )
        update_cpu_compile_result(next_ir, compile_result)
    else:
        exe_name = "sgpo_step_gemm" if build_platform == "linux" else "sgpo_step_gemm.exe"
        compile_result = compile_gemm(
            ir=next_ir,
            source_dir=candidate_code_dir,
            exe_path=build_dir / exe_name,
            timeout_seconds=timeout_seconds,
            vcvars64_path=DEFAULT_VCVARS64,
            build_platform=build_platform,
        )
        update_compile_result(next_ir, compile_result)

    verification = next_ir.setdefault("verification", {})
    summary = verification.setdefault("summary", {})
    compile_status = compile_result.get("status")
    summary.update(
        {
            "step_compile_gate": "pass" if compile_status == "pass" else "fail",
            "compile_status": compile_status,
            "compile_error_message": verification.get("compile", {}).get("error_message"),
            "build_platform": build_platform,
            "step_compile_build_dir": str(build_dir),
        }
    )
    if compile_status != "pass":
        next_ir["chain_step"] = {
            "status": "failed",
            "phase": "step_compile_gate",
            "candidate_code_dir": str(candidate_code_dir),
            "reason": verification.get("compile", {}).get("error_message") or "compile failed",
        }
        verification["accepted"] = False
        verification.setdefault("correctness", {})["status"] = "not_run"
        verification.setdefault("runtime_safety", {})["status"] = "not_run"
        verification.setdefault("runtime_safety", {})["cuda_error"] = None
        summary["chain_step_status"] = "failed"
        summary["failed_phase"] = "step_compile_gate"
    return next_ir


def step_compile_gate_passed(ir: dict[str, Any]) -> bool:
    return ir.get("verification", {}).get("compile", {}).get("status") == "pass"


def infer_strategy_profile(ir: dict[str, Any], strategy_index: dict[str, Any]) -> str:
    if target_backend(ir) == "cpu":
        return (
            ir.get("strategy", {}).get("profile")
            or ir.get("precision", {}).get("profile")
            or strategy_index.get("target", {}).get("default_profile")
            or DEFAULT_STRATEGY_PROFILE
        )
    return (
        ir.get("strategy", {}).get("profile")
        or ir.get("precision", {}).get("profile")
        or DEFAULT_STRATEGY_PROFILE
        or strategy_index.get("target", {}).get("default_profile")
    )


def run_unlocked_performance_phase(
    baseline_terminal_verification: dict[str, Any],
    raw_strategy_index: dict[str, Any],
    dependency_graph: dict[str, Any],
    strategy_library: dict[str, Any],
    client: OpenAICompatibleClient,
    user_question: str,
    profile: str,
    build_platform: str,
    compile_each_step: bool,
    step_compile_timeout_seconds: int,
    benchmark_runs: int,
    benchmark_warmup_runs: int,
) -> dict[str, Any]:
    phase1_top = select_top_correct_candidates(
        baseline_terminal_verification.get("verified_candidates", []) or [],
        DEFAULT_TOP_K_FINAL_RESULTS,
    )
    unlocked_candidates: list[dict[str, Any]] = []
    final_frontier: list[dict[str, Any]] = []
    stage_summaries: list[dict[str, Any]] = []
    chain_reports: list[dict[str, Any]] = []

    def invoke_unlock(item):
        chain_index, candidate = item
        with chain_client(client) as worker_client:
            return run_batch_unlock_for_terminal_candidate(
                base_candidate=copy.deepcopy(candidate),
                chain_index=chain_index,
                raw_strategy_index=raw_strategy_index,
                dependency_graph=dependency_graph,
                strategy_library=strategy_library,
                client=worker_client,
                user_question=user_question,
                profile=profile,
                build_platform=build_platform,
                compile_each_step=compile_each_step,
                step_compile_timeout_seconds=step_compile_timeout_seconds,
                benchmark_runs=benchmark_runs,
                benchmark_warmup_runs=benchmark_warmup_runs,
            )
    indexed_candidates = list(enumerate(phase1_top, start=1))
    outcomes = parallel_chain_map(
        invoke_unlock, indexed_candidates,
        lambda item: f"unlock_{item[0]}",
    )
    for (chain_index, candidate), outcome in zip(indexed_candidates, outcomes):
        for output, data in outcome.latest.items():
            save_json(output, data)
        if outcome.error is not None:
            chain_report = {
                "base_chain_id": candidate.get("strategy_id"), "verified_candidates": [],
                "summary": {"stage": "PerformanceUnlock", "status": "chain_error",
                            "base_chain_id": candidate.get("strategy_id"), "error_message": str(outcome.error)},
            }
        else:
            chain_report = outcome.value
        chain_reports.append(chain_report)
        unlocked_candidates.extend(chain_report.get("verified_candidates", []) or [])
        if chain_report.get("final_state"):
            final_frontier.append(chain_report["final_state"])
        stage_summaries.append(chain_report["summary"])

    resource_config = load_config(Path(DEFAULT_CONFIG)).get("bounded_resource_unlock", {}) or {}
    if resource_config.get("enabled", False):
        from SGPO.verification.bounded_resource_unlock import run_bounded_unlock
        import uuid
        resource_seeds = select_top_correct_candidates(
            [*phase1_top, *unlocked_candidates], DEFAULT_TOP_K_FINAL_RESULTS)
        resource_candidates = run_bounded_unlock(
            resource_seeds, ROOT / "results" / "code" / ("resource_unlock_" + uuid.uuid4().hex[:12]),
            build_platform, benchmark_runs, benchmark_warmup_runs,
            load_layout_trials=resource_config.get("load_layout_trials", True))
        unlocked_candidates.extend(resource_candidates)
        stage_summaries.append({"stage": "PerformanceUnlock.BoundedResources",
                                "candidate_count": len(resource_candidates),
                                "accepted_count": sum(bool(c.get("accepted")) for c in resource_candidates)})
    best_candidate = choose_best_candidate([item for item in unlocked_candidates if item.get("accepted")])
    if best_candidate is None:
        best_candidate = choose_best_candidate(unlocked_candidates)
    top_terminal_results = build_top_terminal_results(unlocked_candidates, DEFAULT_TOP_K_FINAL_RESULTS)
    terminal_verification = {
        "verified_candidates": unlocked_candidates,
        "best_candidate": best_candidate,
        "best_state": None,
        "top_terminal_results": top_terminal_results,
        "summary": {
            "stage": "PerformanceUnlock",
            "phase": "performance_unlock",
            "selection_mode": "category_local_single_objective_unlock",
            "profile": profile,
            "input_correct_phase1_chain_count": len(phase1_top),
            "terminal_chain_count": len(unlocked_candidates),
            "verified_terminal_chain_count": len(unlocked_candidates),
            "accepted_terminal_chain_count": len([item for item in unlocked_candidates if item.get("accepted")]),
            "best_chain_id": best_candidate.get("strategy_id") if best_candidate else None,
            "best_chain_gflops": candidate_gflops(best_candidate),
            "top_k_final_results": DEFAULT_TOP_K_FINAL_RESULTS,
            "top_terminal_results": top_terminal_results,
            "chain_reports": chain_reports,
        },
    }
    return {
        "profile": profile,
        "stage_order": ["PerformanceUnlock.BatchPlan", "PerformanceUnlock.BatchApply"],
        "frontier": final_frontier,
        "terminal_verification": terminal_verification,
        "best_candidate": best_candidate,
        "stage_summaries": stage_summaries + [terminal_verification["summary"]],
    }


def terminal_verification_has_correct_chain(terminal_verification: dict[str, Any]) -> bool:
    for candidate in terminal_verification.get("verified_candidates", []) or []:
        summary = candidate.get("verified_ir", {}).get("verification", {}).get("summary", {}) or {}
        if summary.get("correctness_status") in PERFORMANCE_UNLOCK_SUCCESS_STATUSES:
            return True
    return False


def summarize_performance_phase(performance_phase: dict[str, Any]) -> dict[str, Any]:
    terminal = performance_phase.get("terminal_verification", {}) or {}
    return {
        "profile": performance_phase.get("profile"),
        "stage_order": performance_phase.get("stage_order"),
        "frontier_state_count": len(performance_phase.get("frontier", []) or []),
        "best_strategy_id": (performance_phase.get("best_candidate") or {}).get("strategy_id"),
        "best_code_dir": (performance_phase.get("best_candidate") or {}).get("candidate_code_dir"),
        "best_gflops": candidate_gflops(performance_phase.get("best_candidate")),
        "top_terminal_results": terminal.get("top_terminal_results", []),
        "stage_summaries": performance_phase.get("stage_summaries", []),
    }


def select_top_correct_candidates(candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    correct = [candidate for candidate in candidates if candidate_has_correctness_pass(candidate)]
    return sorted(correct, key=lambda item: candidate_gflops(item) or -1.0, reverse=True)[:top_k]


def run_batch_unlock_for_terminal_candidate(
    base_candidate: dict[str, Any],
    chain_index: int,
    raw_strategy_index: dict[str, Any],
    dependency_graph: dict[str, Any],
    strategy_library: dict[str, Any],
    client: OpenAICompatibleClient,
    user_question: str,
    profile: str,
    build_platform: str,
    compile_each_step: bool,
    step_compile_timeout_seconds: int,
    benchmark_runs: int,
    benchmark_warmup_runs: int,
) -> dict[str, Any]:
    base_ir = copy.deepcopy(base_candidate.get("verified_ir", {}) or {})
    base_history = copy_history(base_candidate.get("history", {}) or base_ir.get("strategy", {}))
    base_history.setdefault("events", []).append(
        {
            "stage": "PerformanceUnlock",
            "status": "started",
            "profile": profile,
            "base_chain_id": base_candidate.get("strategy_id"),
        }
    )
    working_state = make_frontier_state(
        state_id=f"unlock_{chain_index:04d}",
        current_ir=base_ir,
        source_snapshot=base_candidate.get("source_snapshot", {}),
        history=base_history,
        path=list(base_candidate.get("path", []) or base_ir.get("strategy", {}).get("chain_path", []) or []),
        path_code=list(base_candidate.get("path_code", []) or []),
        last_candidate=base_candidate,
        code_dir=base_candidate.get("candidate_code_dir"),
    )
    matrix_profile = build_matrix_profile(base_ir)
    verified_candidates: list[dict[str, Any]] = []
    batch_summaries: list[dict[str, Any]] = []
    batch_plans: list[dict[str, Any]] = []
    no_improvement_rounds = 0
    compilation_count = 0
    candidate_strategy_count = 0
    from SGPO.generate_ir.unlock_feedback import choose_unlock_steps, infrastructure_failure
    pending_steps = []
    from SGPO.llm.unlock_category_selector import CATEGORIES, select_category_plan
    category_visits = {}
    stop_reason = "max_feedback_rounds_reached"
    for round_index in range(1, FEEDBACK_SEARCH_LIMITS["max_rounds"] * len(CATEGORIES) + 1):
        if compilation_count >= FEEDBACK_SEARCH_LIMITS["max_total_compilations"]:
            stop_reason = "compilation_budget_exhausted"
            break
        round_ir = working_state["current_ir"]
        matrix_profile = build_matrix_profile(round_ir)
        round_history = working_state["history"]
        phase2_index = build_phase2_unlock_strategy_index(
            raw_strategy_index=raw_strategy_index,
            ir=round_ir,
            dependency_graph=dependency_graph,
            history=round_history,
            matrix_profile=matrix_profile,
        )
        candidate_strategy_count = phase2_index.get("strategy_count", 0)
        if not candidate_strategy_count:
            stop_reason = "no_remaining_unlock_candidates"
            break
        code_dir = Path(
            working_state.get("code_dir")
            or chain_code_path(working_state["path"], working_state["path_code"], round_ir)
        )
        code_summary = build_unlock_code_summary(
            code_dir, backend_llm_context_files(round_ir), round_ir, round_history,
        )
        verifier_summary = build_unlock_verifier_summary(
            working_state.get("last_candidate") or base_candidate
        )
        verifier_summary["feedback_failed_strategy_counts"] = dict(
            round_history.get("failed_strategy_counts", {}) or {}
        )
        verifier_summary["failed_unlock_bundles"] = list(round_history.get("failed_unlock_bundles", []) or [])
        verifier_summary["ineffective_unlock_bundles"] = list(
            round_history.get("ineffective_unlock_bundles", []) or []
        )
        verifier_summary["recent_unlock_defect"] = {
            "scope": "historical_attempt_not_necessarily_current_source",
            "diagnosis": round_history.get("recent_unlock_defect"),
        }
        batch_plan_error = None
        try:
            batch_plan = select_category_plan(
                client=client,
                strategy_index=phase2_index,
                visits=category_visits,
                history=round_history,
                max_visits=FEEDBACK_SEARCH_LIMITS["max_rounds"],
                optir_summary=compact_ir_summary(round_ir),
                code_summary=code_summary,
                matrix_profile=matrix_profile,
                verification_summary=verifier_summary,
                user_question=user_question,
            )
        except Exception as exc:
            batch_plan_error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "fallback": "skip_category_selection_no_implicit_strategy",
            }
            batch_plan = {"batches": [], "selection_mode": "category_local"}
        if not batch_plan_error and batch_plan.get("category") is None:
            stop_reason = "category_candidates_or_visit_budget_exhausted"
            break
        plan_record = {
            "phase": "performance_unlock",
            "feedback_round": round_index,
            "profile": profile,
            "base_chain_id": base_candidate.get("strategy_id"),
            "base_code_dir": str(code_dir),
            "matrix_profile": matrix_profile,
            "candidate_strategy_count": candidate_strategy_count,
            "selected_strategy_ids": list(round_history.get("applied_strategy_ids", []) or []),
            "unselected_strategy_ids": [
                item.get("strategy_id") for item in phase2_index.get("strategies", []) or []
            ],
            "batch_plan": batch_plan,
            "batch_plan_error": batch_plan_error,
            "verifier_summary": verifier_summary,
            "dynamic_dependency_overlay": build_dynamic_dependency_overlay(batch_plan),
            "feedback_limits": dict(FEEDBACK_SEARCH_LIMITS),
        }
        round_batches, pending_steps, dropped_steps = choose_unlock_steps(
            pending_steps, batch_plan.get("batches", []),
            {item.get("strategy_id") for item in phase2_index.get("strategies", [])},
            round_history, FEEDBACK_SEARCH_LIMITS["experiments_per_round"],
        )
        plan_record["execution_granularity"] = "single_objective_coupled_regions"
        plan_record["category_visits"] = dict(category_visits)
        plan_record["execution_steps"] = copy.deepcopy(round_batches)
        plan_record["pending_steps"] = copy.deepcopy(pending_steps)
        plan_record["pending_steps_filtered"] = dropped_steps
        batch_plans.append(plan_record)
        save_unlock_artifact(
            DEFAULT_UNLOCK_BATCH_PLAN_OUTPUT,
            f"{base_candidate.get('strategy_id') or f'chain_{chain_index:04d}'}.round_{round_index}",
            None,
            plan_record,
        )

        round_improved = False
        round_has_measurements = False
        infrastructure_blocked = False
        for local_batch_number, batch in enumerate(round_batches, start=1):
            if not batch.get("strategy_ids"):
                continue
            if compilation_count >= FEEDBACK_SEARCH_LIMITS["max_total_compilations"]:
                pending_steps = round_batches[local_batch_number - 1:] + pending_steps
                break
            try:
                result = run_unlock_batch(
                    working_state=working_state,
                    batch=batch,
                    batch_number=(round_index - 1) * FEEDBACK_SEARCH_LIMITS["experiments_per_round"] + local_batch_number,
                    base_chain_id=base_candidate.get("strategy_id") or f"chain_{chain_index:04d}",
                    raw_strategy_index=raw_strategy_index,
                    strategy_library=strategy_library,
                    client=client,
                    build_platform=build_platform,
                    compile_each_step=compile_each_step,
                    step_compile_timeout_seconds=step_compile_timeout_seconds,
                    benchmark_runs=benchmark_runs,
                    benchmark_warmup_runs=benchmark_warmup_runs,
                    dependency_graph=dependency_graph,
                    compilation_budget=unlock_experiment_compilation_budget(
                        round_index=round_index,
                        local_batch_number=local_batch_number,
                        compilation_count=compilation_count,
                    ),
                )
            except Exception as exc:
                failure = {
                    "stage": "PerformanceUnlock",
                    "phase": "unlock_candidate_execution",
                    "status": "failed",
                    "batch_id": batch.get("batch_id"),
                    "strategy_ids": list(batch.get("strategy_ids", []) or []),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "feedback_round": round_index,
                    "rolled_back": True,
                }
                save_unlock_artifact(
                    DEFAULT_UNLOCK_BATCH_DEFECT_OUTPUT,
                    base_candidate.get("strategy_id") or f"chain_{chain_index:04d}",
                    batch.get("batch_id"),
                    failure,
                )
                batch_summaries.append(failure)
                record_unlock_feedback(
                    working_state,
                    batch,
                    {"summary": failure, "verified_candidates": [], "compilation_count": 0},
                    round_index,
                )
                if infrastructure_failure(failure):
                    pending_steps = round_batches[local_batch_number - 1:] + pending_steps
                    infrastructure_blocked = True
                    break
                continue
            result["summary"]["feedback_round"] = round_index
            compilation_count += int(result.get("compilation_count", 0) or 0)
            batch_summaries.append(result["summary"])
            verified_candidates.extend(result.get("verified_candidates", []) or [])
            round_has_measurements |= any(
                candidate.get("accepted") and candidate_gflops(candidate) is not None
                for candidate in result.get("verified_candidates", []) or []
            )
            if infrastructure_failure(result["summary"]):
                record_unlock_feedback(working_state, batch, result, round_index)
                pending_steps = round_batches[local_batch_number - 1:] + pending_steps
                infrastructure_blocked = True
                break
            if result.get("accepted_state"):
                working_state = result["accepted_state"]
                round_improved = True
            else:
                record_unlock_feedback(working_state, batch, result, round_index)
        if infrastructure_blocked:
            stop_reason = "infrastructure_error_pending_work_preserved"
            break
        if round_improved:
            no_improvement_rounds = 0
        elif round_has_measurements:
            no_improvement_rounds += 1
        if (no_improvement_rounds >= FEEDBACK_SEARCH_LIMITS["no_improvement_patience"]
                and not pending_steps and not batch_plan.get("unvisited_categories") and not batch_plan_error):
            stop_reason = "no_performance_improvement_patience_exhausted"
            break
        if compilation_count >= FEEDBACK_SEARCH_LIMITS["max_total_compilations"]:
            stop_reason = "compilation_budget_exhausted"
            break

    return {
        "base_chain_id": base_candidate.get("strategy_id"),
        "base_code_dir": base_candidate.get("candidate_code_dir"),
        "matrix_profile": matrix_profile,
        "batch_plan": batch_plans[-1]["batch_plan"] if batch_plans else {"batches": []},
        "feedback_rounds": batch_plans,
        "verified_candidates": verified_candidates,
        "final_state": working_state,
        "summary": {
            "stage": "PerformanceUnlock",
            "phase": "performance_unlock",
            "base_chain_id": base_candidate.get("strategy_id"),
            "candidate_strategy_count": candidate_strategy_count,
            "planned_batch_count": sum(len(item["batch_plan"].get("batches", []) or []) for item in batch_plans),
            "feedback_round_count": len(batch_plans),
            "verified_candidate_count": len(verified_candidates),
            "accepted_candidate_count": len([item for item in verified_candidates if item.get("accepted")]),
            "best_gflops": candidate_gflops(choose_best_candidate(verified_candidates)),
            "matrix_profile": matrix_profile,
            "batch_summaries": batch_summaries,
            "stopped_after_no_improvement": stop_reason == "no_performance_improvement_patience_exhausted",
            "compilation_count": compilation_count,
            "compilation_budget": FEEDBACK_SEARCH_LIMITS["max_total_compilations"],
            "stop_reason": stop_reason,
            "pending_steps": copy.deepcopy(pending_steps),
        },
    }


def unlock_experiment_compilation_budget(
    round_index: int,
    local_batch_number: int,
    compilation_count: int,
) -> int:
    remaining_budget = max(1, FEEDBACK_SEARCH_LIMITS["max_total_compilations"] - compilation_count)
    completed_slots = (
        (round_index - 1) * FEEDBACK_SEARCH_LIMITS["experiments_per_round"]
        + local_batch_number - 1
    )
    total_slots = FEEDBACK_SEARCH_LIMITS["max_rounds"] * FEEDBACK_SEARCH_LIMITS["experiments_per_round"]
    remaining_slots_after_current = max(0, total_slots - completed_slots - 1)
    # Reserve one compile for every future experiment. The remaining three
    # compiles are distributed as one repair opportunity per feedback round.
    repair_bonus = 1 if local_batch_number == 1 and remaining_budget > remaining_slots_after_current + 1 else 0
    return min(remaining_budget, 1 + repair_bonus)


def record_unlock_feedback(
    working_state: dict[str, Any],
    batch: dict[str, Any],
    result: dict[str, Any],
    round_index: int,
) -> None:
    history = working_state.setdefault("history", {})
    strategy_ids = sorted(set(batch.get("strategy_ids", []) or []))
    summary = result.get("summary", {}) or {}
    from SGPO.generate_ir.unlock_feedback import infrastructure_failure
    if infrastructure_failure(summary):
        history.setdefault("infrastructure_failures", []).append({
            "feedback_round": round_index, "strategy_ids": strategy_ids,
            "batch_id": batch.get("batch_id"), "summary": copy.deepcopy(summary),
        })
        history.setdefault("events", []).append({
            "stage": "PerformanceUnlock", "status": "infrastructure_error",
            "strategy_ids": strategy_ids, "feedback_round": round_index,
        })
        return
    status = summary.get("status") or "failed"
    bundle_key = "|".join(strategy_ids)
    bucket = "ineffective_unlock_bundles" if status in {"verified_without_performance_improvement", "unchanged_implementation"} else "failed_unlock_bundles"
    bundles = history.setdefault(bucket, [])
    if bundle_key and bundle_key not in bundles:
        bundles.append(bundle_key)
    if bucket == "failed_unlock_bundles":
        failed_counts = history.setdefault("failed_strategy_counts", {})
        for strategy_id in strategy_ids:
            failed_counts[strategy_id] = failed_counts.get(strategy_id, 0) + 1
    recent_defect = summary.get("last_defect_diagnosis") or summary.get("defect_diagnosis")
    if recent_defect:
        history["recent_unlock_defect"] = recent_defect
    history.setdefault("events", []).append(
        {
            "stage": "PerformanceUnlock",
            "feedback_round": round_index,
            "batch_id": batch.get("batch_id"),
            "strategy_ids": strategy_ids,
            "status": status,
            "reason": summary.get("reason"),
        }
    )
    strategy_state = working_state.setdefault("current_ir", {}).setdefault("strategy", {})
    strategy_state["failed_strategy_counts"] = dict(history.get("failed_strategy_counts", {}) or {})
    strategy_state["failed_unlock_bundles"] = list(history.get("failed_unlock_bundles", []) or [])
    strategy_state["ineffective_unlock_bundles"] = list(history.get("ineffective_unlock_bundles", []) or [])


def build_dynamic_dependency_overlay(batch_plan: dict[str, Any]) -> dict[str, Any]:
    """Represent the LLM plan as a per-chain soft ordering over static hard dependencies."""
    batches = batch_plan.get("batches", []) or []
    nodes = []
    edges = []
    previous_ids: list[str] = []
    for batch in batches:
        strategy_ids = list(batch.get("strategy_ids", []) or [])
        nodes.extend(strategy_id for strategy_id in strategy_ids if strategy_id not in nodes)
        for source in previous_ids:
            for target in strategy_ids:
                edges.append({"from": source, "to": target, "kind": "llm_soft_order"})
        previous_ids = strategy_ids
    return {
        "scope": "current_chain_only",
        "nodes": nodes,
        "edges": edges,
        "hard_dependencies_unchanged": True,
    }


def expand_unlock_plan_to_strategy_steps(batch_plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep LLM batch ordering, but execute every strategy as an isolated transaction."""
    steps: list[dict[str, Any]] = []
    for batch in batch_plan.get("batches", []) or []:
        strategy_ids = list(batch.get("strategy_ids", []) or [])
        for step_index, strategy_id in enumerate(strategy_ids, start=1):
            steps.append(
                {
                    **batch,
                    "batch_id": f"{safe_batch_id(batch.get('batch_id') or 'batch')}.s{step_index}",
                    "planning_batch_id": safe_batch_id(batch.get("batch_id") or "batch"),
                    "strategy_ids": [strategy_id],
                    "execution_granularity": "single_strategy",
                    "step_index": step_index,
                    "step_count": len(strategy_ids),
                }
            )
    return steps


def run_unlock_batch(
    working_state: dict[str, Any],
    batch: dict[str, Any],
    batch_number: int,
    base_chain_id: str,
    raw_strategy_index: dict[str, Any],
    strategy_library: dict[str, Any],
    client: OpenAICompatibleClient,
    build_platform: str,
    compile_each_step: bool,
    step_compile_timeout_seconds: int,
    benchmark_runs: int,
    benchmark_warmup_runs: int,
    dependency_graph: dict[str, Any] | None = None,
    compilation_budget: int | None = None,
) -> dict[str, Any]:
    verified_candidates: list[dict[str, Any]] = []
    compilation_count = 0
    attempted_bundles = [batch, *build_unlock_fallback_batches(batch, dependency_graph)][
        : FEEDBACK_SEARCH_LIMITS["max_rollback_depth"] + 1
    ]
    last_result: dict[str, Any] | None = None
    last_verified_ir: dict[str, Any] | None = None
    last_diagnosis: dict[str, Any] | None = None
    for fallback_index, bundle in enumerate(attempted_bundles, start=0):
        if compilation_budget is not None and compilation_count >= compilation_budget:
            break
        strategy_ids = bundle.get("strategy_ids", []) or []
        if not strategy_ids:
            continue
        bundle_strategy = build_unlock_bundle_strategy(bundle, raw_strategy_index, strategy_library)
        pre_report = check_before_codegen(working_state["current_ir"], bundle_strategy)
        if not pre_report["preconditions_ok"]:
            save_unlock_artifact(DEFAULT_UNLOCK_BATCH_DEFECT_OUTPUT, base_chain_id, bundle.get("batch_id"),
                                 {"status": "preconditions_failed", "checker_report": pre_report})
            continue
        precheck_item = {
            "strategy_id": bundle_strategy["strategy_id"],
            "strategy_applicable": True,
            "preconditions_ok": True,
            "hard_constraints_ok": True,
            "failed_checks": [],
            "checker_report": {"phase": "batch_unlock_precheck", "accepted": True},
        }
        chain_path = [
            *working_state.get("path", []),
            f"UnlockBatch:{bundle.get('batch_id')}",
            *strategy_ids,
        ]
        chain_code = [
            *working_state.get("path_code", []),
            batch_number,
            fallback_index + 1,
        ]
        namespace = chain_code_key(chain_code)
        from SGPO.verification.locked_repair import begin_optimization_transaction
        optimization_ir = begin_optimization_transaction(working_state["current_ir"])
        candidate_result = run_strategy_candidate(
            stage="PerformanceUnlock",
            base_ir=optimization_ir,
            base_source_snapshot=working_state["source_snapshot"],
            strategy=bundle_strategy,
            precheck_item=precheck_item,
            client=client,
            candidate_namespace=namespace,
            chain_path=chain_path,
            chain_code=chain_code,
            semantic_precheck_terminal=False,
            build_platform=build_platform,
            compile_each_step=compile_each_step,
            step_compile_timeout_seconds=step_compile_timeout_seconds,
        )
        last_result = candidate_result
        maybe_copy_unlock_patch(candidate_result, base_chain_id, bundle.get("batch_id"))
        if not candidate_result.get("accepted"):
            last_verified_ir = candidate_result.get("verified_ir")
            last_diagnosis = (last_verified_ir or {}).get("defect_diagnosis", {})
            save_unlock_artifact(
                DEFAULT_UNLOCK_BATCH_DEFECT_OUTPUT,
                base_chain_id,
                bundle.get("batch_id"),
                last_diagnosis or {"status": "generation_failed", "strategy_ids": strategy_ids},
            )
            from SGPO.generate_ir.unlock_feedback import infrastructure_failure
            last_verification = (last_verified_ir or {}).get("verification", {}).get("summary", {})
            if infrastructure_failure(last_verification):
                failure = {
                    "base_chain_id": base_chain_id, "batch_id": bundle.get("batch_id"),
                    "strategy_ids": strategy_ids, "accepted": False,
                    "status": "infrastructure_error", "failure_class": "infrastructure_error",
                    "reason": "Execution infrastructure failed; no strategy performance conclusion",
                    "last_verification": last_verification,
                }
                save_unlock_artifact(DEFAULT_UNLOCK_BATCH_RESULT_OUTPUT, base_chain_id, bundle.get("batch_id"), failure)
                return {"accepted_state": None, "verified_candidates": verified_candidates,
                        "summary": failure, "compilation_count": compilation_count}
            continue

        chain_dir = Path(candidate_result["candidate_code_dir"])
        terminal_ir = copy.deepcopy(candidate_result["verified_ir"])
        if 'Compiler.FastMath.Enabled' in strategy_ids:
            terminal_ir.setdefault('compiler', {})['use_fast_math'] = True
        from SGPO.verification.implementation_identity import unchanged_verified_parent
        duplicate = unchanged_verified_parent(
            working_state['current_ir'], working_state['source_snapshot'], terminal_ir,
            snapshot_source_files(chain_dir, backend_code_files(terminal_ir)), build_platform)
        if duplicate:
            result_record = {
                'base_chain_id': base_chain_id, 'batch_id': bundle.get('batch_id'),
                'strategy_ids': strategy_ids, 'status': 'unchanged_implementation',
                'accepted': False, 'runtime_correct_parent_retained': True,
                'candidate_code_dir': str(chain_dir), 'implementation_fingerprint': duplicate,
                'reason': 'Effective source and CUDA options match the verified parent; no new benchmark or strategy benefit claimed.',
            }
            save_unlock_artifact(DEFAULT_UNLOCK_BATCH_RESULT_OUTPUT, base_chain_id, bundle.get('batch_id'), result_record)
            return {'accepted_state': None, 'verified_candidates': verified_candidates,
                    'summary': result_record, 'compilation_count': compilation_count}
        terminal_ir.setdefault("strategy", {})["applied_strategy_ids"] = append_unique(
            working_state["history"].get("applied_strategy_ids", []),
            strategy_ids,
        )
        terminal_ir.setdefault("strategy", {})["chain_id"] = f"{base_chain_id}.{bundle.get('batch_id')}"
        terminal_ir.setdefault("strategy", {})["chain_code"] = chain_code_key(chain_code)
        terminal_ir.setdefault("strategy", {})["chain_path"] = chain_path
        from SGPO.verification.locked_repair import locked_tile_parameters
        terminal_ir['locked_repair_contract'] = {
            'strategy_ids': list(terminal_ir['strategy']['applied_strategy_ids']),
            'tile_parameters': locked_tile_parameters(terminal_ir, working_state['source_snapshot'].get('cuda_kernel.cuh', '')),
        }
        verified_ir = verify_terminal_chain_once(
            terminal_ir=terminal_ir,
            chain_dir=chain_dir,
            chain_id=f"{base_chain_id}.{bundle.get('batch_id')}",
            strategy_library=strategy_library,
            attempt=1,
            build_platform=build_platform,
            benchmark_runs=benchmark_runs,
            benchmark_warmup_runs=benchmark_warmup_runs,
        )
        compilation_count += 1
        repair_attempts = []
        repair_baseline_snapshot = snapshot_source_files(chain_dir, backend_code_files(terminal_ir))
        seen_repair_hashes: set[str] = set()
        from SGPO.verification.strategy_application import strategy_application
        def retain_runtime_candidate(ir, suffix):
            if not terminal_chain_is_accepted(ir):
                return
            preserved_dir = chain_dir / ('retained_' + suffix)
            snapshot = snapshot_source_files(chain_dir, backend_code_files(ir))
            restore_source_files(preserved_dir, snapshot)
            saved_ir = copy.deepcopy(ir)
            application = strategy_application(saved_ir, strategy_ids)
            saved_ir['strategy_application'] = application
            history = copy_history(working_state['history'])
            history['selected_strategy_ids'] = append_unique(history.get('selected_strategy_ids', []), strategy_ids)
            history['applied_strategy_ids'] = append_unique(history.get('applied_strategy_ids', []), application['realized_strategy_ids'])
            save_json(preserved_dir / 'verified_ir.json', saved_ir)
            verified_candidates.append({
                'accepted': True, 'stage': 'PerformanceUnlock',
                'strategy_id': f"{base_chain_id}.{bundle.get('batch_id')}.{suffix}",
                'verified_ir': saved_ir, 'source_snapshot': snapshot,
                'candidate_code_dir': str(preserved_dir), 'history': history,
                'path': chain_path, 'path_code': chain_code, 'source_phase': 'performance_unlock',
            })
        for repair_attempt in range(1, DEFAULT_UNLOCK_BATCH_REPAIR_ATTEMPTS + 1):
            application = strategy_application(verified_ir, strategy_ids)
            if terminal_chain_is_accepted(verified_ir) and not application['repair_required']:
                break
            retain_runtime_candidate(verified_ir, f'before_repair_{repair_attempt}')
            if application['repair_required']:
                diagnosis = verified_ir.setdefault('defect_diagnosis', {})
                diagnosis.setdefault('defects', []).append({
                    'defect_type': 'StrategyImplementation.MissingSelectedOptimization',
                    'related_strategy': ','.join(application['missing_strategy_ids']),
                    'repair_action': 'Implement the selected optimization without removing other optimizations. Preserve runtime correctness.',
                    'evidence': application,
                })
                diagnosis['defect_count'] = len(diagnosis['defects'])
            if compilation_budget is not None and compilation_count >= compilation_budget:
                break
            before_repair_ir = copy.deepcopy(verified_ir)
            before_repair_source = snapshot_source_files(chain_dir, backend_code_files(verified_ir))
            save_json(chain_dir / f"unlock_repair_checkpoint_{repair_attempt}.json",
                      {"ir": before_repair_ir, "source_snapshot": before_repair_source})
            repair_result = repair_chain_locally(
                source_chain_dir=chain_dir,
                output_chain_dir=chain_dir,
                diagnosis=verified_ir.get("defect_diagnosis", {}),
                ir=verified_ir,
                client=client,
                strategy_library=strategy_library,
                repair_attempt=repair_attempt,
                baseline_snapshot=repair_baseline_snapshot,
            )
            repair_result["repair_attempt"] = repair_attempt
            save_json(chain_dir / f"unlock_semantic_repair_attempt_{repair_attempt}.json", repair_result)
            repair_attempts.append(repair_result)
            repair_hash = repair_result.get("source_hash_after")
            if repair_result.get("status") != "repair_generated" or not repair_hash:
                verified_ir["repair_feedback"] = repair_result
                if repair_result.get("status") == "repair_failed":
                    continue
                break
            if repair_hash in seen_repair_hashes:
                repair_result["status"] = "repair_retry_no_progress"
                repair_result["reason"] = "repair produced a previously tested source snapshot"
                restore_source_files(chain_dir, before_repair_source)
                verified_ir = before_repair_ir
                verified_ir["repair_feedback"] = repair_result
                save_json(chain_dir / f"unlock_semantic_repair_attempt_{repair_attempt}.json", repair_result)
                continue
            seen_repair_hashes.add(repair_hash)
            verified_ir["locked_repair_contract"] = {
                "strategy_ids": repair_result.get("locked_strategy_ids", []),
                "tile_parameters": repair_result.get("locked_tile_parameters", {}),
            }
            verified_ir = verify_terminal_chain_once(
                terminal_ir=verified_ir,
                chain_dir=chain_dir,
                chain_id=f"{base_chain_id}.{bundle.get('batch_id')}",
                strategy_library=strategy_library,
                attempt=repair_attempt + 1,
                build_platform=build_platform,
                benchmark_runs=benchmark_runs,
                benchmark_warmup_runs=benchmark_warmup_runs,
            )
            compilation_count += 1
            from SGPO.verification.repair_transaction import finish_repair
            verified_ir, rollback = finish_repair(
                before_repair_ir, verified_ir, before_repair_source,
                snapshot_source_files(chain_dir, backend_code_files(verified_ir)),
                repair_result, terminal_chain_is_accepted(verified_ir)
                and strategy_application(verified_ir, strategy_ids)['continuation_allowed'])
            if rollback:
                restore_source_files(chain_dir, before_repair_source)
            save_json(chain_dir / f"unlock_semantic_repair_attempt_{repair_attempt}.json", repair_result)
        if repair_attempts:
            verified_ir.setdefault("terminal_repair", {})["attempts"] = repair_attempts
            verified_ir.setdefault("terminal_repair", {})["attempt_count"] = len(repair_attempts)
        # A repair may restore the parent verbatim. Its fresh timing must not
        # be credited as a new optimization merely because it ran faster.
        duplicate = unchanged_verified_parent(
            working_state['current_ir'], working_state['source_snapshot'], verified_ir,
            snapshot_source_files(chain_dir, backend_code_files(verified_ir)), build_platform)
        if duplicate:
            result_record = {
                'base_chain_id': base_chain_id, 'batch_id': bundle.get('batch_id'),
                'strategy_ids': strategy_ids, 'status': 'unchanged_implementation',
                'accepted': False, 'runtime_correct_parent_retained': True,
                'candidate_code_dir': str(chain_dir), 'implementation_fingerprint': duplicate,
                'reason': 'Repair produced the parent implementation; measured timing is not a strategy gain.',
                'repair_attempt_count': len(repair_attempts),
            }
            save_unlock_artifact(DEFAULT_UNLOCK_BATCH_RESULT_OUTPUT, base_chain_id, bundle.get('batch_id'), result_record)
            return {'accepted_state': None, 'verified_candidates': verified_candidates,
                    'summary': result_record, 'compilation_count': compilation_count}
        accepted = terminal_chain_is_accepted(verified_ir)
        application = strategy_application(verified_ir, strategy_ids)
        verified_ir['strategy_application'] = application
        history = copy_history(working_state["history"])
        history['selected_strategy_ids'] = append_unique(history.get('selected_strategy_ids', []), strategy_ids)
        history["applied_strategy_ids"] = append_unique(history.get("applied_strategy_ids", []), application['realized_strategy_ids'])
        history.setdefault("events", []).append(
            {
                "stage": "PerformanceUnlock",
                "batch_id": bundle.get("batch_id"),
                "strategy_ids": strategy_ids,
                "status": "accepted" if accepted else "failed",
                "fallback_index": fallback_index,
                "verification": verified_ir.get("verification", {}).get("summary", {}),
            }
        )
        candidate = {
            "accepted": accepted,
            "stage": "PerformanceUnlock",
            "strategy_id": f"{base_chain_id}.{bundle.get('batch_id')}",
            "verified_ir": verified_ir,
            "source_snapshot": snapshot_source_files(chain_dir, backend_code_files(verified_ir)),
            "candidate_code_dir": str(chain_dir),
            "history": history,
            "path": chain_path,
            "path_code": chain_code,
            "source_phase": "performance_unlock",
        }
        verified_candidates.append(candidate)
        result_record = {
            "base_chain_id": base_chain_id,
            "batch_id": bundle.get("batch_id"),
            "strategy_ids": strategy_ids,
            "accepted": accepted,
            "fallback_index": fallback_index,
            "candidate_code_dir": str(chain_dir),
            "verification": verified_ir.get("verification", {}).get("summary", {}),
            "performance": verified_ir.get("performance", {}),
            "defect_diagnosis": verified_ir.get("defect_diagnosis", {}),
            "strategy_application": application,
        }
        save_unlock_artifact(DEFAULT_UNLOCK_BATCH_RESULT_OUTPUT, base_chain_id, bundle.get("batch_id"), result_record)
        save_unlock_artifact(DEFAULT_UNLOCK_BATCH_DEFECT_OUTPUT, base_chain_id, bundle.get("batch_id"), verified_ir.get("defect_diagnosis", {}))
        if accepted and not application['continuation_allowed']:
            result_record['status'] = 'retained_runtime_only_' + application['status']
            save_unlock_artifact(DEFAULT_UNLOCK_BATCH_RESULT_OUTPUT, base_chain_id, bundle.get('batch_id'), result_record)
            return {'accepted_state': None, 'verified_candidates': verified_candidates,
                    'summary': result_record, 'compilation_count': compilation_count}
        if accepted:
            improved = candidate_gflops(candidate) is not None and (
                candidate_gflops(working_state.get("last_candidate")) is None
                or candidate_gflops(candidate) > candidate_gflops(working_state["last_candidate"])
            )
            if not improved:
                result_record["status"] = "verified_without_performance_improvement"
                save_unlock_artifact(DEFAULT_UNLOCK_BATCH_RESULT_OUTPUT, base_chain_id, bundle.get("batch_id"), result_record)
                return {
                    "accepted_state": None,
                    "verified_candidates": verified_candidates,
                    "summary": result_record,
                    "compilation_count": compilation_count,
                }
            accepted_state = make_frontier_state(
                state_id=f"{working_state['state_id']}->{safe_name(str(bundle.get('batch_id')))}",
                current_ir=verified_ir,
                source_snapshot=candidate["source_snapshot"],
                history=history,
                path=chain_path,
                path_code=chain_code,
                last_candidate=candidate,
                code_dir=str(chain_dir),
            )
            return {
                "accepted_state": accepted_state,
                "verified_candidates": verified_candidates,
                "summary": result_record,
                "compilation_count": compilation_count,
            }
        last_verified_ir = verified_ir
        last_diagnosis = verified_ir.get("defect_diagnosis", {})

    rollback_record = {
        "base_chain_id": base_chain_id,
        "batch_id": batch.get("batch_id"),
        "strategy_ids": batch.get("strategy_ids", []),
        "accepted": False,
        "status": "rolled_back",
        "reason": "all batch and fallback repair attempts failed",
        "last_verification": (last_verified_ir or {}).get("verification", {}).get("summary", {}),
        "last_defect_diagnosis": last_diagnosis,
    }
    save_unlock_artifact(DEFAULT_UNLOCK_BATCH_RESULT_OUTPUT, base_chain_id, batch.get("batch_id"), rollback_record)
    return {
        "accepted_state": None,
        "verified_candidates": verified_candidates,
        "last_result": last_result,
        "summary": rollback_record,
        "compilation_count": compilation_count,
    }


def build_matrix_profile(ir: dict[str, Any]) -> dict[str, Any]:
    problem = ir.get("problem", {}) or {}
    tiling = ir.get("tiling", {}) or {}
    hw = ir.get("hardware", {}) or {}
    gpu = hw.get("gpu", {}) if isinstance(hw.get("gpu"), dict) else hw
    m = int_or_default(problem.get("M") or problem.get("m"), 0)
    n = int_or_default(problem.get("N") or problem.get("n"), 0)
    k = int_or_default(problem.get("K") or problem.get("k"), 0)
    bm = int_or_default(tiling.get("block_m"), max(m, 1))
    bn = int_or_default(tiling.get("block_n"), max(n, 1))
    bk = int_or_default(tiling.get("block_k"), max(k, 1))
    execution_profile = build_gpu_architecture_profile(hw)
    sm_count = int_or_default(execution_profile.get("sm_count"), 1)
    ops = 2 * m * n * k
    cta_count = ceil_div(m, bm) * ceil_div(n, bn)
    k_tiles = ceil_div(k, bk)
    shared_usage = shared_memory_usage(ir)
    shared_memory_bytes = shared_usage["total_bytes"]
    estimated_registers = int_or_default(
        get_nested_value(ir, "resource.register.actual_per_thread")
        or get_nested_value(ir, "resource.register.estimated_per_thread")
        or ir.get("resource", {}).get("estimated_registers_per_thread")
        or ir.get("mapping", {}).get("estimated_registers_per_thread"),
        estimate_registers_per_thread(ir),
    )
    cta_per_sm = float(cta_count) / float(max(sm_count, 1))
    mapping = ir.get("mapping", {}) or {}
    threads_per_block = int_or_default(mapping.get("threads_per_block"), 0)
    warps_per_block = int_or_default(mapping.get("warps_per_block"), 0)
    if not warps_per_block and threads_per_block:
        warps_per_block = ceil_div(threads_per_block, int_or_default(execution_profile.get("warp_size"), 32))
    execution_estimate = {}
    if threads_per_block and warps_per_block and bm and bn and bk:
        dtype = str(problem.get("dtype") or problem.get("dtype_A") or "fp32").lower()
        element_bytes = 2 if "16" in dtype or "bf16" in dtype else 8 if "64" in dtype else 4
        execution_estimate = estimate_tiling_execution(
            problem,
            execution_profile,
            bm=bm,
            bn=bn,
            bk=bk,
            threads_per_block=threads_per_block,
            warps_per_block=warps_per_block,
            shared_memory_bytes=shared_memory_bytes,
            estimated_registers_per_thread=estimated_registers,
            element_bytes=element_bytes,
            estimated_accumulators_per_thread=int_or_default(
                get_nested_value(ir, "resource.tiling_candidate.estimated_accumulators_per_thread"),
                int_or_default(tiling.get("thread_m"), 1) * int_or_default(tiling.get("thread_n"), 1),
            ),
        )
    cta_waves = execution_estimate.get("cta_waves") if execution_estimate else None
    # A square 1024 FP32 GEMM is already throughput-oriented on a many-SM GPU.
    # The old 2^33 threshold classified it as medium and hid all L2/scheduling
    # candidates from the unlock phase.
    throughput_sized_mn = min(m, n) >= 1024 and ops >= 2**31
    large_mn = throughput_sized_mn or (cta_count >= 2 * sm_count and ops >= 2**33)
    large_k = k_tiles >= 128 or k >= 4096
    return {
        "M": m,
        "N": n,
        "K": k,
        "BM": bm,
        "BN": bn,
        "BK": bk,
        "ops": ops,
        "cta_count": cta_count,
        "k_tiles": k_tiles,
        "sm_count": sm_count,
        "architecture_family": execution_profile.get("architecture_family"),
        "execution_profile": execution_profile,
        "cta_per_sm": cta_per_sm,
        "cta_waves": cta_waves,
        "shared_memory_bytes": shared_memory_bytes,
        "shared_memory_per_stage_bytes": shared_usage["per_stage_bytes"],
        "pipeline_stage_count": shared_usage["stage_count"],
        "estimated_registers_per_thread": estimated_registers,
        "small_mn_low_cta": cta_count < sm_count,
        "medium": cta_count >= sm_count and ops < 2**33,
        "large_mn": large_mn,
        "large_k": large_k,
        "large": large_mn or large_k,
        "execution_estimate": execution_estimate,
    }


def build_phase2_unlock_strategy_index(
    raw_strategy_index: dict[str, Any],
    ir: dict[str, Any],
    dependency_graph: dict[str, Any],
    history: dict[str, Any],
    matrix_profile: dict[str, Any],
) -> dict[str, Any]:
    applied = set(history.get("applied_strategy_ids", []) or [])
    failed_counts = history.get("failed_strategy_counts", {}) or {}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_canonical: set[str] = set()
    for strategy in raw_strategy_index.get("strategies", []) or []:
        strategy_id = strategy.get("strategy_id")
        phase = strategy_phase(strategy)
        reason = phase2_unlock_reject_reason(
            strategy=strategy,
            phase=phase,
            ir=ir,
            applied=applied,
            seen_canonical=seen_canonical,
            dependency_graph=dependency_graph,
            matrix_profile=matrix_profile,
            failed_counts=failed_counts,
        )
        if reason:
            rejected.append({"strategy_id": strategy_id, "phase": phase, "reason": reason})
            continue
        item = copy.deepcopy(strategy)
        item.setdefault("phase", phase)
        item.setdefault("canonical_strategy_id", canonical_strategy_id(item))
        item.setdefault("alias_of", None)
        item.setdefault("conflict_group", strategy_conflict_group(item))
        item.setdefault("modifies_regions", infer_modifies_regions(strategy_id or ""))
        item.setdefault("provides_fields", infer_provides_fields(item))
        item.setdefault("requires_fields", infer_requires_fields(item))
        item["filter_reason"] = "eligible for size-aware batch unlock"
        accepted.append(item)
        seen_canonical.add(canonical_strategy_id(item))
    result = copy.deepcopy(raw_strategy_index)
    result["strategies"] = accepted
    result["strategy_count"] = len(accepted)
    result["strategy_ids_by_stage"] = build_strategy_ids_by_stage(accepted)
    result["filter_context"] = {
        "phase": "performance_unlock",
        "profile": DEFAULT_UNLOCKED_PERFORMANCE_PROFILE,
        "matrix_profile": matrix_profile,
        "applied_strategy_ids": sorted(applied),
        "failed_strategy_counts": failed_counts,
        "large_matrix_strategies_enabled": bool(matrix_profile.get("large")),
        "filtered_out": rejected,
    }
    return result


def phase2_unlock_reject_reason(
    strategy: dict[str, Any],
    phase: str,
    ir: dict[str, Any],
    applied: set[str],
    seen_canonical: set[str],
    dependency_graph: dict[str, Any],
    matrix_profile: dict[str, Any],
    failed_counts: dict[str, int],
) -> str | None:
    strategy_id = strategy.get("strategy_id") or ""
    if not strategy_id:
        return "missing strategy_id"
    if strategy_id in applied:
        return "already applied"
    if strategy.get("alias_of"):
        return f"alias of {strategy.get('alias_of')}; canonical strategy only"
    if (strategy.get("materialization") or {}).get("mode") == "bounded_resource_unlock":
        return "executed by the bounded post-unlock resource executor, not an LLM patch"
    if phase == EXCLUDED_DEFAULT_PHASE:
        return "excluded_default phase"
    if phase not in {PHASE2_GENERAL, PHASE2_LARGE_MATRIX}:
        return f"not a phase2 unlock strategy: {phase}"
    if phase == PHASE2_LARGE_MATRIX and not matrix_profile.get("large"):
        return "large-matrix strategy filtered for small/medium matrix profile"
    canonical = canonical_strategy_id(strategy)
    if canonical in seen_canonical:
        return f"duplicate canonical strategy: {canonical}"
    if failed_counts.get(strategy_id, 0) >= DEFAULT_MAX_REPAIR_ATTEMPTS:
        return "strategy failed too many times"
    from SGPO.verification.strategy_application import unproven_dependencies
    unproven = unproven_dependencies(ir, dependency_graph.get('requires', {}).get(strategy_id, []))
    if unproven:
        return 'requires unproven implementation capabilities: ' + '; '.join(unproven)
    missing_graph = find_missing_requirements(strategy_id, ir, applied, dependency_graph)
    if missing_graph:
        return f"missing graph requirements: {'; '.join(missing_graph)}"
    missing_fields = [
        field for field in infer_requires_fields(strategy)
        if field and (get_nested_value(ir, field) is None or get_nested_value(ir, field) is False)
    ]
    if missing_fields:
        return f"missing required IR fields: {', '.join(missing_fields)}"
    resource_reason = resource_profile_reject_reason(strategy_id, ir, matrix_profile)
    if resource_reason:
        return resource_reason
    return None


def resource_profile_reject_reason(strategy_id: str, ir: dict[str, Any], matrix_profile: dict[str, Any]) -> str | None:
    registers = int_or_default(matrix_profile.get("estimated_registers_per_thread"), 0)
    shared_memory_bytes = int_or_default(matrix_profile.get("shared_memory_bytes"), 0)
    max_shared = shared_memory_limit(ir)
    heavy_register = any(token in strategy_id for token in ("Prefetch", "Unroll8", "UnrollBK", "Register.Cache"))
    if registers >= 96 and heavy_register:
        return "register pressure is high; prefetch/unroll/register-cache strategies are filtered"
    stage_multiplier = 0
    match = re.fullmatch(r"Pipeline\.CpAsync\.Multistage([234])\.SharedAB", strategy_id)
    if match:
        stage_multiplier = int(match.group(1))
    elif "DoubleBuffer" in strategy_id:
        stage_multiplier = 2
    requested_bytes = proposed_pipeline_bytes(ir, stage_multiplier) if stage_multiplier else 0
    if stage_multiplier and max_shared > 0 and requested_bytes > max_shared:
        return (
            f"{stage_multiplier}-stage pipeline requires "
            f"{requested_bytes} shared-memory bytes, above hardware limit {max_shared}"
        )
    return None


def sanitize_unlock_batch_plan(
    batch_plan: dict[str, Any],
    strategy_index: dict[str, Any],
    matrix_profile: dict[str, Any],
    history: dict[str, Any],
) -> dict[str, Any]:
    valid_by_id = {item["strategy_id"]: item for item in strategy_index.get("strategies", []) or []}
    applied = set(history.get("applied_strategy_ids", []) or [])
    attempted_bundle_keys = set(history.get("failed_unlock_bundles", []) or []) | set(
        history.get("ineffective_unlock_bundles", []) or []
    )
    plan_batches = batch_plan.get("batches") if isinstance(batch_plan, dict) else None
    if not isinstance(plan_batches, list):
        plan_batches = []
    cleaned_batches: list[dict[str, Any]] = []
    global_selected_conflicts: set[str] = set()
    for index, batch in enumerate(plan_batches, start=1):
        if not isinstance(batch, dict):
            continue
        selected_ids: list[str] = []
        seen_ids: set[str] = set()
        batch_conflicts: set[str] = set()
        rejected: list[dict[str, str]] = []
        for strategy_id in batch.get("strategy_ids", []) or []:
            if strategy_id in seen_ids:
                rejected.append({"strategy_id": strategy_id, "reason": "duplicate in batch"})
                continue
            strategy = valid_by_id.get(strategy_id)
            if strategy is None:
                rejected.append({"strategy_id": str(strategy_id), "reason": "not in filtered unlock candidate index"})
                continue
            if strategy_id in applied:
                rejected.append({"strategy_id": strategy_id, "reason": "already applied"})
                continue
            phase = strategy_phase(strategy)
            if phase == PHASE2_LARGE_MATRIX and not matrix_profile.get("large"):
                rejected.append({"strategy_id": strategy_id, "reason": "large-matrix strategy not allowed for this profile"})
                continue
            conflict_group = strategy_conflict_group(strategy)
            if conflict_group and (conflict_group in batch_conflicts or conflict_group in global_selected_conflicts):
                rejected.append({"strategy_id": strategy_id, "reason": f"conflict_group duplicate: {conflict_group}"})
                continue
            selected_ids.append(strategy_id)
            seen_ids.add(strategy_id)
            if conflict_group:
                batch_conflicts.add(conflict_group)
        bundle_key = "|".join(sorted(set(selected_ids)))
        if selected_ids and bundle_key not in attempted_bundle_keys:
            global_selected_conflicts.update(batch_conflicts)
            cleaned_batches.append(
                {
                    "batch_id": safe_batch_id(batch.get("batch_id") or f"batch_{index}"),
                    "purpose": str(batch.get("purpose", "")).strip(),
                    "strategy_ids": selected_ids,
                    "coupling_reason": str(batch.get("coupling_reason", "")).strip(),
                    "requires_summary": batch.get("requires_summary", []) if isinstance(batch.get("requires_summary"), list) else [],
                    "risk": batch.get("risk") if batch.get("risk") in {"low", "medium", "high"} else "medium",
                    "filtered_out": rejected,
                }
            )
    if not cleaned_batches and valid_by_id:
        cleaned_batches = [
            batch for batch in fallback_unlock_batches(list(valid_by_id.values()))
            if "|".join(sorted(set(batch.get("strategy_ids", []) or []))) not in attempted_bundle_keys
        ]
    for batch in cleaned_batches:
        needed = {required for sid in batch["strategy_ids"]
                  for required in valid_by_id[sid].get("requires_bundle_strategy_ids", [])}
        for required in needed:
            if required not in batch["strategy_ids"]:
                batch["strategy_ids"].insert(0, required)
            for other in cleaned_batches:
                if other is not batch and required in other["strategy_ids"]:
                    other["strategy_ids"].remove(required)
    ensure_async_pipeline_exploration_batch(cleaned_batches, valid_by_id, matrix_profile, applied)
    cleaned_batches = [
        batch for batch in cleaned_batches
        if "|".join(sorted(set(batch.get("strategy_ids", []) or []))) not in attempted_bundle_keys
    ]
    executor_id = "Compiler.ResourceFeedback.PtxasOccupancySweep"
    executor_selected = any(executor_id in batch["strategy_ids"] for batch in cleaned_batches)
    for batch in cleaned_batches:
        batch["strategy_ids"] = [sid for sid in batch["strategy_ids"] if sid != executor_id]
    cleaned_batches = [batch for batch in cleaned_batches if batch["strategy_ids"]]
    if executor_selected:
        cleaned_batches.append({"batch_id": "resource_sweep", "strategy_ids": [executor_id],
                                "purpose": "bounded PTXAS resource sweep after code changes", "risk": "low"})
    return {
        "batch_order": [batch["batch_id"] for batch in cleaned_batches],
        "batches": cleaned_batches,
        "matrix_profile": matrix_profile,
        "sanitized": True,
    }


def ensure_async_pipeline_exploration_batch(
    batches: list[dict[str, Any]],
    valid_by_id: dict[str, dict[str, Any]],
    matrix_profile: dict[str, Any],
    applied: set[str],
) -> None:
    """Schedule a guarded async-copy replacement for long synchronous K loops."""
    if "Pipeline.NoAsyncCopy.V1" not in applied:
        return
    if int_or_default(matrix_profile.get("k_tiles"), 0) < 32:
        return
    selected = {strategy_id for batch in batches for strategy_id in batch.get("strategy_ids", [])}
    if any(strategy_id.startswith("Pipeline.CpAsync.") for strategy_id in selected):
        return
    preferred_ids = (
        "Pipeline.CpAsync.Multistage2.SharedAB",
        "Pipeline.CpAsync.Multistage3.SharedAB",
        "Pipeline.CpAsync.Multistage4.SharedAB",
    )
    strategy_id = next((item for item in preferred_ids if item in valid_by_id), None)
    if strategy_id is None:
        return
    required = list(valid_by_id[strategy_id].get("requires_bundle_strategy_ids", []) or [])
    for batch in batches:
        batch["strategy_ids"] = [item for item in batch.get("strategy_ids", []) if item not in required]
    batches.append(
        {
            "batch_id": "async_pipeline_upgrade",
            "purpose": "replace synchronous shared-memory loading with an async pipeline",
            "strategy_ids": append_unique(required, [strategy_id]),
            "coupling_reason": "NoAsyncCopy is superseded for a long K reduction",
            "requires_summary": ["compile, correctness and runtime gates remain mandatory"],
            "risk": "high",
            "filtered_out": [],
        }
    )


def fallback_unlock_batches(strategies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = {}
    for strategy in strategies:
        region_key = "+".join(strategy.get("modifies_regions") or infer_modifies_regions(strategy.get("strategy_id", ""))) or "general"
        groups.setdefault(region_key, []).append(strategy["strategy_id"])
    batches = []
    for index, (region, strategy_ids) in enumerate(groups.items(), start=1):
        batches.append(
            {
                "batch_id": f"batch_{index}",
                "purpose": f"fallback unlock batch for {region}",
                "strategy_ids": strategy_ids[:DEFAULT_TOP_K_STRATEGIES_PER_SUBPHASE],
                "coupling_reason": "deterministic fallback grouped by modified code region",
                "requires_summary": [],
                "risk": "medium",
                "filtered_out": [],
            }
        )
    return batches


def build_unlock_fallback_batches(batch: dict[str, Any], dependency_graph: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if batch.get("execution_granularity") == "single_objective_coupled_regions":
        # Repair the complete objective; alternative strategies are selected afresh.
        return []
    fallbacks = []
    for index, strategy_id in enumerate(batch.get("strategy_ids", []) or [], start=1):
        if len(batch.get("strategy_ids", [])) == 1:
            break
        fallbacks.append(
            {
                **batch,
                "batch_id": f"{batch.get('batch_id')}_fallback_{index}",
                "strategy_ids": [strategy_id],
                "purpose": f"fallback single-strategy unlock for {strategy_id}",
                "coupling_reason": "single strategy fallback after batch failure",
            }
        )
    seen = {tuple(item.get("strategy_ids", [])) for item in [batch, *fallbacks]}
    for strategy_id in batch.get("strategy_ids", []):
        for edge in (dependency_graph or {}).get("fallback", {}).get(strategy_id, []):
            for fallback_id in edge.get("strategies", []):
                if (fallback_id,) in seen:
                    continue
                seen.add((fallback_id,))
                fallbacks.append({**batch, "batch_id": f"{batch.get('batch_id')}_graph_{len(fallbacks)+1}",
                                  "strategy_ids": [fallback_id], "purpose": f"graph fallback for {strategy_id}",
                                  "execution_granularity": "single_strategy"})
    return fallbacks


def build_unlock_bundle_strategy(
    batch: dict[str, Any],
    raw_strategy_index: dict[str, Any],
    strategy_library: dict[str, Any],
) -> dict[str, Any]:
    strategy_ids = batch.get("strategy_ids", []) or []
    strategies = [find_strategy_object(strategy_id, raw_strategy_index, strategy_library) for strategy_id in strategy_ids]
    if len(strategies) == 1:
        strategy = copy.deepcopy(strategies[0])
        strategy["unlock_execution"] = {
            "planning_batch_id": batch.get("planning_batch_id") or batch.get("batch_id"),
            "execution_granularity": "single_strategy",
        }
        return strategy
    merged_updates: dict[str, Any] = {}
    for strategy in strategies:
        merged_updates.update(copy.deepcopy(strategy.get("ir_updates", {}) or {}))
    return {
        "strategy_id": f"UnlockBatch.{safe_batch_id(batch.get('batch_id') or 'batch')}",
        "stage": "PerformanceUnlock",
        "category": "BatchUnlock",
        "name": batch.get("purpose") or "Batch unlock optimization",
        "phase": PHASE2_GENERAL,
        "bundle_strategy_ids": strategy_ids,
        "bundle_strategies": [copy.deepcopy(strategy) for strategy in strategies],
        "ir_updates": merged_updates,
        "preconditions": {"predicates": [p for s in strategies for p in get_precondition_predicates(s)]},
        "postconditions": {
            "patch_ir_verification": {"predicates": [p for s in strategies for p in get_patch_ir_postcondition_predicates(s)]},
            "code_verification": {"constraints": [p for s in strategies for p in get_code_verification_constraints(s)]},
        },
        "expected_effect": batch.get("purpose", ""),
        "potential_risks": [batch.get("risk", "medium"), batch.get("coupling_reason", "")],
    }


def find_strategy_object(strategy_id: str, raw_strategy_index: dict[str, Any], strategy_library: dict[str, Any]) -> dict[str, Any]:
    try:
        return load_strategy(strategy_library, strategy_id)
    except ValueError:
        for strategy in raw_strategy_index.get("strategies", []) or []:
            if strategy.get("strategy_id") == strategy_id:
                return copy.deepcopy(strategy)
        raise ValueError(f"Strategy not found: {strategy_id}")


def maybe_copy_unlock_patch(candidate_result: dict[str, Any], chain_id: str, batch_id: str | None) -> None:
    for attempt in candidate_result.get("attempts", []) or []:
        patch_output = attempt.get("patch_output")
        if patch_output and Path(patch_output).exists():
            save_unlock_artifact(DEFAULT_UNLOCK_BATCH_PATCH_OUTPUT, chain_id, batch_id, load_json(Path(patch_output)))


def save_unlock_artifact(path: Path, chain_id: str | None, batch_id: str | None, data: dict[str, Any]) -> None:
    suffix_parts = [safe_name(str(chain_id or "chain"))]
    if batch_id:
        suffix_parts.append(safe_name(str(batch_id)))
    artifact_path = path.with_name(f"{compact_artifact_stem(path.stem + '.' + '.'.join(suffix_parts))}{path.suffix}")
    save_json(artifact_path, data)
    publish_latest_json(path, data)


def build_unlock_code_summary(
    code_dir: Path,
    code_files: list[str],
    ir: dict[str, Any] | None = None,
    history: dict[str, Any] | None = None,
) -> dict[str, Any]:
    files = []
    ir = ir or {}
    history = history or {}
    for relative in code_files:
        path = code_dir / relative
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8")
        code_only = strip_cpp_comments(content)
        anchors = [
            region
            for region in [
                "LAUNCH_CONFIG",
                "SHARED_DECL",
                "INDEX_MAPPING",
                "REGISTER_DECL",
                "GLOBAL_TO_SHARED_LOAD",
        "NEXT_TILE_LOAD",
                "SYNC_AFTER_LOAD",
                "MAIN_LOOP",
                "COMPUTE_INNER",
                "STORE",
            ]
            if f"{region}_BEGIN" in content and f"{region}_END" in content
        ]
        files.append(
            {
                "path": relative,
                "line_count": content.count("\n") + 1,
                "available_regions": anchors,
                "contains_shared_memory": "__shared__" in code_only,
                "contains_float4": "float4" in code_only or "FLOAT4" in code_only,
                "contains_double_buffer": "double" in code_only.lower() and "buffer" in code_only.lower(),
                "contains_boundary_guard": bool(re.search(r"\bif\s*\([^)]*(?:<\s*M|<\s*N|<\s*K)", code_only)),
                "contains_k_tail_handling": bool(re.search(r"\b(?:k|global_k|sgpo_k)\b[^;]*(?:<\s*K|%\s*BK)", code_only)),
                "shared_decl_count": len(re.findall(r"__shared__\s+float\s+", code_only)),
                "global_store_count": len(re.findall(r"\bC\s*\[[^\]]+\]\s*=", code_only)),
                "mac_statement_count": len(re.findall(r"\+=\s*[^;]*\*\s*[^;]*;", code_only)),
            }
        )
    return {
        "code_dir": str(code_dir),
        "strategy_contract_state": ir.get('strategy_contract_state', {}),
        "applied_strategy_ids": history.get("applied_strategy_ids", []) or ir.get("strategy", {}).get("applied_strategy_ids", []),
        "tile_config": {
            "BM": ir_get(ir, "tiling.block_m"),
            "BN": ir_get(ir, "tiling.block_n"),
            "BK": ir_get(ir, "tiling.block_k"),
            "WM": ir_get(ir, "tiling.warp_tile.warp_m"),
            "WN": ir_get(ir, "tiling.warp_tile.warp_n"),
            "WMITER": ir_get(ir, "tiling.warp_tile.warp_m_iter"),
            "WNITER": ir_get(ir, "tiling.warp_tile.warp_n_iter"),
            "TM": ir_get(ir, "tiling.thread_m"),
            "TN": ir_get(ir, "tiling.thread_n"),
        },
        "shared_memory_bytes": (
            ir_get(ir, "resource.shared_memory.total_bytes")
            or ir_get(ir, "memory.shared_memory_bytes")
            or estimate_shared_memory_bytes(ir)
        ),
        "register_estimate": (
            ir_get(ir, "resource.register.estimated_per_thread")
            or ir_get(ir, "resource.estimated_registers_per_thread")
            or estimate_registers_per_thread(ir)
        ),
        "files": files,
    }


def build_unlock_verifier_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    from SGPO.verification.unlock_evidence import classify_diagnosis
    verified_ir = candidate.get("verified_ir", {}) or {}
    summary = verified_ir.get("verification", {}).get("summary", {}) or {}
    return {
        "compile_status": summary.get("compile_status"),
        "correctness_status": summary.get("correctness_status"),
        "runtime_safety_status": summary.get("runtime_safety_status"),
        "cuda_error": summary.get("cuda_error"),
        "latency_ms": summary.get("latency_ms") or verified_ir.get("performance", {}).get("latency_ms"),
        "gflops": summary.get("gflops") or verified_ir.get("performance", {}).get("gflops"),
        "semantic_defect_summary": classify_diagnosis(verified_ir.get("defect_diagnosis", {}), summary),
        "strategy_contract_state": verified_ir.get('strategy_contract_state', {}),
        "tuning_observations": tuning_observations(verified_ir),
        "resource_estimate": verified_ir.get("resource", {}),
        "failed_strategy_counts": verified_ir.get("strategy", {}).get("failed_strategy_counts", {}),
        "current_changed_regions": verified_ir.get("strategy", {}).get("changed_regions", []),
    }


def tuning_observations(ir: dict[str, Any]) -> dict[str, Any]:
    profile = build_matrix_profile(ir)
    observations = []
    ctas = profile.get('cta_count')
    sm_count = ir_get(ir, 'hardware.sm_count') or ir_get(ir, 'hardware.multiprocessor_count')
    if isinstance(ctas, (int, float)) and isinstance(sm_count, (int, float)) and 0 < ctas < sm_count:
        observations.append('Fewer CTAs than SMs: compare smaller Block tiles with coupled Warp/Thread mapping; do not assume Split-K is profitable.')
    registers = ir_get(ir, 'verification.compile.ptxas_resources.register_counts', []) or []
    if registers and max(registers) >= 96:
        observations.append('High measured register count: compare fragment size, unroll and prefetch depth; confirm occupancy/spills before attributing the bottleneck.')
    pending = ir.get('strategy_contract_state', {}).get('pending_obligations', [])
    if any(item.get('status') in ('degraded', 'not_realized') for item in pending):
        observations.append('Resolve explicit strategy degradation/missing implementation before stacking dependent optimizations.')
    return {'basis': 'resource observations, not profiler-proven bottlenecks',
            'matrix_profile': profile, 'suggested_controlled_comparisons': observations}


def strategy_phase(strategy: dict[str, Any]) -> str:
    return strategy.get("phase") or infer_strategy_phase(strategy.get("strategy_id", ""))


def infer_strategy_phase(strategy_id: str) -> str:
    if not strategy_id:
        return EXCLUDED_DEFAULT_PHASE
    large_prefixes = (
        "Mapping.CTASwizzle.",
        "Memory.L2Reuse.",
        "Scheduling.PersistentCTA.",
        "Reduction.StreamK.",
        "Reduction.SplitK.",
    )
    if strategy_id.startswith(large_prefixes):
        return PHASE2_LARGE_MATRIX
    if strategy_id.startswith(("Compiler.", "Tuning.")) or strategy_id == "Scheduling.WaveQuantization.SMResidentBlocks":
        return PHASE2_GENERAL
    if strategy_id.startswith("Epilogue.Fusion."):
        return EXCLUDED_DEFAULT_PHASE
    if strategy_id in {
        "Pipeline.DoubleBuffer.SharedAB",
        "Pipeline.DoubleBuffer.SharedAB.V1Enabled",
        "Pipeline.WarpAwareDoubleBuffer.SharedAB",
    }:
        return PHASE1_CORE_COUPLED
    if strategy_id in {
        "Pipeline.WarpRegisterPrefetchAB",
        "Pipeline.SoftwarePrefetch.RegisterA",
    }:
        return PHASE2_GENERAL
    if strategy_id.startswith(("Pipeline.CpAsync.", "Compiler.ResourceFeedback.")):
        return PHASE2_GENERAL
    if strategy_id.startswith(("Memory.Prefetch.", "Vectorization.GlobalLoad", "Vectorization.StoreC.float")):
        return PHASE2_GENERAL
    if strategy_id.startswith(("Epilogue.StoreC.Vectorized", "Vectorization.StoreC.GuardedVectorStore", "Vectorization.StoreC.AlignedNoGuard")):
        return PHASE2_GENERAL
    return PHASE1_CORE_COUPLED


def canonical_strategy_id(strategy: dict[str, Any]) -> str:
    return strategy.get("canonical_strategy_id") or strategy.get("alias_of") or strategy.get("strategy_id") or ""


def strategy_conflict_group(strategy: dict[str, Any]) -> str:
    if strategy.get("conflict_group"):
        return strategy["conflict_group"]
    strategy_id = strategy.get("strategy_id", "")
    if ".BlockTile." in strategy_id:
        return "tiling.block_tile"
    if ".WarpTile." in strategy_id:
        return "tiling.warp_tile"
    if ".ThreadTile." in strategy_id:
        return "tiling.thread_tile"
    if strategy_id.startswith(("Mapping.WarpThreadTile.", "Mapping.Warp.OutputFragment", "Mapping.LaneLayout.")):
        return "mapping.thread_output"
    if strategy_id.startswith("Layout.SharedMemory.TransposeA") or strategy_id.startswith("Layout.SharedMemory.PaddingA"):
        return "layout.shared_A_transform"
    if strategy_id.startswith("Layout.SharedMemory.TransposeB") or strategy_id.startswith("Layout.SharedMemory.PaddingB"):
        return "layout.shared_B_transform"
    if strategy_id.startswith("Register.AccumulatorLayout."):
        return "register.accumulator_layout"
    if strategy_id.startswith("Reordering.KLoop."):
        return "schedule.k_loop_unroll"
    if strategy_id.startswith("Safety.BoundaryPolicy."):
        return "safety.boundary_policy"
    if strategy_id.startswith(("Vectorization.GlobalLoad", "Reordering.CooperativeVectorLoad", "Reordering.WarpCooperativeLoad")):
        return "vectorization.global_load"
    if "StoreC" in strategy_id:
        return "store.C"
    if "DoubleBuffer" in strategy_id:
        return "pipeline.buffering"
    if "Prefetch" in strategy_id:
        return "pipeline.prefetch"
    if strategy_id.startswith(("Mapping.CTASwizzle.", "Memory.L2Reuse.", "Scheduling.PersistentCTA.", "Reduction.StreamK.", "Reduction.SplitK.")):
        return "large_matrix.schedule"
    return ""


def infer_modifies_regions(strategy_id: str) -> list[str]:
    regions: list[str] = []
    if strategy_id.startswith("Tiling."):
        regions.extend(["launch_config", "tile_indexing"])
    if strategy_id.startswith("Mapping."):
        regions.extend(["thread_mapping", "launch_config"])
    if strategy_id.startswith("Layout.SharedMemory."):
        regions.extend(["shared_memory_layout", "shared_memory_indexing"])
    if strategy_id.startswith("Reordering."):
        regions.extend(["global_load", "compute_loop"])
    if strategy_id.startswith("Register."):
        regions.extend(["register_file", "compute_loop"])
    if strategy_id.startswith("Vectorization.GlobalLoad"):
        regions.append("global_load")
    if "StoreC" in strategy_id or strategy_id.startswith("Epilogue."):
        regions.append("epilogue_store")
    if strategy_id.startswith(("Pipeline.", "Memory.Prefetch.")):
        regions.extend(["main_k_loop", "shared_memory_pipeline"])
    if strategy_id.startswith(("Compiler.", "Tuning.")):
        regions.append("build_config")
    return sorted(set(regions))


def infer_provides_fields(strategy: dict[str, Any]) -> list[str]:
    fields = set(strategy.get("provides_fields", []) or [])
    fields.update((strategy.get("ir_updates") or {}).keys())
    return sorted(str(field) for field in fields if field)


def infer_requires_fields(strategy: dict[str, Any]) -> list[str]:
    fields = set(strategy.get("requires_fields", []) or [])
    for predicate in normalize_predicates(strategy.get("preconditions")):
        field = predicate.get("field") or predicate.get("lhs") or predicate.get("path")
        if isinstance(field, str) and "." in field and not any(op in field for op in " +-*/"):
            fields.add(field)
    return sorted(str(field) for field in fields if field)


def normalize_predicates(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        raw = raw.get("predicates") or raw.get("conditions") or raw.get("all") or raw.get("any") or []
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def estimate_shared_memory_bytes(ir: dict[str, Any]) -> int:
    tiling = ir.get("tiling", {}) or {}
    bm = int_or_default(tiling.get("block_m"), 0)
    bn = int_or_default(tiling.get("block_n"), 0)
    bk = int_or_default(tiling.get("block_k"), 0)
    if not (bm and bn and bk):
        return 0
    return 4 * (bm * bk + bk * bn)


def estimate_registers_per_thread(ir: dict[str, Any]) -> int:
    tiling = ir.get("tiling", {}) or {}
    tm = int_or_default(tiling.get("thread_m"), 1)
    tn = int_or_default(tiling.get("thread_n"), 1)
    return tm * tn + tm + tn + 16


def ceil_div(value: int, divisor: int) -> int:
    return int(math.ceil(float(max(value, 0)) / float(max(divisor, 1))))


def int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def get_nested_value(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def append_unique(base: list[str], extra: list[str]) -> list[str]:
    result = list(base or [])
    seen = set(result)
    for item in extra:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def safe_batch_id(value: Any) -> str:
    text = str(value or "batch").strip()
    safe = safe_name(text)
    return safe or "batch"


def merge_terminal_verifications(
    *terminal_verifications: dict[str, Any],
    phases: list[str] | None = None,
) -> dict[str, Any]:
    merged_candidates = []
    merged_frontier_by_chain_id: dict[str, dict[str, Any]] = {}
    summaries = []
    for index, terminal in enumerate(terminal_verifications):
        if not terminal:
            continue
        phase = phases[index] if phases and index < len(phases) else terminal.get("summary", {}).get("phase")
        for candidate in terminal.get("verified_candidates", []) or []:
            item = copy.deepcopy(candidate)
            if phase:
                item["source_phase"] = phase
                item.setdefault("verified_ir", {}).setdefault("strategy", {})["source_phase"] = phase
            merged_candidates.append(item)
        best_state = terminal.get("best_state")
        best_candidate = terminal.get("best_candidate")
        chain_id = (best_candidate or {}).get("strategy_id")
        if chain_id and best_state:
            merged_frontier_by_chain_id[chain_id] = best_state
        summaries.append(terminal.get("summary", {}))

    best_candidate = choose_best_candidate([item for item in merged_candidates if item.get("accepted")])
    best_state = merged_frontier_by_chain_id.get((best_candidate or {}).get("strategy_id"))
    top_terminal_results = build_top_terminal_results(merged_candidates, DEFAULT_TOP_K_FINAL_RESULTS)
    terminal_chain_count = sum(summary.get("terminal_chain_count", 0) for summary in summaries)
    return {
        "verified_candidates": merged_candidates,
        "best_candidate": best_candidate,
        "best_state": best_state,
        "top_terminal_results": top_terminal_results,
        "summary": {
            "stage": "TerminalVerification",
            "selection_mode": DEFAULT_SELECTION_MODE,
            "phase": "merged",
            "merged_phase_count": len(summaries),
            "terminal_chain_count": terminal_chain_count,
            "verified_terminal_chain_count": len(merged_candidates),
            "accepted_terminal_chain_count": len([item for item in merged_candidates if item.get("accepted")]),
            "best_chain_id": best_candidate.get("strategy_id") if best_candidate else None,
            "best_chain_gflops": candidate_gflops(best_candidate),
            "top_k_final_results": DEFAULT_TOP_K_FINAL_RESULTS,
            "top_terminal_results": top_terminal_results,
            "phase_summaries": summaries,
        },
    }


def load_strategy_documents(ir: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    if target_backend(ir) == "cpu":
        return load_json(Path(DEFAULT_CPU_STRATEGY_INDEX)), load_json(Path(DEFAULT_CPU_STRATEGY_LIBRARY))
    strategy_index = load_json(Path(DEFAULT_STRATEGY_INDEX))
    strategy_library = load_json(Path(DEFAULT_STRATEGY_LIBRARY))
    if strategy_index.get("stages") and strategy_library.get("stages"):
        return strategy_index, strategy_library
    return load_merged_strategy_documents(
        current_index_path=Path(DEFAULT_STRATEGY_INDEX),
        current_library_path=Path(DEFAULT_STRATEGY_LIBRARY),
    )


def load_dependency_graph(ir: dict[str, Any] | None = None) -> dict[str, Any]:
    return load_json(Path(DEFAULT_CPU_DEPENDENCY_GRAPH if target_backend(ir) == "cpu" else DEFAULT_DEPENDENCY_GRAPH))


def infer_app_stage_order(
    ir: dict[str, Any],
    dependency_graph: dict[str, Any],
    strategy_index: dict[str, Any],
    strategy_library: dict[str, Any],
) -> list[str]:
    # The dependency graph is the authoritative workflow.  IR snapshots may
    # retain legacy stage names or an older, incomplete stage list.
    explicit = dependency_graph.get("stage_order") or ir.get("strategy", {}).get("stage_order")
    if isinstance(explicit, list) and explicit:
        stage_aliases = {"Reordering": "MappingReordering"}
        normalized_explicit = [stage_aliases.get(stage, stage) for stage in explicit]
        filtered_explicit = [
            stage
            for stage in normalized_explicit
            if stage_has_strategy(stage, strategy_index, strategy_library)
        ]
        if filtered_explicit:
            return apply_strategy_profile_to_stage_order(
                filtered_explicit,
                ir.get("strategy", {}).get("profile"),
            )

    index_order = infer_stage_order_from_index(strategy_index)
    if index_order:
        return apply_strategy_profile_to_stage_order(
            [stage for stage in index_order if stage_has_strategy(stage, strategy_index, strategy_library)],
            ir.get("strategy", {}).get("profile"),
        )

    preferred = ["Tiling", "Layout", "Reordering", "Vectorization", "Pipeline", "TensorCore"]
    stages = []
    for strategy in strategy_library.get("strategies", []) or []:
        stage = strategy.get("stage")
        if stage and stage not in stages:
            stages.append(stage)
    ordered = [stage for stage in preferred if stage in stages]
    ordered.extend(stage for stage in stages if stage not in ordered)
    return apply_strategy_profile_to_stage_order(ordered, ir.get("strategy", {}).get("profile"))


def apply_strategy_profile_to_stage_order(stage_order: list[str], profile: str | None) -> list[str]:
    if all(stage.startswith("CPU") for stage in stage_order):
        return stage_order
    if profile == DEFAULT_UNLOCKED_PERFORMANCE_PROFILE:
        return [stage for stage in THROUGHPUT_EXPLORATION_STAGES if stage in set(stage_order)]
    if profile == CORE_CONSTRUCTION_COUPLED_PROFILE:
        return [stage for stage in CORE_CONSTRUCTION_COUPLED_STAGES if stage in set(stage_order)]
    if profile != "stable_correctness_first":
        return stage_order
    return list(STABLE_BASELINE_STAGES)


def stage_has_strategy(stage: str, strategy_index: dict[str, Any], strategy_library: dict[str, Any]) -> bool:
    if stage in (strategy_index.get("stages") or {}):
        return True
    if any(item.get("stage_id") == stage for item in strategy_library.get("stages", []) or []):
        return True
    return any(strategy.get("stage") == stage for strategy in strategy_index.get("strategies", []) or []) or any(
        strategy.get("stage") == stage for strategy in strategy_library.get("strategies", []) or []
    )


def apply_strategy_profile_to_candidates(strategy_index: dict[str, Any], profile: str | None) -> dict[str, Any]:
    if profile == CORE_CONSTRUCTION_COUPLED_PROFILE:
        result = copy.deepcopy(strategy_index)
        accepted = []
        profile_rejected = []
        for strategy in result.get("strategies", []) or []:
            phase = strategy_phase(strategy)
            strategy_id = strategy.get("strategy_id") or ""
            if strategy.get("alias_of"):
                profile_rejected.append({"strategy_id": strategy_id, "reason": f"alias of {strategy.get('alias_of')}"})
            elif phase != PHASE1_CORE_COUPLED:
                profile_rejected.append({"strategy_id": strategy_id, "reason": f"phase {phase} is not enabled in Phase 1"})
            else:
                item = copy.deepcopy(strategy)
                item.setdefault("phase", phase)
                item.setdefault("canonical_strategy_id", canonical_strategy_id(item))
                item.setdefault("alias_of", None)
                item.setdefault("conflict_group", strategy_conflict_group(item))
                item.setdefault("modifies_regions", infer_modifies_regions(strategy_id))
                item.setdefault("provides_fields", infer_provides_fields(item))
                item.setdefault("requires_fields", infer_requires_fields(item))
                accepted.append(item)
        context = result.setdefault("filter_context", {})
        context["strategy_profile"] = profile
        context["profile_policy"] = {
            "mode": "phase1_core_coupled",
            "allowed_phase": PHASE1_CORE_COUPLED,
            "unlock_deferred_phases": [PHASE2_GENERAL, PHASE2_LARGE_MATRIX],
        }
        context.setdefault("rejected", [])
        context["rejected"].extend(profile_rejected)
        result["strategies"] = accepted
        result["strategy_count"] = len(accepted)
        return result
    if profile == DEFAULT_UNLOCKED_PERFORMANCE_PROFILE:
        result = copy.deepcopy(strategy_index)
        result.setdefault("filter_context", {})["strategy_profile"] = profile
        result.setdefault("filter_context", {})["profile_policy"] = {
            "mode": "throughput_exploration",
            "unlocked_after_correctness": True,
            "preferred_strategy_ids_by_subphase": THROUGHPUT_PROFILE_PREFERRED_IDS,
        }
        result["strategies"] = prioritize_profile_strategies(
            result.get("strategies", []) or [],
            result.get("filter_context", {}).get("current_subphase"),
            profile,
        )
        return result
    if profile != "stable_correctness_first":
        return strategy_index
    result = copy.deepcopy(strategy_index)
    accepted = []
    profile_rejected = []
    for strategy in result.get("strategies", []) or []:
        strategy_id = strategy.get("strategy_id") or ""
        reason = stable_baseline_reject_reason(strategy_id)
        if reason:
            profile_rejected.append({"strategy_id": strategy_id, "reason": reason})
        else:
            accepted.append(strategy)
    context = result.setdefault("filter_context", {})
    context["strategy_profile"] = profile
    context["profile_policy"] = {
        "mode": "correctness_first",
        "allowed_stages": STABLE_BASELINE_STAGES,
        "blocked_before_stable_correctness": [
            "Pipeline",
            "Memory.Prefetch",
            "float/vector load-store vectorization",
            "compiler/resource tuning",
            "epilogue fusion",
        ],
    }
    context.setdefault("rejected", [])
    context["rejected"].extend(profile_rejected)
    result["strategies"] = accepted
    result["strategy_count"] = len(accepted)
    return result


def prioritize_profile_strategies(
    strategies: list[dict[str, Any]],
    subphase_id: str | None,
    profile: str | None,
) -> list[dict[str, Any]]:
    if profile != DEFAULT_UNLOCKED_PERFORMANCE_PROFILE:
        return strategies
    preferred = THROUGHPUT_PROFILE_PREFERRED_IDS.get(subphase_id or "", [])
    priority = {strategy_id: index for index, strategy_id in enumerate(preferred)}
    return sorted(
        strategies,
        key=lambda item: (
            0 if item.get("strategy_id") in priority else 1,
            priority.get(item.get("strategy_id"), len(priority)),
            item.get("strategy_id") or "",
        ),
    )


def stable_baseline_reject_reason(strategy_id: str) -> str | None:
    if strategy_id in STABLE_BASELINE_DENY_IDS:
        return "blocked by stable_correctness_first profile until scalar/shared GEMM correctness passes"
    if any(strategy_id.startswith(prefix) for prefix in STABLE_BASELINE_DENY_PREFIXES):
        return "blocked by stable_correctness_first profile until scalar/shared GEMM correctness passes"
    return None


def run_stage(
    stage: str,
    current_ir: dict[str, Any],
    history: dict[str, Any],
    raw_strategy_index: dict[str, Any],
    dependency_graph: dict[str, Any],
    strategy_library: dict[str, Any],
    client: OpenAICompatibleClient,
    user_question: str,
    profile: str | None = None,
    base_source_snapshot: dict[str, str] | None = None,
    candidate_namespace: str | None = None,
    parent_path: list[str] | None = None,
    parent_path_code: list[int] | None = None,
    build_platform: str = "windows",
    compile_each_step: bool = DEFAULT_COMPILE_EACH_STRATEGY_STEP,
    compile_each_stage: bool = DEFAULT_COMPILE_EACH_STAGE,
    step_compile_timeout_seconds: int = DEFAULT_STEP_COMPILE_TIMEOUT_SECONDS,
    stage_benchmark_runs: int = DEFAULT_STAGE_BENCHMARK_RUNS,
    stage_benchmark_warmup_runs: int = DEFAULT_STAGE_BENCHMARK_WARMUP_RUNS,
    tiling_top_k_per_level: int = DEFAULT_TILING_TOP_K_PER_LEVEL,
    tiling_resource_pool_size: int = DEFAULT_TILING_RESOURCE_POOL_SIZE,
    tiling_llm_top_n: int = DEFAULT_TILING_LLM_TOP_N,
    lazy_fallback_execution: bool = DEFAULT_LAZY_FALLBACK_EXECUTION,
) -> dict[str, Any]:
    compile_each_step = False
    stage_snapshot = base_source_snapshot or snapshot_source_files(backend_code_root(current_ir), backend_code_files(current_ir))
    controller = StageController(
        strategy_index=raw_strategy_index,
        workflow_library=strategy_library,
        current_stage=stage,
        optir=current_ir,
        history=history,
    )
    current_subphase = controller.current_subphase()
    current_subphase_id = controller.current_subphase_id()
    if current_subphase_id and current_subphase_id.endswith(".StageVerification"):
        stage_report = controller.verify_stage(current_ir, include_code_checks=False)
        stage_report["verification_scope"] = "stage_ir_predicates_only"
        verified_stage_ir = current_ir
        verified_stage_snapshot = stage_snapshot
        if stage_report["accepted"] and compile_each_stage:
            verified_stage_ir = verify_stage_completion_build_run(
                ir=current_ir,
                source_snapshot=stage_snapshot,
                stage=stage,
                chain_path=parent_path or [],
                chain_code=parent_path_code or [],
                build_platform=build_platform,
                timeout_seconds=step_compile_timeout_seconds,
                benchmark_runs=stage_benchmark_runs,
                benchmark_warmup_runs=stage_benchmark_warmup_runs,
            )
            stage_report = attach_stage_compile_run_report(stage_report, verified_stage_ir)
            stage_report["runtime_verification"] = "completed_at_stage_boundary"
            stage_report["stage_compile_run"]["gate"] = "advisory_until_phase1_completion"
            stage_report["accepted"] = True
        else:
            stage_report["runtime_verification"] = "deferred_until_phase1_completion"
        save_json(stage_path(DEFAULT_PRECHECK_OUTPUT, current_subphase_id), stage_report)
        save_json(stage_path(DEFAULT_POSTCHECK_OUTPUT, current_subphase_id), stage_level_post_check(stage, current_subphase_id, stage_report))
        return {
            "accepted_candidate": None,
            "accepted_candidates": [],
            "events": [
                {
                    "stage": stage,
                    "subphase": current_subphase_id,
                    "status": "accepted" if stage_report["accepted"] else "failed",
                    "verification": stage_report,
                }
            ],
            "failed_strategy_counts": {} if stage_report["accepted"] else {current_subphase_id: 1},
            "stage_completed_ir": verified_stage_ir if stage_report["accepted"] else None,
            "stage_source_snapshot": verified_stage_snapshot if stage_report["accepted"] else None,
            "summary": {
                "stage": stage,
                "current_subphase": current_subphase_id,
                "selection_mode": "stage_verification",
                "accepted_candidate_count": 0,
                "stage_verification": stage_report,
            },
        }

    available_strategy_index = apply_strategy_profile_to_candidates(
        controller.get_subphase_candidates(),
        profile,
    )
    save_json(stage_path(DEFAULT_FILTERED_INDEX_OUTPUT, current_subphase_id or stage), available_strategy_index)
    precheck_result = precheck_result_from_available(available_strategy_index)
    save_json(stage_path(DEFAULT_PRECHECK_OUTPUT, current_subphase_id or stage), precheck_result)
    chain_record = save_chain_stage_available_strategies(
        chain_path=parent_path or [],
        chain_code=parent_path_code or [],
        stage=current_subphase_id or stage,
        filtered_strategy_index=available_strategy_index,
        precheck_result=precheck_result,
        available_strategy_index=available_strategy_index,
    )

    if available_strategy_index.get("strategy_count", 0) == 0:
        if derivation_only_subphase(current_subphase):
            advanced_ir = controller.advance_derivation_subphase()
            local_check = advanced_ir.get("stage_controller", {}).get("last_local_check", {})
            event = {
                "stage": stage,
                "subphase": current_subphase_id,
                "status": "advanced" if local_check.get("accepted") else "failed",
                "reason": "derivation-only subphase",
                "local_check": local_check,
            }
            return {
                "accepted_candidate": None,
                "accepted_candidates": [],
                "advanced_ir": advanced_ir if local_check.get("accepted") else None,
                "events": [event],
                "failed_strategy_counts": {},
                "summary": {
                    "stage": stage,
                    "current_subphase": current_subphase_id,
                    "selection_mode": "derivation_only_subphase",
                    "accepted_candidate_count": 0,
                    "advanced": bool(local_check.get("accepted")),
                    "chain_record": chain_record,
                    "precheck_summary": precheck_result.get("summary"),
                },
            }
        result = empty_stage_result(stage, available_strategy_index, "no subphase candidates")
        skipped_ir = controller.skip_current_subphase("no subphase candidates; continue along stage graph")
        event = {
            "stage": stage,
            "subphase": current_subphase_id,
            "status": "skipped",
            "reason": "no subphase candidates; continue along stage graph",
            "next_subphase": (current_subphase or {}).get("next_subphase"),
        }
        result["skipped_ir"] = skipped_ir
        result["events"] = [event]
        result["summary"]["selection_mode"] = "skip_empty_subphase_and_continue"
        result["summary"]["skipped"] = True
        result["summary"]["skip_reason"] = event["reason"]
        result["chain_record"] = chain_record
        return result

    hierarchical_tiling_subphases = {
        "Tiling.BlockTileSelection",
        "Tiling.WarpTileSelection",
        "Tiling.ThreadTileSelection",
    }
    micro_selection = None
    if target_backend(current_ir) == "cuda" and current_subphase_id == "Tiling.BlockTileSelection":
        try:
            micro_selection = build_joint_tiling_micro_selection(
                client=client,
                current_ir=current_ir,
                strategy_library=strategy_library,
                available_strategy_index=available_strategy_index,
                top_k=tiling_llm_top_n,
                resource_pool_size=tiling_resource_pool_size,
            )
        except Exception as exc:
            error = {
                "stage": stage,
                "subphase": current_subphase_id,
                "status": "failed",
                "reason": "hierarchical Tiling LLM selection failed after configured retries",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            return {
                "accepted_candidate": None,
                "accepted_candidates": [],
                "events": [error],
                "failed_strategy_counts": {},
                "summary": {
                    "stage": stage,
                    "current_subphase": current_subphase_id,
                    "selection_mode": "joint_tiling_llm_selection_failed",
                    "filtered_strategy_count": available_strategy_index.get("strategy_count", 0),
                    "accepted_candidate_count": 0,
                    "selection_error": error,
                    "chain_record": chain_record,
                },
            }
    elif target_backend(current_ir) == "cuda" and current_subphase_id in {
        "Tiling.WarpTileSelection", "Tiling.ThreadTileSelection"
    }:
        micro_selection = planned_tiling_micro_selection(
            current_ir,
            current_subphase_id,
            available_strategy_index,
        )
        if micro_selection is None:
            micro_selection = build_fallback_micro_selection(
                available_strategy_index,
                current_subphase_id,
                "joint Tiling plan was unavailable; use filtered deterministic fallback",
            )
    else:
        if micro_selection is None:
            try:
                micro_selection = get_micro_strategy_from_llm(
                    client=client,
                    current_stage=stage,
                    current_subphase=current_subphase_id,
                    subphase=current_subphase or {},
                    optir_summary=compact_ir_summary(current_ir),
                    strategy_index=available_strategy_index,
                    prompt_path=Path(DEFAULT_MICRO_STRATEGY_PROMPT),
                )
            except Exception as exc:
                micro_selection = build_fallback_micro_selection(
                    available_strategy_index,
                    current_subphase_id,
                    f"LLM micro-strategy selection failed: {type(exc).__name__}: {exc}",
                )
        micro_selection = inject_profile_preferred_candidates(
            micro_selection,
            available_strategy_index,
            current_subphase_id,
            profile,
        )
    subphase_top_k = 1 if current_subphase_id == "Tiling.MappingDerivation" else (
        tiling_llm_top_n
        if current_subphase_id == "Tiling.BlockTileSelection"
        else 1
        if current_subphase_id in {"Tiling.WarpTileSelection", "Tiling.ThreadTileSelection"}
        else DEFAULT_TOP_K_STRATEGIES_PER_SUBPHASE
    )
    execution_candidates = select_top_k_candidates(
        micro_selection.get("candidates", []),
        subphase_top_k,
    )
    selected_strategy = {"candidates": execution_candidates}
    llm_selected_count = len(selected_strategy.get("candidates", []))
    selected_strategy["filter_context"] = available_strategy_index.get("filter_context")
    selected_strategy["selection_mode"] = micro_selection.get("selection_mode", DEFAULT_SELECTION_MODE)
    selected_strategy["top_k_strategies_per_subphase"] = DEFAULT_TOP_K_STRATEGIES_PER_SUBPHASE
    selected_strategy["current_subphase"] = current_subphase_id
    selected_strategy["micro_selection"] = micro_selection
    selected_strategy["all_llm_candidates"] = micro_selection.get("candidates", [])
    save_json(stage_path(DEFAULT_SELECTED_STRATEGY_OUTPUT, current_subphase_id or stage), selected_strategy)
    publish_latest_json(Path(DEFAULT_SELECTED_STRATEGY_OUTPUT), selected_strategy)
    update_chain_stage_selected_strategies(chain_record, selected_strategy)

    if llm_selected_count == 0:
        skipped_ir = controller.skip_current_subphase("LLM selected no micro-strategy; continue along stage graph")
        event = {
            "stage": stage,
            "subphase": current_subphase_id,
            "status": "skipped",
            "reason": "LLM selected no micro-strategy; continue along stage graph",
            "next_subphase": (current_subphase or {}).get("next_subphase"),
        }
        return {
            "accepted_candidate": None,
            "accepted_candidates": [],
            "skipped_ir": skipped_ir,
            "events": [event],
            "failed_strategy_counts": {},
            "summary": {
                "stage": stage,
                "current_subphase": current_subphase_id,
                "filtered_strategy_count": available_strategy_index.get("strategy_count", 0),
                "llm_selected_count": llm_selected_count,
                "selection_mode": "skip_empty_selection_and_continue",
                "stage_candidate_count": 0,
                "available_strategy_count": available_strategy_index.get("strategy_count", 0),
                "chain_record": chain_record,
                "precheck_summary": precheck_result.get("summary"),
                "evaluated_candidate_count": 0,
                "accepted_candidate_count": 0,
                "skipped": True,
                "skip_reason": event["reason"],
            },
        }

    candidate_results = []
    events = []
    local_failed_events = []
    failed_counts: dict[str, int] = {}
    precheck_by_id = {item["strategy_id"]: item for item in precheck_result.get("accepted", [])}
    for sibling_index, selected_item in enumerate(selected_strategy.get("candidates", []), start=1):
        selected_item["_sibling_index"] = sibling_index
        strategy_id = selected_item["strategy_id"]
        precheck_item = precheck_by_id.get(strategy_id)
        if precheck_item is None:
            continue
        strategy = controller.strategy_object(strategy_id)
        expected_updates = selected_item.get("expected_ir_updates")
        if not isinstance(expected_updates, dict):
            expected_updates = selected_strategy.get("micro_selection", {}).get("expected_ir_updates", {})
        tiling_plan = selected_item.get("tiling_plan")
        if not isinstance(tiling_plan, dict):
            tiling_plan = (current_ir.get("strategy") or {}).get("tiling_joint_plan")
        if isinstance(tiling_plan, dict):
            expected_updates = merge_expected_ir_updates(
                strategy_id,
                expected_updates if isinstance(expected_updates, dict) else {},
                tiling_plan_ir_updates(strategy_id, tiling_plan),
            )
        strategy["ir_updates"] = merge_expected_ir_updates(
            strategy_id,
            strategy.get("ir_updates") or {},
            expected_updates if isinstance(expected_updates, dict) else {},
        )
        local_ir = controller.apply_micro_strategy(
            strategy_id,
            strategy["ir_updates"],
        )
        if isinstance(selected_item.get("tiling_plan"), dict):
            local_ir.setdefault("strategy", {})["tiling_joint_plan"] = copy.deepcopy(selected_item["tiling_plan"])
        local_check = local_ir.get("stage_controller", {}).get("last_local_check", {})
        precheck_item["checker_report"] = local_check
        if local_check.get("accepted") is False:
            failed_counts[strategy_id] = failed_counts.get(strategy_id, 0) + 1
            event = {
                "stage": stage,
                "subphase": current_subphase_id,
                "strategy_id": strategy_id,
                "status": "failed",
                "reason": "micro local check failed",
                "local_check": local_check,
            }
            events.append(event)
            local_failed_events.append(event)
            continue
        candidate_result = run_strategy_candidate(
            stage=stage,
            base_ir=local_ir,
            base_source_snapshot=stage_snapshot,
            strategy=strategy,
            precheck_item=precheck_item,
            client=client,
            chain_path=[*(parent_path or []), strategy_id],
            chain_code=[*(parent_path_code or []), selected_item.get("_sibling_index", 1)],
            candidate_namespace=chain_code_key([*(parent_path_code or []), selected_item.get("_sibling_index", 1)]),
            build_platform=build_platform,
            semantic_precheck_terminal=False,
            defer_semantic_checks=True,
            compile_each_step=False,
            step_compile_timeout_seconds=step_compile_timeout_seconds,
        )
        candidate_results.append(candidate_result)
        events.append(candidate_result["event"])
        if not candidate_result["accepted"]:
            failed_counts[strategy_id] = failed_counts.get(strategy_id, 0) + 1
        elif lazy_fallback_execution and stage != "Tiling":
            # Candidates are ranked by the LLM. Outside deterministic Tiling,
            # materialize only the first successful choice; lower-ranked items
            # remain recorded as fallbacks instead of becoming eager branches.
            break

    accepted_candidates = [item for item in candidate_results if item["accepted"]]
    best_candidate = choose_best_candidate(accepted_candidates)
    if not accepted_candidates and subphase_can_continue_after_candidate_failures(current_subphase):
        skipped_ir = controller.skip_current_subphase("all selected candidates failed; continue along stage graph")
        event = {
            "stage": stage,
            "subphase": current_subphase_id,
            "status": "skipped",
            "reason": "all selected candidates failed; continue along stage graph",
            "next_subphase": (current_subphase or {}).get("next_subphase"),
            "failed_strategy_counts": failed_counts,
        }
        return {
            "accepted_candidate": None,
            "accepted_candidates": [],
            "skipped_ir": skipped_ir,
            "events": [*events, event],
            "failed_strategy_counts": failed_counts,
            "summary": {
                "stage": stage,
                "current_subphase": current_subphase_id,
                "filtered_strategy_count": available_strategy_index.get("strategy_count", 0),
                "llm_selected_count": llm_selected_count,
                "selection_mode": "skip_failed_optional_subphase_and_continue",
                "top_k_strategies_per_subphase": DEFAULT_TOP_K_STRATEGIES_PER_SUBPHASE,
                "stage_candidate_count": len(selected_strategy.get("candidates", [])),
                "available_strategy_count": available_strategy_index.get("strategy_count", 0),
                "chain_record": chain_record,
                "precheck_summary": precheck_result.get("summary"),
                "evaluated_candidate_count": len(candidate_results),
                "local_failed_count": len(local_failed_events),
                "local_failed_events": local_failed_events,
                "accepted_candidate_count": 0,
                "selected_strategy_id": None,
                "selected_candidate_code_dir": None,
                "selected_gflops": None,
                "skipped": True,
                "skip_reason": event["reason"],
            },
        }

    return {
        "accepted_candidate": best_candidate,
        "accepted_candidates": accepted_candidates,
        "events": events,
        "failed_strategy_counts": failed_counts,
        "summary": {
            "stage": stage,
            "current_subphase": current_subphase_id,
            "filtered_strategy_count": available_strategy_index.get("strategy_count", 0),
            "llm_selected_count": llm_selected_count,
            "selection_mode": selected_strategy.get("selection_mode"),
            "top_k_strategies_per_subphase": DEFAULT_TOP_K_STRATEGIES_PER_SUBPHASE,
            "stage_candidate_count": len(selected_strategy.get("candidates", [])),
            "available_strategy_count": available_strategy_index.get("strategy_count", 0),
            "chain_record": chain_record,
            "precheck_summary": precheck_result.get("summary"),
            "evaluated_candidate_count": len(candidate_results),
            "local_failed_count": len(local_failed_events),
            "local_failed_events": local_failed_events,
            "accepted_candidate_count": len(accepted_candidates),
            "selected_strategy_id": best_candidate["strategy_id"] if best_candidate else None,
            "selected_candidate_code_dir": best_candidate.get("candidate_code_dir") if best_candidate else None,
            "selected_gflops": candidate_gflops(best_candidate) if best_candidate else None,
            "lazy_fallback_execution": lazy_fallback_execution,
            "chain_workers": execution_config().chain_workers,
            "fallback_candidate_count": max(0, llm_selected_count - len(candidate_results)),
        },
    }


def run_exhaustive_stage(
    stage: str,
    frontier: list[dict[str, Any]],
    raw_strategy_index: dict[str, Any],
    dependency_graph: dict[str, Any],
    strategy_library: dict[str, Any],
    client: OpenAICompatibleClient,
    user_question: str,
    profile: str | None = None,
    max_frontier_states: int = 0,
    exhaust_pending: bool = False,
    build_platform: str = "windows",
    compile_each_step: bool = DEFAULT_COMPILE_EACH_STRATEGY_STEP,
    compile_each_stage: bool = DEFAULT_COMPILE_EACH_STAGE,
    step_compile_timeout_seconds: int = DEFAULT_STEP_COMPILE_TIMEOUT_SECONDS,
    stage_benchmark_runs: int = DEFAULT_STAGE_BENCHMARK_RUNS,
    stage_benchmark_warmup_runs: int = DEFAULT_STAGE_BENCHMARK_WARMUP_RUNS,
    tiling_top_k_per_level: int = DEFAULT_TILING_TOP_K_PER_LEVEL,
    tiling_resource_pool_size: int = DEFAULT_TILING_RESOURCE_POOL_SIZE,
    tiling_llm_top_n: int = DEFAULT_TILING_LLM_TOP_N,
    preserve_tiling_lineages: bool = False,
    top_k_per_tiling_lineage: int = DEFAULT_TOP_K_PER_TILING_LINEAGE,
    lazy_fallback_execution: bool = DEFAULT_LAZY_FALLBACK_EXECUTION,
) -> dict[str, Any]:
    next_frontier = []
    active_frontier = list(frontier)
    pending_frontier: list[dict[str, Any]] = []
    terminal_states = []
    state_summaries = []
    accepted_candidates = []
    failed_counts: dict[str, int] = {}
    total_events = []
    seen_paths = {tuple(state.get("path", [])) for state in frontier}
    output_paths: set[tuple[str, ...]] = set()
    round_index = 0

    while active_frontier or (
        pending_frontier and (exhaust_pending or max_frontier_states <= 0 or len(next_frontier) < max_frontier_states)
    ):
        if not active_frontier and pending_frontier:
            if preserve_tiling_lineages:
                if not pending_frontier:
                    continue
                active_frontier, pending_frontier = take_frontier_batch_per_tiling_lineage(
                    pending_frontier,
                    top_k_per_tiling_lineage,
                )
            else:
                active_frontier, pending_frontier = take_frontier_batch(pending_frontier, max_frontier_states)
            total_events.append(
                {
                    "stage": stage,
                    "status": "backtrack",
                    "reason": "active beam was exhausted; resumed from pending sibling strategy paths",
                    "resumed_state_count": len(active_frontier),
                    "remaining_pending_state_count": len(pending_frontier),
                }
            )

        round_index += 1
        new_active = []
        def invoke_state(state):
            with chain_client(client) as worker_client:
                return run_stage(
                    stage=stage,
                    current_ir=copy.deepcopy(state["current_ir"]),
                    history=copy_history(state["history"]),
                    raw_strategy_index=raw_strategy_index,
                    dependency_graph=dependency_graph,
                    strategy_library=strategy_library,
                    client=worker_client,
                    user_question=user_question,
                    profile=profile,
                    base_source_snapshot=dict(state["source_snapshot"]),
                    candidate_namespace=chain_code_key(state.get("path_code", [])),
                    parent_path=state.get("path", []),
                    parent_path_code=state.get("path_code", []),
                    build_platform=build_platform,
                    compile_each_step=compile_each_step,
                    compile_each_stage=compile_each_stage,
                    step_compile_timeout_seconds=step_compile_timeout_seconds,
                    stage_benchmark_runs=stage_benchmark_runs,
                    stage_benchmark_warmup_runs=stage_benchmark_warmup_runs,
                    tiling_top_k_per_level=tiling_top_k_per_level,
                    tiling_resource_pool_size=tiling_resource_pool_size,
                    tiling_llm_top_n=tiling_llm_top_n,
                    lazy_fallback_execution=lazy_fallback_execution,
                )
        outcomes = parallel_chain_map(
            invoke_state, active_frontier,
            lambda state: chain_code_key(state.get("path_code", [])),
        )
        for state, outcome in zip(active_frontier, outcomes):
            for output, data in outcome.latest.items():
                save_json(output, data)
            if outcome.error is not None:
                stage_result = {
                    "accepted_candidates": [], "failed_strategy_counts": {},
                    "summary": {"status": "chain_error", "error_message": str(outcome.error)},
                    "events": [{"stage": stage, "status": "chain_error",
                                "input_state_id": state["state_id"], "error_message": str(outcome.error)}],
                }
            else:
                stage_result = outcome.value
            state_summaries.append(
                {
                    "round": round_index,
                    "input_state_id": state["state_id"],
                    "input_path": state.get("path", []),
                    "input_path_code": chain_code_key(state.get("path_code", [])),
                    **stage_result["summary"],
                }
            )
            total_events.extend(stage_result.get("events", []))
            add_failed_counts(failed_counts, stage_result.get("failed_strategy_counts", {}))

            if stage_result.get("advanced_ir") is not None:
                advanced_history = copy_history(state["history"])
                advanced_history["events"].extend(stage_result.get("events", []))
                merge_strategy_progress_from_ir(advanced_history, stage_result["advanced_ir"])
                advanced_state = make_frontier_state(
                    state_id=f"{state['state_id']}->advance_{safe_name(stage)}_r{round_index}",
                    current_ir=stage_result["advanced_ir"],
                    source_snapshot=state["source_snapshot"],
                    history=advanced_history,
                    path=list(state.get("path", [])),
                    path_code=list(state.get("path_code", [])),
                    last_candidate=state.get("last_candidate"),
                    code_dir=state.get("code_dir"),
                )
                new_active.append(advanced_state)
                continue

            if stage_result.get("skipped_ir") is not None:
                skipped_history = copy_history(state["history"])
                skipped_history["events"].extend(stage_result.get("events", []))
                merge_strategy_progress_from_ir(skipped_history, stage_result["skipped_ir"])
                skipped_state = make_frontier_state(
                    state_id=f"{state['state_id']}->skip_{safe_name(stage)}_r{round_index}",
                    current_ir=stage_result["skipped_ir"],
                    source_snapshot=state["source_snapshot"],
                    history=skipped_history,
                    path=list(state.get("path", [])),
                    path_code=list(state.get("path_code", [])),
                    last_candidate=state.get("last_candidate"),
                    code_dir=state.get("code_dir"),
                )
                new_active.append(skipped_state)
                continue

            if stage_result.get("stage_completed_ir") is not None:
                completed_history = copy_history(state["history"])
                completed_history["events"].extend(stage_result.get("events", []))
                completed_history.setdefault("completed_stages", []).append(stage)
                merge_strategy_progress_from_ir(completed_history, stage_result["stage_completed_ir"])
                completed_history.setdefault("stage_checkpoints", []).append(
                    {
                        "stage": stage,
                        "path": list(state.get("path", [])),
                        "path_code": chain_code_key(state.get("path_code", [])),
                        "compile_run": (
                            stage_result.get("summary", {}).get("stage_verification", {}).get("stage_compile_run")
                            or {"enabled": False, "status": "deferred"}
                        ),
                        "source_snapshot_available": bool(stage_result.get("stage_source_snapshot")),
                    }
                )
                completed_path_key = tuple(state.get("path", []))
                if completed_path_key not in output_paths:
                    output_paths.add(completed_path_key)
                    next_frontier.append(
                        make_frontier_state(
                            state_id=f"{state['state_id']}->complete_{safe_name(stage)}",
                            current_ir=stage_result["stage_completed_ir"],
                            source_snapshot=stage_result.get("stage_source_snapshot") or state["source_snapshot"],
                            history=completed_history,
                            path=list(state.get("path", [])),
                            path_code=list(state.get("path_code", [])),
                            last_candidate=state.get("last_candidate"),
                            code_dir=state.get("code_dir"),
                        )
                    )
                continue

            if not stage_result.get("accepted_candidates"):
                terminal_state = copy.deepcopy(state)
                terminal_state["terminal_reason"] = "all_children_failed_or_stage_gate_failed"
                terminal_states.append(terminal_state)
                carried_history = copy_history(state["history"])
                carried_history["events"].extend(stage_result.get("events", []))
                merge_failed_counts(carried_history, stage_result.get("failed_strategy_counts", {}))
                total_events.append(
                    {
                        "stage": stage,
                        "input_state_id": state["state_id"],
                        "input_path": state.get("path", []),
                        "status": "blocked",
                        "reason": "stage produced no accepted successor; incomplete chains are not carried to the next stage",
                    }
                )

            for accepted_index, accepted in enumerate(stage_result.get("accepted_candidates", []), start=1):
                new_path = [*state.get("path", []), accepted["strategy_id"]]
                new_path_code = accepted.get("path_code") or [*state.get("path_code", []), accepted_index]
                path_key = tuple(new_path)
                if path_key in seen_paths:
                    continue
                seen_paths.add(path_key)

                accepted_history = copy_history(state["history"])
                accepted_history["events"].extend(stage_result.get("events", []))
                merge_failed_counts(accepted_history, stage_result.get("failed_strategy_counts", {}))
                accepted_history.setdefault("applied_strategy_ids", []).append(accepted["strategy_id"])
                merge_strategy_progress_from_ir(accepted_history, accepted["verified_ir"])
                accepted["history"] = accepted_history
                accepted["path"] = new_path
                accepted["path_code"] = new_path_code
                accepted["verified_ir"].setdefault("strategy", {})["applied_strategy_ids"] = accepted_history[
                    "applied_strategy_ids"
                ]
                accepted["verified_ir"].setdefault("strategy", {})["failed_strategy_counts"] = accepted_history[
                    "failed_strategy_counts"
                ]
                accepted["verified_ir"].setdefault("strategy", {})["history"] = accepted_history["events"]
                accepted["verified_ir"].setdefault("strategy", {})["completed_subphases"] = accepted_history[
                    "completed_subphases"
                ]
                accepted["verified_ir"].setdefault("strategy", {})["applied_micro_strategies"] = accepted_history[
                    "applied_micro_strategies"
                ]
                accepted["verified_ir"].setdefault("strategy", {})["optional_skipped_subphases"] = accepted_history.get(
                    "optional_skipped_subphases", []
                )
                accepted["verified_ir"].setdefault("strategy", {})["chain_code"] = chain_code_key(new_path_code)
                accepted["verified_ir"].setdefault("strategy", {})["chain_path"] = new_path

                new_state = make_frontier_state(
                    state_id=f"{state['state_id']}->{safe_name(accepted['strategy_id'])}",
                    current_ir=accepted["verified_ir"],
                    source_snapshot=accepted["source_snapshot"],
                    history=accepted_history,
                    path=new_path,
                    path_code=new_path_code,
                    last_candidate=accepted,
                    code_dir=accepted.get("candidate_code_dir"),
                )
                new_active.append(new_state)
                accepted_candidates.append(accepted)
                if DEFAULT_SINGLE_PATH_MODE:
                    break
            if DEFAULT_SINGLE_PATH_MODE and new_active:
                break

        if max_frontier_states > 0:
            if preserve_tiling_lineages:
                active_frontier, pruned_frontier = take_frontier_batch_per_tiling_lineage(
                    new_active,
                    top_k_per_tiling_lineage,
                )
            else:
                active_frontier, pruned_frontier = take_frontier_batch(new_active, max_frontier_states)
            pending_frontier.extend(pruned_frontier)
        else:
            active_frontier = new_active

    carry_reason = (
        "single-path mode carries only one selected chain state"
        if DEFAULT_SINGLE_PATH_MODE
        else (
            f"top-{max_frontier_states or DEFAULT_MAX_FRONTIER_STATES_PER_STAGE} beam search carries active states "
            "and backtracks through pending sibling paths"
        )
    )

    stage_candidate_count = len(next_frontier)
    next_frontier, stage_pruned = select_stage_survivors(next_frontier, max_frontier_states)

    return {
        "frontier": next_frontier,
        "terminal_states": terminal_states,
        "accepted_candidates": accepted_candidates,
        "events": total_events,
        "failed_strategy_counts": failed_counts,
        "summary": {
            "stage": stage,
            "selection_mode": DEFAULT_SELECTION_MODE,
            "input_frontier_count": len(frontier),
            "output_frontier_count": len(next_frontier),
            "stage_candidate_count": stage_candidate_count,
            "stage_pruned_paths": [
                {"path_code": chain_code_key(state.get("path_code", [])),
                 "reason": "stage_budget_after_block_diversity"}
                for state in stage_pruned
            ],
            "stage_selection_basis": "best_quality_then_distinct_blocks_then_mmr",
            "accepted_candidate_count": len(accepted_candidates),
            "best_stage_strategy_id": choose_best_candidate(accepted_candidates)["strategy_id"]
            if accepted_candidates
            else None,
            "best_stage_gflops": candidate_gflops(choose_best_candidate(accepted_candidates)),
            "max_frontier_states": max_frontier_states or None,
            "exhaust_pending": exhaust_pending,
            "pending_frontier_count": len(pending_frontier),
            "unexplored_paths": [
                {"path_code": chain_code_key(state.get("path_code", [])), "reason": "search_budget_pruned"}
                for state in pending_frontier
            ],
            "terminal_fallback_count": len(terminal_states),
            "execution": vars(execution_config()),
            "carry_reason": carry_reason,
            "preserve_tiling_lineages": preserve_tiling_lineages,
            "tiling_lineage_count": len(tiling_lineage_keys(next_frontier)),
            "block_tile_root_count": len(tiling_lineage_keys(next_frontier)),
            "top_k_per_tiling_lineage": top_k_per_tiling_lineage if preserve_tiling_lineages else None,
            "lazy_fallback_execution": lazy_fallback_execution,
            "state_summaries": state_summaries,
        },
    }


def deterministic_materialization_mode(strategy: dict[str, Any]) -> str:
    materialization = strategy.get("materialization")
    if isinstance(materialization, dict):
        mode = materialization.get("mode")
        if isinstance(mode, str) and mode:
            return mode
    return DETERMINISTIC_MATERIALIZATION_MODE if is_deterministic_strategy(strategy.get("strategy_id", "")) else LLM_PATCH_MATERIALIZATION_MODE


def is_deterministic_strategy(strategy_id: str) -> bool:
    if strategy_id in {
        "Vectorization.AlignmentGuard",
        "Safety.AssumeDivisibleAligned",
        "Safety.BoundaryPolicy.StaticDivisibleNoGuard",
    }:
        return False
    if not strategy_id or strategy_id.startswith("CPU."):
        return False
    return strategy_id in DETERMINISTIC_STRATEGY_IDS or any(
        strategy_id.startswith(prefix) for prefix in DETERMINISTIC_STRATEGY_PREFIXES
    )


def run_deterministic_strategy_candidate(
    stage: str,
    base_ir: dict[str, Any],
    base_source_snapshot: dict[str, str],
    strategy: dict[str, Any],
    precheck_item: dict[str, Any],
    candidate_namespace: str | None = None,
    chain_path: list[str] | None = None,
    chain_code: list[int] | None = None,
    build_platform: str = "windows",
    compile_each_step: bool = DEFAULT_COMPILE_EACH_STRATEGY_STEP,
    step_compile_timeout_seconds: int = DEFAULT_STEP_COMPILE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    compile_each_step = False
    strategy_id = strategy["strategy_id"]
    attempt = 1
    candidate_code_dir = chain_code_path(chain_path or [strategy_id], chain_code, base_ir)
    restore_source_files(candidate_code_dir, base_source_snapshot)

    patch_result = build_deterministic_patch_result(strategy, base_ir)
    patch_result["repair_attempt"] = attempt
    patch_result["stage"] = stage
    save_candidate_json(DEFAULT_PATCH_OUTPUT, stage, strategy_id, attempt, patch_result, candidate_namespace)

    patch_ir = build_patch_ir(
        ir=base_ir,
        strategy=strategy,
        precheck_item=precheck_item,
        patch_result=patch_result,
    )
    derive_fields(patch_ir, strategy_id=strategy_id)
    patch_ir["repair_attempt"] = attempt
    patch_ir["candidate_stage"] = stage
    save_candidate_json(DEFAULT_PATCH_IR_OUTPUT, stage, strategy_id, attempt, patch_ir, candidate_namespace)

    post_check_result = check_after_codegen(patch_ir, strategy, include_hard_constraints=False)
    post_check_result.update(
        {
            "phase": "after_deterministic_patch_ir_generation",
            "repair_attempt": attempt,
            "stage": stage,
            "strategy_id": strategy_id,
            "semantic_defect_check": "skipped",
            "message": "Deterministic strategies are gated only by preconditions and patch-IR postconditions.",
        }
    )
    save_candidate_json(DEFAULT_POSTCHECK_OUTPUT, stage, strategy_id, attempt, post_check_result, candidate_namespace)
    if post_check_result.get("accepted_by_ir_checker") is False:
        verified_ir = mark_deterministic_pre_post_failed(patch_ir, post_check_result, candidate_code_dir)
        diagnosis = deterministic_semantic_diagnosis_skipped(stage, strategy_id, accepted=False, message="deterministic patch-IR postconditions failed")
        verified_ir = attach_diagnosis(verified_ir, diagnosis)
        save_candidate_json(DEFAULT_DIAGNOSIS_OUTPUT, stage, strategy_id, attempt, diagnosis, candidate_namespace)
        save_candidate_json(DEFAULT_VERIFIED_IR_OUTPUT, stage, strategy_id, attempt, verified_ir, candidate_namespace)
        attempts = [
            {
                "attempt": attempt,
                "patch_output": str(candidate_path(DEFAULT_PATCH_OUTPUT, stage, strategy_id, attempt, candidate_namespace)),
                "postcheck_output": str(candidate_path(DEFAULT_POSTCHECK_OUTPUT, stage, strategy_id, attempt, candidate_namespace)),
                "diagnosis_output": str(candidate_path(DEFAULT_DIAGNOSIS_OUTPUT, stage, strategy_id, attempt, candidate_namespace)),
                "patched_code_output": None,
                "code_ast_output": None,
                "candidate_code_dir": str(candidate_code_dir),
                "path_code": chain_code or [],
                "verification": verified_ir.get("verification", {}).get("summary", {}),
                "defect_count": 0,
                "materialization_mode": DETERMINISTIC_MATERIALIZATION_MODE,
            }
        ]
        return {
            "accepted": False,
            "stage": stage,
            "strategy_id": strategy_id,
            "verified_ir": verified_ir,
            "source_snapshot": base_source_snapshot,
            "candidate_code_dir": None,
            "path_code": chain_code or [],
            "attempts": attempts,
            "event": candidate_event(stage, strategy_id, "failed", verified_ir, attempts),
        }

    try:
        code_apply_result, generated_code = apply_deterministic_code_patch(candidate_code_dir, strategy_id, patch_ir)
    except Exception as exc:
        verified_ir = mark_llm_generation_failed(
            patch_ir,
            stage=stage,
            strategy_id=strategy_id,
            phase="deterministic_materialization",
            error=exc,
            candidate_code_dir=candidate_code_dir,
        )
        diagnosis = deterministic_semantic_diagnosis_skipped(
            stage,
            strategy_id,
            accepted=False,
            message=f"deterministic materialization failed: {type(exc).__name__}: {exc}",
        )
        verified_ir = attach_diagnosis(verified_ir, diagnosis)
        save_candidate_json(DEFAULT_DIAGNOSIS_OUTPUT, stage, strategy_id, attempt, diagnosis, candidate_namespace)
        save_candidate_json(DEFAULT_VERIFIED_IR_OUTPUT, stage, strategy_id, attempt, verified_ir, candidate_namespace)
        attempts = [
            failed_generation_attempt_record(
                stage,
                strategy_id,
                attempt,
                candidate_code_dir,
                chain_code,
                verified_ir,
                diagnosis,
                "deterministic_materialization",
            )
        ]
        return {
            "accepted": False,
            "stage": stage,
            "strategy_id": strategy_id,
            "verified_ir": verified_ir,
            "source_snapshot": base_source_snapshot,
            "candidate_code_dir": None,
            "path_code": chain_code or [],
            "attempts": attempts,
            "event": candidate_event(stage, strategy_id, "failed", verified_ir, attempts),
        }

    generated_code.update({"repair_attempt": attempt, "stage": stage, "candidate_code_dir": str(candidate_code_dir)})
    save_candidate_json(DEFAULT_PATCHED_CODE_OUTPUT, stage, strategy_id, attempt, generated_code, candidate_namespace)

    code_apply_result["candidate_code_dir"] = str(candidate_code_dir)
    patch_ir["patch_to_code_application"] = code_apply_result
    if code_apply_result["status"] != "pass":
        verified_ir = mark_patch_apply_failed(patch_ir, code_apply_result)
    else:
        verified_ir = mark_chain_step_generated(patch_ir, code_apply_result)
        syntax_report = check_generated_source_syntax(candidate_code_dir, backend_ast_files(verified_ir))
        attach_source_syntax_report(verified_ir, syntax_report, candidate_code_dir)
        if compile_each_step and not source_syntax_gate_failed(verified_ir):
            verified_ir = verify_strategy_step_compile_gate(
                verified_ir,
                candidate_code_dir,
                build_platform,
                step_compile_timeout_seconds,
            )
        verified_ir["code_completeness"] = {
            "accepted": True,
            "results": [],
            "semantic_obligations": {
                "accepted": None,
                "skipped_for_deterministic_strategy": True,
                "message": "Semantic defect verification is skipped for deterministic micro-strategies; pre/post IR checks are authoritative here.",
            },
        }
        verified_ir.setdefault("verification", {}).setdefault("summary", {})["semantic_defect_check"] = "skipped"

    if source_syntax_gate_failed(verified_ir) or (compile_each_step and not step_compile_gate_passed(verified_ir)):
        diagnosis = diagnose_defects(verified_ir, post_check_result)
        diagnosis["semantic_defect_check"] = "skipped_for_deterministic_strategy"
    else:
        diagnosis = deterministic_semantic_diagnosis_skipped(
            stage,
            strategy_id,
            accepted=candidate_is_accepted(verified_ir),
            message="Deterministic strategy accepted by pre/post IR checks and step compile gate; semantic defect diagnosis skipped."
            if candidate_is_accepted(verified_ir)
            else code_apply_result.get("error_message"),
        )
    verified_ir = attach_diagnosis(verified_ir, diagnosis)
    save_candidate_json(DEFAULT_DIAGNOSIS_OUTPUT, stage, strategy_id, attempt, diagnosis, candidate_namespace)
    save_candidate_json(DEFAULT_VERIFIED_IR_OUTPUT, stage, strategy_id, attempt, verified_ir, candidate_namespace)

    attempts = [
        {
            "attempt": attempt,
            "patch_output": str(candidate_path(DEFAULT_PATCH_OUTPUT, stage, strategy_id, attempt, candidate_namespace)),
            "postcheck_output": str(candidate_path(DEFAULT_POSTCHECK_OUTPUT, stage, strategy_id, attempt, candidate_namespace)),
            "diagnosis_output": str(candidate_path(DEFAULT_DIAGNOSIS_OUTPUT, stage, strategy_id, attempt, candidate_namespace)),
            "patched_code_output": str(candidate_path(DEFAULT_PATCHED_CODE_OUTPUT, stage, strategy_id, attempt, candidate_namespace)),
            "code_ast_output": None,
            "candidate_code_dir": str(candidate_code_dir),
            "path_code": chain_code or [],
            "verification": verified_ir.get("verification", {}).get("summary", {}),
            "defect_count": diagnosis.get("defect_count"),
            "materialization_mode": DETERMINISTIC_MATERIALIZATION_MODE,
            "step_compile_gate": verified_ir.get("verification", {}).get("summary", {}).get("step_compile_gate"),
        }
    ]

    if candidate_is_accepted(verified_ir):
        source_snapshot = snapshot_source_files(candidate_code_dir, backend_code_files(verified_ir))
        archive_generated_chain_step(candidate_code_dir, stage, strategy_id, attempt, verified_ir)
        return {
            "accepted": True,
            "stage": stage,
            "strategy_id": strategy_id,
            "verified_ir": verified_ir,
            "source_snapshot": source_snapshot,
            "candidate_code_dir": str(candidate_code_dir),
            "path_code": chain_code or [],
            "attempts": attempts,
            "event": candidate_event(stage, strategy_id, "accepted", verified_ir, attempts),
        }

    return {
        "accepted": False,
        "stage": stage,
        "strategy_id": strategy_id,
        "verified_ir": verified_ir,
        "source_snapshot": base_source_snapshot,
        "candidate_code_dir": None,
        "path_code": chain_code or [],
        "attempts": attempts,
        "event": candidate_event(stage, strategy_id, "failed", verified_ir, attempts),
    }


def build_deterministic_patch_result(strategy: dict[str, Any], ir: dict[str, Any]) -> dict[str, Any]:
    strategy_id = strategy["strategy_id"]
    deterministic_updates = copy.deepcopy(strategy.get("ir_updates") or {})
    deterministic_updates.update(synthesize_ir_updates(strategy_id))
    deterministic_updates = clean_expected_ir_updates(deterministic_updates)
    regions = [
        {
            "file": "cuda_kernel.cuh",
            "anchor": f"{region}_BEGIN",
            "change_summary": f"Deterministically materialize {region} for {strategy_id}.",
        }
        for region in deterministic_code_regions(strategy_id, ir)
    ]
    return {
        "strategy_id": strategy_id,
        "generation_method": "deterministic_region_materialization",
        "modified_code_regions": regions,
        "modified_ir_fields": list(deterministic_updates.keys()),
        "ir_updates": deterministic_updates,
        "code_patch": {
            "diff": "",
            "note": "This patch is materialized by deterministic SGPO anchor replacement, not by LLM free-form code generation.",
        },
        "expected_effect": "Update stable GEMM construction parameters or simple code regions without LLM variability.",
        "potential_risks": [
            "Only local anchor regions are changed; final correctness/performance is verified after the full chain."
        ],
    }


def mark_deterministic_pre_post_failed(
    patch_ir: dict[str, Any],
    post_check_result: dict[str, Any],
    candidate_code_dir: Path,
) -> dict[str, Any]:
    verified_ir = copy.deepcopy(patch_ir)
    verified_ir["post_check"] = post_check_result
    verified_ir["chain_step"] = {
        "status": "failed",
        "phase": "deterministic_pre_post_check",
        "candidate_code_dir": str(candidate_code_dir),
        "reason": "deterministic strategy failed patch-IR postcondition check",
    }
    verification = verified_ir.setdefault("verification", {})
    verification["accepted"] = False
    verification["compile"] = {"status": "not_run"}
    verification["correctness"] = {"status": "not_run"}
    verification["runtime_safety"] = {"status": "not_run", "cuda_error": None}
    verification["summary"] = {
        "chain_step_status": "failed",
        "failed_phase": "deterministic_pre_post_check",
        "postconditions_ok": post_check_result.get("postconditions_ok"),
        "semantic_defect_check": "skipped",
    }
    return verified_ir


def deterministic_semantic_diagnosis_skipped(
    stage: str,
    strategy_id: str,
    accepted: bool,
    message: str | None = None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "status": "skipped" if accepted else "failed_without_semantic_diagnosis",
        "related_strategy": strategy_id,
        "defect_count": 0,
        "defects": [],
        "semantic_defect_check": "skipped_for_deterministic_strategy",
        "message": message
        or "Deterministic strategy uses only precondition and patch-IR postcondition checks.",
    }


def deterministic_code_regions(strategy_id: str, ir: dict[str, Any]) -> list[str]:
    if strategy_id == "Mapping.WarpStore.CoalescedC":
        return ["STORE"]
    if strategy_id.startswith("Tiling."):
        return ["LAUNCH_CONFIG"]
    if strategy_id.startswith("Mapping."):
        return ["LAUNCH_CONFIG", "INDEX_MAPPING", "REGISTER_DECL", "COMPUTE_INNER", "STORE"]
    if strategy_id == "Layout.SharedMemory.AB.Basic":
        return ["SHARED_DECL"]
    if strategy_id in {"Layout.RegisterTile.C", "Register.AccumulatorLayout.2DArray"}:
        return ["REGISTER_DECL", "COMPUTE_INNER", "STORE"]
    if strategy_id in {
        "Safety.BoundaryPolicy.GeneralGuarded",
        "Safety.BoundaryPolicy.StaticDivisibleNoGuard",
        "Safety.AssumeDivisibleAligned",
        "Epilogue.AlphaBeta.General",
        "Epilogue.BetaZero.FastPath",
        "Epilogue.BetaOne.FastPath",
        "Epilogue.StoreC.CoalescedScalar",
        "Vectorization.StoreC.SafeScalar",
        "Mapping.WarpStore.CoalescedC",
    }:
        return ["STORE"]
    return []


def load_strategy_scoped_code_context(
    ir: dict[str, Any],
    code_dir: Path,
    code_files: list[str],
    strategy: dict[str, Any],
    patch_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if target_backend(ir) == "cpu":
        return load_code_context(code_dir, code_files)
    context = load_code_context(code_dir, [name for name in code_files if name != "main.cpp"])
    context["context_scope"] = "complete_current_parent_kernel"
    context["suggested_regions"] = strategy_relevant_regions(strategy, patch_result)
    context["region_selection"] = "Choose all coupled existing regions needed; suggestions are not restrictions."
    from SGPO.verification.locked_repair import cooperative_load_reference
    context["cooperative_load_reference"] = cooperative_load_reference(
        context.get("files", {}).get("cuda_kernel.cuh", ""), ir)
    return context


def strategy_relevant_regions(
    strategy: dict[str, Any],
    patch_result: dict[str, Any] | None = None,
) -> list[str]:
    if strategy.get("strategy_id") == "Reordering.LoadCompute.SeparatePhases":
        # These declarations are read-only context for the local loop rewrite.
        return ["LAUNCH_CONFIG", "SHARED_DECL", "INDEX_MAPPING", "REGISTER_DECL",
                "GLOBAL_TO_SHARED_LOAD", "SYNC_AFTER_LOAD", "MAIN_LOOP", "STORE"]
    regions: list[str] = []
    for item in (patch_result or {}).get("modified_code_regions", []) or []:
        if isinstance(item, dict):
            regions.extend(region_names_from_text(" ".join(str(item.get(key, "")) for key in ["anchor", "region", "change_summary"])))
    regions.extend(regions_from_strategy_metadata(strategy))
    regions.extend(regions_from_strategy_id(strategy.get("strategy_id", "")))
    for bundled_strategy in strategy.get("bundle_strategies", []) or []:
        if isinstance(bundled_strategy, dict):
            regions.extend(regions_from_strategy_metadata(bundled_strategy))
            regions.extend(regions_from_strategy_id(bundled_strategy.get("strategy_id", "")))
    return unique_region_names(regions)


def regions_from_strategy_metadata(strategy: dict[str, Any]) -> list[str]:
    regions: list[str] = []
    for value in strategy.get("modifies_regions", []) or []:
        regions.extend(regions_from_semantic_region(str(value)))
    subphase = str(strategy.get("subphase") or strategy.get("subphase_id") or "")
    regions.extend(regions_from_subphase_name(subphase))
    return regions


def regions_from_semantic_region(value: str) -> list[str]:
    text = value.lower()
    regions: list[str] = []
    if any(token in text for token in ["launch", "tile_config", "tiling"]):
        regions.append("LAUNCH_CONFIG")
    if any(token in text for token in ["shared_memory_layout", "shared_decl", "shared_memory_declaration"]):
        regions.append("SHARED_DECL")
    if any(token in text for token in ["global_load", "shared_load", "cooperative", "load"]):
        regions.append("GLOBAL_TO_SHARED_LOAD")
    if any(token in text for token in ["sync", "barrier"]):
        regions.append("SYNC_AFTER_LOAD")
    if any(token in text for token in ["index", "mapping", "thread_map", "lane", "warp"]):
        regions.append("INDEX_MAPPING")
    if any(token in text for token in ["register", "accumulator"]):
        regions.append("REGISTER_DECL")
    if any(token in text for token in ["compute", "k_loop", "ffma", "unroll", "prefetch", "pipeline"]):
        regions.extend(["MAIN_LOOP", "COMPUTE_INNER"])
    if any(token in text for token in ["store", "epilogue", "c_write"]):
        regions.append("STORE")
    return regions


def regions_from_subphase_name(subphase: str) -> list[str]:
    text = subphase.lower()
    if not text:
        return []
    if "blocktile" in text or "warptile" in text or "threadtile" in text:
        return ["LAUNCH_CONFIG"]
    if "mapping" in text or "lane" in text:
        return ["INDEX_MAPPING"]
    if "sharedmemorydeclaration" in text:
        return ["SHARED_DECL"]
    if "sharedmemorytransform" in text or "cooperativeload" in text or "loadvectorization" in text:
        return ["GLOBAL_TO_SHARED_LOAD", "SYNC_AFTER_LOAD"]
    if "compute" in text or "prefetch" in text or "buffering" in text or "pipeline" in text:
        return ["MAIN_LOOP", "COMPUTE_INNER", "SYNC_AFTER_LOAD"]
    if "store" in text or "epilogue" in text or "alphabetapolicy" in text:
        return ["STORE"]
    return []


def regions_from_strategy_id(strategy_id: str) -> list[str]:
    if not strategy_id:
        return []
    if strategy_id.startswith("Tiling."):
        return ["LAUNCH_CONFIG"]
    if strategy_id.startswith("Mapping."):
        return ["INDEX_MAPPING", "STORE"] if "Store" in strategy_id else ["INDEX_MAPPING"]
    if strategy_id.startswith("Layout.SharedMemory.AB"):
        return ["SHARED_DECL", "GLOBAL_TO_SHARED_LOAD", "SYNC_AFTER_LOAD"]
    if strategy_id.startswith("Layout.SharedMemory."):
        return ["SHARED_DECL", "GLOBAL_TO_SHARED_LOAD", "COMPUTE_INNER"]
    if strategy_id.startswith("Layout.Register") or strategy_id.startswith("Register.Accumulator"):
        return ["REGISTER_DECL", "COMPUTE_INNER", "STORE"]
    if strategy_id.startswith("Reordering.ThreadMapping") or strategy_id.startswith("Reordering.Cooperative"):
        return ["INDEX_MAPPING", "GLOBAL_TO_SHARED_LOAD", "SYNC_AFTER_LOAD"]
    if strategy_id.startswith("Reordering.KLoop") or strategy_id.startswith("Reordering.WarpCompute") or strategy_id.startswith("Register.FFMA"):
        return ["MAIN_LOOP", "COMPUTE_INNER"]
    if strategy_id.startswith("Register.Cache"):
        return ["REGISTER_DECL", "COMPUTE_INNER"]
    if strategy_id.startswith("Vectorization.GlobalLoad"):
        return ["GLOBAL_TO_SHARED_LOAD"]
    if strategy_id.startswith("Vectorization.StoreC") or strategy_id.startswith("Epilogue."):
        return ["STORE"]
    if strategy_id.startswith("Pipeline."):
        return ["SHARED_DECL", "GLOBAL_TO_SHARED_LOAD", "MAIN_LOOP", "COMPUTE_INNER", "SYNC_AFTER_LOAD"]
    if strategy_id.startswith(("Memory.Prefetch.", "Scheduling.", "Reduction.StreamK.", "Memory.L2Reuse.")):
        return ["LAUNCH_CONFIG", "MAIN_LOOP", "COMPUTE_INNER"]
    if strategy_id.startswith("Safety."):
        return ["GLOBAL_TO_SHARED_LOAD", "STORE"]
    if strategy_id.startswith("Compiler."):
        return ["LAUNCH_CONFIG"]
    return ["COMPUTE_INNER"]


def region_names_from_text(text: str) -> list[str]:
    names = [
        "LAUNCH_CONFIG",
        "SHARED_DECL",
        "INDEX_MAPPING",
        "REGISTER_DECL",
        "GLOBAL_TO_SHARED_LOAD",
        "NEXT_TILE_LOAD",
        "SYNC_AFTER_LOAD",
        "MAIN_LOOP",
        "COMPUTE_INNER",
        "STORE",
    ]
    return [name for name in names if name in text]


def unique_region_names(regions: list[str]) -> list[str]:
    regions = list(regions)
    # A physical shared layout is consumed by all tile loads and reductions.
    if "SHARED_DECL" in regions:
        regions.extend(["GLOBAL_TO_SHARED_LOAD", "NEXT_TILE_LOAD", "COMPUTE_INNER", "MAIN_LOOP"])
    if "GLOBAL_TO_SHARED_LOAD" in regions:
        regions.append("NEXT_TILE_LOAD")
    valid = {
        "LAUNCH_CONFIG",
        "SHARED_DECL",
        "INDEX_MAPPING",
        "REGISTER_DECL",
        "GLOBAL_TO_SHARED_LOAD",
        "NEXT_TILE_LOAD",
        "SYNC_AFTER_LOAD",
        "MAIN_LOOP",
        "COMPUTE_INNER",
        "STORE",
    }
    result = []
    seen = set()
    for region in regions:
        if region in valid and region not in seen:
            result.append(region)
            seen.add(region)
    return result or ["COMPUTE_INNER"]


def apply_deterministic_code_patch(
    code_root: Path,
    strategy_id: str,
    patch_ir: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    kernel_path = code_root / "cuda_kernel.cuh"
    if not kernel_path.exists():
        raise FileNotFoundError(f"deterministic materializer requires cuda_kernel.cuh: {kernel_path}")

    disk_original = kernel_path.read_text(encoding="utf-8")
    original = disk_original
    if "SGPO_NAIVE_BASELINE" in disk_original:
        scaffold = ROOT / "gemm_code" / "skeleton_template" / "cuda_kernel.cuh"
        original = scaffold.read_text(encoding="utf-8")
    content = original
    applied = []
    if strategy_id == "Reordering.LoadCompute.SeparatePhases":
        raise ValueError("SeparatePhases requires an LLM local-region patch; full-kernel materialization is forbidden")
    for region, replacement in deterministic_region_replacements(strategy_id, patch_ir).items():
        content = replace_anchor_region(content, region, replacement)
        applied.append({"file": "cuda_kernel.cuh", "region": region, "change_summary": f"Replaced {region} region."})

    if content != disk_original:
        kernel_path.write_text(content, encoding="utf-8")

    generated_code = {
        "strategy_id": strategy_id,
        "generation_method": "deterministic_region_materialization",
        "files": [
            {
                "path": "cuda_kernel.cuh",
                "content": content,
                "change_summary": "Applied deterministic SGPO anchor replacements.",
            }
        ],
        "modified_regions": applied,
    }
    return (
        {
            "status": "pass",
            "method": "deterministic_region_materialization",
            "applied": applied,
            "materialization": {"status": "pass", "message": "Deterministic materialization completed."},
        },
        generated_code,
    )


def deterministic_region_replacements(strategy_id: str, ir: dict[str, Any]) -> dict[str, list[str]]:
    replacements: dict[str, list[str]] = {}
    if "LAUNCH_CONFIG" in deterministic_code_regions(strategy_id, ir):
        replacements["LAUNCH_CONFIG"] = launch_config_lines(ir)
    if "INDEX_MAPPING" in deterministic_code_regions(strategy_id, ir):
        replacements["INDEX_MAPPING"] = index_mapping_lines(ir)
    if "SHARED_DECL" in deterministic_code_regions(strategy_id, ir):
        replacements["SHARED_DECL"] = shared_decl_lines(ir)
    if "REGISTER_DECL" in deterministic_code_regions(strategy_id, ir):
        replacements["REGISTER_DECL"] = register_decl_lines(ir)
    if "COMPUTE_INNER" in deterministic_code_regions(strategy_id, ir):
        replacements["COMPUTE_INNER"] = compute_inner_lines(ir)
    if "STORE" in deterministic_code_regions(strategy_id, ir):
        replacements["STORE"] = store_lines(ir)
    return replacements


def replace_anchor_region(content: str, region: str, replacement_lines: list[str]) -> str:
    lines = content.splitlines()
    begin_marker = f"{region}_BEGIN"
    end_marker = f"{region}_END"
    if sum(begin_marker in line for line in lines) > 1 or sum(end_marker in line for line in lines) > 1:
        raise ValueError(f"Ambiguous duplicate anchor: {region}; cannot replace only the first occurrence")
    begin_index = next((i for i, line in enumerate(lines) if begin_marker in line), None)
    end_index = next((i for i, line in enumerate(lines) if end_marker in line and begin_index is not None and i > begin_index), None)
    if begin_index is None or end_index is None:
        raise ValueError(f"Anchor region not found in cuda_kernel.cuh: {region}")

    body_start = begin_index + 1
    while body_start < end_index and "*/" not in lines[body_start]:
        body_start += 1
    if body_start < end_index and "*/" in lines[body_start]:
        body_start += 1

    end_open_index = find_region_end_open(lines, body_start, end_index)
    body_end = end_open_index if end_open_index is not None else end_index

    indent = ""
    if body_start < len(lines):
        match = re.match(r"^(\s*)", lines[body_start])
        indent = match.group(1) if match else ""
    new_body = [f"{indent}{line}" if line else "" for line in replacement_lines]
    suffix = lines[body_end:]
    if end_open_index is None:
        suffix = [f"{indent}/*"] + suffix
    updated = lines[:body_start] + new_body + suffix
    trailing_newline = "\n" if content.endswith(("\n", "\r\n")) else ""
    return "\n".join(updated) + trailing_newline


def find_region_end_open(lines: list[str], body_start: int, end_index: int) -> int | None:
    for index in range(end_index - 1, body_start - 1, -1):
        stripped = lines[index].strip()
        if not stripped:
            continue
        if re.match(r"^/\*", stripped):
            return index
        return None
    return None


def launch_config_lines(ir: dict[str, Any]) -> list[str]:
    bm = ir_int(ir, "tiling.block_m", 64)
    bn = ir_int(ir, "tiling.block_n", 64)
    bk = ir_int(ir, "tiling.block_k", 8)
    wm = ir_int(ir, "tiling.warp_tile.warp_m", bm)
    wn = ir_int(ir, "tiling.warp_tile.warp_n", bn)
    wmiter = ir_int(ir, "tiling.warp_tile.warp_m_iter", 1)
    wniter = ir_int(ir, "tiling.warp_tile.warp_n_iter", 1)
    tm = ir_int(ir, "tiling.thread_m", 1)
    tn = ir_int(ir, "tiling.thread_n", 1)
    warp_size = ir_int(ir, "hardware.warp_size", 32)
    if not warp_iteration_is_valid(wm, wn, tm, tn, wmiter, wniter, warp_size):
        extents = choose_warp_iteration_extents(wm, wn, tm, tn, warp_size)
        if extents is None:
            raise ValueError(f"No legal warp fragment for {wm}x{wn} / {tm}x{tn}")
        wmiter, wniter = extents
    use_tiled_launch = bool(ir.get("tiling", {}).get("enabled")) or bm > 32 or bn > 32
    launch_lines = [
        f"static const int BM = {bm};",
        f"static const int BN = {bn};",
        f"static const int BK = {bk};",
        f"static const int WM = {wm};",
        f"static const int WN = {wn};",
        f"static const int WMITER = {wmiter};",
        f"static const int WNITER = {wniter};",
        f"static const int TM = {tm};",
        f"static const int TN = {tn};",
    ]
    if use_tiled_launch:
        launch_lines.extend(
            [
                "dim3 block((BM * BN) / (WM * WN) * 32, 1, 1);",
                "dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);",
            ]
        )
    else:
        launch_lines.extend(
            [
                "dim3 block(BN, BM, 1);",
                "dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM), 1);",
            ]
        )
    return launch_lines

def shared_decl_lines(ir: dict[str, Any]) -> list[str]:
    bm = ir_int(ir, "tiling.block_m", 64)
    bn = ir_int(ir, "tiling.block_n", 64)
    bk = ir_int(ir, "tiling.block_k", 8)
    pad_a = ir_int(ir, "memory.shared_A.padding", 0)
    pad_b = ir_int(ir, "memory.shared_B.padding", 0)
    return [
        f"__shared__ float As[2][BK + {pad_a}][BM];" if pad_a else "__shared__ float As[2][BK][BM];",
        f"__shared__ float Bs[2][BK + {pad_b}][BN];" if pad_b else "__shared__ float Bs[2][BK][BN];",
        f"static_assert(BM == {bm} && BN == {bn} && BK == {bk}, \"SGPO launch config and shared-memory IR disagree\");",
    ]


def index_mapping_lines(ir: dict[str, Any]) -> list[str]:
    return [
        "const int elements_per_tile = BM * BN;",
        "const int warp_tiles_n = BN / WN;",
        "const int Wrow = wid / warp_tiles_n;",
        "const int Wcol = wid - Wrow * warp_tiles_n;",
        "const int lane_cols = WNITER / TN;",
        "const int Trow = lane / lane_cols;",
        "const int Tcol = lane - Trow * lane_cols;",
        "const int local_m_base = Wrow * WM + Trow * TM;",
        "const int local_n_base = Wcol * WN + Tcol * TN;",
        "const int global_m_base = tile_m0 + local_m_base;",
        "const int global_n_base = tile_n0 + local_n_base;",
        "const int global_m = global_m_base;",
        "const int global_n = global_n_base;",
        "const int load_a_smem_m = tid % BM;",
        "const int load_a_smem_k = (tid / BM) % BK;",
        "const int load_b_smem_n = tid % BN;",
        "const int load_b_smem_k = (tid / BN) % BK;",
        "const int hightA = CEIL_DIV(BM * BK, thread_num);",
        "const int hightB = CEIL_DIV(BN * BK, thread_num);",
        "(void)global_m_base;",
        "(void)global_n_base;",
        "(void)load_a_smem_m;",
        "(void)load_a_smem_k;",
        "(void)load_b_smem_n;",
        "(void)load_b_smem_k;",
        "(void)hightA;",
        "(void)hightB;",
    ]


def register_decl_lines(ir: dict[str, Any]) -> list[str]:
    return [
        "float results[WM / WMITER * TM][WN / WNITER * TN] = {0.0f};",
        "float regM[TM] = {0.0f};",
        "float regN[TN] = {0.0f};",
    ]


def compute_inner_lines(ir: dict[str, Any]) -> list[str]:
    return [
        "#pragma unroll",
        "for (int k = 0; k < BK; ++k) {",
        "    #pragma unroll",
        "    for (int wm = 0; wm < WM / WMITER; ++wm) {",
        "        #pragma unroll",
        "        for (int wn = 0; wn < WN / WNITER; ++wn) {",
        "            #pragma unroll",
        "            for (int i = 0; i < TM; ++i) {",
        "                regM[i] = As[comp_flag][k][Wrow * WM + wm * WMITER + Trow * TM + i];",
        "            }",
        "            #pragma unroll",
        "            for (int j = 0; j < TN; ++j) {",
        "                regN[j] = Bs[comp_flag][k][Wcol * WN + wn * WNITER + Tcol * TN + j];",
        "            }",
        "            #pragma unroll",
        "            for (int i = 0; i < TM; ++i) {",
        "                #pragma unroll",
        "                for (int j = 0; j < TN; ++j) {",
        "                    results[wm * TM + i][wn * TN + j] += regM[i] * regN[j];",
        "                }",
        "            }",
        "        }",
        "    }",
        "}",
    ]


def store_lines(ir: dict[str, Any]) -> list[str]:
    mode = ir_get(ir, "epilogue.mode")
    accumulator = "results[m + wm * TM][n + wn * TN]"
    value = f"alpha * {accumulator} + beta * C[c_index]"
    if mode == "beta_zero_fast_path":
        value = f"(beta == 0.0f ? alpha * {accumulator} : alpha * {accumulator} + beta * C[c_index])"
    elif mode == "beta_one_fast_path":
        value = f"alpha * {accumulator} + (beta == 1.0f ? C[c_index] : beta * C[c_index])"
    return [
        "#pragma unroll",
        "for (int wm = 0; wm < WM / WMITER; ++wm) {",
        "    #pragma unroll",
        "    for (int wn = 0; wn < WN / WNITER; ++wn) {",
        "        #pragma unroll",
        "        for (int m = 0; m < TM; ++m) {",
        "            #pragma unroll",
        "            for (int n = 0; n < TN; ++n) {",
        "                const int global_m = tile_m0 + Wrow * WM + wm * WMITER + Trow * TM + m;",
        "                const int global_n = tile_n0 + Wcol * WN + wn * WNITER + Tcol * TN + n;",
        "                if (global_m < M && global_n < N) {",
        "                    const int c_index = OFFSET(global_m, global_n, N);",
        f"                    C[c_index] = {value};",
        "                }",
        "            }",
        "        }",
        "    }",
        "}",
    ]


def ir_int(ir: dict[str, Any], dotted_path: str, default: int) -> int:
    value = ir_get(ir, dotted_path)
    return int_or_default(value, default)


def ir_get(ir: dict[str, Any], dotted_path: str, default: Any = None) -> Any:
    current: Any = ir
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def run_strategy_candidate(
    stage: str,
    base_ir: dict[str, Any],
    base_source_snapshot: dict[str, str],
    strategy: dict[str, Any],
    precheck_item: dict[str, Any],
    client: OpenAICompatibleClient,
    candidate_namespace: str | None = None,
    chain_path: list[str] | None = None,
    chain_code: list[int] | None = None,
    semantic_precheck_terminal: bool = True,
    defer_semantic_checks: bool = False,
    build_platform: str = "windows",
    compile_each_step: bool = DEFAULT_COMPILE_EACH_STRATEGY_STEP,
    step_compile_timeout_seconds: int = DEFAULT_STEP_COMPILE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    strategy_id = strategy["strategy_id"]
    bundle_ids = strategy.get("bundle_strategy_ids", [strategy_id])
    host_only = bool(bundle_ids) and all(sid.startswith("Memory.SharedMemory.Carveout.") for sid in bundle_ids)
    if bundle_ids == ["Compiler.ResourceFeedback.PtxasOccupancySweep"] or host_only:
        candidate_code_dir = chain_code_path(chain_path or [strategy_id], chain_code, base_ir)
        restore_source_files(candidate_code_dir, base_source_snapshot)
        patch_result = {"strategy_id": strategy_id, "ir_updates": strategy["ir_updates"],
                        "modified_code_regions": [], "modified_ir_fields": list(strategy["ir_updates"]),
                        "code_patch": {}, "materialization": "host_launch_config" if host_only else "compiler_resource_sweep"}
        patch_ir = build_patch_ir(base_ir, strategy, precheck_item, patch_result)
        if host_only:
            from SGPO.verification.cuda_launch_config import configure_shared_launch
            path = candidate_code_dir / "cuda_kernel.cuh"
            try:
                configured = configure_shared_launch(path.read_text(encoding="utf-8"), patch_ir)
            except ValueError as exc:
                return {"accepted": False, "stage": stage, "strategy_id": strategy_id,
                        "verified_ir": patch_ir, "candidate_code_dir": None, "attempts": [],
                        "error_message": str(exc), "source_snapshot": base_source_snapshot}
            path.write_text(configured, encoding="utf-8")
        save_candidate_json(DEFAULT_PATCH_OUTPUT, stage, strategy_id, 1, patch_result, candidate_namespace)
        return {"accepted": True, "stage": stage, "strategy_id": strategy_id,
                "verified_ir": patch_ir, "candidate_code_dir": str(candidate_code_dir),
                "source_snapshot": snapshot_source_files(candidate_code_dir, backend_code_files(patch_ir)),
                "attempts": [{"attempt": 1, "patch_output": str(candidate_path(DEFAULT_PATCH_OUTPUT, stage, strategy_id, 1, candidate_namespace))}],
                "path_code": chain_code or []}
    if target_backend(base_ir) != "cpu" and deterministic_materialization_mode(strategy) == DETERMINISTIC_MATERIALIZATION_MODE:
        return run_deterministic_strategy_candidate(
            stage=stage,
            base_ir=base_ir,
            base_source_snapshot=base_source_snapshot,
            strategy=strategy,
            precheck_item=precheck_item,
            candidate_namespace=candidate_namespace,
            chain_path=chain_path,
            chain_code=chain_code,
            build_platform=build_platform,
            compile_each_step=compile_each_step,
            step_compile_timeout_seconds=step_compile_timeout_seconds,
        )

    repair_context = None
    attempts = []
    verified_ir = None
    diagnosis = None
    post_check_result = None
    patch_result = None

    for attempt in range(1, DEFAULT_MAX_REPAIR_ATTEMPTS + 1):
        candidate_code_dir = chain_code_path(chain_path or [strategy_id], chain_code, base_ir)
        restore_source_files(candidate_code_dir, base_source_snapshot)
        code_files = backend_code_files(base_ir)
        context_files = backend_llm_context_files(base_ir)
        if target_backend(base_ir) != "cpu":
            # Code-region ownership may expand; IR/strategy ownership remains locked.
            strategy = copy.deepcopy(strategy)
            contract = dict(strategy.get("patch_contract") or {})
            strategy["patch_contract"] = contract
            current_source = base_source_snapshot.get("cuda_kernel.cuh", "")
            contract["allowed_regions"] = [name for name in CUDA_REPAIR_REGIONS
                if f"{name}_BEGIN" in current_source and f"{name}_END" in current_source]
            contract["preserved_regions"] = []
            contract["region_edits_only"] = True
            contract["preserve_existing_strategy_semantics"] = True
        try:
            patch_code_context = load_strategy_scoped_code_context(
                base_ir,
                candidate_code_dir,
                context_files,
                strategy,
            )
            patch_result = generate_patch_with_llm(
                client=client,
                ir=base_ir,
                strategy=strategy,
                precheck_item=precheck_item,
                code_context=patch_code_context,
                prompt_path=backend_patch_prompt(base_ir),
                repair_context=repair_context,
            )
        except Exception as exc:
            verified_ir = mark_llm_generation_failed(
                base_ir,
                stage=stage,
                strategy_id=strategy_id,
                phase="patch_generation",
                error=exc,
                candidate_code_dir=candidate_code_dir,
            )
            diagnosis = diagnose_defects(verified_ir)
            verified_ir = attach_diagnosis(verified_ir, diagnosis)
            save_candidate_json(DEFAULT_DIAGNOSIS_OUTPUT, stage, strategy_id, attempt, diagnosis, candidate_namespace)
            save_candidate_json(DEFAULT_VERIFIED_IR_OUTPUT, stage, strategy_id, attempt, verified_ir, candidate_namespace)
            attempts.append(failed_generation_attempt_record(stage, strategy_id, attempt, candidate_code_dir, chain_code, verified_ir, diagnosis, "patch_generation"))
            repair_context = build_generation_repair_context(attempt, phase="patch_generation", error=exc, diagnosis=diagnosis)
            if non_repairable_llm_generation_error(exc):
                break
            continue
        patch_result["repair_attempt"] = attempt
        patch_result["stage"] = stage
        if strategy.get("ir_updates"):
            merged_updates = clean_llm_patch_ir_updates(
                patch_result.get("ir_updates") or {},
                protected_paths=set(strategy.get("ir_updates", {}).keys()),
            )
            merged_updates.update(copy.deepcopy(strategy["ir_updates"]))
            patch_result["ir_updates"] = merged_updates
        save_candidate_json(DEFAULT_PATCH_OUTPUT, stage, strategy_id, attempt, patch_result, candidate_namespace)

        patch_ir = build_patch_ir(
            ir=base_ir,
            strategy=strategy,
            precheck_item=precheck_item,
            patch_result=patch_result,
        )
        patch_ir["repair_attempt"] = attempt
        patch_ir["candidate_stage"] = stage
        save_candidate_json(DEFAULT_PATCH_IR_OUTPUT, stage, strategy_id, attempt, patch_ir, candidate_namespace)

        post_check_result = deferred_micro_post_check(stage, strategy_id, patch_ir)
        post_check_result["repair_attempt"] = attempt
        post_check_result["stage"] = stage
        post_check_result["strategy_id"] = strategy_id
        save_candidate_json(DEFAULT_POSTCHECK_OUTPUT, stage, strategy_id, attempt, post_check_result, candidate_namespace)

        patch_file = candidate_path(DEFAULT_PATCH_OUTPUT, stage, strategy_id, attempt, candidate_namespace)
        patch_from_file = load_json(patch_file)
        try:
            materialization_code_context = load_strategy_scoped_code_context(
                base_ir,
                candidate_code_dir,
                context_files,
                strategy,
                patch_from_file,
            )
            if target_backend(base_ir) == "cpu":
                generated_code = generate_cpu_c_code_from_patch_with_llm(
                    client=client,
                    patch_ir=patch_ir,
                    patch_result=patch_from_file,
                    strategy=strategy,
                    code_context=materialization_code_context,
                    prompt_path=backend_patch_to_code_prompt(base_ir),
                    repair_context=repair_context,
                    patch_file=patch_file,
                )
            else:
                generated_code = generate_code_files_from_patch_with_llm(
                    client=client,
                    patch_ir=patch_ir,
                    patch_result=patch_from_file,
                    strategy=strategy,
                    code_context=materialization_code_context,
                    prompt_path=backend_patch_to_code_prompt(base_ir),
                    repair_context=repair_context,
                    patch_file=patch_file,
                )
        except Exception as exc:
            verified_ir = mark_llm_generation_failed(
                patch_ir,
                stage=stage,
                strategy_id=strategy_id,
                phase="code_generation",
                error=exc,
                candidate_code_dir=candidate_code_dir,
            )
            diagnosis = diagnose_defects(verified_ir, post_check_result)
            verified_ir = attach_diagnosis(verified_ir, diagnosis)
            save_candidate_json(DEFAULT_DIAGNOSIS_OUTPUT, stage, strategy_id, attempt, diagnosis, candidate_namespace)
            save_candidate_json(DEFAULT_VERIFIED_IR_OUTPUT, stage, strategy_id, attempt, verified_ir, candidate_namespace)
            attempts.append(failed_generation_attempt_record(stage, strategy_id, attempt, candidate_code_dir, chain_code, verified_ir, diagnosis, "code_generation"))
            repair_context = build_generation_repair_context(attempt, phase="code_generation", error=exc, diagnosis=diagnosis)
            if non_repairable_llm_generation_error(exc):
                break
            continue
        generated_code["repair_attempt"] = attempt
        generated_code["stage"] = stage
        generated_code["candidate_code_dir"] = str(candidate_code_dir)
        save_candidate_json(DEFAULT_PATCHED_CODE_OUTPUT, stage, strategy_id, attempt, generated_code, candidate_namespace)

        if target_backend(base_ir) == "cpu":
            code_apply_result = apply_cpu_c_code_files(generated_code, candidate_code_dir)
        else:
            code_apply_result = apply_generated_code_files(
                generated_code,
                candidate_code_dir,
                patch_ir=patch_ir,
                strategy=strategy,
            )
        code_apply_result["candidate_code_dir"] = str(candidate_code_dir)
        patch_ir["patch_to_code_application"] = code_apply_result
        if code_apply_result["status"] != "pass":
            verified_ir = mark_patch_apply_failed(patch_ir, code_apply_result)
        else:
            verified_ir = mark_chain_step_generated(patch_ir, code_apply_result)
            syntax_report = check_generated_source_syntax(candidate_code_dir, backend_ast_files(verified_ir))
            attach_source_syntax_report(verified_ir, syntax_report, candidate_code_dir)
            if compile_each_step and not source_syntax_gate_failed(verified_ir):
                verified_ir = verify_strategy_step_compile_gate(
                    verified_ir,
                    candidate_code_dir,
                    build_platform,
                    step_compile_timeout_seconds,
                )
            if source_syntax_gate_failed(verified_ir):
                pass
            elif compile_each_step and not step_compile_gate_passed(verified_ir):
                pass
            elif not defer_semantic_checks:
                code_ast = (
                    extract_code_ast(candidate_code_dir, backend_ast_files(base_ir))
                    if target_backend(base_ir) != "cpu"
                    else extract_cpu_code_ast(candidate_code_dir)
                )
                save_candidate_json(DEFAULT_CODE_AST_OUTPUT, stage, strategy_id, attempt, code_ast, candidate_namespace)
                verified_ir["code_ast"] = code_ast
                # Phase 1 treats source-level semantic failures as local blockers.
                # Phase 2 uses them as advisory defects and still runs compile/runtime
                # oracles before deciding accept/repair/rollback.
                if target_backend(base_ir) != "cpu":
                    semantic_report = check_gemm_semantic_obligations(candidate_code_dir, verified_ir)
                    verified_ir["code_completeness"] = {
                        "accepted": semantic_report.get("accepted") is not False,
                        "results": [],
                        "semantic_obligations": semantic_report,
                    }
                    if semantic_report.get("accepted") is False and semantic_precheck_terminal:
                        verified_ir["chain_step"] = {
                            "status": "failed",
                            "phase": "semantic_precheck",
                            "candidate_code_dir": str(candidate_code_dir),
                            "reason": "generated CUDA violates source-level semantic obligations",
                        }
                        verified_ir.setdefault("verification", {})["accepted"] = False
                        summary = verified_ir["verification"].setdefault("summary", {})
                        summary.update(
                            {
                                "chain_step_status": "failed",
                                "failure_phase": "semantic_precheck",
                            }
                        )
                    elif semantic_report.get("accepted") is False:
                        summary = verified_ir.setdefault("verification", {}).setdefault("summary", {})
                        summary["code_checker_advisory_failure"] = True
                        summary["checker_failure_is_terminal"] = False
                        summary["semantic_precheck_status"] = "advisory_fail"

        if post_check_result.get("deferred_until_stage_verification") and candidate_is_accepted(verified_ir):
            diagnosis = {
                "stage": stage,
                "status": "deferred",
                "related_strategy": strategy_id,
                "defect_count": 0,
                "defects": [],
                "message": "No micro-step defect diagnosis; stage-level verification is deferred.",
            }
        else:
            diagnosis = diagnose_defects(verified_ir, post_check_result)
        diagnosis["repair_attempt"] = attempt
        diagnosis["stage"] = stage
        verified_ir = attach_diagnosis(verified_ir, diagnosis)
        save_candidate_json(DEFAULT_DIAGNOSIS_OUTPUT, stage, strategy_id, attempt, diagnosis, candidate_namespace)
        save_candidate_json(DEFAULT_VERIFIED_IR_OUTPUT, stage, strategy_id, attempt, verified_ir, candidate_namespace)

        attempts.append(
            {
                "attempt": attempt,
                "patch_output": str(candidate_path(DEFAULT_PATCH_OUTPUT, stage, strategy_id, attempt, candidate_namespace)),
                "postcheck_output": str(candidate_path(DEFAULT_POSTCHECK_OUTPUT, stage, strategy_id, attempt, candidate_namespace)),
                "diagnosis_output": str(candidate_path(DEFAULT_DIAGNOSIS_OUTPUT, stage, strategy_id, attempt, candidate_namespace)),
                "patched_code_output": str(candidate_path(DEFAULT_PATCHED_CODE_OUTPUT, stage, strategy_id, attempt, candidate_namespace)),
                "code_ast_output": str(candidate_path(DEFAULT_CODE_AST_OUTPUT, stage, strategy_id, attempt, candidate_namespace)),
                "candidate_code_dir": str(candidate_code_dir),
                "path_code": chain_code or [],
                "verification": verified_ir.get("verification", {}).get("summary", {}),
                "defect_count": diagnosis.get("defect_count"),
                "step_compile_gate": verified_ir.get("verification", {}).get("summary", {}).get("step_compile_gate"),
                "compile_status": verified_ir.get("verification", {}).get("compile", {}).get("status"),
            }
        )

        if candidate_is_accepted(verified_ir):
            source_snapshot = snapshot_source_files(candidate_code_dir, code_files)
            archive_generated_chain_step(candidate_code_dir, stage, strategy_id, attempt, verified_ir)
            return {
                "accepted": True,
                "stage": stage,
                "strategy_id": strategy_id,
                "verified_ir": verified_ir,
                "source_snapshot": source_snapshot,
                "candidate_code_dir": str(candidate_code_dir),
                "path_code": chain_code or [],
                "attempts": attempts,
                "event": candidate_event(stage, strategy_id, "accepted", verified_ir, attempts),
            }

        repair_context = build_repair_context(attempt, patch_result, post_check_result, verified_ir, diagnosis)

    return {
        "accepted": False,
        "stage": stage,
        "strategy_id": strategy_id,
        "verified_ir": verified_ir,
        "source_snapshot": base_source_snapshot,
        "candidate_code_dir": None,
        "path_code": chain_code or [],
        "attempts": attempts,
        "event": candidate_event(stage, strategy_id, "failed", verified_ir, attempts),
    }


def non_repairable_llm_generation_error(error: Exception) -> bool:
    error_type = type(error).__name__
    from SGPO.generate_ir.unlock_feedback import INFRASTRUCTURE_ERRORS
    if error_type in INFRASTRUCTURE_ERRORS:
        return True
    error_message = str(error).lower()
    if error_type in {
        "APIConnectionError",
        "APITimeoutError",
        "APIError",
        "ConnectError",
        "ReadTimeout",
        "TimeoutException",
        "JSONDecodeError",
    }:
        return True
    return any(
        token in error_message
        for token in [
            "connection error",
            "timed out",
            "timeout",
            "unterminated string",
            "does not contain a json object",
            "eof occurred in violation of protocol",
        ]
    )


def check_generated_source_syntax(code_dir: Path, code_files: list[str]) -> dict[str, Any]:
    results = []
    for relative_path in code_files:
        path = code_dir / relative_path
        if not path.exists():
            results.append(
                {
                    "id": f"SOURCE_EXISTS:{relative_path}",
                    "status": "fail",
                    "message": f"{relative_path} does not exist",
                }
            )
            continue
        content = path.read_text(encoding="utf-8")
        results.extend(check_balanced_source_delimiters(relative_path, content))
        results.extend(check_patch_anchor_pairs(relative_path, content))
        if relative_path.endswith((".cu", ".cuh", ".cpp", ".c")):
            results.extend(check_duplicate_launch_constants(relative_path, content))
    accepted = all(item["status"] != "fail" for item in results)
    return {
        "checker": "sgpo_lightweight_source_syntax",
        "accepted": accepted,
        "results": results,
    }


def check_balanced_source_delimiters(relative_path: str, content: str) -> list[dict[str, Any]]:
    code = strip_strings_for_light_syntax(strip_comments_for_light_syntax(content))
    pairs = [("{", "}"), ("(", ")"), ("[", "]")]
    results = []
    for left, right in pairs:
        left_count = code.count(left)
        right_count = code.count(right)
        results.append(
            {
                "id": f"BALANCED_{left}{right}:{relative_path}",
                "status": "pass" if left_count == right_count else "fail",
                "message": f"{left_count} '{left}' and {right_count} '{right}' delimiters",
            }
        )
    return results


def check_patch_anchor_pairs(relative_path: str, content: str) -> list[dict[str, Any]]:
    begin_pattern = re.compile(r"SGPO_PATCH_([A-Z0-9_]+)_BEGIN")
    end_pattern = re.compile(r"SGPO_PATCH_([A-Z0-9_]+)_END")
    begin_counts: dict[str, int] = {}
    end_counts: dict[str, int] = {}
    for name in begin_pattern.findall(content):
        begin_counts[name] = begin_counts.get(name, 0) + 1
    for name in end_pattern.findall(content):
        end_counts[name] = end_counts.get(name, 0) + 1
    generic_begin = re.compile(
        r"\b(LAUNCH_CONFIG|SHARED_DECL|INDEX_MAPPING|REGISTER_DECL|GLOBAL_TO_SHARED_LOAD|"
        r"NEXT_TILE_LOAD|SYNC_AFTER_LOAD|MAIN_LOOP|COMPUTE_INNER|STORE)_BEGIN\b"
    )
    generic_end = re.compile(
        r"\b(LAUNCH_CONFIG|SHARED_DECL|INDEX_MAPPING|REGISTER_DECL|GLOBAL_TO_SHARED_LOAD|"
        r"NEXT_TILE_LOAD|SYNC_AFTER_LOAD|MAIN_LOOP|COMPUTE_INNER|STORE)_END\b"
    )
    for name in generic_begin.findall(content):
        begin_counts[name] = begin_counts.get(name, 0) + 1
    for name in generic_end.findall(content):
        end_counts[name] = end_counts.get(name, 0) + 1
    names = sorted(set(begin_counts) | set(end_counts))
    return [
        {
            "id": f"PATCH_ANCHOR_PAIR:{relative_path}:{name}",
            "status": "pass" if begin_counts.get(name, 0) == 1 and end_counts.get(name, 0) == 1 else "fail",
            "message": f"{name}: begin={begin_counts.get(name, 0)}, end={end_counts.get(name, 0)}",
        }
        for name in names
    ]


def check_duplicate_launch_constants(relative_path: str, content: str) -> list[dict[str, Any]]:
    results = []
    for name in ("BM", "BN", "BK", "WM", "WN", "WMITER", "WNITER", "TM", "TN"):
        count = len(re.findall(rf"\bstatic\s+const\s+int\s+{name}\s*=", content))
        results.append(
            {
                "id": f"DUPLICATE_LAUNCH_CONST:{relative_path}:{name}",
                "status": "pass" if count <= 1 else "fail",
                "message": f"{name} static const definition count = {count}",
            }
        )
    return results


def attach_source_syntax_report(verified_ir: dict[str, Any], syntax_report: dict[str, Any], candidate_code_dir: Path) -> None:
    verified_ir["source_syntax"] = syntax_report
    summary = verified_ir.setdefault("verification", {}).setdefault("summary", {})
    summary["source_syntax_check"] = "pass" if syntax_report.get("accepted") else "fail"
    if syntax_report.get("accepted"):
        return
    verified_ir["chain_step"] = {
        "status": "failed",
        "phase": "source_syntax_check",
        "candidate_code_dir": str(candidate_code_dir),
        "reason": "generated source failed lightweight syntax checks",
    }
    verified_ir.setdefault("verification", {})["accepted"] = False
    summary.update(
        {
            "chain_step_status": "failed",
            "failure_phase": "source_syntax_check",
        }
    )


def source_syntax_gate_failed(ir: dict[str, Any]) -> bool:
    return ir.get("source_syntax", {}).get("accepted") is False


def strip_comments_for_light_syntax(content: str) -> str:
    without_block = re.sub(r"/\*.*?\*/", lambda match: "\n" * match.group(0).count("\n"), content, flags=re.DOTALL)
    return re.sub(r"//.*", "", without_block)


def strip_strings_for_light_syntax(content: str) -> str:
    return re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', '""', content)


def subphase_can_continue_after_candidate_failures(subphase: dict[str, Any] | None) -> bool:
    if not subphase:
        return False
    subphase_id = subphase.get("subphase_id") or subphase.get("subphase") or ""
    if not subphase_id or subphase_id.endswith(".StageVerification"):
        return False
    if subphase.get("optional") is True:
        return True
    return subphase_id not in REQUIRED_CONSTRUCTION_SUBPHASES


def select_all_filtered_candidates(
    filtered_strategy_index: dict[str, Any],
) -> dict[str, Any]:
    candidates = []
    seen = set()
    for strategy in filtered_strategy_index.get("strategies", []) or []:
        strategy_id = strategy.get("strategy_id")
        if not strategy_id or strategy_id in seen:
            continue
        candidates.append(
            {
                "strategy_id": strategy_id,
                "reason": "Added for exhaustive stage candidate code generation after graph filtering.",
                "confidence": 0.0,
            }
        )
        seen.add(strategy_id)

    return {
        "candidates": candidates,
        "selection_mode": DEFAULT_SELECTION_MODE,
        "llm_used_for_strategy_selection": False,
    }


def build_fallback_micro_selection(
    available_strategy_index: dict[str, Any],
    subphase_id: str | None,
    reason: str,
) -> dict[str, Any]:
    candidates = []
    for index, strategy in enumerate(available_strategy_index.get("strategies", []) or []):
        strategy_id = strategy.get("strategy_id")
        if not strategy_id:
            continue
        candidates.append(
            {
                "strategy_id": strategy_id,
                "reason": reason,
                "confidence": fallback_strategy_confidence(strategy, index),
                "fallback_selected": True,
            }
        )
    return {
        "selected_strategy_id": candidates[0]["strategy_id"] if candidates else None,
        "reason": reason,
        "expected_ir_updates": {},
        "candidates": candidates,
        "current_subphase": subphase_id,
        "selection_mode": "programmatic_fallback_after_llm_failure",
        "llm_used_for_strategy_selection": False,
        "llm_error": reason,
    }


def all_available_strategies_are_deterministic(strategy_index: dict[str, Any]) -> bool:
    strategies = strategy_index.get("strategies", []) or []
    return bool(strategies) and all(
        deterministic_materialization_mode(strategy) == DETERMINISTIC_MATERIALIZATION_MODE
        for strategy in strategies
    )


def fallback_strategy_confidence(strategy: dict[str, Any], original_index: int) -> float:
    try:
        priority = float(strategy.get("priority", 0.0))
    except (TypeError, ValueError):
        priority = 0.0
    return priority / 1000.0 - original_index * 0.0001


def inject_profile_preferred_candidates(
    micro_selection: dict[str, Any],
    available_strategy_index: dict[str, Any],
    subphase_id: str | None,
    profile: str | None,
) -> dict[str, Any]:
    if profile != DEFAULT_UNLOCKED_PERFORMANCE_PROFILE:
        return micro_selection
    preferred = THROUGHPUT_PROFILE_PREFERRED_IDS.get(subphase_id or "", [])
    if not preferred:
        return micro_selection

    available_by_id = {
        item.get("strategy_id"): item
        for item in available_strategy_index.get("strategies", []) or []
        if item.get("strategy_id")
    }
    candidates = []
    seen = set()
    for rank, strategy_id in enumerate(preferred, start=1):
        if strategy_id not in available_by_id or strategy_id in seen:
            continue
        candidates.append(
            {
                "strategy_id": strategy_id,
                "reason": "Preferred by throughput_exploration profile after correctness passed.",
                "confidence": 1.0 - rank * 0.01,
                "profile_preferred": True,
            }
        )
        seen.add(strategy_id)
    for item in micro_selection.get("candidates", []) or []:
        strategy_id = item.get("strategy_id")
        if not strategy_id or strategy_id in seen:
            continue
        candidates.append(copy.deepcopy(item))
        seen.add(strategy_id)

    result = copy.deepcopy(micro_selection)
    result["candidates"] = candidates
    result["profile_injected_candidates"] = [item["strategy_id"] for item in candidates if item.get("profile_preferred")]
    result["profile"] = profile
    return result


def select_single_path_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not candidates:
        return []
    return [copy.deepcopy(candidates[0])]


def build_joint_tiling_micro_selection(
    client: OpenAICompatibleClient,
    current_ir: dict[str, Any],
    strategy_library: dict[str, Any],
    available_strategy_index: dict[str, Any],
    top_k: int,
    resource_pool_size: int = DEFAULT_TILING_RESOURCE_POOL_SIZE,
) -> dict[str, Any]:
    allowed_blocks = {
        item.get("strategy_id")
        for item in available_strategy_index.get("strategies", []) or []
        if isinstance(item, dict)
    }
    all_legal_candidates = [
        item
        for item in build_joint_tiling_candidates(strategy_library, current_ir)
        if item["block_strategy_id"] in allowed_blocks
    ]
    legal_candidates = make_diverse_tiling_pool(all_legal_candidates, resource_pool_size)
    selected_ids = []
    llm_reason = ""
    llm_error = None
    try:
        response = get_joint_tiling_selection_from_llm(
            client=client,
            optir_summary=compact_joint_tiling_ir_summary(current_ir),
            tiling_candidates=legal_candidates,
            top_k=top_k,
        )
        selected_ids = response.get("candidate_ids", [])
        llm_reason = response.get("reason", "")
    except Exception as exc:
        llm_error = f"{type(exc).__name__}: {exc}"

    if llm_error is not None:
        raise RuntimeError(
            "Joint Tiling selection requires an LLM response; "
            f"selection failed with {llm_error}."
        )
    selected_by_id = {item["candidate_id"]: item for item in legal_candidates}
    selected_plans = [selected_by_id[item] for item in selected_ids if item in selected_by_id]
    if not selected_plans:
        raise RuntimeError("Joint Tiling selection returned no legal Block/Warp/Thread combination.")
    candidates = []
    for rank, plan in enumerate(selected_plans):
        candidates.append(
            {
                "strategy_id": plan["block_strategy_id"],
                "reason": llm_reason or "LLM-selected legal Block/Warp/Thread combination.",
                "confidence": 1.0 - rank * 0.01,
                "tiling_plan": tiling_plan_for_ir(plan),
            }
        )
    return {
        "selected_strategy_id": candidates[0]["strategy_id"] if candidates else None,
        "reason": llm_reason,
        "expected_ir_updates": {},
        "candidates": candidates,
        "selection_mode": "llm_joint_tiling_tuple_selection",
        "legal_tuple_count": len(all_legal_candidates),
        "resource_pool_count": len(legal_candidates),
        "resource_pool_size": resource_pool_size,
        "prompt_option_count": {
            "block": len({item["block_strategy_id"] for item in legal_candidates}),
            "warp": len({item["warp_strategy_id"] for item in legal_candidates}),
            "thread": len({item["thread_strategy_id"] for item in legal_candidates}),
        },
        "llm_error": llm_error,
    }


def planned_tiling_micro_selection(
    current_ir: dict[str, Any],
    current_subphase_id: str | None,
    available_strategy_index: dict[str, Any],
) -> dict[str, Any] | None:
    plan = (current_ir.get("strategy") or {}).get("tiling_joint_plan")
    plan_key = {
        "Tiling.WarpTileSelection": "warp_strategy_id",
        "Tiling.ThreadTileSelection": "thread_strategy_id",
    }.get(current_subphase_id)
    if not isinstance(plan, dict) or plan_key is None:
        return None
    strategy_id = plan.get(plan_key)
    allowed_ids = {
        item.get("strategy_id")
        for item in available_strategy_index.get("strategies", []) or []
        if isinstance(item, dict)
    }
    if strategy_id not in allowed_ids:
        return None
    return {
        "selected_strategy_id": strategy_id,
        "reason": f"Execute the {current_subphase_id} component of the previously selected complete tiling tuple.",
        "expected_ir_updates": {},
        "candidates": [
            {
                "strategy_id": strategy_id,
                "reason": "Planned complete tiling tuple component.",
                "confidence": 1.0,
            }
        ],
        "selection_mode": "planned_joint_tiling",
    }


def tiling_plan_ir_updates(strategy_id: str, plan: dict[str, Any]) -> dict[str, Any]:
    if strategy_id == plan.get("block_strategy_id"):
        return {
            "tiling.enabled": True,
            "tiling.block_m": plan.get("BM"),
            "tiling.block_n": plan.get("BN"),
            "tiling.block_k": plan.get("BK"),
        }
    if strategy_id == plan.get("warp_strategy_id"):
        return {
            "tiling.warp_tile.warp_m": plan.get("WM"),
            "tiling.warp_tile.warp_n": plan.get("WN"),
        }
    if strategy_id == plan.get("thread_strategy_id"):
        return {
            "tiling.thread_m": plan.get("TM"),
            "tiling.thread_n": plan.get("TN"),
            "tiling.warp_tile.warp_m_iter": plan.get("WMITER"),
            "tiling.warp_tile.warp_n_iter": plan.get("WNITER"),
            "resource.tiling_candidate.architecture_family": plan.get("architecture_family"),
            "resource.tiling_candidate.resident_ctas_per_sm": plan.get("resident_ctas_per_sm"),
            "resource.tiling_candidate.active_warps_per_sm": plan.get("active_warps_per_sm"),
            "resource.tiling_candidate.estimated_occupancy": plan.get("estimated_occupancy"),
            "resource.tiling_candidate.cta_count": plan.get("cta_count"),
            "resource.tiling_candidate.cta_waves": plan.get("cta_waves"),
            "resource.tiling_candidate.sm_coverage": plan.get("sm_coverage"),
            "resource.tiling_candidate.last_wave_utilization": plan.get("last_wave_utilization"),
            "resource.tiling_candidate.architecture_score": plan.get("architecture_score"),
        }
    return {}


def select_top_k_candidates(candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if not candidates or top_k <= 0:
        return []
    indexed = [(index, item) for index, item in enumerate(candidates)]
    indexed.sort(key=lambda pair: candidate_selection_score(pair[1], pair[0]), reverse=True)
    return [copy.deepcopy(item) for _, item in indexed[:top_k]]


def frontier_limit_for_stage(
    stage: str,
    backend: str,
    normal_limit: int,
    core_limit: int,
    first_performance_prune_stage: str,
) -> int:
    if backend != "cuda":
        return normal_limit
    core_stages = ["Tiling", first_performance_prune_stage]
    return max(normal_limit, core_limit) if stage in core_stages else normal_limit


def prune_frontier_after_core_construction(
    frontier: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    if top_k <= 0 or len(frontier) <= top_k:
        return list(frontier)
    return sorted(frontier, key=frontier_state_score, reverse=True)[:top_k]


def tiling_lineage_key(state: dict[str, Any]) -> tuple[str, ...] | None:
    path = state.get("path", []) or []
    block_strategy_id = next(
        (item for item in path if isinstance(item, str) and item.startswith("Tiling.BlockTile.")),
        None,
    )
    if block_strategy_id is None:
        return None
    # The first BlockTile choices are the independent search roots (chain.1/2/3).
    # WarpTile and ThreadTile variants compete only within their own BlockTile root.
    return (block_strategy_id,)


def tiling_lineage_keys(states: list[dict[str, Any]]) -> set[tuple[str, ...]]:
    return {key for state in states if (key := tiling_lineage_key(state)) is not None}


def prune_frontier_per_tiling_lineage(
    frontier: list[dict[str, Any]],
    top_k_per_lineage: int,
) -> list[dict[str, Any]]:
    if top_k_per_lineage <= 0:
        return list(frontier)
    grouped: dict[tuple[str, ...] | None, list[dict[str, Any]]] = {}
    for state in frontier:
        grouped.setdefault(tiling_lineage_key(state), []).append(state)
    kept = []
    for group in grouped.values():
        kept.extend(sorted(group, key=frontier_state_score, reverse=True)[:top_k_per_lineage])
    return sorted(kept, key=frontier_state_score, reverse=True)


def take_frontier_batch_per_tiling_lineage(
    states: list[dict[str, Any]],
    top_k_per_lineage: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not states or top_k_per_lineage <= 0:
        return list(states), []
    selected = prune_frontier_per_tiling_lineage(states, top_k_per_lineage)
    selected_ids = {id(state) for state in selected}
    return selected, [state for state in states if id(state) not in selected_ids]


def take_frontier_batch(
    states: list[dict[str, Any]],
    max_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not states:
        return [], []
    if max_count <= 0 or len(states) <= max_count:
        return list(states), []
    ordered = diversify_frontier_states(states)
    return ordered[:max_count], ordered[max_count:]


def diversify_frontier_states(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    group_order: list[tuple[int, ...]] = []
    for state in states:
        code = tuple(state.get("path_code", []) or [])
        parent_code = code[:-1]
        if parent_code not in groups:
            groups[parent_code] = []
            group_order.append(parent_code)
        groups[parent_code].append(state)

    for key, group in groups.items():
        group.sort(key=lambda item: frontier_state_score(item), reverse=True)

    original_group_order = {key: index for index, key in enumerate(group_order)}
    group_order.sort(
        key=lambda key: (
            max(frontier_state_score(item) for item in groups[key]),
            -original_group_order[key],
        ),
        reverse=True,
    )
    ordered = []
    while any(groups.values()):
        for key in group_order:
            if groups[key]:
                ordered.append(groups[key].pop(0))
    return ordered


def candidate_selection_score(candidate: dict[str, Any], original_index: int) -> tuple[float, int]:
    value = candidate.get("confidence", candidate.get("score", candidate.get("rank_score")))
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    return (score, -original_index)


def strategy_index_from_precheck(
    filtered_strategy_index: dict[str, Any],
    precheck_result: dict[str, Any],
) -> dict[str, Any]:
    accepted_ids = {item["strategy_id"] for item in precheck_result.get("accepted", [])}
    result = copy.deepcopy(filtered_strategy_index)
    result["strategies"] = [
        strategy
        for strategy in filtered_strategy_index.get("strategies", []) or []
        if strategy.get("strategy_id") in accepted_ids
    ]
    result["strategy_count"] = len(result["strategies"])
    result["strategy_ids_by_stage"] = build_strategy_ids_by_stage(result["strategies"])
    result["precheck_summary"] = precheck_result.get("summary", {})
    return result


def precheck_result_from_available(strategy_index: dict[str, Any]) -> dict[str, Any]:
    accepted = [
        {
            "strategy_id": strategy.get("strategy_id"),
            "category": strategy.get("category"),
            "stage": strategy.get("stage"),
            "name": strategy.get("name"),
            "preconditions_ok": True,
            "hard_constraints_ok": None,
            "strategy_applicable": True,
            "failed_checks": [],
            "checker_report": {
                "phase": "micro_strategy_local_candidate_filter",
                "current_subphase": strategy_index.get("filter_context", {}).get("current_subphase"),
                "strategy_applicable": True,
            },
        }
        for strategy in strategy_index.get("strategies", []) or []
    ]
    return {
        "accepted": accepted,
        "rejected": strategy_index.get("filter_context", {}).get("rejected", []),
        "unknown_strategy_ids": [],
        "summary": {
            "candidate_count": len(accepted) + len(strategy_index.get("filter_context", {}).get("rejected", [])),
            "accepted_count": len(accepted),
            "rejected_count": len(strategy_index.get("filter_context", {}).get("rejected", [])),
            "unknown_count": 0,
        },
    }


def deferred_micro_post_check(stage: str, strategy_id: str, patch_ir: dict[str, Any]) -> dict[str, Any]:
    subphase = patch_ir.get("strategy", {}).get("current_subphase")
    return {
        "phase": "after_micro_patch_generation",
        "stage": stage,
        "subphase": subphase,
        "strategy_id": strategy_id,
        "accepted_by_ir_checker": True,
        "postconditions_ok": None,
        "hard_constraints_checked": False,
        "hard_constraints_ok": None,
        "deferred_until_stage_verification": True,
        "message": "Micro-strategy postconditions are checked at stage completion, not after each local patch.",
        "results": [],
    }


def stage_level_post_check(stage: str, subphase: str, stage_report: dict[str, Any]) -> dict[str, Any]:
    return {
        "phase": "stage_completion_post_check",
        "stage": stage,
        "subphase": subphase,
        "accepted_by_ir_checker": bool(stage_report.get("accepted")),
        "postconditions_ok": bool(stage_report.get("accepted")),
        "hard_constraints_checked": False,
        "hard_constraints_ok": None,
        "deferred_until_stage_verification": False,
        "results": stage_report.get("512-cuda-result/results", []),
        "stage_verification": stage_report,
    }


def derivation_only_subphase(subphase: dict[str, Any] | None) -> bool:
    if not subphase:
        return False
    allowed_ids = subphase.get("allowed_strategy_ids") or []
    allowed_patterns = subphase.get("allowed_strategy_patterns") or []
    return not allowed_ids and not allowed_patterns and bool(subphase.get("provides_fields") or subphase.get("derived_fields"))


def clean_expected_ir_updates(updates: dict[str, Any]) -> dict[str, Any]:
    placeholders = {"int", "float", "bool", "boolean", "string", "number", "null", "none"}
    return {
        path: value
        for path, value in updates.items()
        if not (isinstance(value, str) and value.strip().lower() in placeholders)
    }


def merge_expected_ir_updates(
    strategy_id: str,
    deterministic_updates: dict[str, Any],
    llm_expected_updates: dict[str, Any],
) -> dict[str, Any]:
    cleaned_llm_updates = clean_expected_ir_updates(llm_expected_updates or {})
    authoritative_updates = copy.deepcopy(deterministic_updates or {})
    authoritative_updates.update(synthesize_ir_updates(strategy_id))
    merged = copy.deepcopy(cleaned_llm_updates)
    for path in list(merged):
        if "." not in path:
            merged.pop(path, None)
    merged.update(authoritative_updates)
    return merged


def clean_llm_patch_ir_updates(updates: dict[str, Any], protected_paths: set[str]) -> dict[str, Any]:
    placeholders = {"int", "float", "bool", "boolean", "string", "number", "null", "none"}
    cleaned = {}
    for path, value in updates.items():
        if path in protected_paths and value is None:
            continue
        if path in protected_paths and isinstance(value, str) and value.strip().lower() in placeholders:
            continue
        cleaned[path] = value
    return cleaned


def compact_ir_summary(ir: dict[str, Any]) -> dict[str, Any]:
    keep = ["problem", "hardware", "tiling", "mapping", "memory", "vectorization", "resource", "strategy"]
    return {key: ir.get(key) for key in keep if key in ir}


def compact_joint_tiling_ir_summary(ir: dict[str, Any]) -> dict[str, Any]:
    hardware = ir.get("hardware", {}) or {}
    gpu = hardware.get("gpu", {}) or {}
    execution_profile = build_gpu_architecture_profile(hardware)
    return {
        "problem": copy.deepcopy(ir.get("problem", {}) or {}),
        "hardware": {
            "warp_size": hardware.get("warp_size", gpu.get("warp_size")),
            "max_threads_per_block": hardware.get("max_threads_per_block", gpu.get("max_threads_per_block")),
            "max_shared_memory_per_block_bytes": hardware.get(
                "max_shared_memory_per_block_bytes",
                gpu.get("max_shared_memory_per_block_bytes"),
            ),
            "sm_count": hardware.get("sm_count", gpu.get("sm_count")),
            "compute_capability": hardware.get("compute_capability", gpu.get("compute_capability")),
            "execution_profile": execution_profile,
        },
    }


def compact_hierarchical_tiling_ir_summary(ir: dict[str, Any]) -> dict[str, Any]:
    summary = compact_joint_tiling_ir_summary(ir)
    summary["tiling"] = copy.deepcopy(ir.get("tiling", {}) or {})
    summary["mapping"] = {
        key: copy.deepcopy((ir.get("mapping", {}) or {}).get(key))
        for key in ("warps_m", "warps_n", "warps_per_block", "threads_per_block")
        if (ir.get("mapping", {}) or {}).get(key) is not None
    }
    return summary


def save_chain_stage_available_strategies(
    chain_path: list[str],
    chain_code: list[int],
    stage: str,
    filtered_strategy_index: dict[str, Any],
    precheck_result: dict[str, Any],
    available_strategy_index: dict[str, Any],
) -> str:
    chain_key = chain_key_for_path(chain_path, chain_code)
    output = Path(DEFAULT_CHAIN_DIR) / f"{safe_name(chain_key)}.{safe_name(stage)}.available.json"
    record = {
        "chain_key": chain_key,
        "chain_code": chain_code_key(chain_code),
        "chain_path": chain_path,
        "stage": stage,
        "filtered_strategy_count": filtered_strategy_index.get("strategy_count", 0),
        "available_strategy_count": available_strategy_index.get("strategy_count", 0),
        "available_strategy_ids": [
            strategy.get("strategy_id") for strategy in available_strategy_index.get("strategies", []) or []
        ],
        "rejected_strategy_ids": [
            item.get("strategy_id") for item in precheck_result.get("rejected", []) or []
        ],
        "precheck_summary": precheck_result.get("summary", {}),
        "filter_context": filtered_strategy_index.get("filter_context", {}),
        "llm_selected_strategy_ids": [],
    }
    save_json(output, record)
    if not output.exists():
        raise FileNotFoundError(f"Failed to create chain stage record: {output}")
    return str(output)


def update_chain_stage_selected_strategies(chain_record_path: str, selected_strategy: dict[str, Any]) -> None:
    path = Path(chain_record_path)
    if path.exists():
        record = load_json(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "chain_record_missing_before_update": True,
            "chain_record_path": str(path),
            "llm_selected_strategy_ids": [],
        }
    record["llm_selected_strategy_ids"] = [
        item.get("strategy_id") for item in selected_strategy.get("candidates", []) or []
    ]
    record["llm_selected_candidates"] = selected_strategy.get("candidates", [])
    record["selection_mode"] = selected_strategy.get("selection_mode")
    save_json(path, record)


def build_strategy_ids_by_stage(strategies: list[dict[str, Any]]) -> dict[str, list[str]]:
    by_stage: dict[str, list[str]] = {}
    for strategy in strategies:
        by_stage.setdefault(strategy.get("stage"), []).append(strategy["strategy_id"])
    return by_stage


def build_repair_context(
    attempt: int,
    patch_result: dict[str, Any],
    post_check_result: dict[str, Any],
    verified_ir: dict[str, Any],
    diagnosis: dict[str, Any],
) -> dict[str, Any]:
    return {
        "repair_attempt": attempt + 1,
        "previous_patch": patch_result,
        "post_check_result": post_check_result,
        "verification": verified_ir.get("verification", {}),
        "defect_diagnosis": diagnosis,
        "code_verification": verified_ir.get("code_verification"),
        "code_ast": verified_ir.get("code_ast"),
        "repair_instruction": (
            "Generate a repaired local patch for the same strategy. "
            "Address every defect_diagnosis item. Do not repeat a patch "
            "that violates the same post-check or verification failure."
        ),
    }


def candidate_is_accepted(verified_ir: dict[str, Any]) -> bool:
    chain_step = verified_ir.get("chain_step", {})
    if chain_step.get("status") == "generated":
        summary = verified_ir.get("verification", {}).get("summary", {}) or {}
        if summary.get("checker_failure_is_terminal") is False:
            return True
        return verified_ir.get("code_completeness", {}).get("accepted") is not False
    return oracle_verification_passed(verified_ir)


def oracle_verification_passed(verified_ir: dict[str, Any]) -> bool:
    return verified_ir.get("verification", {}).get("accepted") is True


def stage_failure_requires_strategy_fallback(verified_ir: dict[str, Any]) -> bool:
    verification = verified_ir.get("verification", {}) or {}
    runtime = verification.get("runtime_safety", {}) or {}
    compile_result = verification.get("compile", {}) or {}
    text = " ".join(
        str(value or "")
        for value in (
            runtime.get("cuda_error"),
            runtime.get("host_error"),
            compile_result.get("error_message"),
            compile_result.get("stderr"),
        )
    ).lower()
    return any(
        marker in text
        for marker in (
            "too many resources requested for launch",
            "launch out of resources",
        )
    )


def mark_chain_step_generated(ir: dict[str, Any], code_apply_result: dict[str, Any]) -> dict[str, Any]:
    next_ir = copy.deepcopy(ir)
    next_ir["chain_step"] = {
        "status": "generated",
        "verification_deferred": True,
        "reason": "compile/correctness/runtime/performance verification runs at stage completion and after the full strategy chain",
        "candidate_code_dir": code_apply_result.get("candidate_code_dir"),
    }
    verification = next_ir.setdefault("verification", {})
    verification["accepted"] = None
    verification["compile"] = {"status": "deferred"}
    verification["correctness"] = {"status": "deferred"}
    verification["runtime_safety"] = {"status": "deferred", "cuda_error": None}
    verification["summary"] = {
        "chain_step_status": "generated",
        "final_verification": "deferred",
    }
    return next_ir


def mark_llm_generation_failed(
    ir: dict[str, Any],
    stage: str,
    strategy_id: str,
    phase: str,
    error: Exception,
    candidate_code_dir: Path,
) -> dict[str, Any]:
    next_ir = copy.deepcopy(ir)
    message = f"{type(error).__name__}: {error}"
    next_ir.setdefault("strategy", {})["current_stage"] = stage
    next_ir.setdefault("strategy", {})["current_strategy_id"] = strategy_id
    next_ir["chain_step"] = {
        "status": "failed",
        "phase": phase,
        "candidate_code_dir": str(candidate_code_dir),
        "reason": message,
    }
    verification = next_ir.setdefault("verification", {})
    verification["accepted"] = False
    verification["codegen"] = {
        "status": "fail",
        "phase": phase,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    verification["compile"] = {"status": "not_run"}
    verification["correctness"] = {"status": "not_run"}
    verification["runtime_safety"] = {"status": "not_run", "cuda_error": None}
    verification["summary"] = {
        "chain_step_status": "failed",
        "failed_phase": phase,
        "codegen_status": "fail",
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    return next_ir


def build_generation_repair_context(
    attempt: int,
    phase: str,
    error: Exception,
    diagnosis: dict[str, Any],
) -> dict[str, Any]:
    return {
        "repair_attempt": attempt + 1,
        "last_error": f"{type(error).__name__}: {error}",
        "failed_phase": phase,
        "defect_diagnosis": diagnosis,
        "repair_instruction": (
            "Regenerate the same candidate with strict JSON. For codegen edits, "
            "prefer replacement_lines array entries instead of one multiline replacement string. "
            "For CUDA codegen, return only cuda_kernel.cuh anchor-region edits, not full source files."
        ),
    }


def failed_generation_attempt_record(
    stage: str,
    strategy_id: str,
    attempt: int,
    candidate_code_dir: Path,
    chain_code: list[int] | None,
    verified_ir: dict[str, Any],
    diagnosis: dict[str, Any],
    phase: str,
) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "stage": stage,
        "strategy_id": strategy_id,
        "candidate_code_dir": str(candidate_code_dir),
        "path_code": chain_code or [],
        "failed_phase": phase,
        "verification": verified_ir.get("verification", {}).get("summary", {}),
        "defect_count": diagnosis.get("defect_count"),
    }


def choose_best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return max(candidates, key=lambda item: candidate_gflops(item) or -1.0)


def is_better_candidate(candidate: dict[str, Any] | None, current_best: dict[str, Any] | None) -> bool:
    if not candidate:
        return False
    if not current_best:
        return True
    candidate_score = candidate_gflops(candidate)
    best_score = candidate_gflops(current_best)
    if candidate_score is None:
        return best_score is None
    if best_score is None:
        return True
    return candidate_score > best_score


def candidate_gflops(candidate: dict[str, Any] | None) -> float | None:
    if not candidate:
        return None
    performance = candidate.get("verified_ir", {}).get("performance", {}) or {}
    value = performance.get("gflops_trimmed_mean")
    if value is None:
        value = performance.get("gflops_mean")
    if value is None:
        value = performance.get("gflops_median")
    if value is None:
        value = performance.get("gflops")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def candidate_event(
    stage: str,
    strategy_id: str,
    status: str,
    verified_ir: dict[str, Any] | None,
    attempts: list[dict[str, Any]],
) -> dict[str, Any]:
    performance = (verified_ir or {}).get("performance", {}) or {}
    return {
        "stage": stage,
        "strategy_id": strategy_id,
        "status": status,
        "attempt_count": len(attempts),
        "candidate_code_dir": attempts[-1].get("candidate_code_dir") if attempts else None,
        "verification": (verified_ir or {}).get("verification", {}).get("summary", {}),
        "gflops": performance.get("gflops_mean") or performance.get("gflops"),
    }


def repair_chain_locally(source_chain_dir, output_chain_dir, diagnosis=None, ir=None,
                         client=None, strategy_library=None, repair_attempt=1,
                         baseline_snapshot=None):
    """Repair the current failed CUDA source without reselecting strategies."""
    if target_backend(ir or {}) == "cpu":
        return generate_semantic_repair_candidate(source_chain_dir, output_chain_dir, diagnosis, ir)
    strategy_ids = (ir or {}).get("strategy", {}).get("applied_strategy_ids", [])
    strategies = []
    for strategy_id in strategy_ids:
        try:
            loaded = load_strategy(strategy_library or {}, strategy_id)
            strategies.append(
                {
                    "strategy_id": strategy_id,
                    "modifies_regions": loaded.get("modifies_regions", []),
                    "provides_fields": loaded.get("provides_fields", []),
                    "implementation_example_ids": loaded.get("implementation_example_ids", []),
                    "preconditions": loaded.get("preconditions", {}),
                    "postconditions": loaded.get("postconditions", {}),
                    "ir_updates": loaded.get("ir_updates", {}),
                }
            )
        except ValueError:
            strategies.append({"strategy_id": strategy_id})
    kernel_path = Path(source_chain_dir) / "cuda_kernel.cuh"
    kernel_content = kernel_path.read_text(encoding="utf-8")
    repair_level = min(max(int(repair_attempt), 1), 3)
    full_kernel_repair = repair_level >= 3
    suggested_regions = repair_regions_from_diagnosis(kernel_content, diagnosis or {}, repair_level)
    allowed_regions = [name for name in CUDA_REPAIR_REGIONS
                       if f"{name}_BEGIN" in kernel_content and f"{name}_END" in kernel_content]
    from SGPO.verification.locked_repair import locked_tile_parameters, tile_lock_defects, repair_evidence
    locked_parameters = locked_tile_parameters(ir or {}, kernel_content)
    symbol_contract = extract_cuda_symbol_contract(kernel_content)
    strategy = {
        "strategy_id": "Repair.PreserveAppliedStrategies",
        "applied_strategies": strategies,
        "locked_strategy_ids": list(strategy_ids),
        "locked_tile_parameters": locked_parameters,
        "patch_contract": {
            "region_edits_only": not full_kernel_repair,
            "allowed_regions": allowed_regions,
            "preserved_regions": [],
        },
    }
    patch = {"strategy_id": strategy["strategy_id"], "code_patch": [],
             "repair_instruction": "Repair implementation only: compile, correctness and runtime safety must pass and ALL locked strategies must be implemented. Never reselect strategies or change locked parameters or IR. Report concrete constraint omissions separately; failed repair does not establish a strategy conflict."}
    originals = dict(snapshot_source_files(source_chain_dir, backend_code_files(ir or {})))
    evidence = repair_evidence(ir or {}, diagnosis or {}, kernel_content,
                              (baseline_snapshot or originals).get("cuda_kernel.cuh", kernel_content))
    try:
        deterministic = apply_deterministic_compile_repairs(kernel_content, diagnosis or {})
        from SGPO.verification.targeted_cuda_repair import repair_known_cuda_defects
        from SGPO.verification.optimization_preservation import preservation_defects
        deterministic, targeted_changes = repair_known_cuda_defects(deterministic, ir or {})
        if deterministic != kernel_content and (not full_kernel_repair or client is None):
            violations = tile_lock_defects(deterministic, locked_parameters) + preservation_defects(
                kernel_content, deterministic, strategy)
            if violations:
                raise ValueError("; ".join(violations))
            restore_source_files(output_chain_dir, originals)
            (Path(output_chain_dir) / "cuda_kernel.cuh").write_text(deterministic, encoding="utf-8")
            syntax = check_generated_source_syntax(output_chain_dir, backend_ast_files(ir or {}))
            if syntax.get("accepted") is False:
                raise ValueError("Deterministic repair introduced source syntax defects")
            return {
                "status": "repair_generated",
                "method": "deterministic_compile_repair",
                "targeted_changes": targeted_changes,
                "repair_level": repair_level,
                "allowed_regions": allowed_regions,
                "locked_strategy_ids": list(strategy_ids),
                "locked_tile_parameters": locked_parameters,
                "source_hash_after": source_snapshot_hash(
                    snapshot_source_files(Path(output_chain_dir), backend_code_files(ir or {}))
                ),
            }
        if client is None:
            return {"status": "repair_unavailable", "reason": "no matching deterministic repair; LLM client required"}
        generated = generate_code_files_from_patch_with_llm(
            client=client, patch_ir=ir or {}, patch_result=patch, strategy=strategy,
            code_context=load_code_context(source_chain_dir, ["cuda_kernel.cuh", "kernel.h"]),
            prompt_path=ROOT / "llm" / "prompts" / (
                "repair_locked_cuda_full_prompt.txt" if full_kernel_repair
                else "repair_locked_cuda_prompt.txt"),
            repair_context={**evidence, "requires_more_regions": True,
                            "repair_instruction": patch["repair_instruction"],
                            "repair_level": repair_level,
                            "suggested_regions": suggested_regions,
                            "allowed_regions": allowed_regions,
                            "symbol_contract": symbol_contract},
        )
        if full_kernel_repair:
            files = generated.get("files", [])
            if generated.get("edits") or len(files) != 1 or files[0].get("path") != "cuda_kernel.cuh":
                raise ValueError("Full repair must return only cuda_kernel.cuh")
        elif not generated.get("edits"):
            raise ValueError("Production repair must return local region edits")
        if any(edit.get("path") != "cuda_kernel.cuh" for edit in generated.get("edits", [])):
            raise ValueError("Repair cannot modify files other than cuda_kernel.cuh")
        if any(key in generated for key in ("ir_updates", "selected_strategy_ids", "applied_strategy_ids")):
            raise ValueError("Repair must not update IR or reselect strategies")
        restore_source_files(output_chain_dir, originals)
        applied = apply_generated_code_files(generated, output_chain_dir, ir or {}, strategy)
        if applied.get("status") != "pass":
            raise ValueError(applied.get("error_message", "local repair application failed"))
        violations = tile_lock_defects((Path(output_chain_dir) / "cuda_kernel.cuh").read_text(encoding="utf-8"), locked_parameters)
        if violations:
            raise ValueError("; ".join(violations))
        syntax = check_generated_source_syntax(output_chain_dir, backend_ast_files(ir or {}))
        if syntax.get("accepted") is False:
            raise ValueError("Repair introduced source syntax defects: " + json.dumps(syntax["results"]))
        after_snapshot = snapshot_source_files(Path(output_chain_dir), backend_code_files(ir or {}))
        after_hash = source_snapshot_hash(after_snapshot)
        if after_hash == source_snapshot_hash(originals):
            raise ValueError("Repair produced no source change")
        return {
            "status": "repair_generated",
            "method": "llm_full_kernel" if full_kernel_repair else "llm_coupled_region_patch",
            "repair_level": repair_level,
            "allowed_regions": allowed_regions,
            "symbol_contract": symbol_contract,
            "locked_strategy_ids": list(strategy_ids),
            "locked_tile_parameters": locked_parameters,
            "localization": evidence,
            "source_hash_after": after_hash,
            "patch": generated,
        }
    except Exception as exc:
        restore_source_files(output_chain_dir, originals)
        return {"status": "repair_failed", "error_message": str(exc),
                "failure_class": "implementation_repair_failed_not_strategy_conflict",
                "rejected_patch": locals().get("generated"),
                "locked_strategy_ids": list(strategy_ids), "locked_tile_parameters": locked_parameters,
                "localization": evidence}


CUDA_REPAIR_REGIONS = [
    "LAUNCH_CONFIG", "SHARED_DECL", "INDEX_MAPPING", "REGISTER_DECL",
    "GLOBAL_TO_SHARED_LOAD", "NEXT_TILE_LOAD", "SYNC_AFTER_LOAD",
    "MAIN_LOOP", "COMPUTE_INNER", "STORE",
]


def source_snapshot_hash(snapshot: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode("utf-8")).hexdigest()


def repair_regions_from_diagnosis(source: str, diagnosis: dict[str, Any], repair_level: int) -> list[str]:
    line_regions = anchor_regions_for_compiler_lines(source, compiler_error_lines(diagnosis))
    text = json.dumps(diagnosis, ensure_ascii=False).lower()
    inferred: list[str] = []
    if any(token in text for token in ["undeclared", "undefined", "identifier", "rega", "regb"]):
        inferred.extend(["REGISTER_DECL", "COMPUTE_INNER"])
    if any(token in text for token in ["sa\"", "sb\"", "shared", "cooperative load"]):
        inferred.extend(["SHARED_DECL", "GLOBAL_TO_SHARED_LOAD", "NEXT_TILE_LOAD"])
    if any(token in text for token in ["launch", "<<<", "blockdim", "griddim"]):
        inferred.append("LAUNCH_CONFIG")
    if any(token in text for token in ["store", "output", "epilogue"]):
        inferred.append("STORE")
    ordered = unique_region_names([*line_regions, *inferred])
    limits = {1: 2, 2: 4, 3: len(CUDA_REPAIR_REGIONS)}
    return ordered[:limits.get(repair_level, 4)]


def compiler_error_lines(diagnosis: dict[str, Any]) -> list[int]:
    text = json.dumps(diagnosis, ensure_ascii=False)
    values = re.findall(r"cuda_kernel\.cuh(?:\(|:)(\d+)(?:\)|:)", text, flags=re.IGNORECASE)
    return sorted({int(value) for value in values})


def anchor_regions_for_compiler_lines(source: str, line_numbers: list[int]) -> list[str]:
    lines = source.splitlines()
    spans: list[tuple[str, int, int]] = []
    for region in CUDA_REPAIR_REGIONS:
        begin = next((index for index, line in enumerate(lines, start=1) if f"{region}_BEGIN" in line), None)
        end = next((index for index, line in enumerate(lines, start=1)
                    if begin is not None and index > begin and f"{region}_END" in line), None)
        if begin is not None and end is not None:
            spans.append((region, begin, end))
    result: list[str] = []
    for line_number in line_numbers:
        containing = [(region, end - begin) for region, begin, end in spans if begin <= line_number <= end]
        if containing:
            region = min(containing, key=lambda item: item[1])[0]
            if region not in result:
                result.append(region)
    return result


def extract_cuda_symbol_contract(source: str) -> dict[str, Any]:
    code = strip_comments_for_light_syntax(source)
    declarations = sorted(set(re.findall(
        r"\b(?:const\s+)?(?:unsigned\s+)?(?:int|float|double|float[234]|dim3)\s+([A-Za-z_]\w*)",
        code,
    )))
    arrays = sorted(set(re.findall(
        r"\b(?:__shared__\s+)?(?:float|double|float[234])\s+([A-Za-z_]\w*)\s*(?=\[)",
        code,
    )))
    return {
        "declared_identifiers": declarations[:120],
        "array_identifiers": arrays[:60],
        "rule": "Reuse declared symbols with compatible scope and dimensions; do not invent aliases.",
    }


def apply_deterministic_compile_repairs(source: str, diagnosis: dict[str, Any]) -> str:
    text = json.dumps(diagnosis, ensure_ascii=False).lower()
    if "expected a" not in text and "compile" not in text:
        return source
    # CUDA launch syntax cannot contain a function qualifier between the
    # template-id and <<<...>>> launch configuration.
    return re.sub(
        r"(\bgemm\s*<[^;{}]+?>\s*)\b__forceinline(?:__)?\s*(?=<<<)",
        r"\1",
        source,
        flags=re.DOTALL,
    )


def deduplicate_code_candidates(candidates):
    unique = {}
    for candidate in candidates:
        snapshot = candidate.get("source_snapshot") or {}
        if not snapshot:
            code_dir = candidate.get("candidate_code_dir")
            if code_dir and Path(code_dir).exists():
                snapshot = snapshot_source_files(Path(code_dir), backend_code_files(candidate.get("verified_ir", {})))
        if not snapshot:
            key = candidate.get("strategy_id") or id(candidate)
        else:
            ir = candidate.get("verified_ir", {})
            payload = {"source": snapshot, "problem": ir.get("problem"),
                       "target": ir.get("hardware", {}).get("compute_capability"),
                       "build_options": candidate.get("build_options", {})}
            key = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
        if key not in unique or is_better_candidate(candidate, unique[key]):
            unique[key] = candidate
    return list(unique.values())


def verify_terminal_chains(
    frontier: list[dict[str, Any]],
    strategy_library: dict[str, Any],
    build_platform: str = "windows",
    benchmark_runs: int = DEFAULT_BENCHMARK_RUNS,
    benchmark_warmup_runs: int = DEFAULT_BENCHMARK_WARMUP_RUNS,
    client: OpenAICompatibleClient | None = None,
) -> dict[str, Any]:
    verified_candidates = []
    terminal_summaries = []
    best_candidate = None
    best_state = None

    seen_sources = set()
    for index, state in enumerate(frontier, start=1):
        if not state.get("path"):
            continue
        source_key = hashlib.sha256(json.dumps({
            "source": state.get("source_snapshot", {}),
            "problem": state.get("current_ir", {}).get("problem"),
            "target": state.get("current_ir", {}).get("hardware", {}).get("compute_capability"),
            "platform": build_platform,
        }, sort_keys=True, default=str).encode()).hexdigest()
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        chain_id = chain_id_for_state(index, state)
        chain_dir = Path(
            state.get("code_dir")
            or chain_code_path(state.get("path", []), state.get("path_code", []), state.get("current_ir", {}))
        )
        restore_source_files(chain_dir, state["source_snapshot"])

        terminal_ir = copy.deepcopy(state["current_ir"])
        terminal_ir.setdefault("strategy", {})["applied_strategy_ids"] = state["history"].get(
            "applied_strategy_ids", []
        )
        terminal_ir.setdefault("strategy", {})["history"] = state["history"].get("events", [])
        terminal_ir.setdefault("strategy", {})["chain_id"] = chain_id
        terminal_ir.setdefault("strategy", {})["chain_code"] = chain_code_key(state.get("path_code", []))
        terminal_ir.setdefault("strategy", {})["chain_path"] = state.get("path", [])

        verified_ir = verify_terminal_chain_once(
            terminal_ir=terminal_ir,
            chain_dir=chain_dir,
            chain_id=chain_id,
            strategy_library=strategy_library,
            attempt=1,
            build_platform=build_platform,
            benchmark_runs=benchmark_runs,
            benchmark_warmup_runs=benchmark_warmup_runs,
        )
        repair_attempts = []
        repair_baseline_snapshot = snapshot_source_files(chain_dir, backend_code_files(terminal_ir))
        seen_repair_hashes: set[str] = set()
        for repair_attempt in range(1, DEFAULT_MAX_REPAIR_ATTEMPTS + 1):
            if terminal_chain_is_accepted(verified_ir):
                break
            diagnosis = verified_ir.get("defect_diagnosis", {})
            before_repair_ir = copy.deepcopy(verified_ir)
            before_repair_source = snapshot_source_files(chain_dir, backend_code_files(verified_ir))
            save_json(chain_dir / f"repair_checkpoint_{repair_attempt}.json",
                      {"ir": before_repair_ir, "source_snapshot": before_repair_source})
            repair_result = repair_chain_locally(
                source_chain_dir=chain_dir,
                output_chain_dir=chain_dir,
                diagnosis=diagnosis,
                ir=verified_ir,
                client=client,
                strategy_library=strategy_library,
                repair_attempt=repair_attempt,
                baseline_snapshot=repair_baseline_snapshot,
            )
            repair_result["repair_attempt"] = repair_attempt
            save_json(chain_dir / f"semantic_repair_attempt_{repair_attempt}.json", repair_result)
            repair_attempts.append(repair_result)
            repair_hash = repair_result.get("source_hash_after")
            if repair_result.get("status") != "repair_generated" or not repair_hash:
                verified_ir["repair_feedback"] = repair_result
                if repair_result.get("status") == "repair_failed":
                    continue
                break
            if repair_hash in seen_repair_hashes:
                repair_result["status"] = "repair_retry_no_progress"
                repair_result["reason"] = "repair produced a previously tested source snapshot"
                restore_source_files(chain_dir, before_repair_source)
                verified_ir = before_repair_ir
                verified_ir["repair_feedback"] = repair_result
                save_json(chain_dir / f"semantic_repair_attempt_{repair_attempt}.json", repair_result)
                continue
            seen_repair_hashes.add(repair_hash)
            verified_ir["locked_repair_contract"] = {
                "strategy_ids": repair_result.get("locked_strategy_ids", []),
                "tile_parameters": repair_result.get("locked_tile_parameters", {}),
            }
            verified_ir = verify_terminal_chain_once(
                terminal_ir=verified_ir,
                chain_dir=chain_dir,
                chain_id=chain_id,
                strategy_library=strategy_library,
                attempt=repair_attempt + 1,
                build_platform=build_platform,
                benchmark_runs=benchmark_runs,
                benchmark_warmup_runs=benchmark_warmup_runs,
            )
            from SGPO.verification.repair_transaction import finish_repair
            verified_ir, rollback = finish_repair(
                before_repair_ir, verified_ir, before_repair_source,
                snapshot_source_files(chain_dir, backend_code_files(verified_ir)),
                repair_result, terminal_chain_is_accepted(verified_ir))
            if rollback:
                restore_source_files(chain_dir, before_repair_source)
            save_json(chain_dir / f"semantic_repair_attempt_{repair_attempt}.json", repair_result)
        if repair_attempts:
            verified_ir.setdefault("terminal_repair", {})["attempts"] = repair_attempts
            verified_ir.setdefault("terminal_repair", {})["attempt_count"] = len(repair_attempts)

        manifest = {
            "status": "verified_terminal_chain",
            "terminal_reason": state.get("terminal_reason", "construction_completed"),
            "chain_id": chain_id,
            "chain_code": chain_code_key(state.get("path_code", [])),
            "path": state.get("path", []),
            "code_dir": str(chain_dir),
            "verification": verified_ir.get("verification", {}).get("summary", {}),
            "performance": verified_ir.get("performance", {}),
        }
        save_json(chain_dir / "terminal_chain.json", manifest)
        save_json(chain_result_path(chain_id), build_chain_result_record(state, manifest, verified_ir))
        save_candidate_json(DEFAULT_VERIFIED_IR_OUTPUT, "Chain", chain_id, 1, verified_ir)

        candidate = {
            "accepted": terminal_chain_is_accepted(verified_ir),
            "stage": "Chain",
            "strategy_id": chain_id,
            "verified_ir": verified_ir,
            "source_snapshot": snapshot_source_files(chain_dir, backend_code_files(verified_ir)),
            "candidate_code_dir": str(chain_dir),
            "history": state["history"],
            "path": state.get("path", []),
            "path_code": state.get("path_code", []),
        }
        verified_candidates.append(candidate)
        terminal_summaries.append(manifest)
        if candidate["accepted"] and is_better_candidate(candidate, best_candidate):
            best_candidate = candidate
            best_state = state

    top_terminal_results = build_top_terminal_results(verified_candidates, DEFAULT_TOP_K_FINAL_RESULTS)
    return {
        "verified_candidates": verified_candidates,
        "best_candidate": best_candidate,
        "best_state": best_state,
        "top_terminal_results": top_terminal_results,
        "summary": {
            "stage": "TerminalVerification",
            "selection_mode": DEFAULT_SELECTION_MODE,
            "terminal_chain_count": len([state for state in frontier if state.get("path")]),
            "verified_terminal_chain_count": len(verified_candidates),
            "accepted_terminal_chain_count": len([item for item in verified_candidates if item["accepted"]]),
            "best_chain_id": best_candidate.get("strategy_id") if best_candidate else None,
            "best_chain_gflops": candidate_gflops(best_candidate),
            "top_k_final_results": DEFAULT_TOP_K_FINAL_RESULTS,
            "top_terminal_results": top_terminal_results,
            "terminal_chains": terminal_summaries,
        },
    }


def build_top_terminal_results(candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    accepted = [candidate for candidate in candidates if candidate.get("accepted")]
    ranked_pool = deduplicate_code_candidates(accepted)
    ranked = sorted(ranked_pool, key=lambda item: candidate_gflops(item) or -1.0, reverse=True)[:top_k]
    results = []
    for rank, candidate in enumerate(ranked, start=1):
        verified_ir = candidate.get("verified_ir", {}) or {}
        verification_summary = verified_ir.get("verification", {}).get("summary", {}) or {}
        performance = verified_ir.get("performance", {}) or {}
        results.append(
            {
                "rank": rank,
                "chain_id": candidate.get("strategy_id"),
                "source_phase": candidate.get("source_phase"),
                "accepted": candidate.get("accepted"),
                "code_dir": candidate.get("candidate_code_dir"),
                "chain_code": chain_code_key(candidate.get("path_code", [])),
                "path": candidate.get("path", []),
                "compile_status": verification_summary.get("compile_status"),
                "correctness_status": verification_summary.get("correctness_status"),
                "runtime_safety_status": verification_summary.get("runtime_safety_status"),
                "cuda_error": verification_summary.get("cuda_error"),
                "oracle_acceptance_status": verification_summary.get("oracle_acceptance_status"),
                "acceptance_basis": verification_summary.get("acceptance_basis"),
                "code_verification_basis": verification_summary.get("code_verification_basis"),
                "optimization_realization_status": verification_summary.get("optimization_realization_status"),
                "strategy_application": verified_ir.get("strategy_application"),
                "code_verification_status": verification_summary.get("code_verification_status"),
                "checker_failure_is_terminal": verification_summary.get("checker_failure_is_terminal"),
                "code_checker_advisory_failure": verification_summary.get("code_checker_advisory_failure"),
                "static_findings_are_advisory": verification_summary.get("static_findings_are_advisory"),
                "locked_repair_status": verified_ir.get("locked_repair_verification", {}).get("status"),
                "hard_constraints_ok": verified_ir.get("phase1_hard_constraints", {}).get("hard_constraints_ok"),
                "latency_ms": performance.get("latency_ms") or verification_summary.get("latency_ms"),
                "gflops": performance.get("gflops") or verification_summary.get("gflops"),
                "benchmark_runs": performance.get("benchmark_runs"),
                "benchmark_warmup_runs": performance.get("warmup_runs"),
                "benchmark_successful_runs": performance.get("benchmark_successful_runs"),
                "latency_ms_mean": performance.get("latency_ms_mean"),
                "latency_ms_trimmed_mean": performance.get("latency_ms_trimmed_mean"),
                "latency_ms_median": performance.get("latency_ms_median"),
                "latency_ms_std": performance.get("latency_ms_std"),
                "latency_ms_best": performance.get("latency_ms_best"),
                "gflops_mean": performance.get("gflops_mean"),
                "gflops_trimmed_mean": performance.get("gflops_trimmed_mean"),
                "gflops_median": performance.get("gflops_median"),
                "gflops_std": performance.get("gflops_std"),
                "gflops_best": performance.get("gflops_best"),
                "relative_to_cublas": performance.get("relative_to_cublas"),
                "cublas_latency_ms_mean": performance.get("cublas_latency_ms_mean"),
                "cublas_latency_ms_trimmed_mean": performance.get("cublas_latency_ms_trimmed_mean"),
                "cublas_gflops_mean": performance.get("cublas_gflops_mean"),
                "cublas_gflops_trimmed_mean": performance.get("cublas_gflops_trimmed_mean"),
                "cpu_blas_latency_ms": performance.get("cpu_blas_latency_ms"),
                "cpu_blas_gflops": performance.get("cpu_blas_gflops"),
                "relative_to_cpu_blas": performance.get("relative_to_cpu_blas"),
                "terminal_repair_attempt_count": verified_ir.get("terminal_repair", {}).get("attempt_count", 0),
            }
        )
    return results


def candidate_has_correctness_pass(candidate: dict[str, Any]) -> bool:
    summary = candidate.get("verified_ir", {}).get("verification", {}).get("summary", {}) or {}
    runtime_status = summary.get("runtime_safety_status")
    return (
        summary.get("compile_status") == "pass"
        and summary.get("correctness_status") == "pass"
        and runtime_status != "fail"
        and summary.get("cuda_error") is None
    )


def verify_terminal_chain_once(
    terminal_ir: dict[str, Any],
    chain_dir: Path,
    chain_id: str,
    strategy_library: dict[str, Any],
    attempt: int,
    build_platform: str = "windows",
    benchmark_runs: int = DEFAULT_BENCHMARK_RUNS,
    benchmark_warmup_runs: int = DEFAULT_BENCHMARK_WARMUP_RUNS,
) -> dict[str, Any]:
    current_ir = copy.deepcopy(terminal_ir)
    if target_backend(current_ir) != "cpu":
        from SGPO.verification.store_alignment_proof import prove_fragment_store_alignment
        kernel_path, harness_path = chain_dir / "cuda_kernel.cuh", chain_dir / "main.cpp"
        if kernel_path.exists() and harness_path.exists():
            from SGPO.verification.pipeline_evidence import pipeline_evidence
            current_ir['pipeline_realization'] = pipeline_evidence(kernel_path.read_text(encoding='utf-8'))
            save_json(chain_dir / 'pipeline_realization.json', current_ir['pipeline_realization'])
            proof = prove_fragment_store_alignment(kernel_path.read_text(encoding="utf-8"),
                                                    harness_path.read_text(encoding="utf-8"), current_ir)
            old_node = current_ir.get("vectorization", {}).get("C", {})
            if old_node.get("alignment_evidence", {}).get("scope") == "current_shape_and_cudaMalloc_benchmark":
                old_node["alignment_proven"] = False
                old_node.pop("alignment_evidence", None)
            if proof:
                node = current_ir.setdefault("vectorization", {}).setdefault("C", {})
                node["alignment_proven"] = True
                node["alignment_evidence"] = proof
    if target_backend(current_ir) != "cpu":
        from SGPO.verification.memory_access_plan import enforce_memory_access_plan
        access_plan = enforce_memory_access_plan(chain_dir, allow_scalar_fallback=False)
        current_ir["memory_access_plan_verification"] = access_plan
        from SGPO.verification.cooperative_load_repair import diagnose_cooperative_loads
        current_ir["cooperative_load_diagnostics"] = diagnose_cooperative_loads(
            (chain_dir / "cuda_kernel.cuh").read_text(encoding="utf-8"), current_ir)
        save_json(chain_dir / "cooperative_load_diagnostics.json", current_ir["cooperative_load_diagnostics"])
        save_json(chain_dir / "memory_access_plan_verification.json", access_plan)
    if target_backend(current_ir) != "cpu":
        current_ir["phase1_hard_constraints"] = check_after_codegen(
            current_ir, include_hard_constraints=True,
        )
        save_json(chain_dir / "phase1_hard_constraints.json", current_ir["phase1_hard_constraints"])
    code_ast = (
        extract_code_ast(chain_dir, backend_ast_files(current_ir))
        if target_backend(current_ir) != "cpu"
        else extract_cpu_code_ast(chain_dir)
    )
    save_json(chain_dir / "code_ast.json", code_ast)
    current_ir["code_ast"] = code_ast
    current_ir["code_completeness"] = check_terminal_code_completeness(chain_dir, current_ir)

    if target_backend(current_ir) == "cpu":
        verified_ir = verify_cpu_build_and_run(
            ir=current_ir,
            source_dir=chain_dir,
            build_dir=chain_dir / "build",
            build_platform=build_platform,
            benchmark_runs=benchmark_runs,
            benchmark_warmup_runs=benchmark_warmup_runs,
        )
    else:
        verified_ir = verify_build_and_run(
            ir=current_ir,
            source_dir=chain_dir,
            build_dir=chain_dir / "build",
            build_platform=build_platform,
            benchmark_runs=benchmark_runs,
            benchmark_warmup_runs=benchmark_warmup_runs,
        )
    verified_ir.setdefault("verification", {})["summary"] = summarize_verification(verified_ir)
    code_verification = check_chain_code_verification(verified_ir, strategy_library)
    verified_ir["code_verification"] = code_verification
    summary = verified_ir.setdefault("verification", {}).setdefault("summary", {})
    oracle_passed = oracle_verification_passed(verified_ir)
    code_checker_passed = code_verification.get("accepted_by_code_verifier") is not False
    summary["code_verification_status"] = "pass" if code_checker_passed else "fail"
    summary["oracle_acceptance_status"] = "pass" if oracle_passed else "fail"
    summary["acceptance_basis"] = "compile_correctness_runtime_oracle"
    summary["code_verification_basis"] = "advisory_after_oracle"
    summary["checker_failure_is_terminal"] = not oracle_passed
    summary["code_checker_advisory_failure"] = oracle_passed and not code_checker_passed
    applied_strategy_ids = list((verified_ir.get("strategy", {}) or {}).get("applied_strategy_ids", []) or [])
    applied_strategy_ids = append_unique(
        applied_strategy_ids,
        verified_ir.get('strategy_contract_state', {}).get('selected_strategies', []))
    realization = verify_strategy_realization(chain_dir, applied_strategy_ids, verified_ir)
    verified_ir["strategy_realization"] = realization
    from SGPO.verification.strategy_application import update_strategy_contract_state
    update_strategy_contract_state(verified_ir)
    if target_backend(verified_ir) != "cpu" and "locked_repair_contract" in verified_ir:
        from SGPO.verification.locked_repair import tile_lock_defects
        locked = verified_ir["locked_repair_contract"]
        lock_errors = tile_lock_defects((chain_dir / "cuda_kernel.cuh").read_text(encoding="utf-8"), locked["tile_parameters"])
        if verified_ir.get('strategy', {}).get('applied_strategy_ids', []) != locked["strategy_ids"]:
            lock_errors.append("Applied strategy list changed during repair")
        if not realization["hard_gate_passed"]:
            lock_errors.append("Selected strategy implementation is missing")
        if not code_checker_passed and not oracle_passed:
            lock_errors.append("Selected strategy code verification did not pass")
        verified_ir["locked_repair_verification"] = {
            "status": "fail" if lock_errors else "pass", "errors": lock_errors,
            "basis": "parameter_lock_and_available_strategy_verifiers",
            "strategy_reports": realization.get("strategy_reports", []),
            "code_checker_advisory_failure": oracle_passed and not code_checker_passed,
        }
    summary["optimization_realization_status"] = (
        "fail" if not realization["hard_gate_passed"] else realization["realization_status"]
    )
    summary['static_findings_are_advisory'] = True
    summary['acceptance_basis'] = 'compile_correctness_runtime_oracle'
    summary['checker_failure_is_terminal'] = not oracle_passed
    summary['code_checker_advisory_failure'] = oracle_passed and (
        not code_checker_passed or not realization['hard_gate_passed']
        or verified_ir.get('locked_repair_verification', {}).get('status') == 'fail'
        or verified_ir.get('phase1_hard_constraints', {}).get('hard_constraints_ok') is False
    )
    diagnosis = diagnose_defects(verified_ir, verified_ir.get("phase1_hard_constraints"))
    lock_report = verified_ir.get("locked_repair_verification", {})
    if lock_report.get("status") == "fail":
        diagnosis.setdefault("defects", []).append({
            "defect_type": "StrategyCompliance.LockedRepairContractViolation",
            "related_strategy": "Repair.PreserveAppliedStrategies",
            "related_fields": ["strategy.applied_strategy_ids", "tiling"],
            "repair_action": "Implement the locked strategies and parameters without reselection; consult strategy reports and code verification.",
            "evidence": lock_report,
        })
        diagnosis["defect_count"] = len(diagnosis["defects"])
    diagnosis["stage"] = "Chain"
    diagnosis["chain_id"] = chain_id
    diagnosis["terminal_verification_attempt"] = attempt
    verified_ir = attach_diagnosis(verified_ir, diagnosis)
    save_candidate_json(DEFAULT_DIAGNOSIS_OUTPUT, "Chain", chain_id, attempt, diagnosis)
    return verified_ir


def check_chain_code_verification(
    verified_ir: dict[str, Any],
    strategy_library: dict[str, Any],
) -> dict[str, Any]:
    if target_backend(verified_ir) == "cpu":
        code_completeness = verified_ir.get("code_completeness", {})
        return {
            "phase": "terminal_chain_after_code_patch_applied",
            "accepted_by_code_verifier": code_completeness.get("accepted") is not False,
            "backend": "cpu",
            "strategy_count": 0,
            "strategy_reports": [],
            "code_completeness": code_completeness,
        }
    strategy_ids = verified_ir.get("strategy", {}).get("applied_strategy_ids", []) or []
    reports = []
    accepted = True
    for strategy_id in strategy_ids:
        try:
            strategy = load_strategy(strategy_library, strategy_id)
        except ValueError:
            continue
        report = check_code_verification(verified_ir, strategy)
        reports.append(report)
        if report.get("accepted_by_code_verifier") is False:
            accepted = False
    code_completeness = verified_ir.get("code_completeness", {})
    if code_completeness.get("accepted") is False:
        accepted = False
    return {
        "phase": "terminal_chain_after_code_patch_applied",
        "accepted_by_code_verifier": accepted,
        "strategy_count": len(reports),
        "strategy_reports": reports,
        "code_completeness": code_completeness,
    }


def check_terminal_code_completeness(chain_dir: Path, ir: dict[str, Any]) -> dict[str, Any]:
    if target_backend(ir) == "cpu":
        return check_terminal_cpu_code_completeness(chain_dir, ir)
    kernel_path = chain_dir / "cuda_kernel.cuh"
    if not kernel_path.exists():
        return {"accepted": False, "results": [{"id": "CUDA_KERNEL_CUH_EXISTS", "status": "fail"}]}
    content = kernel_path.read_text(encoding="utf-8")
    code_only = strip_cpp_comments(content)
    checks = []
    access_plan = ir.get("memory_access_plan_verification", {}) or {}
    checks.append({
        "id": "MEMORY_ACCESS_PLAN_VERIFIED",
        "status": "pass" if access_plan.get("status") == "pass" and access_plan.get("checked") else "fail",
        "message": access_plan.get("reason") or (
            "generated CUDA memory accesses passed deterministic source checks"
            if access_plan.get("status") == "pass"
            else "generated CUDA memory access plan could not be verified"
        ),
        "failure_type": "GEMM.Semantic.MemoryAccessContractViolation",
        "repair_action": "repair block offsets, shared-memory rank, register extents, K-tile coverage and accumulator layout",
        "detail": {"defects": access_plan.get("defects", [])},
    })
    checks.append({
        "id": "NO_PLACEHOLDER_ONLY_REGIONS",
        "status": "pass" if "Insert " not in code_only else "fail",
        "message": "skeleton placeholder text must not remain as active code",
    })
    checks.append({
        "id": "HAS_C_STORE",
        "status": "pass" if re.search(r"\bC\s*\[[^\]]+\]\s*=", code_only) else "fail",
        "message": "kernel must write computed values back to C",
    })
    checks.append({
        "id": "HAS_ACCUMULATION",
        "status": "pass"
        if re.search(r"(acc|results|c_reg|cReg|Creg)[A-Za-z0-9_]*(?:\s*\[[^\]]+\])*\s*\+=", code_only)
        else "fail",
        "message": "kernel must accumulate partial sums in registers",
    })
    checks.append({
        "id": "HAS_MULTIPLY_ACCUMULATE_EXPRESSION",
        "status": "pass" if re.search(r"\+=\s*[^;]*\*\s*[^;]*;", code_only) else "fail",
        "message": "kernel must contain a multiply-accumulate expression",
    })
    checks.append({
        "id": "HAS_KERNEL_LAUNCH",
        "status": "pass" if "<<<" in code_only and ">>>" in code_only else "fail",
        "message": "cuda_gemm must launch the GEMM kernel",
    })
    if ir.get("memory", {}).get("use_shared_memory") is True:
        shared_names = []
        for file_ast in (ir.get("code_ast", {}).get("files", {}) or {}).values():
            shared_names.extend((item.get("name") for item in file_ast.get("shared_memory", []) or []))
        from SGPO.verification.optimization_preservation import staged_buffers
        shared_names.extend(staged_buffers(code_only))
        checks.append({
            "id": "HAS_SHARED_A_B",
            "status": "pass" if has_name_like(shared_names, ["As", "shared_A", "sA"]) and has_name_like(shared_names, ["Bs", "shared_B", "sB"]) else "fail",
            "message": "shared-memory IR requires real A/B __shared__ buffers",
        })
    semantic_report = check_gemm_semantic_obligations(chain_dir, ir)
    return {
        "accepted": all(item["status"] == "pass" for item in checks) and semantic_report.get("accepted") is not False,
        "results": checks,
        "semantic_obligations": semantic_report,
    }


def check_terminal_cpu_code_completeness(chain_dir: Path, ir: dict[str, Any] | None = None) -> dict[str, Any]:
    kernel_path = chain_dir / "cpu_kernel.c"
    if not kernel_path.exists():
        return {"accepted": False, "backend": "cpu", "results": [{"id": "CPU_KERNEL_C_EXISTS", "status": "fail"}]}
    content = kernel_path.read_text(encoding="utf-8")
    code_only = strip_cpp_comments(content)
    forbidden_patterns = [
        r"#\s*include\s*<cuda",
        r"#\s*include\s*<cublas",
        r"\b__global__\b",
        r"\b__device__\b",
        r"\b__shared__\b",
        r"\bthreadIdx\b",
        r"\bblockIdx\b",
        r"\bcuda[A-Za-z0-9_]*\b",
        r"\bcublas[A-Za-z0-9_]*\b",
        r"<<<",
    ]
    checks = [
        {
            "id": "CPU_KERNEL_C_EXISTS",
            "status": "pass",
            "message": "cpu_kernel.c exists",
        },
        {
            "id": "CPU_HAS_CPU_GEMM",
            "status": "pass" if re.search(r"\bvoid\s+cpu_gemm\s*\(", code_only) else "fail",
            "message": "cpu_kernel.c must define cpu_gemm",
        },
        {
            "id": "CPU_HAS_LOOP_NEST",
            "status": "pass" if re.search(r"\bfor\s*\(", code_only) else "fail",
            "message": "CPU GEMM must contain executable C loops",
        },
        {
            "id": "CPU_HAS_MULTIPLY_ACCUMULATE",
            "status": "pass" if re.search(r"\+=\s*[^;]*\*\s*[^;]*;", code_only) else "fail",
            "message": "CPU GEMM must contain multiply-accumulate statements",
        },
        {
            "id": "CPU_HAS_C_STORE",
            "status": "pass" if re.search(r"\b(?:C|C_data)\s*\[[^\]]+\]\s*=", code_only) else "fail",
            "message": "CPU GEMM must store results into C",
        },
        {
            "id": "CPU_NO_CUDA_TOKENS",
            "status": "pass" if not any(re.search(pattern, code_only) for pattern in forbidden_patterns) else "fail",
            "message": "CPU C backend must not contain CUDA-only constructs",
        },
    ]
    checks.extend(check_cpu_static_semantic_obligations(code_only))
    checks.extend(check_cpu_strategy_code_obligations(code_only, ir or {}))
    return {
        "accepted": all(item["status"] == "pass" for item in checks),
        "backend": "cpu",
        "results": checks,
    }


def check_cpu_strategy_code_obligations(code_only: str, ir: dict[str, Any]) -> list[dict[str, Any]]:
    strategy_ids = list((ir.get("strategy", {}) or {}).get("applied_strategy_ids", []) or [])
    joined = "\n".join(strategy_ids)
    checks: list[dict[str, Any]] = []

    def selected(*needles: str) -> bool:
        return any(needle in strategy_id for strategy_id in strategy_ids for needle in needles)

    def add_check(check_id: str, ok: bool, message: str, failure_type: str, repair_action: str) -> None:
        checks.append(
            {
                "id": check_id,
                "status": "pass" if ok else "fail",
                "message": message,
                "failure_type": None if ok else failure_type,
                "repair_action": repair_action,
                "selected_strategies": [strategy_id for strategy_id in strategy_ids if check_id_matches_strategy(check_id, strategy_id)],
            }
        )

    if "AVX512" in joined:
        add_check(
            "CPU_STRATEGY_AVX512_INTRINSICS_PRESENT",
            bool(re.search(r"\b__m512\b|_mm512_", code_only))
            and "_mm512_fmadd_ps" in code_only
            and bool(re.search(r"_mm512_(?:loadu|load)_ps", code_only)),
            "selected AVX512 strategy must materialize real AVX512 FMA/load/store intrinsics",
            "CPU.StrategyImplementation.AVX512Missing",
            "generate an AVX512 micro-kernel using _mm512_broadcastss_ps/_mm512_loadu_ps/_mm512_fmadd_ps/_mm512_storeu_ps",
        )
    if "AVX2" in joined:
        add_check(
            "CPU_STRATEGY_AVX2_INTRINSICS_PRESENT",
            bool(re.search(r"\b__m256\b|_mm256_", code_only))
            and "_mm256_fmadd_ps" in code_only
            and bool(re.search(r"_mm256_(?:loadu|load)_ps", code_only)),
            "selected AVX2 strategy must materialize real AVX2 FMA/load/store intrinsics",
            "CPU.StrategyImplementation.AVX2Missing",
            "generate an AVX2 micro-kernel using _mm256_broadcast_ss/_mm256_loadu_ps/_mm256_fmadd_ps/_mm256_storeu_ps",
        )
    if selected("OpenMP", "Parallel.OpenMP", "Threading.OpenMP"):
        add_check(
            "CPU_STRATEGY_OPENMP_PRESENT",
            bool(re.search(r"#\s*pragma\s+omp\s+parallel", code_only)),
            "selected OpenMP strategy must materialize a real OpenMP parallel region or parallel loop",
            "CPU.StrategyImplementation.OpenMPMissing",
            "parallelize independent output tiles with #pragma omp parallel for and avoid overlapping C writes",
        )
    if selected("PackAB", "PackA", "PackB", "Packing.", "Memory.Pack"):
        has_pack_a = bool(re.search(r"\b(?:a_panel|pack_a|packed_a|A_pack|sa)\b", code_only))
        has_pack_b = bool(re.search(r"\b(?:b_panel|pack_b|packed_b|B_pack|sb)\b", code_only))
        has_pack_copy = bool(re.search(r"(?:a_panel|packed_a|sa)\s*\[[^\]]+\]\s*=", code_only)) or bool(
            re.search(r"(?:b_panel|packed_b|sb)\s*\[[^\]]+\]\s*=", code_only)
        )
        has_pack_consumption = bool(
            re.search(r"(?:a_panel|packed_a|sa)\s*\[[^\]]+\]\s*\*", code_only)
            or re.search(r"\*\s*(?:b_panel|packed_b|sb)\s*\[[^\]]+\]", code_only)
            or re.search(r"_mm(?:256|512)_[a-z0-9_]*ps\s*\([^;]*(?:a_panel|packed_a|b_panel|packed_b|sa|sb)", code_only)
        )
        add_check(
            "CPU_STRATEGY_PACKING_PRESENT_AND_USED",
            has_pack_a and has_pack_b and has_pack_copy and has_pack_consumption,
            "selected packing strategy must allocate/fill packed A/B panels and consume packed panels in compute",
            "CPU.StrategyImplementation.PackingMissingOrUnused",
            "add packed A/B panel buffers, fill them with copy loops, and make the micro-kernel read the packed buffers",
        )
    if selected("OpenBLASStyle", "PanelDriver"):
        has_panel_order = (
            bool(re.search(r"for\s*\([^;]*\bn0\b[^;]*<\s*N", code_only))
            and bool(re.search(r"for\s*\([^;]*\bk0\b[^;]*<\s*K", code_only))
            and bool(re.search(r"for\s*\([^;]*\bm0\b[^;]*<\s*M", code_only))
        )
        has_micro_tile = bool(re.search(r"for\s*\([^;]*\bm\w*\b[^;]*\+=\s*(?:MR|RM)", code_only)) and bool(
            re.search(r"for\s*\([^;]*\bn\w*\b[^;]*\+=\s*(?:NR|RN)", code_only)
        )
        add_check(
            "CPU_STRATEGY_OPENBLAS_PANEL_DRIVER_PRESENT",
            has_panel_order and has_micro_tile,
            "selected OpenBLAS-style panel driver must have N/K/M panel loops and MR/NR micro-tile loops",
            "CPU.StrategyImplementation.PanelDriverMissing",
            "emit N-panel, K-panel, M-panel loops with packed panels feeding an MR x NR micro-kernel",
        )
    if selected("MicroKernel"):
        uses_vector_kernel = bool(re.search(r"_mm(?:256|512)_fmadd_ps", code_only))
        uses_register_tile = bool(re.search(r"\bacc(?:\s*\[|[A-Za-z0-9_]*)", code_only)) and bool(
            re.search(r"\b(?:MR|RM|REGISTER_M)\b", code_only)
        ) and bool(re.search(r"\b(?:NR|RN|REGISTER_N)\b", code_only))
        add_check(
            "CPU_STRATEGY_MICROKERNEL_PRESENT",
            uses_register_tile and (uses_vector_kernel or "PragmaSIMD" in joined),
            "selected CPU micro-kernel strategy must materialize register blocking and vector or SIMD compute",
            "CPU.StrategyImplementation.MicroKernelMissing",
            "generate an explicit MR x NR register-blocked micro-kernel instead of falling back to plain scalar loops",
        )
    return checks


def check_id_matches_strategy(check_id: str, strategy_id: str) -> bool:
    mapping = {
        "AVX512": "AVX512",
        "AVX2": "AVX2",
        "OPENMP": "OpenMP",
        "PACKING": "Pack",
        "OPENBLAS": "OpenBLAS",
        "MICROKERNEL": "MicroKernel",
    }
    return any(token in check_id and needle in strategy_id for token, needle in mapping.items())


def check_cpu_static_semantic_obligations(code_only: str) -> list[dict[str, Any]]:
    uses_simd = bool(re.search(r"\b__m(?:256|512)\b|_mm(?:256|512)_", code_only))
    loads_a_k_vector = bool(re.search(r"_mm(?:256|512)_loadu_ps\s*\(\s*&?\s*A\s*\[[^\]]*\b(?:k|kk)\b", code_only))
    has_a_broadcast = bool(re.search(r"_mm(?:256|512)_broadcast", code_only))
    has_vector_b_load = bool(re.search(r"_mm(?:256|512)_loadu_ps\s*\(\s*&?\s*(?:B|B_data|b_panel)\s*\[", code_only))
    pragma_pos = code_only.find("#pragma omp parallel")
    pack_alloc = re.search(r"float\s*\*\s*(?:a_panel|b_panel|sa|sb)\s*=", code_only)
    shared_parallel_pack = pragma_pos >= 0 and pack_alloc is not None and pack_alloc.start() < pragma_pos
    return [
        {
            "id": "CPU_SIMD_LANES_MAP_N_OUTPUT",
            "status": "pass" if not uses_simd or (not loads_a_k_vector and has_a_broadcast and has_vector_b_load) else "fail",
            "message": "explicit SIMD lanes must map to contiguous N outputs: broadcast A scalar, vector-load B columns, reduce over K sequentially",
            "failure_type": "CPU.Semantic.SIMDLaneMappingViolation" if uses_simd and (loads_a_k_vector or not has_a_broadcast or not has_vector_b_load) else None,
            "repair_action": "rewrite the SIMD micro-kernel so each accumulator vector represents one output row over contiguous N columns",
        },
        {
            "id": "CPU_OPENMP_PACK_BUFFER_PRIVATE",
            "status": "pass" if not shared_parallel_pack else "fail",
            "message": "OpenMP packed panel buffers must be thread-private or allocated inside independent tile work",
            "failure_type": "CPU.Parallel.SharedPackedBufferRace" if shared_parallel_pack else None,
            "repair_action": "move packed panel allocation into the OpenMP tile body or allocate one buffer per thread",
        },
    ]


def strip_cpp_comments(content: str) -> str:
    without_block_comments = re.sub(r"/\*.*?\*/", lambda match: "\n" * match.group(0).count("\n"), content, flags=re.DOTALL)
    return re.sub(r"//.*", "", without_block_comments)


def has_name_like(names: list[Any], candidates: list[str]) -> bool:
    lowered = [str(name).lower() for name in names if name is not None]
    return any(any(name == candidate.lower() or name.startswith(candidate.lower()) for name in lowered) for candidate in candidates)


def terminal_chain_is_accepted(verified_ir: dict[str, Any]) -> bool:
    # Preserve measured working kernels even when static proofs or selected
    # strategy realization are incomplete. Keep those findings in the report.
    return oracle_verification_passed(verified_ir)


def chain_id_for_state(index: int, state: dict[str, Any]) -> str:
    return f"chain_{index:04d}.{chain_code_key(state.get('path_code', []))}"


def chain_key_for_path(path: list[str], path_code: list[int] | None = None) -> str:
    if not path and not path_code:
        return "chain_root"
    return f"chain.{chain_code_key(path_code or [])}"


def chain_code_path(path: list[str], path_code: list[int] | None = None, ir: dict[str, Any] | None = None) -> Path:
    return backend_generated_code_root(ir or {}) / chain_key_for_path(path, path_code)


def chain_result_path(chain_id: str) -> Path:
    return Path(DEFAULT_CHAIN_DIR) / f"{safe_name(chain_id)}.json"


def chain_code_key(path_code: list[int] | None) -> str:
    if not path_code:
        return "root"
    return "-".join(str(part) for part in path_code)


def build_chain_result_record(
    state: dict[str, Any],
    manifest: dict[str, Any],
    verified_ir: dict[str, Any],
) -> dict[str, Any]:
    history = state.get("history", {})
    return {
        **manifest,
        "applied_strategy_ids": history.get("applied_strategy_ids", []),
        "failed_strategy_counts": history.get("failed_strategy_counts", {}),
        "events": history.get("events", []),
        "verification_detail": verified_ir.get("verification", {}),
        "code_verification": verified_ir.get("code_verification", {}),
        "defect_diagnosis": verified_ir.get("defect_diagnosis", {}),
        "verified_ir_output": str(candidate_path(DEFAULT_VERIFIED_IR_OUTPUT, "Chain", manifest["chain_id"], 1)),
    }


def empty_stage_result(stage: str, filtered_strategy_index: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "accepted_candidate": None,
        "events": [],
        "failed_strategy_counts": {},
        "summary": {
            "stage": stage,
            "filtered_strategy_count": filtered_strategy_index.get("strategy_count", 0),
            "llm_selected_count": 0,
            "precheck_summary": None,
            "evaluated_candidate_count": 0,
            "accepted_candidate_count": 0,
            "selected_strategy_id": None,
            "selected_gflops": None,
            "reason": reason,
        },
    }


def merge_failed_counts(history: dict[str, Any], new_counts: dict[str, int]) -> None:
    failed = history.setdefault("failed_strategy_counts", {})
    add_failed_counts(failed, new_counts)


def add_failed_counts(failed: dict[str, int], new_counts: dict[str, int]) -> None:
    for strategy_id, count in new_counts.items():
        failed[strategy_id] = failed.get(strategy_id, 0) + count


def make_frontier_state(
    state_id: str,
    current_ir: dict[str, Any],
    source_snapshot: dict[str, str],
    history: dict[str, Any],
    path: list[str],
    path_code: list[int] | None = None,
    last_candidate: dict[str, Any] | None = None,
    code_dir: str | None = None,
) -> dict[str, Any]:
    return {
        "state_id": state_id,
        "current_ir": copy.deepcopy(current_ir),
        "source_snapshot": copy.deepcopy(source_snapshot),
        "history": copy_history(history),
        "path": list(path),
        "path_code": list(path_code or []),
        "last_candidate": last_candidate,
        "code_dir": code_dir or (last_candidate or {}).get("candidate_code_dir"),
    }


def copy_history(history: dict[str, Any]) -> dict[str, Any]:
    return {
        "applied_strategy_ids": list(history.get("applied_strategy_ids", []) or []),
        "applied_micro_strategies": copy.deepcopy(history.get("applied_micro_strategies", []) or []),
        "failed_strategy_counts": dict(history.get("failed_strategy_counts", {}) or {}),
        "events": copy.deepcopy(history.get("events", []) or []),
        "completed_stages": list(history.get("completed_stages", []) or []),
        "completed_subphases": list(history.get("completed_subphases", []) or []),
        "optional_skipped_subphases": list(history.get("optional_skipped_subphases", []) or []),
        "stage_checkpoints": copy.deepcopy(history.get("stage_checkpoints", []) or []),
        "failed_unlock_bundles": list(history.get("failed_unlock_bundles", []) or []),
        "ineffective_unlock_bundles": list(history.get("ineffective_unlock_bundles", []) or []),
        "recent_unlock_defect": copy.deepcopy(history.get("recent_unlock_defect")),
    }


def merge_strategy_progress_from_ir(history: dict[str, Any], ir: dict[str, Any]) -> None:
    strategy = ir.get("strategy", {}) or {}
    completed = history.setdefault("completed_subphases", [])
    for subphase in strategy.get("completed_subphases", []) or []:
        if subphase not in completed:
            completed.append(subphase)
    skipped = history.setdefault("optional_skipped_subphases", [])
    for subphase in strategy.get("optional_skipped_subphases", []) or []:
        if subphase not in skipped:
            skipped.append(subphase)
    applied = history.setdefault("applied_micro_strategies", [])
    seen = {
        (item.get("subphase"), item.get("strategy_id"))
        for item in applied
        if isinstance(item, dict)
    }
    for item in strategy.get("applied_micro_strategies", []) or []:
        if not isinstance(item, dict):
            continue
        key = (item.get("subphase"), item.get("strategy_id"))
        if key not in seen:
            applied.append(copy.deepcopy(item))
            seen.add(key)


def select_best_frontier_state(frontier: list[dict[str, Any]]) -> dict[str, Any]:
    if not frontier:
        raise ValueError("Frontier is empty; cannot select final state.")
    return max(frontier, key=frontier_state_score)


def mark_no_terminal_frontier(ir: dict[str, Any], stage_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    final_ir = copy.deepcopy(ir)
    verification = final_ir.setdefault("verification", {})
    verification["accepted"] = False
    verification["summary"] = {
        "compile_status": "not_run",
        "correctness_status": "not_run",
        "cuda_error": None,
        "latency_ms": None,
        "gflops": None,
        "search_status": "blocked",
        "reason": "No terminal strategy chain was produced.",
    }
    final_ir.setdefault("performance", {})["latency_ms"] = None
    final_ir.setdefault("performance", {})["gflops"] = None
    final_ir.setdefault("strategy", {})["search_status"] = "blocked"
    final_ir.setdefault("strategy", {})["stage_summaries"] = stage_summaries
    return final_ir


def mark_partial_frontier(
    ir: dict[str, Any],
    stage_summaries: list[dict[str, Any]],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    final_ir = copy.deepcopy(ir)
    verification = final_ir.setdefault("verification", {})
    verification["accepted"] = False
    verification.setdefault("summary", {})["search_status"] = "partial_chain_selected"
    verification.setdefault("summary", {})["reason"] = (
        "No terminal strategy chain was produced; restored the deepest generated partial chain instead of the empty skeleton."
    )
    final_ir.setdefault("strategy", {})["search_status"] = "partial_chain_selected"
    final_ir.setdefault("strategy", {})["stage_summaries"] = stage_summaries
    final_ir.setdefault("strategy", {})["final_selected_strategy_id"] = candidate.get("strategy_id")
    final_ir.setdefault("strategy", {})["final_selected_code_dir"] = candidate.get("candidate_code_dir")
    final_ir.setdefault("strategy", {})["final_selected_is_partial"] = True
    return final_ir


def choose_better_partial_candidate(
    current: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if candidate is None:
        return current
    if current is None:
        return candidate
    return candidate if partial_candidate_score(candidate) > partial_candidate_score(current) else current


def partial_candidate_score(candidate: dict[str, Any]) -> tuple[int, float]:
    path_len = len(candidate.get("path", []) or candidate.get("history", {}).get("applied_strategy_ids", []) or [])
    gflops = candidate_gflops(candidate)
    return (path_len, gflops if gflops is not None else -1.0)


def select_stage_survivors(states: list[dict[str, Any]], top_k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select a quality/diversity stage frontier with Pareto filtering and MMR."""
    if top_k <= 0 or len(states) <= top_k:
        return list(states), []
    features = [_stage_state_features(state) for state in states]
    measured = [item for item in features if item["performance"] is not None]
    if measured:
        best = max(item["performance"] for item in measured)
        eligible = [
            item for item in features
            if item["performance"] is None
            or item["performance"] >= best * STAGE_PERFORMANCE_FLOOR_RATIO
        ]
    else:
        eligible = features
    pareto = [
        item for item in eligible
        if not any(_dominates(other, item) for other in eligible if other is not item)
    ]
    quality_values = [item["quality"] for item in eligible]
    quality_min = min(quality_values)
    quality_span = max(quality_values) - quality_min
    for item in eligible:
        item["normalized_quality"] = (
            (item["quality"] - quality_min) / quality_span if quality_span > 0 else 1.0
        )
    pool = pareto + [item for item in eligible if item not in pareto]
    selected = []
    while pool and len(selected) < top_k:
        if not selected:
            choice = max(pool, key=lambda item: (item["normalized_quality"], -item["mean_rank"]))
        else:
            # Reserve distinct Block regimes before spending the remaining
            # construction budget on siblings of the same Block configuration.
            represented = {item['signature'][:3] for item in selected}
            distinct = [item for item in pool
                        if all(value is not None for value in item['signature'][:3])
                        and item['signature'][:3] not in represented]
            choice = max(
                distinct or pool,
                key=lambda item: (
                    STAGE_MMR_LAMBDA * item["normalized_quality"]
                    + (1.0 - STAGE_MMR_LAMBDA)
                    * min(_structure_distance(item["signature"], kept["signature"]) for kept in selected),
                    -item["mean_rank"],
                ),
            )
        selected.append(choice)
        pool.remove(choice)
    kept = [item["state"] for item in selected]
    removed = [state for state in states if state not in kept]
    for index, item in enumerate(selected):
        item["state"]["survivor_role"] = "best_quality" if index == 0 else "diversity_preserved"
        item["state"]["stage_selection_metrics"] = {
            key: value for key, value in item.items() if key not in {"state", "signature"}
        }
    return kept, removed


def _stage_state_features(state: dict[str, Any]) -> dict[str, Any]:
    ir = state.get("current_ir", {}) or {}
    performance = _float_or_none(ir_get(ir, "performance.gflops_trimmed_mean"))
    if performance is None:
        performance = _float_or_none(ir_get(ir, "performance.gflops_mean"))
    if performance is None:
        performance = _float_or_none(ir_get(ir, "performance.gflops"))
    if performance is not None and performance <= 0:
        performance = None
    architecture = _float_or_none(ir_get(ir, "resource.tiling_candidate.architecture_score")) or 0.0
    registers = _float_or_none(ir_get(ir, "resource.registers.estimated_per_thread"))
    if registers is None:
        registers = _float_or_none(ir_get(ir, "resource.estimated_registers_per_thread")) or 0.0
    shared = _float_or_none(ir_get(ir, "resource.shared_memory.total_bytes")) or 0.0
    opportunities = len(ir_get(ir, "strategy.future_compatible_strategy_ids", []) or [])
    ranks = [int_or_default(value, 1) for value in state.get("path_code", [])]
    mean_rank = sum(ranks) / len(ranks) if ranks else 1.0
    quality = math.log1p(max(performance, 0.0)) if performance is not None else architecture / 20.0
    quality += min(opportunities, 10) * 0.03
    quality -= min(registers, 255) / 2550.0 + min(shared, 98304) / 983040.0
    return {
        "state": state,
        "performance": performance,
        "architecture": architecture,
        "registers": registers,
        "shared": shared,
        "opportunities": opportunities,
        "mean_rank": mean_rank,
        "quality": quality,
        "signature": _stage_structure_signature(ir),
    }


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_perf = left["performance"] if left["performance"] is not None else left["architecture"]
    right_perf = right["performance"] if right["performance"] is not None else right["architecture"]
    not_worse = (
        left_perf >= right_perf
        and left["opportunities"] >= right["opportunities"]
        and left["registers"] <= right["registers"]
        and left["shared"] <= right["shared"]
    )
    strictly_better = (
        left_perf > right_perf
        or left["opportunities"] > right["opportunities"]
        or left["registers"] < right["registers"]
        or left["shared"] < right["shared"]
    )
    return not_worse and strictly_better


def _stage_structure_signature(ir: dict[str, Any]) -> tuple[Any, ...]:
    paths = (
        "tiling.block_m", "tiling.block_n", "tiling.block_k",
        "tiling.warp_tile.warp_m", "tiling.warp_tile.warp_n",
        "tiling.thread_tile.thread_m", "tiling.thread_tile.thread_n",
        "layout.shared_A", "layout.shared_B", "pipeline.mode",
        "vectorization.A.vector_width", "vectorization.B.vector_width", "vectorization.C.vector_width",
    )
    return tuple(ir_get(ir, path) for path in paths)


def _structure_distance(left: tuple[Any, ...], right: tuple[Any, ...]) -> float:
    comparable = [(a, b) for a, b in zip(left, right) if a is not None or b is not None]
    if not comparable:
        return 0.0
    return sum(a != b for a, b in comparable) / len(comparable)


def build_phase1_compile_shortlist(
    completed_states: list[dict[str, Any]],
    fallback_states: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Keep one global compile budget, preferring fully constructed unique paths."""
    if top_k <= 0:
        return [*completed_states, *fallback_states]
    unique: dict[tuple[str, ...], dict[str, Any]] = {}
    for state in [*completed_states, *fallback_states]:
        path_key = tuple(state.get("path", []) or [])
        if not path_key:
            continue
        unique.setdefault(path_key, state)

    completed_keys = {tuple(state.get("path", []) or []) for state in completed_states}

    def rank(state: dict[str, Any]) -> tuple[Any, ...]:
        path = tuple(state.get("path", []) or [])
        architecture_score = ir_get(
            state.get("current_ir", {}),
            "resource.tiling_candidate.architecture_score",
            float("-inf"),
        )
        try:
            architecture_score = float(architecture_score)
        except (TypeError, ValueError):
            architecture_score = float("-inf")
        path_ranks = tuple(int_or_default(value, 1) for value in state.get("path_code", []) or [])
        return (
            0 if path in completed_keys else 1,
            -len(path),
            -architecture_score,
            path_ranks,
        )

    ordered = sorted(unique.values(), key=rank)
    shortlist: list[dict[str, Any]] = []
    selected_paths: set[tuple[str, ...]] = set()

    # Keep materially different tile regimes alive until target-device
    # measurement. A scalar resource score is not reliable enough to discard
    # every high-reuse candidate before compilation.
    for tiling_class in ("high_parallel", "balanced", "high_reuse"):
        representative = next(
            (state for state in ordered if phase1_tiling_class(state) == tiling_class),
            None,
        )
        if representative is None:
            continue
        path = tuple(representative.get("path", []) or [])
        shortlist.append(representative)
        selected_paths.add(path)
        if len(shortlist) >= top_k:
            return shortlist

    for state in ordered:
        path = tuple(state.get("path", []) or [])
        if path in selected_paths:
            continue
        shortlist.append(state)
        selected_paths.add(path)
        if len(shortlist) >= top_k:
            break
    return shortlist


def phase1_tiling_class(state: dict[str, Any]) -> str:
    ir = state.get("current_ir", {}) or {}
    bm = int_or_default(ir_get(ir, "tiling.block_m"), 0)
    bn = int_or_default(ir_get(ir, "tiling.block_n"), 0)
    area = bm * bn
    if area < 4096:
        return "high_parallel"
    if area < 8192:
        return "balanced"
    return "high_reuse"


def frontier_state_score(state: dict[str, Any]) -> float:
    candidate = state.get("last_candidate")
    score = candidate_gflops(candidate)
    if score is not None:
        return score
    architecture_score = ir_get(
        state.get("current_ir", {}),
        "resource.tiling_candidate.architecture_score",
    )
    try:
        if architecture_score is not None:
            return float(architecture_score)
    except (TypeError, ValueError):
        pass
    try:
        return float(state.get("current_ir", {}).get("performance", {}).get("gflops"))
    except (TypeError, ValueError):
        return -1.0


def target_backend(ir: dict[str, Any] | None) -> str:
    target = (ir or {}).get("target", {}) if isinstance(ir, dict) else {}
    backend = target.get("backend") or target.get("device")
    if isinstance(backend, str) and backend.lower() in {"cpu", "c"}:
        return "cpu"
    return "cuda"


def backend_code_root(ir: dict[str, Any] | None) -> Path:
    return Path(DEFAULT_CPU_CODE_ROOT) if target_backend(ir) == "cpu" else Path(DEFAULT_CODE_ROOT)


def backend_generated_code_root(ir: dict[str, Any] | None) -> Path:
    return Path(DEFAULT_GENERATED_CPU_CODE_ROOT) if target_backend(ir) == "cpu" else Path(DEFAULT_GENERATED_CODE_ROOT)


def backend_code_files(ir: dict[str, Any] | None) -> list[str]:
    return list(CPU_CODE_FILES) if target_backend(ir) == "cpu" else list(DEFAULT_CODE_FILES)


def backend_llm_context_files(ir: dict[str, Any] | None) -> list[str]:
    return list(CPU_CODE_FILES) if target_backend(ir) == "cpu" else list(DEFAULT_LLM_CONTEXT_FILES)


def backend_ast_files(ir: dict[str, Any] | None) -> list[str]:
    return list(CPU_CODE_FILES) if target_backend(ir) == "cpu" else list(DEFAULT_AST_CODE_FILES)


def backend_patch_prompt(ir: dict[str, Any] | None) -> Path:
    return Path(DEFAULT_CPU_PATCH_PROMPT) if target_backend(ir) == "cpu" else Path(DEFAULT_PATCH_PROMPT)


def backend_patch_to_code_prompt(ir: dict[str, Any] | None) -> Path:
    return Path(DEFAULT_CPU_PATCH_TO_CODE_PROMPT) if target_backend(ir) == "cpu" else Path(DEFAULT_PATCH_TO_CODE_PROMPT)


def extract_cpu_code_ast(code_root: Path) -> dict[str, Any]:
    files = {}
    for relative_path in CPU_CODE_FILES:
        path = code_root / relative_path
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8")
        files[relative_path] = {
            "line_count": len(content.splitlines()),
            "functions": re.findall(r"\b(?:void|int|float|double)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", content),
            "for_loop_count": len(re.findall(r"\bfor\s*\(", content)),
            "has_multiply_accumulate": bool(re.search(r"\+=\s*[^;]*\*\s*[^;]*;", strip_cpp_comments(content))),
            "has_cuda_tokens": bool(
                re.search(
                    r"\b(?:__global__|__device__|__shared__|threadIdx|blockIdx|cuda[A-Za-z0-9_]*|cublas[A-Za-z0-9_]*)\b|<<<",
                    content,
                )
            ),
        }
    return {
        "ast_kind": "sgpo_lightweight_cpu_c_ast",
        "backend": "cpu",
        "code_root": str(code_root),
        "files": files,
    }


def snapshot_source_files(code_root: Path, code_files: list[str] | None = None) -> dict[str, str]:
    snapshot = {}
    for relative_path in code_files or DEFAULT_CODE_FILES:
        path = code_root / relative_path
        if path.exists():
            snapshot[relative_path] = path.read_text(encoding="utf-8")
    return snapshot


def restore_source_files(code_root: Path, snapshot: dict[str, str]) -> None:
    for relative_path, content in snapshot.items():
        path = code_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def candidate_code_path(stage: str, strategy_id: str, attempt: int, namespace: str | None = None) -> Path:
    parts = [safe_name(stage)]
    if namespace:
        parts.append(safe_name(namespace))
    parts.extend([safe_name(strategy_id), f"attempt_{attempt}"])
    return Path(DEFAULT_GENERATED_CODE_ROOT) / ".".join(parts)


def archive_accepted_code(
    code_dir: Path,
    stage: str,
    strategy_id: str,
    attempt: int,
    verified_ir: dict[str, Any],
) -> None:
    manifest = {
        "status": "accepted",
        "stage": stage,
        "strategy_id": strategy_id,
        "attempt": attempt,
        "code_dir": str(code_dir),
        "verification": verified_ir.get("verification", {}).get("summary", {}),
        "performance": verified_ir.get("performance", {}),
        "files": [
            relative_path
            for relative_path in backend_code_files(verified_ir)
            if (code_dir / relative_path).exists()
        ],
    }
    save_json(code_dir / "accepted_candidate.json", manifest)


def archive_generated_chain_step(
    code_dir: Path,
    stage: str,
    strategy_id: str,
    attempt: int,
    verified_ir: dict[str, Any],
) -> None:
    manifest = {
        "status": "generated_chain_step",
        "stage": stage,
        "strategy_id": strategy_id,
        "attempt": attempt,
        "code_dir": str(code_dir),
        "verification": verified_ir.get("verification", {}).get("summary", {}),
        "chain_step": verified_ir.get("chain_step", {}),
        "files": [
            relative_path
            for relative_path in backend_code_files(verified_ir)
            if (code_dir / relative_path).exists()
        ],
    }
    save_json(code_dir / "generated_chain_step.json", manifest)


def collect_final_code(
    code_root: Path,
    ir: dict[str, Any],
    history: dict[str, Any],
    stage_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    files = []
    for relative_path in backend_code_files(ir):
        path = code_root / relative_path
        if path.exists():
            files.append(
                {
                    "path": relative_path,
                    "content": path.read_text(encoding="utf-8"),
                }
            )
    return {
        "generation_method": "stagewise_strategy_candidate_search",
        "applied_strategy_ids": history.get("applied_strategy_ids", []),
        "failed_strategy_counts": history.get("failed_strategy_counts", {}),
        "final_selected_strategy_id": ir.get("strategy", {}).get("final_selected_strategy_id"),
        "final_selected_code_dir": ir.get("strategy", {}).get("final_selected_code_dir"),
        "stage_summaries": stage_summaries,
        "verification": ir.get("verification", {}),
        "performance": ir.get("performance", {}),
        "files": files,
    }


def collect_final_code_from_snapshot(
    source_snapshot: dict[str, str],
    ir: dict[str, Any],
    history: dict[str, Any],
    stage_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    files = [
        {
            "path": relative_path,
            "content": content,
        }
        for relative_path, content in (source_snapshot or {}).items()
        if relative_path in backend_code_files(ir)
    ]
    return {
        "generation_method": "stagewise_strategy_candidate_search",
        "applied_strategy_ids": history.get("applied_strategy_ids", []),
        "failed_strategy_counts": history.get("failed_strategy_counts", {}),
        "final_selected_strategy_id": ir.get("strategy", {}).get("final_selected_strategy_id"),
        "final_selected_code_dir": ir.get("strategy", {}).get("final_selected_code_dir"),
        "stage_summaries": stage_summaries,
        "verification": ir.get("verification", {}),
        "performance": ir.get("performance", {}),
        "files": files,
    }


def write_final_code_bundle(path: Path, final_code: dict[str, Any]) -> None:
    lines = [
        "// SGPO final generated code bundle",
        "// This file is for inspection. Build uses the original file layout.",
        "",
    ]
    for item in final_code.get("files", []):
        lines.extend(
            [
                f"// ===== BEGIN {item['path']} =====",
                item["content"].rstrip(),
                f"// ===== END {item['path']} =====",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def materialize_final_generated_files(
    code_root: Path,
    source_snapshot: dict[str, str],
    ir: dict[str, Any],
) -> list[str]:
    """Publish the selected generated kernel without replacing the fixed benchmark driver."""
    published = []
    fixed_driver_names = {"main.cpp", "main.c"}
    allowed_files = set(backend_code_files(ir))
    for relative_path, content in (source_snapshot or {}).items():
        if relative_path not in allowed_files or Path(relative_path).name in fixed_driver_names:
            continue
        destination = code_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        published.append(relative_path)
    return published


def summarize_terminal_results_for_console(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "rank",
        "chain_id",
        "source_phase",
        "compile_status",
        "correctness_status",
        "runtime_safety_status",
        "latency_ms_mean",
        "gflops_mean",
        "cublas_gflops_mean",
        "relative_to_cublas",
        "code_dir",
    )
    return [{key: item.get(key) for key in keys} for item in results]


def save_candidate_json(
    path: Path,
    stage: str,
    strategy_id: str,
    attempt: int,
    data: dict[str, Any],
    namespace: str | None = None,
) -> None:
    candidate_output = candidate_path(path, stage, strategy_id, attempt, namespace)
    save_json(candidate_output, data)
    publish_latest_json(path, data)


def publish_latest_json(path: Path, data: dict[str, Any]) -> None:
    if not defer_latest(path, data):
        save_json(path, data)


def stage_path(path: Path, stage: str) -> Path:
    namespace = chain_namespace()
    suffix = f".{safe_name(namespace)}" if namespace else ""
    stem = compact_artifact_stem(f"{path.stem}.{safe_name(stage)}{suffix}")
    return path.with_name(f"{stem}{path.suffix}")


def candidate_path(path: Path, stage: str, strategy_id: str, attempt: int, namespace: str | None = None) -> Path:
    if namespace:
        parts = [path.stem, safe_name(stage), safe_name(namespace), f"attempt_{attempt}"]
    else:
        parts = [path.stem, safe_name(stage), safe_name(strategy_id), f"attempt_{attempt}"]
    stem = compact_artifact_stem(".".join(parts))
    return path.with_name(f"{stem}{path.suffix}")


def compact_artifact_stem(stem: str, limit: int = MAX_ARTIFACT_STEM_LENGTH) -> str:
    if len(stem) <= limit:
        return stem
    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:12]
    head_limit = max(24, limit - len(digest) - 3)
    return f"{stem[:head_limit]}__{digest}"


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)


if __name__ == "__main__":
    main()
