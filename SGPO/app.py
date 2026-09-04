from __future__ import annotations

import argparse
import json
import copy
import hashlib
import re
from pathlib import Path
from typing import Any

from SGPO.diagnosis.defect_diagnosis import attach_diagnosis, diagnose_defects
from SGPO.generate_ir.ir_checker import check_code_verification
from SGPO.generate_ir.ir_extraction import build_extracted_ir
from SGPO.generate_ir.strategy_filter import filter_strategies_by_preconditions
from SGPO.generate_ir.strategy_index_filter import filter_strategy_index, infer_stage_order_from_index
from SGPO.generate_ir.strategy_library_merge import load_merged_strategy_documents
from SGPO.generate_ir.stage_controller import StageController, synthesize_ir_updates
from SGPO.llm.openai_client import OpenAICompatibleClient
from SGPO.llm.patch_generator import (
    build_patch_ir,
    generate_patch_with_llm,
    load_code_context,
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
from SGPO.llm.strategy_selector import get_micro_strategy_from_llm
from SGPO.utils.common_utils import load_config, load_json, save_json
from SGPO.verification.build_run_verifier import verify_build_and_run
from SGPO.verification.c_build_run_verifier import verify_cpu_build_and_run
from SGPO.verification.code_ast_extractor import extract_code_ast
from SGPO.verification.gemm_semantic_checker import check_gemm_semantic_obligations
from SGPO.verification.gemm_semantic_repair import generate_semantic_repair_candidate
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
DEFAULT_PROMPT = ROOT / "llm" / "prompts" / "get_strategy_prompt.txt"
DEFAULT_MICRO_STRATEGY_PROMPT = ROOT / "llm" / "prompts" / "get_micro_strategy_prompt.txt"
DEFAULT_PATCH_PROMPT = ROOT / "llm" / "prompts" / "generate_patch_prompt.txt"
DEFAULT_PATCH_TO_CODE_PROMPT = ROOT / "llm" / "prompts" / "generate_concrete_code_prompt.txt"
DEFAULT_CPU_PATCH_PROMPT = ROOT / "llm" / "prompts" / "generate_cpu_c_patch_prompt.txt"
DEFAULT_CPU_PATCH_TO_CODE_PROMPT = ROOT / "llm" / "prompts" / "generate_cpu_c_code_prompt.txt"
DEFAULT_CONFIG = ROOT / "conf.yaml"
DEFAULT_MAX_REPAIR_ATTEMPTS = 3
DEFAULT_STRATEGY_PROFILE = "stable_correctness_first"
DEFAULT_UNLOCKED_PERFORMANCE_PROFILE = "throughput_exploration"
DEFAULT_SELECTION_MODE = "top3_beam_search_graph_filtered"
DEFAULT_MAX_FRONTIER_STATES_PER_STAGE = 3
DEFAULT_SINGLE_PATH_MODE = False
DEFAULT_TOP_K_STRATEGIES_PER_SUBPHASE = 3
DEFAULT_TOP_K_FINAL_RESULTS = 3
MAX_ARTIFACT_STEM_LENGTH = 140
STABLE_BASELINE_STAGES = ["Tiling", "Layout", "MappingReordering", "Vectorization", "Epilogue"]
THROUGHPUT_EXPLORATION_STAGES = ["Tiling", "Layout", "MappingReordering", "Vectorization", "Pipeline", "Epilogue"]
PERFORMANCE_UNLOCK_SUCCESS_STATUSES = {"pass"}
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
    description = DEFAULT_DESCRIPTION
    user_question = DEFAULT_DESCRIPTION

    template = load_json(Path(DEFAULT_TEMPLATE))
    current_ir = build_extracted_ir(template, description)
    save_json(Path(DEFAULT_IR_OUTPUT), current_ir)

    raw_strategy_index, strategy_library = load_strategy_documents(current_ir)
    dependency_graph = load_dependency_graph(current_ir)
    strategy_profile = infer_strategy_profile(current_ir, raw_strategy_index)
    current_ir.setdefault("strategy", {})["profile"] = strategy_profile
    stage_order = infer_app_stage_order(current_ir, dependency_graph, raw_strategy_index, strategy_library)
    code_root = backend_code_root(current_ir)
    code_files = backend_code_files(current_ir)

    config = load_config(Path(DEFAULT_CONFIG))
    build_platform = resolve_build_platform(args, config)
    client = OpenAICompatibleClient(config["llm"])
    search_config = config.get("search", {}) or {}
    performance_config = config.get("performance_unlock", {}) or {}
    max_frontier_states = 1 if DEFAULT_SINGLE_PATH_MODE else int(
        search_config.get("max_frontier_states_per_stage", DEFAULT_MAX_FRONTIER_STATES_PER_STAGE) or 0
    )
    cpu_unlock_enabled = performance_config.get("enable_cpu", True)
    auto_unlock_performance = bool(performance_config.get("enabled", False)) and (
        target_backend(current_ir) != "cpu" or bool(cpu_unlock_enabled)
    )
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
        stage_result = run_exhaustive_stage(
            stage=stage,
            frontier=frontier,
            raw_strategy_index=raw_strategy_index,
            dependency_graph=dependency_graph,
            strategy_library=strategy_library,
            client=client,
            user_question=user_question,
            profile=strategy_profile,
            max_frontier_states=max_frontier_states,
            exhaust_pending=stage == stage_order[-1],
        )
        stage_summaries.append(stage_result["summary"])
        best_partial_candidate = choose_better_partial_candidate(
            best_partial_candidate,
            choose_best_candidate(stage_result.get("accepted_candidates", [])),
        )
        frontier = stage_result["frontier"]

    baseline_terminal_verification = verify_terminal_chains(
        frontier=frontier,
        strategy_library=strategy_library,
        build_platform=build_platform,
    )
    terminal_verification = baseline_terminal_verification
    performance_phase = None
    if auto_unlock_performance and terminal_verification_has_correct_chain(baseline_terminal_verification):
        performance_phase = run_unlocked_performance_phase(
            initial_ir=current_ir,
            initial_source_snapshot=initial_source_snapshot,
            raw_strategy_index=raw_strategy_index,
            dependency_graph=dependency_graph,
            strategy_library=strategy_library,
            client=client,
            user_question=user_question,
            profile=unlocked_profile,
            max_frontier_states=max_frontier_states,
            build_platform=build_platform,
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
    restore_source_files(code_root, final_source_snapshot)

    final_code = collect_final_code(code_root, final_ir, final_history, stage_summaries)
    save_json(Path(DEFAULT_FINAL_CODE_OUTPUT), final_code)
    write_final_code_bundle(Path(DEFAULT_FINAL_CODE_BUNDLE_OUTPUT), final_code)
    save_json(Path(DEFAULT_EVOLUTION_OUTPUT), {
        "selection_mode": DEFAULT_SELECTION_MODE,
        "stage_order": stage_order,
        "strategy_profile": strategy_profile,
        "performance_unlock": {
            "enabled": auto_unlock_performance,
            "build_platform": build_platform,
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
                "top_k_strategies_per_subphase": DEFAULT_TOP_K_STRATEGIES_PER_SUBPHASE,
                "top_k_final_results": DEFAULT_TOP_K_FINAL_RESULTS,
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
                "top_3_terminal_results": top_terminal_results,
                "applied_strategy_ids": final_history.get("applied_strategy_ids", []),
                "failed_strategy_counts": final_history.get("failed_strategy_counts", {}),
                "stage_summaries": stage_summaries,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SGPO GEMM strategy-guided optimization.")
    parser.add_argument(
        "--build-platform",
        choices=["windows", "linux"],
        default=None,
        help="Compilation/runtime platform for generated code. Defaults to build.platform in conf.yaml, then windows.",
    )
    return parser.parse_args()


def resolve_build_platform(args: argparse.Namespace, config: dict[str, Any]) -> str:
    configured = (config.get("build", {}) or {}).get("platform")
    value = args.build_platform or configured or "windows"
    value = str(value).strip().lower()
    return "linux" if value in {"linux", "unix", "posix"} else "windows"


def infer_strategy_profile(ir: dict[str, Any], strategy_index: dict[str, Any]) -> str:
    return (
        ir.get("strategy", {}).get("profile")
        or ir.get("precision", {}).get("profile")
        or DEFAULT_STRATEGY_PROFILE
        or strategy_index.get("target", {}).get("default_profile")
    )


def run_unlocked_performance_phase(
    initial_ir: dict[str, Any],
    initial_source_snapshot: dict[str, str],
    raw_strategy_index: dict[str, Any],
    dependency_graph: dict[str, Any],
    strategy_library: dict[str, Any],
    client: OpenAICompatibleClient,
    user_question: str,
    profile: str,
    max_frontier_states: int,
    build_platform: str,
) -> dict[str, Any]:
    phase_ir = copy.deepcopy(initial_ir)
    phase_ir.setdefault("strategy", {})["profile"] = profile
    phase_stage_order = infer_app_stage_order(phase_ir, dependency_graph, raw_strategy_index, strategy_library)
    frontier = [
        make_frontier_state(
            state_id=f"{safe_name(profile)}_root",
            current_ir=phase_ir,
            source_snapshot=initial_source_snapshot,
            history={
                "applied_strategy_ids": [],
                "applied_micro_strategies": [],
                "failed_strategy_counts": {},
                "completed_subphases": [],
                "events": [
                    {
                        "stage": "PerformanceUnlock",
                        "status": "started",
                        "profile": profile,
                        "reason": "stable correctness chain passed; performance constraints are unlocked",
                    }
                ],
            },
            path=[],
            path_code=[9],
        )
    ]
    stage_summaries = []
    for stage in phase_stage_order:
        if not frontier:
            stage_summaries.append(
                {
                    "stage": stage,
                    "selection_mode": DEFAULT_SELECTION_MODE,
                    "input_frontier_count": 0,
                    "output_frontier_count": 0,
                    "accepted_candidate_count": 0,
                    "status": "skipped",
                    "reason": "performance-unlock frontier is empty",
                    "profile": profile,
                }
            )
            continue
        stage_result = run_exhaustive_stage(
            stage=stage,
            frontier=frontier,
            raw_strategy_index=raw_strategy_index,
            dependency_graph=dependency_graph,
            strategy_library=strategy_library,
            client=client,
            user_question=user_question,
            profile=profile,
            max_frontier_states=max_frontier_states,
            exhaust_pending=stage == phase_stage_order[-1],
        )
        stage_result["summary"]["profile"] = profile
        stage_result["summary"]["phase"] = "performance_unlock"
        stage_summaries.append(stage_result["summary"])
        frontier = stage_result["frontier"]

    terminal_verification = verify_terminal_chains(
        frontier=frontier,
        strategy_library=strategy_library,
        build_platform=build_platform,
    )
    terminal_verification["summary"]["profile"] = profile
    terminal_verification["summary"]["phase"] = "performance_unlock"
    stage_summaries.append(terminal_verification["summary"])
    return {
        "profile": profile,
        "stage_order": phase_stage_order,
        "frontier": frontier,
        "terminal_verification": terminal_verification,
        "best_candidate": terminal_verification.get("best_candidate"),
        "stage_summaries": stage_summaries,
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
    if best_candidate is None:
        best_candidate = choose_best_candidate(merged_candidates)
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
    explicit = ir.get("strategy", {}).get("stage_order") or dependency_graph.get("stage_order")
    if isinstance(explicit, list) and explicit:
        filtered_explicit = [stage for stage in explicit if stage_has_strategy(stage, strategy_index, strategy_library)]
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
) -> dict[str, Any]:
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
        stage_report = controller.verify_stage(current_ir)
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
            "failed_strategy_counts": {},
            "stage_completed_ir": current_ir if stage_report["accepted"] else None,
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

    micro_selection = get_micro_strategy_from_llm(
        client=client,
        current_stage=stage,
        current_subphase=current_subphase_id,
        subphase=current_subphase or {},
        optir_summary=compact_ir_summary(current_ir),
        strategy_index=available_strategy_index,
        prompt_path=Path(DEFAULT_MICRO_STRATEGY_PROMPT),
    )
    micro_selection = inject_profile_preferred_candidates(
        micro_selection,
        available_strategy_index,
        current_subphase_id,
        profile,
    )
    execution_candidates = select_top_k_candidates(
        micro_selection.get("candidates", []),
        DEFAULT_TOP_K_STRATEGIES_PER_SUBPHASE,
    )
    selected_strategy = {"candidates": execution_candidates}
    llm_selected_count = len(selected_strategy.get("candidates", []))
    selected_strategy["filter_context"] = available_strategy_index.get("filter_context")
    selected_strategy["selection_mode"] = DEFAULT_SELECTION_MODE
    selected_strategy["top_k_strategies_per_subphase"] = DEFAULT_TOP_K_STRATEGIES_PER_SUBPHASE
    selected_strategy["current_subphase"] = current_subphase_id
    selected_strategy["micro_selection"] = micro_selection
    selected_strategy["all_llm_candidates"] = micro_selection.get("candidates", [])
    save_json(stage_path(DEFAULT_SELECTED_STRATEGY_OUTPUT, current_subphase_id or stage), selected_strategy)
    save_json(Path(DEFAULT_SELECTED_STRATEGY_OUTPUT), selected_strategy)
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
        strategy["ir_updates"] = merge_expected_ir_updates(
            strategy_id,
            strategy.get("ir_updates") or {},
            expected_updates if isinstance(expected_updates, dict) else {},
        )
        local_ir = controller.apply_micro_strategy(
            strategy_id,
            strategy["ir_updates"],
        )
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
        )
        candidate_results.append(candidate_result)
        events.append(candidate_result["event"])
        if not candidate_result["accepted"]:
            failed_counts[strategy_id] = failed_counts.get(strategy_id, 0) + 1

    accepted_candidates = [item for item in candidate_results if item["accepted"]]
    best_candidate = choose_best_candidate(accepted_candidates)

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
) -> dict[str, Any]:
    next_frontier = []
    active_frontier = list(frontier)
    pending_frontier: list[dict[str, Any]] = []
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
        for state in active_frontier:
            stage_result = run_stage(
                stage=stage,
                current_ir=state["current_ir"],
                history=state["history"],
                raw_strategy_index=raw_strategy_index,
                dependency_graph=dependency_graph,
                strategy_library=strategy_library,
                client=client,
                user_question=user_question,
                profile=profile,
                base_source_snapshot=state["source_snapshot"],
                candidate_namespace=chain_code_key(state.get("path_code", [])),
                parent_path=state.get("path", []),
                parent_path_code=state.get("path_code", []),
            )
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
                completed_path_key = tuple(state.get("path", []))
                if completed_path_key not in output_paths:
                    output_paths.add(completed_path_key)
                    next_frontier.append(
                        make_frontier_state(
                            state_id=f"{state['state_id']}->complete_{safe_name(stage)}",
                            current_ir=stage_result["stage_completed_ir"],
                            source_snapshot=state["source_snapshot"],
                            history=completed_history,
                            path=list(state.get("path", [])),
                            path_code=list(state.get("path_code", [])),
                            last_candidate=state.get("last_candidate"),
                            code_dir=state.get("code_dir"),
                        )
                    )
                continue

            if not stage_result.get("accepted_candidates"):
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

    if not exhaust_pending and max_frontier_states > 0 and len(next_frontier) > max_frontier_states:
        next_frontier = sorted(next_frontier, key=frontier_state_score, reverse=True)[:max_frontier_states]

    return {
        "frontier": next_frontier,
        "accepted_candidates": accepted_candidates,
        "events": total_events,
        "failed_strategy_counts": failed_counts,
        "summary": {
            "stage": stage,
            "selection_mode": DEFAULT_SELECTION_MODE,
            "input_frontier_count": len(frontier),
            "output_frontier_count": len(next_frontier),
            "accepted_candidate_count": len(accepted_candidates),
            "best_stage_strategy_id": choose_best_candidate(accepted_candidates)["strategy_id"]
            if accepted_candidates
            else None,
            "best_stage_gflops": candidate_gflops(choose_best_candidate(accepted_candidates)),
            "max_frontier_states": max_frontier_states or None,
            "exhaust_pending": exhaust_pending,
            "pending_frontier_count": len(pending_frontier),
            "carry_reason": carry_reason,
            "state_summaries": state_summaries,
        },
    }


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
) -> dict[str, Any]:
    strategy_id = strategy["strategy_id"]
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
        try:
            patch_result = generate_patch_with_llm(
                client=client,
                ir=base_ir,
                strategy=strategy,
                precheck_item=precheck_item,
                code_context=load_code_context(candidate_code_dir, code_files),
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
            if target_backend(base_ir) == "cpu":
                generated_code = generate_cpu_c_code_from_patch_with_llm(
                    client=client,
                    patch_ir=patch_ir,
                    patch_result=patch_from_file,
                    strategy=strategy,
                    code_context=load_code_context(candidate_code_dir, code_files),
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
                    code_context=load_code_context(candidate_code_dir, code_files),
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
            code_ast = extract_code_ast(candidate_code_dir) if target_backend(base_ir) != "cpu" else extract_cpu_code_ast(candidate_code_dir)
            save_candidate_json(DEFAULT_CODE_AST_OUTPUT, stage, strategy_id, attempt, code_ast, candidate_namespace)
            patch_ir["code_ast"] = code_ast
            verified_ir = mark_chain_step_generated(patch_ir, code_apply_result)

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


def select_top_k_candidates(candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if not candidates or top_k <= 0:
        return []
    indexed = [(index, item) for index, item in enumerate(candidates)]
    indexed.sort(key=lambda pair: candidate_selection_score(pair[1], pair[0]), reverse=True)
    return [copy.deepcopy(item) for _, item in indexed[:top_k]]


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
        return True
    return oracle_verification_passed(verified_ir)


def oracle_verification_passed(verified_ir: dict[str, Any]) -> bool:
    return verified_ir.get("verification", {}).get("accepted") is True


def mark_chain_step_generated(ir: dict[str, Any], code_apply_result: dict[str, Any]) -> dict[str, Any]:
    next_ir = copy.deepcopy(ir)
    next_ir["chain_step"] = {
        "status": "generated",
        "verification_deferred": True,
        "reason": "compile/correctness/runtime/performance verification runs after the full strategy chain is generated",
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
            "Regenerate the same candidate with strict JSON. For CPU codegen edits, "
            "prefer replacement_lines array entries instead of one multiline replacement string."
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
    value = candidate.get("verified_ir", {}).get("performance", {}).get("gflops")
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
    return {
        "stage": stage,
        "strategy_id": strategy_id,
        "status": status,
        "attempt_count": len(attempts),
        "candidate_code_dir": attempts[-1].get("candidate_code_dir") if attempts else None,
        "verification": (verified_ir or {}).get("verification", {}).get("summary", {}),
        "gflops": (verified_ir or {}).get("performance", {}).get("gflops"),
    }


def verify_terminal_chains(
    frontier: list[dict[str, Any]],
    strategy_library: dict[str, Any],
    build_platform: str = "windows",
) -> dict[str, Any]:
    verified_candidates = []
    terminal_summaries = []
    best_candidate = None
    best_state = None

    for index, state in enumerate(frontier, start=1):
        if not state.get("path"):
            continue
        chain_id = chain_id_for_state(index, state)
        chain_dir = Path(
            state.get("code_dir")
            or chain_code_path(state.get("path", []), state.get("path_code", []), state.get("current_ir", {}))
        )
        if not chain_dir.exists():
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
        )
        repair_attempts = []
        for repair_attempt in range(1, DEFAULT_MAX_REPAIR_ATTEMPTS + 1):
            if terminal_chain_is_accepted(verified_ir):
                break
            diagnosis = verified_ir.get("defect_diagnosis", {})
            repair_result = generate_semantic_repair_candidate(
                source_chain_dir=chain_dir,
                output_chain_dir=chain_dir,
                diagnosis=diagnosis,
                ir=verified_ir,
            )
            repair_result["repair_attempt"] = repair_attempt
            save_json(chain_dir / f"semantic_repair_attempt_{repair_attempt}.json", repair_result)
            repair_attempts.append(repair_result)
            verified_ir = verify_terminal_chain_once(
                terminal_ir=verified_ir,
                chain_dir=chain_dir,
                chain_id=chain_id,
                strategy_library=strategy_library,
                attempt=repair_attempt + 1,
                build_platform=build_platform,
            )
        if repair_attempts:
            verified_ir.setdefault("terminal_repair", {})["attempts"] = repair_attempts
            verified_ir.setdefault("terminal_repair", {})["attempt_count"] = len(repair_attempts)

        manifest = {
            "status": "verified_terminal_chain",
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
    runnable_correct = [candidate for candidate in candidates if candidate_has_correctness_pass(candidate)]
    accepted = [candidate for candidate in candidates if candidate.get("accepted")]
    ranked_pool = accepted or runnable_correct or candidates
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
                "code_verification_status": verification_summary.get("code_verification_status"),
                "checker_failure_is_terminal": verification_summary.get("checker_failure_is_terminal"),
                "code_checker_advisory_failure": verification_summary.get("code_checker_advisory_failure"),
                "latency_ms": performance.get("latency_ms") or verification_summary.get("latency_ms"),
                "gflops": performance.get("gflops") or verification_summary.get("gflops"),
                "relative_to_cublas": performance.get("relative_to_cublas"),
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
) -> dict[str, Any]:
    current_ir = copy.deepcopy(terminal_ir)
    code_ast = extract_code_ast(chain_dir) if target_backend(current_ir) != "cpu" else extract_cpu_code_ast(chain_dir)
    save_json(chain_dir / "code_ast.json", code_ast)
    current_ir["code_ast"] = code_ast
    current_ir["code_completeness"] = check_terminal_code_completeness(chain_dir, current_ir)

    if target_backend(current_ir) == "cpu":
        verified_ir = verify_cpu_build_and_run(
            ir=current_ir,
            source_dir=chain_dir,
            build_dir=chain_dir / "build",
            build_platform=build_platform,
        )
    else:
        verified_ir = verify_build_and_run(
            ir=current_ir,
            source_dir=chain_dir,
            build_dir=chain_dir / "build",
            build_platform=build_platform,
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
    diagnosis = diagnose_defects(verified_ir)
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


def frontier_state_score(state: dict[str, Any]) -> float:
    candidate = state.get("last_candidate")
    score = candidate_gflops(candidate)
    if score is not None:
        return score
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
    save_json(path, data)


def stage_path(path: Path, stage: str) -> Path:
    stem = compact_artifact_stem(f"{path.stem}.{safe_name(stage)}")
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
