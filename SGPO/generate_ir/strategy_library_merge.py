from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

from SGPO.utils.common_utils import load_json, save_json


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
DEFAULT_CURRENT_INDEX = ROOT / "data" / "lib" / "strategy_index.json"
DEFAULT_CURRENT_LIBRARY = ROOT / "data" / "lib" / "strategy_library.json"
DEFAULT_BASE_INDEX = REPO_ROOT / "late_data" / "strategy_index.json"
DEFAULT_QIMENG_LIBRARY = REPO_ROOT / "late_data" / "strategy_library.json"
DEFAULT_MERGED_INDEX = ROOT / "data" / "lib" / "merged" / "strategy_index.merged.json"
DEFAULT_MERGED_LIBRARY = ROOT / "data" / "lib" / "merged" / "strategy_library.merged.json"


def load_merged_strategy_documents(
    current_index_path: Path = DEFAULT_CURRENT_INDEX,
    current_library_path: Path = DEFAULT_CURRENT_LIBRARY,
    base_index_path: Path = DEFAULT_BASE_INDEX,
    qimeng_library_path: Path = DEFAULT_QIMENG_LIBRARY,
    merged_index_output: Path = DEFAULT_MERGED_INDEX,
    merged_library_output: Path = DEFAULT_MERGED_LIBRARY,
) -> tuple[dict[str, Any], dict[str, Any]]:
    current_index = load_json(current_index_path)
    current_library = load_json(current_library_path)
    base_index = load_json(base_index_path) if base_index_path.exists() else {}
    qimeng_library = load_json(qimeng_library_path) if qimeng_library_path.exists() else {}

    merged_library = merge_strategy_libraries(current_library, base_index, qimeng_library)
    merged_index = merge_strategy_indices(current_index, base_index, merged_library)

    save_json(merged_index_output, merged_index)
    save_json(merged_library_output, merged_library)
    return merged_index, merged_library


def merge_strategy_libraries(
    current_library: dict[str, Any],
    base_index: dict[str, Any],
    qimeng_library: dict[str, Any],
) -> dict[str, Any]:
    merged = copy.deepcopy(current_library)
    merged["library_name"] = "SGPO merged GEMM strategy library"
    merged["library_version"] = "merged-current-base-qimeng"
    merged["merge_sources"] = [
        current_library.get("library_name") or "SGPO current strategy_library.json",
        base_index.get("index_name") or "late_data strategy_index.json",
        qimeng_library.get("library_name") or "late_data strategy_library.json",
    ]

    by_id: dict[str, dict[str, Any]] = {}
    for strategy in current_library.get("strategies", []) or []:
        if strategy.get("strategy_id"):
            by_id[strategy["strategy_id"]] = copy.deepcopy(strategy)

    for strategy in qimeng_library.get("strategies", []) or []:
        if strategy.get("strategy_id"):
            by_id[strategy["strategy_id"]] = merge_strategy_record(
                by_id.get(strategy["strategy_id"]),
                strategy,
            )

    for summary in base_index.get("strategies", []) or []:
        strategy_id = summary.get("strategy_id")
        if not strategy_id or strategy_id in by_id:
            continue
        by_id[strategy_id] = synthesize_strategy_from_index_summary(summary)

    merged["strategies"] = sorted(by_id.values(), key=strategy_sort_key)
    merged["strategy_count"] = len(merged["strategies"])
    merged["strategy_ids_by_stage"] = build_stage_index(merged["strategies"])
    merged["strategy_ids_by_category"] = build_category_index(merged["strategies"])
    return merged


def merge_strategy_indices(
    current_index: dict[str, Any],
    base_index: dict[str, Any],
    merged_library: dict[str, Any],
) -> dict[str, Any]:
    merged = copy.deepcopy(current_index)
    merged["index_name"] = "SGPO merged GEMM strategy index"
    merged["index_version"] = "merged-current-base-qimeng"
    merged["source_library_name"] = merged_library.get("library_name")
    merged["source_library_version"] = merged_library.get("library_version")

    summaries: dict[str, dict[str, Any]] = {}
    for source in [base_index, current_index]:
        for item in source.get("strategies", []) or []:
            strategy_id = item.get("strategy_id")
            if strategy_id:
                summaries[strategy_id] = copy.deepcopy(item)

    for strategy in merged_library.get("strategies", []) or []:
        strategy_id = strategy.get("strategy_id")
        if not strategy_id:
            continue
        summary = summaries.get(strategy_id, {})
        summaries[strategy_id] = {
            "strategy_id": strategy_id,
            "category": strategy.get("category") or summary.get("category"),
            "stage": strategy.get("stage") or summary.get("stage"),
            "name": strategy.get("name") or summary.get("name") or strategy_id,
            "short_description": summary.get("short_description") or strategy.get("intent") or strategy.get("applicable_scope", ""),
            "intent": strategy.get("intent") or summary.get("intent") or summary.get("short_description", ""),
            "when_to_use": summary.get("when_to_use") or strategy.get("applicable_scope", ""),
            "risk_summary": summary.get("risk_summary") or ", ".join(strategy.get("risk_types", []) or []),
            "priority": summary.get("priority", strategy.get("priority", 5)),
            "maturity": strategy.get("maturity") or summary.get("maturity", "v1"),
        }

    strategies = sorted(summaries.values(), key=strategy_sort_key)
    merged["strategies"] = strategies
    merged["strategy_count"] = len(strategies)
    merged["strategy_ids_by_stage"] = build_stage_index(strategies)
    merged["strategy_ids_by_category"] = build_category_index(strategies)
    merged["target"] = merge_target(current_index.get("target", {}), base_index.get("target", {}))
    merged["strategy_profiles"] = merge_strategy_profiles(
        base_index.get("strategy_profiles", {}),
        current_index.get("strategy_profiles", {}),
    )
    copy_profile_loading_fields(merged, current_index)
    add_profile_group_from_strategy_profiles(merged)
    return merged


def merge_strategy_record(existing: dict[str, Any] | None, incoming: dict[str, Any]) -> dict[str, Any]:
    if not existing:
        return copy.deepcopy(incoming)
    merged = copy.deepcopy(existing)
    for key, value in incoming.items():
        if value not in (None, "", [], {}):
            merged[key] = copy.deepcopy(value)
    return merged


def synthesize_strategy_from_index_summary(summary: dict[str, Any]) -> dict[str, Any]:
    strategy = copy.deepcopy(summary)
    strategy.setdefault("name", strategy.get("strategy_id"))
    strategy.setdefault("intent", strategy.get("short_description", ""))
    strategy.setdefault("applicable_scope", strategy.get("when_to_use", ""))
    strategy.setdefault("maturity", "v1")
    ir_updates = synthesize_ir_updates(strategy["strategy_id"])
    strategy["preconditions"] = {
        "verification_stage": "before_patch_generation",
        "input": ["OptIR_t"],
        "verifier": "ir_predicate_eval",
        "predicates": synthesize_pre_predicates(strategy["strategy_id"]),
    }
    strategy["ir_updates"] = ir_updates
    strategy["affected_fields"] = sorted(ir_updates)
    strategy["allowed_modified_regions"] = synthesize_allowed_regions(strategy["strategy_id"])
    strategy["code_requirements"] = synthesize_code_requirements(strategy["strategy_id"])
    strategy["postconditions"] = {
        "patch_ir_verification": {
            "verification_stage": "after_patch_json_before_code_application",
            "input": ["OptIR_t", "Patch.ir_updates"],
            "verifier": "ir_predicate_eval",
            "predicates": synthesize_post_predicates(ir_updates),
        },
        "code_verification": {
            "verification_stage": "after_code_patch_applied",
            "constraints": [],
        },
    }
    strategy["synthesized_from_index_summary"] = True
    return strategy


def synthesize_ir_updates(strategy_id: str) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    block = re.fullmatch(r"Tiling\.BlockTile\.(\d+)x(\d+)x(\d+)", strategy_id)
    if block:
        bm, bn, bk = [int(x) for x in block.groups()]
        updates.update(
            {
                "tiling.enabled": True,
                "tiling.block_m": bm,
                "tiling.block_n": bn,
                "tiling.block_k": bk,
            }
        )
    thread = re.fullmatch(r"Tiling\.ThreadTile\.(\d+)x(\d+)", strategy_id)
    if thread:
        tm, tn = [int(x) for x in thread.groups()]
        updates.update({"tiling.thread_m": tm, "tiling.thread_n": tn, "memory.use_register_tile": True})
    if strategy_id == "Layout.SharedMemory.AB.Basic":
        updates.update(
            {
                "memory.use_shared_memory": True,
                "memory.shared_A.enabled": True,
                "memory.shared_B.enabled": True,
            }
        )
    if strategy_id == "Layout.RegisterTile.C":
        updates["memory.use_register_tile"] = True
    if strategy_id == "Layout.SharedMemory.PaddingB.Plus1":
        updates["memory.shared_B.padding"] = 1
    if strategy_id == "Layout.SharedMemory.TransposeB":
        updates["memory.shared_B.transposed"] = True
    if strategy_id == "Vectorization.GlobalLoadA.float2":
        updates.update(vector_updates("A", 2, "float2", "vectorized_load"))
    if strategy_id == "Vectorization.GlobalLoadB.float2":
        updates.update(vector_updates("B", 2, "float2", "vectorized_load"))
    if strategy_id == "Vectorization.GlobalLoadAB.float4":
        updates.update(vector_updates("A", 4, "float4", "vectorized_load"))
        updates.update(vector_updates("B", 4, "float4", "vectorized_load"))
    if strategy_id == "Vectorization.StoreC.SafeScalar":
        updates.update({"vectorization.C.vectorized_store": False, "vectorization.C.vector_width": 1})
    if strategy_id == "Vectorization.AlignmentGuard":
        for tensor in ("A", "B", "C"):
            updates[f"vectorization.{tensor}.alignment_guard"] = True
            updates[f"vectorization.{tensor}.tail_handling"] = True
    if strategy_id == "Pipeline.DoubleBuffer.SharedAB":
        updates.update(
            {
                "pipeline.enabled": True,
                "pipeline.double_buffer": True,
                "memory.shared_A.buffers": 2,
                "memory.shared_B.buffers": 2,
            }
        )
    if strategy_id == "Pipeline.SoftwarePrefetch.RegisterA":
        updates["pipeline.software_prefetch_A"] = True
    if strategy_id == "Pipeline.NoAsyncCopy.V1":
        updates["pipeline.async_copy"] = False
    return updates


def synthesize_pre_predicates(strategy_id: str) -> list[dict[str, Any]]:
    predicates: list[dict[str, Any]] = []
    block = re.fullmatch(r"Tiling\.BlockTile\.(\d+)x(\d+)x(\d+)", strategy_id)
    if block:
        bm, bn, bk = [int(x) for x in block.groups()]
        predicates.extend(
            [
                ge_predicate("PRE_M", "problem.M", bm),
                ge_predicate("PRE_N", "problem.N", bn),
                ge_predicate("PRE_K", "problem.K", bk),
            ]
        )
    if strategy_id.startswith("Tiling.ThreadTile."):
        predicates.extend(
            [
                eq_predicate("PRE_TILING_ENABLED", "tiling.enabled", True),
                not_null_predicate("PRE_BLOCK_M", "tiling.block_m"),
                not_null_predicate("PRE_BLOCK_N", "tiling.block_n"),
                not_null_predicate("PRE_BLOCK_K", "tiling.block_k"),
            ]
        )
    if strategy_id == "Layout.SharedMemory.AB.Basic":
        predicates.extend(
            [
                eq_predicate("PRE_TILING_ENABLED", "tiling.enabled", True),
                not_null_predicate("PRE_BLOCK_M", "tiling.block_m"),
                not_null_predicate("PRE_BLOCK_N", "tiling.block_n"),
                not_null_predicate("PRE_BLOCK_K", "tiling.block_k"),
            ]
        )
    if strategy_id == "Layout.RegisterTile.C":
        predicates.extend(
            [
                eq_predicate("PRE_TILING_ENABLED", "tiling.enabled", True),
                not_null_predicate("PRE_THREAD_M", "tiling.thread_m"),
                not_null_predicate("PRE_THREAD_N", "tiling.thread_n"),
            ]
        )
    if strategy_id.startswith("Layout.SharedMemory.Padding") or strategy_id.startswith("Layout.SharedMemory.Transpose"):
        predicates.append(eq_predicate("PRE_SHARED_MEMORY", "memory.use_shared_memory", True))
    if strategy_id.startswith("Vectorization.GlobalLoad"):
        predicates.append(eq_predicate("PRE_ALIGNMENT_GUARD_OR_PROOF", "vectorization.A.tail_handling", True))
    if strategy_id == "Pipeline.DoubleBuffer.SharedAB":
        predicates.append(eq_predicate("PRE_SHARED_MEMORY", "memory.use_shared_memory", True))
    return predicates


def ge_predicate(predicate_id: str, field: str, const: int) -> dict[str, Any]:
    return {
        "id": predicate_id,
        "kind": "ir_predicate",
        "op": "ge",
        "lhs": {"field": field},
        "rhs": {"const": const},
        "verifier": "ir_predicate_eval",
    }


def eq_predicate(predicate_id: str, field: str, const: Any) -> dict[str, Any]:
    return {
        "id": predicate_id,
        "kind": "ir_predicate",
        "op": "eq",
        "lhs": {"field": field},
        "rhs": {"const": const},
        "verifier": "ir_predicate_eval",
    }


def not_null_predicate(predicate_id: str, field: str) -> dict[str, Any]:
    return {
        "id": predicate_id,
        "kind": "ir_predicate",
        "op": "ne",
        "lhs": {"field": field},
        "rhs": {"const": None},
        "verifier": "ir_predicate_eval",
    }


def vector_updates(tensor: str, width: int, vector_type: str, flag: str) -> dict[str, Any]:
    return {
        "vectorization.enabled": True,
        f"vectorization.{tensor}.{flag}": True,
        f"vectorization.{tensor}.vector_width": width,
        f"vectorization.{tensor}.vector_type": vector_type,
        f"vectorization.{tensor}.alignment_required_bytes": 4 * width,
    }


def synthesize_post_predicates(ir_updates: dict[str, Any]) -> list[dict[str, Any]]:
    predicates = []
    for index, (field, value) in enumerate(ir_updates.items(), start=1):
        predicates.append(
            {
                "id": f"POST_SYNTH_{index}",
                "kind": "ir_predicate",
                "op": "eq",
                "lhs": {"field": field},
                "rhs": {"const": value},
                "verifier": "ir_predicate_eval",
            }
        )
    return predicates


def synthesize_allowed_regions(strategy_id: str) -> list[str]:
    if strategy_id.startswith("Tiling."):
        return ["LAUNCH_CONFIG"]
    if strategy_id.startswith("Layout.SharedMemory"):
        return ["SHARED_DECL", "GLOBAL_TO_SHARED_LOAD", "SYNC_AFTER_LOAD"]
    if strategy_id.startswith("Layout.RegisterTile") or strategy_id.startswith("Reordering."):
        return ["COMPUTE_INNER", "MAIN_LOOP"]
    if strategy_id.startswith("Vectorization."):
        return ["GLOBAL_TO_SHARED_LOAD", "STORE"]
    if strategy_id.startswith("Pipeline."):
        return ["MAIN_LOOP", "GLOBAL_TO_SHARED_LOAD", "COMPUTE_INNER"]
    return []


def synthesize_code_requirements(strategy_id: str) -> list[str]:
    if strategy_id.startswith("Tiling.BlockTile."):
        return ["Update block tile constants and preserve row-major GEMM semantics."]
    if strategy_id == "Layout.SharedMemory.AB.Basic":
        return ["Introduce shared memory staging for A and B with synchronization."]
    if strategy_id == "Vectorization.GlobalLoadAB.float4":
        return ["Use guarded or proven-aligned float4 global loads for A and B."]
    if strategy_id == "Pipeline.DoubleBuffer.SharedAB":
        return ["Use two shared-memory buffers for staged load/compute overlap."]
    return []


def merge_target(current_target: dict[str, Any], base_target: dict[str, Any]) -> dict[str, Any]:
    target = copy.deepcopy(base_target)
    target.update(copy.deepcopy(current_target))
    order = target.get("default_stage_order") or ["Tiling", "Layout", "Reordering", "Vectorization", "Pipeline"]
    target["default_stage_order"] = [stage for stage in order if stage != "TensorCore"] + [
        stage for stage in order if stage == "TensorCore"
    ]
    target["default_profile"] = "all"
    return target


def merge_strategy_profiles(*profiles: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for source in profiles:
        for name, value in (source or {}).items():
            merged[name] = copy.deepcopy(value)
    return merged


def copy_profile_loading_fields(merged: dict[str, Any], current_index: dict[str, Any]) -> None:
    for key in ["profile_loading_recommendations", "strategy_groups"]:
        if key in current_index:
            merged[key] = copy.deepcopy(current_index[key])


def add_profile_group_from_strategy_profiles(index: dict[str, Any]) -> None:
    profiles = index.get("strategy_profiles", {}) or {}
    groups = index.setdefault("strategy_groups", {})
    recommendations = index.setdefault("profile_loading_recommendations", {})
    for profile_name, profile in profiles.items():
        ids = profile.get("strategy_ids") if isinstance(profile, dict) else None
        if not isinstance(ids, list):
            continue
        group_name = f"profile.{profile_name}"
        groups[group_name] = ids
        recommendations.setdefault(profile_name, [])
        if group_name not in recommendations[profile_name]:
            recommendations[profile_name].insert(0, group_name)


def build_stage_index(strategies: list[dict[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for strategy in strategies:
        stage = strategy.get("stage")
        if stage:
            result.setdefault(stage, []).append(strategy["strategy_id"])
    return result


def build_category_index(strategies: list[dict[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for strategy in strategies:
        category = strategy.get("category")
        if category:
            result.setdefault(category, []).append(strategy["strategy_id"])
    return result


def strategy_sort_key(strategy: dict[str, Any]) -> tuple[int, int, str]:
    stage_order = {"Tiling": 0, "Layout": 1, "Reordering": 2, "Vectorization": 3, "Pipeline": 4, "TensorCore": 5}
    return (stage_order.get(strategy.get("stage"), 99), int(strategy.get("priority", 5) or 5), strategy.get("strategy_id", ""))


def main() -> None:
    index, library = load_merged_strategy_documents()
    print(
        {
            "merged_index": str(DEFAULT_MERGED_INDEX),
            "merged_library": str(DEFAULT_MERGED_LIBRARY),
            "strategy_count": index.get("strategy_count"),
            "library_count": library.get("strategy_count"),
            "strategy_ids_by_stage": index.get("strategy_ids_by_stage"),
        }
    )


if __name__ == "__main__":
    main()
