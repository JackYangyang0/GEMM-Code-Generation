from __future__ import annotations

import argparse
import copy
import fnmatch
import json
from pathlib import Path
from typing import Any

from SGPO.utils.common_utils import load_json, save_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IR = ROOT / "data" / "IRs" / "ir_patch" / "optir.extracted.json"
DEFAULT_STRATEGY_INDEX = ROOT / "data" / "lib" / "strategy_index.json"
DEFAULT_DEPENDENCY_GRAPH = ROOT / "data" / "graph" / "dependency_graph.json"
DEFAULT_OUTPUT = ROOT / "data" / "lib" / "filter" / "strategy_index.filtered.json"


def filter_strategy_index(
    strategy_index,
    ir,
    dependency_graph=None,
    history=None,
    current_stage=None,
    max_failures=2,
    allowed_maturity=None,
    profile=None,
):
    allowed_maturity = allowed_maturity or {"v1"}
    applied = extract_applied_strategy_ids(ir, history)
    failed_counts = extract_failed_strategy_counts(ir, history)
    graph = dependency_graph or {}
    profile = profile or infer_profile(ir, strategy_index)
    stage = current_stage or infer_current_stage(ir, graph, strategy_index)
    strategies = strategy_index.get("strategies", [])
    by_id = {item["strategy_id"]: item for item in strategies}
    profile_allowed_ids = collect_profile_strategy_ids(strategy_index, profile)

    fallback_ids = collect_fallback_ids(failed_counts, max_failures, graph)
    kept = []
    rejected = []
    for strategy in strategies:
        decision = evaluate_strategy_for_index(
            strategy=strategy,
            ir=ir,
            dependency_graph=graph,
            stage=stage,
            applied=applied,
            failed_counts=failed_counts,
            fallback_ids=fallback_ids,
            max_failures=max_failures,
            allowed_maturity=allowed_maturity,
            profile_allowed_ids=profile_allowed_ids,
            profile=profile,
        )
        if decision["keep"]:
            item = copy.deepcopy(strategy)
            item["filter_reason"] = decision["reason"]
            if strategy["strategy_id"] in fallback_ids:
                item["is_fallback"] = True
            kept.append(item)
        else:
            rejected.append(
                {
                    "strategy_id": strategy["strategy_id"],
                    "stage": strategy.get("stage"),
                    "category": strategy.get("category"),
                    "reason": decision["reason"],
                }
            )

    result = copy.deepcopy(strategy_index)
    result["strategies"] = kept
    result["strategy_count"] = len(kept)
    result["strategy_ids_by_stage"] = build_stage_index(kept)
    result["filter_context"] = {
        "current_stage": stage,
        "applied_strategy_ids": sorted(applied),
        "failed_strategy_counts": failed_counts,
        "max_failures": max_failures,
        "allowed_maturity": sorted(allowed_maturity),
        "profile": profile,
        "profile_filter_enabled": bool(profile_allowed_ids),
        "dependency_graph_loaded": bool(graph),
        "dependency_graph_nodes": len(graph.get("nodes", []) or []),
        "dependency_graph_edges": graph.get("edge_count", len(graph.get("edges", []) or [])),
    }
    result["filtered_out"] = rejected
    result["fallback_candidates"] = [
        strategy_id for strategy_id in sorted(fallback_ids) if strategy_id in by_id
    ]
    return result


def evaluate_strategy_for_index(
    strategy,
    ir,
    dependency_graph,
    stage,
    applied,
    failed_counts,
    fallback_ids,
    max_failures,
    allowed_maturity,
    profile_allowed_ids=None,
    profile=None,
):
    strategy_id = strategy["strategy_id"]
    if strategy.get("stage") != stage:
        return reject(f"stage mismatch: expected {stage}, got {strategy.get('stage')}")

    if strategy.get("alias_of"):
        return reject(f"alias of canonical strategy {strategy.get('alias_of')}")

    if strategy.get("phase") == "excluded_default":
        return reject("excluded_default phase")

    if profile_allowed_ids and strategy_id not in profile_allowed_ids:
        return reject(f"profile {profile} does not include this strategy")

    if strategy_id in applied:
        return reject("already applied")

    maturity = strategy.get("maturity", "v1")
    if maturity not in allowed_maturity:
        return reject(f"maturity {maturity} is not enabled")

    if failed_counts.get(strategy_id, 0) >= max_failures:
        return reject(f"failed {failed_counts[strategy_id]} times; temporarily blocked")

    conflicts = get_hard_conflict_strategy_ids(strategy_id, dependency_graph)
    superseded = set(dependency_graph.get("supersedes", {}).get(strategy_id, []) or [])
    matched_conflicts = [
        conflict_id
        for conflict_id in conflicts
        if conflict_id not in superseded and any_match(applied, conflict_id)
    ]
    if matched_conflicts:
        return reject(f"conflicts with applied strategy: {', '.join(matched_conflicts)}")

    missing_requirements = find_missing_requirements(strategy_id, ir, applied, dependency_graph)
    if missing_requirements:
        return reject(f"missing graph requirements: {'; '.join(missing_requirements)}")

    if strategy_id in fallback_ids:
        return keep("fallback for failed strategy")
    return keep("eligible")


def infer_current_stage(
    ir: dict[str, Any],
    dependency_graph: dict[str, Any] | None = None,
    strategy_index: dict[str, Any] | None = None,
) -> str:
    strategy = ir.get("strategy", {})
    current = strategy.get("current_stage")
    if current and current != "initial":
        return current
    stage_order = (
        strategy.get("stage_order")
        or (dependency_graph or {}).get("stage_order")
        or infer_stage_order_from_index(strategy_index or {})
        or ["Tiling", "Layout", "Reordering", "Vectorization", "Pipeline"]
    )
    available_stages = {
        item.get("stage")
        for item in (strategy_index or {}).get("strategies", []) or []
        if item.get("stage")
    }
    if available_stages:
        for stage in stage_order:
            if stage in available_stages:
                return stage
    return stage_order[0]


def infer_stage_order_from_index(strategy_index: dict[str, Any]) -> list[str]:
    target_order = strategy_index.get("target", {}).get("default_stage_order")
    if isinstance(target_order, list) and target_order:
        return [stage for stage in target_order if isinstance(stage, str)]
    by_stage = strategy_index.get("strategy_ids_by_stage")
    if isinstance(by_stage, dict) and by_stage:
        preferred = ["Tiling", "Layout", "Reordering", "Vectorization", "Pipeline", "TensorCore"]
        stages = [stage for stage in preferred if stage in by_stage]
        stages.extend(stage for stage in by_stage if stage not in stages)
        return stages
    seen = []
    for strategy in strategy_index.get("strategies", []) or []:
        stage = strategy.get("stage")
        if stage and stage not in seen:
            seen.append(stage)
    return seen


def infer_profile(ir: dict[str, Any], strategy_index: dict[str, Any]) -> str | None:
    strategy_profile = ir.get("strategy", {}).get("profile")
    if strategy_profile:
        return strategy_profile
    precision_profile = ir.get("precision", {}).get("profile")
    if precision_profile:
        return precision_profile
    return strategy_index.get("target", {}).get("default_profile")


def collect_profile_strategy_ids(strategy_index: dict[str, Any], profile: str | None) -> set[str]:
    if not profile:
        return set()
    recommendations = strategy_index.get("profile_loading_recommendations", {})
    groups = strategy_index.get("strategy_groups", {})
    group_names = recommendations.get(profile)
    ids: set[str] = set()
    if isinstance(group_names, list):
        for group_name in group_names:
            for strategy_id in groups.get(group_name, []) or []:
                if is_strategy_id(strategy_id):
                    ids.add(strategy_id)

    strategy_profiles = strategy_index.get("strategy_profiles", {})
    profile_entry = strategy_profiles.get(profile) if isinstance(strategy_profiles, dict) else None
    if isinstance(profile_entry, dict):
        for strategy_id in profile_entry.get("strategy_ids", []) or []:
            if is_strategy_id(strategy_id):
                ids.add(strategy_id)
    return ids


def extract_applied_strategy_ids(ir, history):
    ids = set()
    for source in [ir.get("strategy", {}), history or {}]:
        for key in ("applied_strategy_ids", "applied", "accepted_strategy_ids"):
            ids.update(normalize_strategy_id_list(source.get(key)))
        for key in ("history", "strategy_history", "events"):
            for item in source.get(key, []) or []:
                if isinstance(item, dict) and item.get("status") in {None, "accepted", "pass", "applied"}:
                    strategy_id = item.get("strategy_id")
                    if strategy_id:
                        ids.add(strategy_id)
    current_id = ir.get("strategy", {}).get("current_strategy_id")
    if ir.get("verification", {}).get("accepted") and current_id:
        ids.add(current_id)
    return ids


def extract_failed_strategy_counts(ir, history):
    counts = {}
    for source in [ir.get("strategy", {}), history or {}]:
        raw_counts = source.get("failed_strategy_counts") or source.get("failed") or {}
        if isinstance(raw_counts, dict):
            for strategy_id, count in raw_counts.items():
                counts[strategy_id] = counts.get(strategy_id, 0) + int(count)
        for key in ("history", "strategy_history", "events"):
            for item in source.get(key, []) or []:
                if isinstance(item, dict) and item.get("status") in {"failed", "fail", "rejected"}:
                    strategy_id = item.get("strategy_id")
                    if strategy_id:
                        counts[strategy_id] = counts.get(strategy_id, 0) + 1
    return counts


def collect_fallback_ids(
    failed_counts,
    max_failures,
    dependency_graph,
):
    fallback_ids = set()
    fallback_graph = dependency_graph.get("fallback", {})
    for strategy_id, count in failed_counts.items():
        if count >= max_failures:
            for entry in normalize_graph_entries(fallback_graph.get(strategy_id, []) or []):
                for fallback_id in entry.get("strategies", []) or []:
                    if is_strategy_id(fallback_id):
                        fallback_ids.add(fallback_id)
    return fallback_ids


def get_hard_conflict_strategy_ids(strategy_id: str, dependency_graph: dict[str, Any]) -> set[str]:
    conflicts: set[str] = set()
    for entry in normalize_graph_entries(dependency_graph.get("conflicts", {}).get(strategy_id, []) or []):
        if entry.get("severity", "hard") != "hard":
            continue
        conflicts.update(entry.get("strategies", []) or [])
    return conflicts


def find_missing_requirements(
    strategy_id: str,
    ir: dict[str, Any],
    applied: set[str],
    dependency_graph: dict[str, Any],
) -> list[str]:
    missing = []
    for entry in normalize_graph_entries(dependency_graph.get("requires", {}).get(strategy_id, []) or []):
        mode = entry.get("mode", "all")
        required_strategy_ids = entry.get("strategies", []) or []
        strategy_ok = requirement_strategy_ok(required_strategy_ids, mode, applied, ir)
        state_ok = requirement_state_ok(entry.get("required_state", {}) or {}, ir)
        if not (strategy_ok or state_ok):
            missing.append(render_requirement(entry))
    return missing


def normalize_graph_entries(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, str):
        return [{"mode": "all", "strategies": [raw]}]
    if isinstance(raw, list):
        entries = []
        for item in raw:
            if isinstance(item, dict):
                entries.append(item)
            elif isinstance(item, str):
                entries.append({"mode": "all", "strategies": [item]})
        return entries
    return []


def requirement_strategy_ok(required_strategy_ids: list[str], mode: str, applied: set[str], ir: dict[str, Any]) -> bool:
    if not required_strategy_ids:
        return False
    def satisfied(strategy_id: str) -> bool:
        return any_match(applied, strategy_id) or strategy_requirement_satisfied_by_ir(strategy_id, ir)

    if mode == "any":
        return any(satisfied(strategy_id) for strategy_id in required_strategy_ids)
    return all(satisfied(strategy_id) for strategy_id in required_strategy_ids)


def strategy_requirement_satisfied_by_ir(strategy_id: str, ir: dict[str, Any]) -> bool:
    if fnmatch.fnmatchcase(strategy_id, "Tiling.BlockTile.*"):
        tiling = ir.get("tiling", {})
        return bool(tiling.get("enabled")) and tiling.get("block_m") is not None and tiling.get("block_n") is not None
    if fnmatch.fnmatchcase(strategy_id, "Tiling.ThreadTile.*"):
        tiling = ir.get("tiling", {})
        return tiling.get("thread_m") is not None and tiling.get("thread_n") is not None
    if strategy_id == "Layout.SharedMemory.AB.Basic":
        memory = ir.get("memory", {})
        return (
            bool(memory.get("use_shared_memory"))
            or bool(memory.get("shared_A", {}).get("enabled"))
            or bool(memory.get("shared_B", {}).get("enabled"))
        )
    if strategy_id == "Layout.RegisterTile.C":
        memory = ir.get("memory", {})
        tiling = ir.get("tiling", {})
        return (
            bool(memory.get("use_register_tile"))
            or tiling.get("thread_m") not in {None, 1}
            or tiling.get("thread_n") not in {None, 1}
        )
    if strategy_id == "Vectorization.AlignmentGuard":
        vectorization = ir.get("vectorization", {})
        return any(
            bool(vectorization.get(tensor, {}).get("alignment_guard"))
            or bool(vectorization.get(tensor, {}).get("alignment_proven"))
            for tensor in ("A", "B", "C")
        )
    if strategy_id == "Safety.AssumeDivisibleAligned":
        return bool(ir.get("safety", {}).get("assume_divisible_aligned")) or bool(
            ir.get("problem", {}).get("assume_divisible_aligned")
        )
    if strategy_id == "Safety.BoundaryPolicy.GeneralGuarded":
        boundary_guard = ir.get("problem", {}).get("boundary_guard", {})
        return bool(boundary_guard.get("M")) and bool(boundary_guard.get("N"))
    if strategy_id == "Compiler.TemplateSpecialization.ShapeStatic":
        return bool(ir.get("compiler", {}).get("template_specialization", {}).get("shape_static"))
    return False


def requirement_state_ok(required_state: dict[str, Any], ir: dict[str, Any]) -> bool:
    if not required_state:
        return False
    return all(check_state_condition(get_nested(ir, path), expected) for path, expected in required_state.items())


def check_state_condition(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    if expected == "not null":
        return actual is not None
    if isinstance(expected, str) and expected.startswith(">="):
        try:
            return actual is not None and float(actual) >= float(expected[2:])
        except (TypeError, ValueError):
            return False
    return actual == expected


def get_nested(data: dict[str, Any], dotted_path: str) -> Any:
    current: Any = data
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def render_requirement(entry: dict[str, Any]) -> str:
    strategies = entry.get("strategies", []) or []
    required_state = entry.get("required_state", {}) or {}
    strategy_text = f"{entry.get('mode', 'all')}({', '.join(strategies)})" if strategies else ""
    state_text = ", ".join(f"{key}={value}" for key, value in required_state.items())
    parts = [part for part in [strategy_text, state_text] if part]
    return " or state ".join(parts) if parts else entry.get("reason", "unsatisfied requirement")


def is_strategy_id(value: str) -> bool:
    return isinstance(value, str) and "." in value and not value.startswith("rollback_")


def any_match(strategy_ids: set[str], pattern: str) -> bool:
    return any(fnmatch.fnmatchcase(strategy_id, pattern) for strategy_id in strategy_ids)


def normalize_strategy_id_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def build_stage_index(strategies: list[dict[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for strategy in strategies:
        result.setdefault(strategy.get("stage"), []).append(strategy["strategy_id"])
    return result


def keep(reason: str) -> dict[str, Any]:
    return {"keep": True, "reason": reason}


def reject(reason: str) -> dict[str, Any]:
    return {"keep": False, "reason": reason}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter strategy_index before sending candidates to the LLM.")
    parser.add_argument("--ir", type=Path, default=DEFAULT_IR)
    parser.add_argument("--strategy-index", type=Path, default=DEFAULT_STRATEGY_INDEX)
    parser.add_argument("--dependency-graph", type=Path, default=DEFAULT_DEPENDENCY_GRAPH)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--stage", help="Override current optimization stage.")
    parser.add_argument("--max-failures", type=int, default=2)
    parser.add_argument("--allowed-maturity", nargs="+", default=["v1", "v2"])
    parser.add_argument("--profile")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ir = load_json(args.ir)
    strategy_index = load_json(args.strategy_index)
    dependency_graph = load_json(args.dependency_graph)
    history = load_json(args.history) if args.history else None
    filtered = filter_strategy_index(
        strategy_index=strategy_index,
        ir=ir,
        dependency_graph=dependency_graph,
        history=history,
        current_stage=args.stage,
        max_failures=args.max_failures,
        allowed_maturity=set(args.allowed_maturity),
        profile=args.profile,
    )
    save_json(args.output, filtered)
    print(json.dumps({
        "output": str(args.output),
        "current_stage": filtered["filter_context"]["current_stage"],
        "strategy_count": filtered["strategy_count"],
        "strategy_ids": [item["strategy_id"] for item in filtered["strategies"]],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
