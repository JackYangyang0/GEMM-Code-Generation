from __future__ import annotations

import copy
import fnmatch
import re
from dataclasses import dataclass
from typing import Any

from SGPO.generate_ir.ir_checker import check_before_codegen
from SGPO.generate_ir.gpu_resources import shared_memory_usage


@dataclass
class LocalCheckResult:
    id: str
    status: str
    message: str


FIELD_ALIASES = {
    "M": "problem.M",
    "N": "problem.N",
    "K": "problem.K",
    "block_m": "tiling.block_m",
    "block_n": "tiling.block_n",
    "block_k": "tiling.block_k",
    "warp_m": "tiling.warp_tile.warp_m",
    "warp_n": "tiling.warp_tile.warp_n",
    "thread_m": "tiling.thread_m",
    "thread_n": "tiling.thread_n",
    "warp_m_iter": "tiling.warp_tile.warp_m_iter",
    "warp_n_iter": "tiling.warp_tile.warp_n_iter",
    "shared_memory_bytes": "resource.shared_memory.total_bytes",
    "pipeline_multiplier": "resource.shared_memory.pipeline_multiplier",
    "max_shared_memory_per_block_bytes": "hardware.max_shared_memory_per_block_bytes",
    "threads_per_block": "mapping.threads_per_block",
    "warps_per_block": "mapping.warps_per_block",
    "warp_size": "hardware.warp_size",
}


class StageController:
    def __init__(
        self,
        strategy_index: dict[str, Any],
        workflow_library: dict[str, Any],
        current_stage: str,
        optir: dict[str, Any],
        history: dict[str, Any] | None = None,
    ) -> None:
        self.strategy_index = strategy_index
        self.workflow_library = workflow_library
        self.current_stage = current_stage
        self.optir = optir
        self.history = history or {}
        self.stage_spec = get_stage_spec(strategy_index, workflow_library, current_stage)
        self.subphases = get_ordered_subphases(self.stage_spec)
        initialize_default_field_meta(self.optir)

    def state(self) -> dict[str, Any]:
        return {
            "current_stage": self.current_stage,
            "current_subphase": self.current_subphase_id(),
            "optir": self.optir,
            "stage_plan": self.stage_plan(),
            "applied_micro_strategies": self.history.get("applied_micro_strategies", []),
            "completed_subphases": self.history.get("completed_subphases", []),
            "failed_micro_strategies": self.history.get("failed_strategy_counts", {}),
            "completed_stages": self.history.get("completed_stages", []),
            "stage_checkpoints": self.history.get("stage_checkpoints", []),
        }

    def stage_plan(self) -> list[str]:
        return [
            strategy_id
            for strategy_id in self.history.get("applied_strategy_ids", []) or []
            if strategy_stage(strategy_id) == self.current_stage or is_mapping_for_stage(strategy_id, self.current_stage)
        ]

    def current_subphase(self) -> dict[str, Any] | None:
        for subphase in self.subphases:
            subphase_id = subphase_id_of(subphase)
            if subphase_id.endswith(".StageVerification"):
                return subphase
            if not subphase_is_satisfied(subphase, self.optir, self.history):
                return subphase
        return self.subphases[-1] if self.subphases else None

    def current_subphase_id(self) -> str | None:
        subphase = self.current_subphase()
        return subphase_id_of(subphase) if subphase else None

    def get_subphase_candidates(self) -> dict[str, Any]:
        subphase = self.current_subphase()
        if not subphase:
            return empty_candidate_index(self.strategy_index, self.current_stage, None)
        if subphase_id_of(subphase).endswith(".StageVerification"):
            return empty_candidate_index(self.strategy_index, self.current_stage, subphase_id_of(subphase))

        allowed_ids = subphase.get("allowed_strategy_ids") or ids_from_patterns(
            self.strategy_index,
            subphase.get("allowed_strategy_patterns", []) or [],
        )
        allowed_ids = normalize_allowed_ids_for_subphase(subphase_id_of(subphase), allowed_ids)
        available = []
        rejected = []
        for strategy_id in allowed_ids:
            reason = self.reject_reason(strategy_id, subphase)
            item = strategy_index_item(self.strategy_index, strategy_id)
            if reason is None:
                available.append(item or synthesize_index_item(strategy_id, self.current_stage, subphase))
            else:
                rejected.append({"strategy_id": strategy_id, "reason": reason})

        result = copy.deepcopy(self.strategy_index)
        result["strategies"] = available
        result["strategy_count"] = len(available)
        result["filter_context"] = {
            "current_stage": self.current_stage,
            "current_subphase": subphase_id_of(subphase),
            "subphase_purpose": subphase.get("purpose"),
            "requires_fields": subphase.get("requires_fields", []),
            "provides_fields": subphase.get("provides_fields", []),
            "allowed_strategy_ids": allowed_ids,
            "rejected": rejected,
        }
        return result

    def reject_reason(self, strategy_id: str, subphase: dict[str, Any]) -> str | None:
        if strategy_id in self.history.get("applied_strategy_ids", []):
            return "already applied"
        failed = self.history.get("failed_strategy_counts", {}) or {}
        if failed.get(strategy_id, 0) >= 2:
            return "failed too many times"
        dependency_reason = tiling_hierarchy_reject_reason(strategy_id, self.optir)
        if dependency_reason is not None:
            return dependency_reason
        strategy = self.strategy_object_for_subphase(strategy_id, subphase)
        report = check_before_codegen(self.optir, strategy)
        if not report.get("strategy_applicable", False):
            failed_checks = [
                format_precondition_rejection(item)
                for item in report.get("results", []) or []
                if item.get("status") in {"fail", "unknown"}
            ]
            details = ", ".join(failed_checks[:3]) or "checker returned strategy_applicable=false without details"
            return "preconditions failed: " + details
        missing = [
            field
            for field in subphase.get("requires_fields", []) or []
            if not requirement_is_satisfied(self.optir, field)
        ]
        if missing:
            return f"missing required fields: {', '.join(missing)}"
        return None


    def strategy_object(self, strategy_id: str) -> dict[str, Any]:
        subphase = self.current_subphase() or {}
        return self.strategy_object_for_subphase(strategy_id, subphase)

    def strategy_object_for_subphase(self, strategy_id: str, subphase: dict[str, Any]) -> dict[str, Any]:
        strategy = find_strategy_in_tree(self.workflow_library, strategy_id)
        if strategy:
            return strategy
        return synthesize_micro_strategy(strategy_id, self.current_stage, subphase)

    def apply_micro_strategy(self, strategy_id: str, ir_updates: dict[str, Any] | None = None) -> dict[str, Any]:
        subphase = self.current_subphase() or {}
        next_ir = copy.deepcopy(self.optir)
        updates = clean_ir_updates(ir_updates or {})
        updates.update(synthesize_ir_updates(strategy_id))
        for path, value in updates.items():
            set_path(next_ir, path, value)
            set_field_meta(
                next_ir,
                path,
                origin="micro_strategy",
                resolved=True,
                resolved_by=strategy_id,
                subphase=subphase_id_of(subphase),
            )
        apply_defaults(next_ir, subphase_id_of(subphase))
        normalize_numeric_fields(next_ir)
        derive_fields(next_ir, subphase_id_of(subphase), strategy_id)
        local_results = run_local_checks(next_ir, subphase)
        deferred = list(subphase.get("deferred_checks", []) or [])
        next_ir.setdefault("strategy", {})["current_stage"] = self.current_stage
        next_ir.setdefault("strategy", {})["current_subphase"] = subphase_id_of(subphase)
        next_ir.setdefault("strategy", {})["current_strategy_id"] = strategy_id
        next_ir.setdefault("stage_controller", {})["last_local_check"] = {
            "stage": self.current_stage,
            "subphase": subphase_id_of(subphase),
            "strategy_id": strategy_id,
            "results": [item.__dict__ for item in local_results],
            "deferred_checks": deferred,
            "accepted": all(item.status == "pass" for item in local_results),
        }
        record_micro_strategy_application(next_ir, self.current_stage, subphase_id_of(subphase), strategy_id, updates, local_results)
        return next_ir

    def advance_derivation_subphase(self) -> dict[str, Any]:
        subphase = self.current_subphase() or {}
        next_ir = copy.deepcopy(self.optir)
        apply_defaults(next_ir, subphase_id_of(subphase))
        derive_fields(next_ir, subphase_id_of(subphase), None)
        local_results = run_local_checks(next_ir, subphase)
        next_ir.setdefault("strategy", {})["current_stage"] = self.current_stage
        next_ir.setdefault("strategy", {})["current_subphase"] = subphase_id_of(subphase)
        next_ir.setdefault("stage_controller", {})["last_local_check"] = {
            "stage": self.current_stage,
            "subphase": subphase_id_of(subphase),
            "strategy_id": None,
            "results": [item.__dict__ for item in local_results],
            "deferred_checks": list(subphase.get("deferred_checks", []) or []),
            "accepted": all(item.status == "pass" for item in local_results),
            "derivation_only": True,
        }
        if all(item.status == "pass" for item in local_results):
            mark_completed_subphase(next_ir, subphase_id_of(subphase))
        return next_ir

    def skip_current_subphase(self, reason: str) -> dict[str, Any]:
        subphase = self.current_subphase() or {}
        subphase_id = subphase_id_of(subphase)
        next_ir = copy.deepcopy(self.optir)
        mark_skipped_subphase(next_ir, subphase_id)
        next_ir.setdefault("strategy", {})["current_stage"] = self.current_stage
        next_ir.setdefault("strategy", {})["current_subphase"] = subphase_id
        next_ir.setdefault("stage_controller", {})["last_skip"] = {
            "stage": self.current_stage,
            "subphase": subphase_id,
            "reason": reason,
            "next_subphase": subphase.get("next_subphase"),
        }
        return next_ir

    def verify_stage(self, optir: dict[str, Any] | None = None, include_code_checks: bool = True) -> dict[str, Any]:
        ir = optir or self.optir
        stage_verification = get_stage_verification(self.stage_spec)
        predicates = stage_verification.get("predicates") or stage_verification.get("stage_completion_predicates") or []
        required = stage_verification.get("required_fields", []) or []
        results = []
        for field in required:
            ok = get_path(ir, field) is not None
            results.append(LocalCheckResult(f"FIELD:{field}", "pass" if ok else "fail", f"{field} is {'set' if ok else 'missing'}"))
        for predicate in predicates:
            if isinstance(predicate, str):
                results.append(evaluate_text_condition(ir, predicate))
            else:
                results.append(evaluate_predicate_like(ir, predicate))
        if include_code_checks:
            results.extend(run_stage_static_code_checks(self.current_stage, ir))
        return {
            "stage": self.current_stage,
            "subphase": f"{self.current_stage}.StageVerification",
            "accepted": all(item.status == "pass" for item in results),
            "results": [item.__dict__ for item in results],
        }


def get_subphase_candidates(strategy_index, current_stage, current_subphase, optir, history):
    workflow = {"stages": strategy_index.get("stages", {})}
    controller = StageController(strategy_index, workflow, current_stage, optir, history)
    if current_subphase:
        controller.current_subphase = lambda: find_subphase(controller.subphases, current_subphase)  # type: ignore[method-assign]
    return controller.get_subphase_candidates()


def get_stage_spec(strategy_index: dict[str, Any], workflow_library: dict[str, Any], stage: str) -> dict[str, Any]:
    stages = strategy_index.get("stages")
    if isinstance(stages, dict) and stage in stages:
        return stages[stage]
    for item in workflow_library.get("stages", []) or []:
        if item.get("stage_id") == stage:
            return item
    return {"stage_id": stage, "subphases": [], "ordered_subphases": []}


def get_ordered_subphases(stage_spec: dict[str, Any]) -> list[dict[str, Any]]:
    subphases = stage_spec.get("subphases") or stage_spec.get("ordered_subphases") or []
    if isinstance(subphases, list):
        ordered = sorted(subphases, key=lambda item: item.get("order", 999))
    else:
        ordered = []
    verification = stage_spec.get("stage_verification")
    if verification:
        ordered.append({"subphase_id": verification.get("stage_id", f"{stage_spec.get('stage_id')}.StageVerification"), **verification})
    return ordered


def get_stage_verification(stage_spec: dict[str, Any]) -> dict[str, Any]:
    verification = stage_spec.get("stage_verification")
    if isinstance(verification, dict):
        return verification
    for subphase in get_ordered_subphases(stage_spec):
        subphase_id = subphase_id_of(subphase) or ""
        if subphase_id.endswith(".StageVerification"):
            return subphase
    return {}


def find_subphase(subphases: list[dict[str, Any]], subphase_id: str) -> dict[str, Any] | None:
    for subphase in subphases:
        if subphase_id_of(subphase) == subphase_id:
            return subphase
    return None


def subphase_id_of(subphase: dict[str, Any] | None) -> str | None:
    if not subphase:
        return None
    return subphase.get("subphase_id") or subphase.get("subphase")


def subphase_is_satisfied(subphase: dict[str, Any], ir: dict[str, Any], history: dict[str, Any]) -> bool:
    sid = subphase_id_of(subphase) or ""
    if sid.endswith(".StageVerification"):
        return False
    if sid in set(history.get("optional_skipped_subphases", []) or []):
        return True
    if sid in set(ir.get("strategy", {}).get("optional_skipped_subphases", []) or []):
        return True
    if subphase_completed_in_history(sid, history):
        return True
    if subphase_completed_in_ir(sid, ir):
        return True
    if subphase_has_required_micro_strategy(sid, subphase, history, ir):
        return True
    provides = subphase.get("provides_fields", []) or []
    if provides and all(field_is_resolved(ir, field) for field in provides):
        return True
    if subphase.get("allowed_strategy_ids") or subphase.get("allowed_strategy_patterns"):
        return False
    return False


def subphase_completed_in_history(subphase_id: str, history: dict[str, Any]) -> bool:
    return subphase_id in set(history.get("completed_subphases", []) or [])


def subphase_completed_in_ir(subphase_id: str, ir: dict[str, Any]) -> bool:
    return subphase_id in set(ir.get("strategy", {}).get("completed_subphases", []) or [])


def subphase_has_required_micro_strategy(
    subphase_id: str,
    subphase: dict[str, Any],
    history: dict[str, Any],
    ir: dict[str, Any],
) -> bool:
    applied_records = list(history.get("applied_micro_strategies", []) or [])
    applied_records.extend(ir.get("strategy", {}).get("applied_micro_strategies", []) or [])
    applied_ids = set(history.get("applied_strategy_ids", []) or [])
    if subphase_id == "Tiling.ThreadTileSelection":
        has_thread_tile = any(
            record.get("subphase") == subphase_id and str(record.get("strategy_id", "")).startswith("Tiling.ThreadTile.")
            for record in applied_records
        ) or any(strategy_id.startswith("Tiling.ThreadTile.") for strategy_id in applied_ids)
        return has_thread_tile and field_resolved_by_subphase(ir, "tiling.thread_m", subphase_id) and field_resolved_by_subphase(ir, "tiling.thread_n", subphase_id)
    allowed_ids = set(subphase.get("allowed_strategy_ids", []) or [])
    if not allowed_ids:
        return False
    applied_for_subphase = {
        record.get("strategy_id")
        for record in applied_records
        if record.get("subphase") == subphase_id and record.get("status") in {None, "local_pass", "generated"}
    }
    return bool(allowed_ids & (applied_ids | applied_for_subphase))


def ids_from_patterns(strategy_index: dict[str, Any], patterns: list[str]) -> list[str]:
    ids = set()
    for stage in (strategy_index.get("stages", {}) or {}).values():
        for subphase in stage.get("subphases", []) or []:
            for strategy_id in subphase.get("allowed_strategy_ids", []) or []:
                if any(fnmatch.fnmatchcase(strategy_id, pattern) for pattern in patterns):
                    ids.add(strategy_id)
    for section in ["strategy_ids_by_stage", "strategy_ids_by_category"]:
        for values in (strategy_index.get("legacy_flat_views", {}).get(section, {}) or {}).values():
            for strategy_id in values:
                if any(fnmatch.fnmatchcase(strategy_id, pattern) for pattern in patterns):
                    ids.add(strategy_id)
    return sorted(ids)


def normalize_allowed_ids_for_subphase(subphase_id: str | None, strategy_ids: list[str]) -> list[str]:
    if subphase_id == "Tiling.WarpTileSelection":
        return [strategy_id for strategy_id in strategy_ids if ".Iteration." not in strategy_id]
    return strategy_ids


def strategy_index_item(strategy_index: dict[str, Any], strategy_id: str) -> dict[str, Any] | None:
    for item in strategy_index.get("strategies", []) or []:
        if item.get("strategy_id") == strategy_id:
            return copy.deepcopy(item)
    return None


def find_strategy_in_tree(node: Any, strategy_id: str) -> dict[str, Any] | None:
    if isinstance(node, dict):
        if node.get("strategy_id") == strategy_id:
            return copy.deepcopy(node)
        if strategy_id in node and isinstance(node[strategy_id], dict):
            result = copy.deepcopy(node[strategy_id])
            result.setdefault("strategy_id", strategy_id)
            return result
        for value in node.values():
            found = find_strategy_in_tree(value, strategy_id)
            if found:
                return found
    elif isinstance(node, list):
        for value in node:
            found = find_strategy_in_tree(value, strategy_id)
            if found:
                return found
    return None


def synthesize_index_item(strategy_id: str, stage: str, subphase: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategy_id": strategy_id,
        "stage": stage,
        "category": strategy_id.split(".")[0],
        "name": strategy_id,
        "intent": subphase.get("purpose", ""),
        "maturity": "v1",
    }


def synthesize_micro_strategy(strategy_id: str, stage: str, subphase: dict[str, Any]) -> dict[str, Any]:
    updates = synthesize_ir_updates(strategy_id)
    code_requirements = code_requirements_for_strategy(strategy_id)
    return {
        "strategy_id": strategy_id,
        "stage": stage,
        "category": strategy_id.split(".")[0],
        "name": strategy_id,
        "intent": subphase.get("purpose", ""),
        "subphase": subphase_id_of(subphase),
        "ir_updates": updates,
        "affected_fields": sorted(updates),
        "allowed_modified_regions": allowed_regions_for_subphase(subphase),
        "preconditions": {"verifier": "ir_predicate_eval", "predicates": []},
        "postconditions": {
            "patch_ir_verification": {
                "verifier": "ir_predicate_eval",
                "predicates": [
                    {"id": f"POST_{i}", "kind": "ir_predicate", "op": "eq", "lhs": {"field": field}, "rhs": {"const": value}}
                    for i, (field, value) in enumerate(updates.items(), start=1)
                ],
            },
            "code_verification": {"constraints": code_requirements},
        },
        "code_requirements": code_requirements,
        "deferred_checks": subphase.get("deferred_checks", []),
    }


def code_requirements_for_strategy(strategy_id: str) -> list[dict[str, Any]]:
    if strategy_id == "Layout.SharedMemory.AB.Basic":
        return [
            {
                "id": "SHARED_A_DECLARED",
                "verifier": "static_code_ast",
                "kind": "shared_memory_declaration",
                "names_any": ["As", "shared_A", "sA"],
                "required": True,
                "region": "SHARED_DECL",
            },
            {
                "id": "SHARED_B_DECLARED",
                "verifier": "static_code_ast",
                "kind": "shared_memory_declaration",
                "names_any": ["Bs", "shared_B", "sB"],
                "required": True,
                "region": "SHARED_DECL",
            },
        ]
    return []


def allowed_regions_for_subphase(subphase: dict[str, Any]) -> list[str]:
    sid = subphase_id_of(subphase) or ""
    if "BlockTile" in sid or "WarpTile" in sid or "ThreadTile" in sid:
        return ["LAUNCH_CONFIG"]
    if "Mapping" in sid:
        return ["INDEX_MAPPING"]
    if "SharedMemoryDeclaration" in sid:
        return ["SHARED_DECL"]
    if "Transform" in sid or "CooperativeLoad" in sid:
        return ["GLOBAL_TO_SHARED_LOAD", "SYNC_AFTER_LOAD"]
    if "Compute" in sid:
        return ["MAIN_LOOP", "COMPUTE_INNER"]
    if "Store" in sid or "Epilogue" in sid:
        return ["STORE"]
    return []


def synthesize_ir_updates(strategy_id: str) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if strategy_id.startswith("CPU."):
        updates.update(synthesize_cpu_ir_updates(strategy_id))
        return updates
    block = re.fullmatch(r"Tiling\.BlockTile\.(\d+)x(\d+)x(\d+)", strategy_id)
    if block:
        bm, bn, bk = map(int, block.groups())
        updates.update({"tiling.enabled": True, "tiling.block_m": bm, "tiling.block_n": bn, "tiling.block_k": bk})
    warp = re.fullmatch(r"Tiling\.WarpTile\.(\d+)x(\d+)", strategy_id)
    if warp:
        wm, wn = map(int, warp.groups())
        updates.update({"tiling.warp_tile.warp_m": wm, "tiling.warp_tile.warp_n": wn})
    if strategy_id in {"Tiling.WarpTile.WMxWN", "Tiling.WarpTile.ParametricWMxWN"}:
        updates.update({"tiling.warp_tile.warp_m": 16, "tiling.warp_tile.warp_n": 32})
    thread = re.fullmatch(r"Tiling\.ThreadTile\.(\d+)x(\d+)", strategy_id)
    if thread:
        tm, tn = map(int, thread.groups())
        updates.update({"tiling.thread_m": tm, "tiling.thread_n": tn})
    if strategy_id.startswith("Mapping.Warp"):
        updates.update({"mapping.warp.enabled": True, "mapping.thread_to_output": "warp_thread_tile", "mapping.warp_to_output": "2d_tile"})
    if strategy_id == "Mapping.Warp.BasicWidLane":
        updates.update({"mapping.warp.linear_thread_id": "threadIdx.x", "mapping.warp.wid_lane": True})
    if strategy_id in {"Mapping.WarpThreadTile.WarpLaneFragmentBasic", "Mapping.WarpThreadTile.WarpLaneFragment2D"}:
        updates.update({
            "mapping.thread_to_output": "warp_lane_fragment_tile",
            "mapping.lane_layout": "warp_lane_fragment_2d" if strategy_id.endswith("2D") else "warp_lane_fragment_basic",
        })
    if strategy_id == "Mapping.LaneLayout.2D_8x4":
        updates.update({
            "mapping.warp.enabled": True,
            "mapping.warp_to_output": "2d_warp_tile",
            "mapping.lane_layout": "lane_2d_8x4",
            "mapping.lane_m": 8,
            "mapping.lane_n": 4,
        })
    if strategy_id == "Mapping.LaneLayout.1DContiguousM":
        updates.update({
            "mapping.warp.enabled": True,
            "mapping.warp_to_output": "2d_warp_tile",
            "mapping.lane_layout": "lane_1d_contiguous_m",
        })
    if strategy_id == "Mapping.LaneLayout.1DContiguousN":
        updates.update({
            "mapping.warp.enabled": True,
            "mapping.warp_to_output": "2d_warp_tile",
            "mapping.lane_layout": "lane_1d_contiguous_n",
        })
    if strategy_id == "Layout.SharedMemory.AB.Basic":
        updates.update({
            "memory.use_shared_memory": True,
            "memory.shared_A.enabled": True,
            "memory.shared_A.declaration_required": True,
            "memory.shared_B.enabled": True,
            "memory.shared_B.declaration_required": True,
            "memory.shared_A.shape": ["block_k", "block_m"],
            "memory.shared_B.shape": ["block_k", "block_n"],
        })
    if strategy_id.startswith("Layout.SharedMemory.Padding"):
        updates.update({"memory.shared_A.padding": 1, "memory.shared_B.padding": 1})
    if strategy_id.startswith("Layout.SharedMemory.Transpose"):
        updates.update({"memory.shared_B.transposed": True})
    if strategy_id == "Layout.RegisterTile.C":
        updates.update({"memory.use_register_tile": True, "register.accumulator.layout": "2d_array"})
    if strategy_id == "Register.AccumulatorLayout.2DArray":
        updates.update({"memory.use_register_tile": True, "register.accumulator.layout": "2d_array"})
    if strategy_id == "Register.AccumulatorLayout.FlatArray":
        updates.update({"memory.use_register_tile": True, "register.accumulator.layout": "flat_array"})
    if strategy_id == "Register.AccumulatorLayout.ScalarUnrolled":
        updates.update({"memory.use_register_tile": True, "register.accumulator.layout": "scalar_unrolled"})
    if strategy_id == "Reordering.LoadCompute.SeparatePhases":
        updates.update({
            "schedule.loop_structure": "separate_load_compute",
            "synchronization.sync_policy.after_global_to_shared_load": True,
            "synchronization.sync_policy.before_shared_buffer_reuse": True,
        })
    if strategy_id.startswith("Reordering.ThreadMapping.CoalescedLoad"):
        target = "A" if strategy_id.endswith("A") else "B" if strategy_id.endswith("B") else "AB"
        if "A" in target:
            updates["memory.global_load_A.pattern"] = "coalesced"
        if "B" in target:
            updates["memory.global_load_B.pattern"] = "coalesced"
        updates["memory.shared_load_mapping"] = "thread_coalesced"
    if strategy_id in {"Reordering.CooperativeVectorLoadAB.float4", "Reordering.WarpCooperativeLoadAB.float4"}:
        updates.update({
            "memory.global_load_A.pattern": "cooperative_vector_float4",
            "memory.global_load_B.pattern": "cooperative_vector_float4",
            "memory.shared_load_mapping": "cooperative_vector",
            "vectorization.A.vector_width": 4,
            "vectorization.B.vector_width": 4,
        })
    if strategy_id.startswith("Reordering.WarpCooperativeLoadA"):
        updates.update({"memory.global_load_A.pattern": "warp_cooperative", "memory.shared_load_mapping": "warp_cooperative"})
    if strategy_id.startswith("Reordering.WarpCooperativeLoadB"):
        updates.update({"memory.global_load_B.pattern": "warp_cooperative", "memory.shared_load_mapping": "warp_cooperative"})
    if strategy_id.startswith("Reordering.KLoop."):
        updates["schedule.unrolling"] = strategy_id.rsplit(".", 1)[-1]
    if strategy_id.startswith("Register.FFMA."):
        updates["schedule.ffma"] = strategy_id
    if strategy_id.startswith("Register.CacheA"):
        updates["register.cache_A"] = True
    if strategy_id.startswith("Register.CacheB"):
        updates["register.cache_B"] = True
    if strategy_id.startswith("Reordering.WarpCompute"):
        updates["schedule.warp_compute"] = strategy_id
    if strategy_id == "Vectorization.AlignmentGuard":
        updates.update({
            "vectorization.A.alignment_guard": True,
            "vectorization.B.alignment_guard": True,
            "vectorization.C.alignment_guard": True,
            "safety.boundary_policy": "general_guarded",
            "safety.tail_handling": True,
        })
    if strategy_id.startswith("Safety.BoundaryPolicy."):
        updates.update({"safety.boundary_policy": strategy_id.rsplit(".", 1)[-1], "safety.tail_handling": True})
    if strategy_id == "Safety.AssumeDivisibleAligned":
        updates.update({
            "vectorization.A.alignment_proven": True,
            "vectorization.B.alignment_proven": True,
            "vectorization.C.alignment_proven": True,
            "safety.boundary_policy": "static_divisible_no_guard",
        })
    if strategy_id.startswith("Vectorization.GlobalLoadA.float"):
        updates.update({"vectorization.A.vector_width": vector_width_from_strategy(strategy_id), "memory.global_load_A.pattern": strategy_id})
    if strategy_id.startswith("Vectorization.GlobalLoadB.float"):
        updates.update({"vectorization.B.vector_width": vector_width_from_strategy(strategy_id), "memory.global_load_B.pattern": strategy_id})
    if strategy_id.startswith("Vectorization.GlobalLoadAB.float"):
        width = vector_width_from_strategy(strategy_id)
        updates.update({
            "vectorization.A.vector_width": width,
            "vectorization.B.vector_width": width,
            "memory.global_load_A.pattern": strategy_id,
            "memory.global_load_B.pattern": strategy_id,
        })
    if strategy_id.startswith("Vectorization.StoreC") or strategy_id.startswith("Epilogue.StoreC"):
        updates.update({
            "vectorization.C.vector_width": vector_width_from_strategy(strategy_id, default=1),
            "memory.global_store_C.pattern": strategy_id,
        })
    if strategy_id == "Mapping.WarpStore.CoalescedC":
        updates.update({
            "vectorization.C.vector_width": 1,
            "memory.global_store_C.pattern": strategy_id,
        })
    if strategy_id.startswith("Epilogue.AlphaBeta."):
        updates["epilogue.mode"] = "alpha_beta_general"
    if strategy_id.startswith("Epilogue.BetaZero"):
        updates["epilogue.mode"] = "beta_zero_fast_path"
    if strategy_id.startswith("Epilogue.BetaOne"):
        updates["epilogue.mode"] = "beta_one_fast_path"
    if strategy_id.startswith("Epilogue.Fusion."):
        updates["epilogue.fusion"] = strategy_id.rsplit(".", 1)[-1]
    if strategy_id.startswith("Pipeline.DoubleBuffer") or strategy_id.startswith("Pipeline.WarpAwareDoubleBuffer"):
        updates.update({"pipeline.enabled": True, "pipeline.stage_count": 2, "pipeline.double_buffering": True, "resource.shared_memory.pipeline_multiplier": 2})
    if strategy_id == "Pipeline.NoAsyncCopy.V1":
        # This is a construction choice, not a rollback of existing buffering.
        pass
    if strategy_id.startswith("Memory.Prefetch.") or strategy_id == "Pipeline.WarpRegisterPrefetchAB":
        updates.update({"pipeline.prefetch": strategy_id, "resource.register.prefetch_enabled": True})
    if strategy_id.startswith("Mapping.CTASwizzle."):
        updates["scheduling.cta_swizzle"] = strategy_id.rsplit(".", 1)[-1]
    if strategy_id.startswith("Memory.L2Reuse."):
        updates["memory.l2_reuse_policy"] = strategy_id.rsplit(".", 1)[-1]
    if strategy_id.startswith("Scheduling.WaveQuantization."):
        updates["scheduling.wave_quantization"] = True
    if strategy_id.startswith("Scheduling.PersistentCTA."):
        updates["scheduling.persistent_cta"] = strategy_id.rsplit(".", 1)[-1]
    if strategy_id.startswith("Reduction.StreamK."):
        updates["scheduling.work_decomposition"] = "stream_k"
    return updates


def synthesize_cpu_ir_updates(strategy_id: str) -> dict[str, Any]:
    updates: dict[str, Any] = {
        "target.backend": "cpu",
        "target.language": "c",
        "target.device": "cpu",
    }
    l2_block = re.fullmatch(r"CPU\.Tiling\.L2Block\.(\d+)x(\d+)x(\d+)", strategy_id)
    if l2_block:
        bm, bn, bk = map(int, l2_block.groups())
        updates.update({
            "cpu_tiling.l2_block_m": bm,
            "cpu_tiling.l2_block_n": bn,
            "cpu_tiling.l2_block_k": bk,
            "cpu_tiling.l2_policy": "capacity_checked",
        })
    l1_block = re.fullmatch(r"CPU\.Tiling\.L1Block\.(\d+)x(\d+)x(\d+)", strategy_id)
    if l1_block:
        bm, bn, bk = map(int, l1_block.groups())
        updates.update({
            "cpu_tiling.l1_block_m": bm,
            "cpu_tiling.l1_block_n": bn,
            "cpu_tiling.l1_block_k": bk,
            "cpu_tiling.l1_policy": "capacity_checked",
        })
    legacy_cache_block = re.fullmatch(r"CPU\.Tiling\.CacheBlock\.(\d+)x(\d+)x(\d+)", strategy_id)
    if legacy_cache_block:
        bm, bn, bk = map(int, legacy_cache_block.groups())
        updates.update({
            "cpu_tiling.cache_block_m": bm,
            "cpu_tiling.cache_block_n": bn,
            "cpu_tiling.cache_block_k": bk,
            "cpu_tiling.l2_block_m": bm,
            "cpu_tiling.l2_block_n": bn,
            "cpu_tiling.l2_block_k": bk,
        })
    register_block = re.fullmatch(r"CPU\.Tiling\.RegisterBlock\.(\d+)x(\d+)", strategy_id)
    if register_block:
        rm, rn = map(int, register_block.groups())
        updates.update({"cpu_tiling.register_m": rm, "cpu_tiling.register_n": rn})
    if strategy_id.startswith("CPU.LoopOrder."):
        updates["cpu_schedule.loop_order"] = strategy_id.rsplit(".", 1)[-1].lower()
    if strategy_id.startswith("CPU.KLoop.Unroll"):
        factor = strategy_id.rsplit("Unroll", 1)[-1]
        updates["cpu_schedule.k_unroll"] = coerce_int(factor) or 1
    if strategy_id == "CPU.KLoop.NoUnroll":
        updates["cpu_schedule.k_unroll"] = 1
    if strategy_id.startswith("CPU.Vectorization."):
        policy = strategy_id.rsplit(".", 1)[-1]
        updates.update({
            "cpu_vectorization.policy": snake_case(policy),
            "cpu_vectorization.tail_handling": True,
            "cpu_vectorization.lane_mapping": "n_contiguous",
            "cpu_vectorization.a_load_policy": "scalar_broadcast",
        })
    if strategy_id.startswith("CPU.Parallel."):
        updates["cpu_parallel.policy"] = snake_case(strategy_id.removeprefix("CPU.Parallel."))
    if strategy_id == "CPU.Memory.NoPack":
        updates.update({
            "cpu_memory.pack_a": False,
            "cpu_memory.pack_b": False,
            "cpu_memory.pack_layout": "none",
            "cpu_memory.pack_buffer_scope": "none",
            "cpu_memory.prefetch": "disabled",
        })
    if strategy_id == "CPU.Memory.PackB.ContiguousPanel":
        updates.update({
            "cpu_memory.pack_a": False,
            "cpu_memory.pack_b": True,
            "cpu_memory.pack_b_layout": "contiguous_panel",
            "cpu_memory.pack_layout": "pack_b_panel",
            "cpu_memory.pack_b_access_equivalent": True,
        })
    if strategy_id == "CPU.Memory.PackA.ContiguousPanel":
        updates.update({
            "cpu_memory.pack_a": True,
            "cpu_memory.pack_b": False,
            "cpu_memory.pack_a_layout": "contiguous_panel",
            "cpu_memory.pack_layout": "pack_a_panel",
            "cpu_memory.pack_a_access_equivalent": True,
        })
    if strategy_id == "CPU.Memory.PackAB.PanelMajor":
        updates.update({
            "cpu_memory.pack_a": True,
            "cpu_memory.pack_b": True,
            "cpu_memory.pack_a_layout": "panel_major",
            "cpu_memory.pack_b_layout": "panel_major",
            "cpu_memory.pack_layout": "pack_ab_panel_major",
            "cpu_memory.pack_a_access_equivalent": True,
            "cpu_memory.pack_b_access_equivalent": True,
        })
    if strategy_id == "CPU.Packing.PackA.MRxKC":
        updates.update({
            "cpu_memory.pack_a": True,
            "cpu_memory.pack_b": get_path_value_default_false(updates, "cpu_memory.pack_b"),
            "cpu_memory.pack_a_layout": "mr_x_kc",
            "cpu_memory.pack_layout": "pack_a_mr_x_kc",
            "cpu_memory.pack_a_access_equivalent": True,
        })
    if strategy_id == "CPU.Packing.PackB.KCxNR":
        updates.update({
            "cpu_memory.pack_a": get_path_value_default_false(updates, "cpu_memory.pack_a"),
            "cpu_memory.pack_b": True,
            "cpu_memory.pack_b_layout": "kc_x_nr",
            "cpu_memory.pack_layout": "pack_b_kc_x_nr",
            "cpu_memory.pack_b_access_equivalent": True,
        })
    if strategy_id == "CPU.Packing.PackAB.MRxKC_KCxNR":
        updates.update({
            "cpu_memory.pack_a": True,
            "cpu_memory.pack_b": True,
            "cpu_memory.pack_a_layout": "mr_x_kc",
            "cpu_memory.pack_b_layout": "kc_x_nr",
            "cpu_memory.pack_layout": "pack_ab_mr_x_kc_kc_x_nr",
            "cpu_memory.pack_a_access_equivalent": True,
            "cpu_memory.pack_b_access_equivalent": True,
        })
    if strategy_id == "CPU.Memory.PackBuffer.ThreadPrivate":
        updates["cpu_memory.pack_buffer_scope"] = "thread_private"
    if strategy_id == "CPU.Memory.PackBuffer.TilePrivate":
        updates["cpu_memory.pack_buffer_scope"] = "tile_private"
    if strategy_id.startswith("CPU.Memory.Prefetch."):
        updates["cpu_memory.prefetch"] = snake_case(strategy_id.rsplit(".", 1)[-1])
    microkernel = re.fullmatch(r"CPU\.MicroKernel\.(Scalar|AVX2|AVX512)(?:\.FMA)?\.(\d+)x(\d+)", strategy_id)
    if microkernel:
        family, mr, nr = microkernel.groups()
        family_name = family.lower()
        if family_name == "avx2":
            family_name = "avx2_fma"
        if family_name == "avx512":
            family_name = "avx512_fma"
        updates.update({
            "cpu_microkernel.family": family_name,
            "cpu_microkernel.mr": int(mr),
            "cpu_microkernel.nr": int(nr),
            "cpu_microkernel.uses_intrinsics": family.lower() != "Scalar",
            "cpu_microkernel.k_reduction_policy": "sequential_complete",
        })
        if family.lower() != "Scalar":
            updates.update({
                "cpu_microkernel.simd_lane_mapping": "n_contiguous",
                "cpu_microkernel.a_operand_policy": "scalar_broadcast",
            })
    if strategy_id == "CPU.Vectorization.AVX2.FMA.Explicit":
        updates.update({
            "cpu_vectorization.policy": "explicit_intrinsics",
            "cpu_vectorization.isa": "avx2_fma",
            "cpu_vectorization.tail_handling": True,
            "cpu_vectorization.lane_mapping": "n_contiguous",
            "cpu_vectorization.a_load_policy": "scalar_broadcast",
        })
    if strategy_id == "CPU.Vectorization.AVX512.FMA.Explicit":
        updates.update({
            "cpu_vectorization.policy": "explicit_intrinsics",
            "cpu_vectorization.isa": "avx512_fma",
            "cpu_vectorization.tail_handling": True,
            "cpu_vectorization.lane_mapping": "n_contiguous",
            "cpu_vectorization.a_load_policy": "scalar_broadcast",
        })
    if strategy_id == "CPU.LoopOrder.PackedPanelMajor":
        updates["cpu_schedule.loop_order"] = "packed_panel_major"
    if strategy_id == "CPU.LoopOrder.OpenBLASPanelMajor":
        updates["cpu_schedule.loop_order"] = "openblas_panel_major"
    if strategy_id == "CPU.MacroKernel.OpenBLASStyle.PanelDriver":
        updates.update({
            "cpu_macro_kernel.driver": "openblas_style_panel_driver",
            "cpu_macro_kernel.n_panel_outer": True,
            "cpu_macro_kernel.k_panel_middle": True,
            "cpu_macro_kernel.m_panel_inner": True,
            "cpu_macro_kernel.pack_b_reuse_across_m": True,
            "cpu_macro_kernel.calls_microkernel_grid": True,
        })
    if strategy_id == "CPU.MacroKernel.DirectTiled.Driver":
        updates.update({
            "cpu_macro_kernel.driver": "direct_tiled_driver",
            "cpu_macro_kernel.direct_row_major": True,
            "cpu_macro_kernel.pack_b_reuse_across_m": False,
            "cpu_macro_kernel.calls_microkernel_grid": True,
        })
    if strategy_id in {"CPU.Parallel.OpenMP.RowBlock", "CPU.Parallel.OpenMP.Collapse2", "CPU.Threading.OpenMP.TilePartition"}:
        updates["cpu_parallel.output_tile_exclusive"] = True
    if strategy_id == "CPU.Threading.OpenMP.TilePartition":
        updates["cpu_parallel.policy"] = "openmp_tile_partition"
    if strategy_id == "CPU.Compiler.PortableO2":
        updates.update({
            "cpu_compiler.optimization_level": "O2",
            "cpu_compiler.native_arch": False,
            "cpu_compiler.openmp": False,
            "cpu_compiler.vector_isa": "portable",
        })
    if strategy_id == "CPU.Compiler.NativeO3":
        updates.update({
            "cpu_compiler.optimization_level": "O3",
            "cpu_compiler.native_arch": True,
            "cpu_compiler.openmp": False,
            "cpu_compiler.vector_isa": "native",
        })
    if strategy_id == "CPU.Compiler.NativeO3OpenMP":
        updates.update({
            "cpu_compiler.optimization_level": "O3",
            "cpu_compiler.native_arch": True,
            "cpu_compiler.openmp": True,
            "cpu_compiler.vector_isa": "native",
        })
    if strategy_id == "CPU.Compiler.MSVC.AVX2OpenMP":
        updates.update({
            "cpu_compiler.optimization_level": "O2",
            "cpu_compiler.native_arch": False,
            "cpu_compiler.openmp": True,
            "cpu_compiler.vector_isa": "avx2",
        })
    if strategy_id.startswith("CPU.Epilogue.Store."):
        updates.update({
            "cpu_epilogue.store_policy": snake_case(strategy_id.rsplit(".", 1)[-1]),
            "cpu_tail.full_tile_fast_path": False,
            "cpu_tail.scalar_cleanup": True,
        })
    if strategy_id == "CPU.TailKernel.FullTileFastPathScalarCleanup":
        updates.update({
            "cpu_tail.full_tile_fast_path": True,
            "cpu_tail.scalar_cleanup": True,
            "cpu_safety.boundary_policy": "full_tile_fast_path_with_scalar_cleanup",
        })
    return updates


def get_path_value_default_false(updates: dict[str, Any], key: str) -> Any:
    return updates.get(key, False)


def snake_case(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return value.replace(".", "_").replace("-", "_").lower()


def initialize_default_field_meta(ir: dict[str, Any]) -> None:
    for path in [
        "tiling.thread_m",
        "tiling.thread_n",
        "tiling.warp_tile.warp_m_iter",
        "tiling.warp_tile.warp_n_iter",
    ]:
        if get_path(ir, path) is not None and get_field_meta(ir, path) is None:
            set_field_meta(
                ir,
                path,
                origin="default_template",
                resolved=False,
                resolved_by=None,
                subphase=None,
            )


def get_field_meta(ir: dict[str, Any], path: str) -> dict[str, Any] | None:
    meta = ir.get("field_meta", {})
    item = meta.get(path)
    return item if isinstance(item, dict) else None


def set_field_meta(
    ir: dict[str, Any],
    path: str,
    origin: str,
    resolved: bool,
    resolved_by: str | None,
    subphase: str | None,
) -> None:
    ir.setdefault("field_meta", {})[path] = {
        "origin": origin,
        "resolved": resolved,
        "resolved_by": resolved_by,
        "subphase": subphase,
    }


def field_is_resolved(ir: dict[str, Any], path: str) -> bool:
    if get_path(ir, path) is None:
        return False
    meta = get_field_meta(ir, path)
    if meta is None:
        return True
    return meta.get("resolved") is True and meta.get("origin") != "default_template"


def field_resolved_by_subphase(ir: dict[str, Any], path: str, subphase_id: str) -> bool:
    if get_path(ir, path) is None:
        return False
    meta = get_field_meta(ir, path)
    if meta is None:
        return False
    if meta.get("origin") == "default_template":
        return False
    if meta.get("subphase") == subphase_id:
        return meta.get("resolved") is True
    return meta.get("origin") == "derived" and meta.get("resolved") is True


def mark_completed_subphase(ir: dict[str, Any], subphase_id: str | None) -> None:
    if not subphase_id:
        return
    completed = ir.setdefault("strategy", {}).setdefault("completed_subphases", [])
    if subphase_id not in completed:
        completed.append(subphase_id)


def mark_skipped_subphase(ir: dict[str, Any], subphase_id: str | None) -> None:
    if not subphase_id:
        return
    skipped = ir.setdefault("strategy", {}).setdefault("optional_skipped_subphases", [])
    if subphase_id not in skipped:
        skipped.append(subphase_id)


def record_micro_strategy_application(
    ir: dict[str, Any],
    stage: str,
    subphase_id: str | None,
    strategy_id: str,
    ir_updates: dict[str, Any],
    local_results: list[LocalCheckResult],
) -> None:
    accepted = all(item.status == "pass" for item in local_results)
    record = {
        "stage": stage,
        "subphase": subphase_id,
        "strategy_id": strategy_id,
        "ir_updates": copy.deepcopy(ir_updates),
        "status": "local_pass" if accepted else "local_fail",
    }
    applied = ir.setdefault("strategy", {}).setdefault("applied_micro_strategies", [])
    applied.append(record)
    if accepted and subphase_completion_record_is_sufficient(subphase_id, strategy_id):
        mark_completed_subphase(ir, subphase_id)


def subphase_completion_record_is_sufficient(subphase_id: str | None, strategy_id: str) -> bool:
    if subphase_id == "Tiling.ThreadTileSelection":
        return strategy_id.startswith("Tiling.ThreadTile.")
    return subphase_id is not None


def apply_defaults(ir: dict[str, Any], subphase_id: str | None = None) -> None:
    if get_path(ir, "tiling.warp_tile.warp_m_iter") is None:
        set_path(ir, "tiling.warp_tile.warp_m_iter", 1)
        set_field_meta(
            ir,
            "tiling.warp_tile.warp_m_iter",
            origin="derived_default",
            resolved=True,
            resolved_by=f"{subphase_id}.default_iteration" if subphase_id else "default_iteration",
            subphase=subphase_id,
        )
    if get_path(ir, "tiling.warp_tile.warp_n_iter") is None:
        set_path(ir, "tiling.warp_tile.warp_n_iter", 1)
        set_field_meta(
            ir,
            "tiling.warp_tile.warp_n_iter",
            origin="derived_default",
            resolved=True,
            resolved_by=f"{subphase_id}.default_iteration" if subphase_id else "default_iteration",
            subphase=subphase_id,
        )


def clean_ir_updates(updates: dict[str, Any]) -> dict[str, Any]:
    placeholders = {"int", "float", "bool", "boolean", "string", "number", "null", "none"}
    cleaned = {}
    for path, value in updates.items():
        if isinstance(value, str) and value.strip().lower() in placeholders:
            continue
        cleaned[path] = value
    return cleaned


def normalize_numeric_fields(ir: dict[str, Any]) -> None:
    numeric_paths = [
        "tiling.block_m",
        "tiling.block_n",
        "tiling.block_k",
        "tiling.warp_tile.warp_m",
        "tiling.warp_tile.warp_n",
        "tiling.warp_tile.warp_m_iter",
        "tiling.warp_tile.warp_n_iter",
        "tiling.thread_m",
        "tiling.thread_n",
        "mapping.warps_m",
        "mapping.warps_n",
        "mapping.warps_per_block",
        "mapping.threads_per_block",
        "hardware.warp_size",
        "hardware.max_threads_per_block",
        "resource.shared_memory.pipeline_multiplier",
        "cpu_tiling.l2_block_m",
        "cpu_tiling.l2_block_n",
        "cpu_tiling.l2_block_k",
        "cpu_tiling.l1_block_m",
        "cpu_tiling.l1_block_n",
        "cpu_tiling.l1_block_k",
        "cpu_tiling.cache_block_m",
        "cpu_tiling.cache_block_n",
        "cpu_tiling.cache_block_k",
        "cpu_tiling.register_m",
        "cpu_tiling.register_n",
        "cpu_resource.l1_working_set_bytes",
        "cpu_resource.l2_working_set_bytes",
        "cpu_resource.l1_budget_bytes",
        "cpu_resource.l2_budget_bytes",
        "cpu_resource.register_accumulator_count",
    ]
    for path in numeric_paths:
        value = get_path(ir, path)
        coerced = coerce_int(value)
        if coerced is not None:
            set_path(ir, path, coerced)


def derive_fields(ir: dict[str, Any], subphase_id: str | None = None, strategy_id: str | None = None) -> None:
    bm = coerce_int(get_path(ir, "tiling.block_m"))
    bn = coerce_int(get_path(ir, "tiling.block_n"))
    wm = coerce_int(get_path(ir, "tiling.warp_tile.warp_m"))
    wn = coerce_int(get_path(ir, "tiling.warp_tile.warp_n"))
    warp_size = coerce_int(get_path(ir, "hardware.warp_size")) or 32
    tm = coerce_int(get_path(ir, "tiling.thread_m"))
    tn = coerce_int(get_path(ir, "tiling.thread_n"))
    wmi = coerce_int(get_path(ir, "tiling.warp_tile.warp_m_iter")) or 1
    wni = coerce_int(get_path(ir, "tiling.warp_tile.warp_n_iter")) or 1
    if bm and wm and bm % wm == 0:
        set_path(ir, "mapping.warps_m", bm // wm)
        set_field_meta(ir, "mapping.warps_m", "derived", True, strategy_id, subphase_id)
    if bn and wn and bn % wn == 0:
        set_path(ir, "mapping.warps_n", bn // wn)
        set_field_meta(ir, "mapping.warps_n", "derived", True, strategy_id, subphase_id)
    warps_m = get_path(ir, "mapping.warps_m")
    warps_n = get_path(ir, "mapping.warps_n")
    if warps_m and warps_n:
        set_path(ir, "mapping.warps_per_block", warps_m * warps_n)
        set_path(ir, "mapping.threads_per_block", warps_m * warps_n * warp_size)
        set_field_meta(ir, "mapping.warps_per_block", "derived", True, strategy_id, subphase_id)
        set_field_meta(ir, "mapping.threads_per_block", "derived", True, strategy_id, subphase_id)
    if wm and wn and tm and tn:
        warp_iter = choose_warp_iteration_extents(wm, wn, tm, tn, warp_size)
        if warp_iter and (iteration_fields_are_auto(ir) or not warp_iteration_is_valid(wm, wn, tm, tn, wmi, wni, warp_size)):
            wmi, wni = warp_iter
            set_path(ir, "tiling.warp_tile.warp_m_iter", wmi)
            set_path(ir, "tiling.warp_tile.warp_n_iter", wni)
            set_field_meta(ir, "tiling.warp_tile.warp_m_iter", "derived_default", True, f"{subphase_id}.default_iteration" if subphase_id else strategy_id, subphase_id)
            set_field_meta(ir, "tiling.warp_tile.warp_n_iter", "derived_default", True, f"{subphase_id}.default_iteration" if subphase_id else strategy_id, subphase_id)
    if tm and tn:
        fragments = (wm // wmi) * (wn // wni) if wm and wn else 1
        set_path(ir, "mapping.outputs_per_thread", tm * tn * fragments)
        set_path(ir, "mapping.outputs_per_warp", warp_size * tm * tn * fragments)
        set_field_meta(ir, "mapping.outputs_per_thread", "derived", True, strategy_id, subphase_id)
        set_field_meta(ir, "mapping.outputs_per_warp", "derived", True, strategy_id, subphase_id)
    derive_resource_fields(ir, subphase_id, strategy_id)
    derive_cpu_resource_fields(ir, subphase_id, strategy_id)


def thread_tile_is_resolved(ir: dict[str, Any]) -> bool:
    return field_is_resolved(ir, "tiling.thread_m") and field_is_resolved(ir, "tiling.thread_n")


def tiling_hierarchy_reject_reason(strategy_id: str, ir: dict[str, Any]) -> str | None:
    if strategy_id in {
        "Tiling.WarpTile.WMxWN",
        "Tiling.WarpTile.ParametricWMxWN",
    }:
        return "parametric WarpTile placeholders are not selectable in hierarchical concrete tiling search"

    warp = re.fullmatch(r"Tiling\.WarpTile\.(\d+)x(\d+)", strategy_id)
    if warp:
        bm = coerce_int(get_path(ir, "tiling.block_m"))
        bn = coerce_int(get_path(ir, "tiling.block_n"))
        if bm is None or bn is None:
            return "BlockTile must be selected before WarpTile"
        wm, wn = map(int, warp.groups())
        if bm % wm != 0 or bn % wn != 0:
            return f"WarpTile {wm}x{wn} does not divide BlockTile {bm}x{bn}"
        warp_size = coerce_int(get_path(ir, "hardware.warp_size")) or 32
        max_threads = coerce_int(get_path(ir, "hardware.max_threads_per_block")) or 1024
        threads = (bm // wm) * (bn // wn) * warp_size
        if threads > max_threads:
            return f"derived threads_per_block {threads} exceeds hardware limit {max_threads}"

    thread = re.fullmatch(r"Tiling\.ThreadTile\.(\d+)x(\d+)", strategy_id)
    if thread:
        wm = coerce_int(get_path(ir, "tiling.warp_tile.warp_m"))
        wn = coerce_int(get_path(ir, "tiling.warp_tile.warp_n"))
        if wm is None or wn is None:
            return "WarpTile must be selected before ThreadTile"
        tm, tn = map(int, thread.groups())
        warp_size = coerce_int(get_path(ir, "hardware.warp_size")) or 32
        if choose_warp_iteration_extents(wm, wn, tm, tn, warp_size) is None:
            return f"ThreadTile {tm}x{tn} has no legal {warp_size}-lane mapping inside WarpTile {wm}x{wn}"
    return None


def choose_warp_iteration_extents(wm: int, wn: int, tm: int, tn: int, warp_size: int = 32) -> tuple[int, int] | None:
    candidates: list[tuple[float, int, int, int]] = []
    target_aspect = wm / wn if wn else 1.0
    for wmi in divisors(wm):
        if wmi % tm != 0:
            continue
        for wni in divisors(wn):
            if wni % tn != 0:
                continue
            if (wmi // tm) * (wni // tn) != warp_size:
                continue
            aspect_penalty = abs((wmi / wni) - target_aspect) if wni else float("inf")
            # For equal-aspect candidates, favor N-contiguous fragments because
            # consecutive lanes then touch adjacent C/B columns more often.
            candidates.append((aspect_penalty, -wni, -wmi, wmi, wni))
    if not candidates:
        return None
    _, _, _, wmi, wni = min(candidates)
    return wmi, wni


def warp_iteration_is_valid(wm: int, wn: int, tm: int, tn: int, wmi: int, wni: int, warp_size: int = 32) -> bool:
    if min(wm, wn, tm, tn, wmi, wni, warp_size) <= 0:
        return False
    return (
        wm % wmi == 0
        and wn % wni == 0
        and wmi % tm == 0
        and wni % tn == 0
        and (wmi // tm) * (wni // tn) == warp_size
    )


def divisors(value: int) -> list[int]:
    return [item for item in range(1, value + 1) if value % item == 0]


def iteration_fields_are_auto(ir: dict[str, Any]) -> bool:
    for path in ["tiling.warp_tile.warp_m_iter", "tiling.warp_tile.warp_n_iter"]:
        meta = get_field_meta(ir, path)
        if meta and meta.get("origin") == "micro_strategy":
            return False
    return True


def derive_resource_fields(ir: dict[str, Any], subphase_id: str | None = None, strategy_id: str | None = None) -> None:
    from SGPO.generate_ir.gpu_resources import pipeline_stages
    stages = pipeline_stages(ir)
    set_path(ir, "resource.shared_memory.pipeline_multiplier", stages)
    bm = coerce_int(get_path(ir, "tiling.block_m"))
    bn = coerce_int(get_path(ir, "tiling.block_n"))
    bk = coerce_int(get_path(ir, "tiling.block_k"))
    if bm and bn and bk and get_path(ir, "memory.use_shared_memory") is True:
        usage = shared_memory_usage(ir)
        set_path(ir, "resource.shared_memory.per_stage_bytes", usage["per_stage_bytes"])
        set_path(ir, "resource.shared_memory.total_bytes",
                 usage["per_stage_bytes"] * usage["stage_count"] + usage["auxiliary_bytes"])
        set_field_meta(ir, "resource.shared_memory.total_bytes", "derived", True, strategy_id, subphase_id)
    tm = coerce_int(get_path(ir, "tiling.thread_m")) or 1
    tn = coerce_int(get_path(ir, "tiling.thread_n")) or 1
    if get_path(ir, "resource.register.estimated_per_thread") is None:
        outputs = coerce_int(get_path(ir, "mapping.outputs_per_thread")) or tm * tn
        set_path(ir, "resource.register.estimated_per_thread", outputs + tm + tn + 8)
        set_field_meta(ir, "resource.register.estimated_per_thread", "derived", True, strategy_id, subphase_id)


def derive_cpu_resource_fields(ir: dict[str, Any], subphase_id: str | None = None, strategy_id: str | None = None) -> None:
    if get_path(ir, "target.backend") != "cpu":
        return
    dtype_bytes = 4
    l1_cache = coerce_int(get_path(ir, "hardware.cpu_cache.l1_data_cache_bytes")) or 32 * 1024
    l2_cache = coerce_int(get_path(ir, "hardware.cpu_cache.l2_cache_bytes")) or 1024 * 1024
    l1_budget = int(l1_cache * 0.75)
    l2_budget = int(l2_cache * 0.80)
    set_path(ir, "cpu_resource.l1_budget_bytes", l1_budget)
    set_path(ir, "cpu_resource.l2_budget_bytes", l2_budget)
    set_field_meta(ir, "cpu_resource.l1_budget_bytes", "derived", True, strategy_id, subphase_id)
    set_field_meta(ir, "cpu_resource.l2_budget_bytes", "derived", True, strategy_id, subphase_id)

    l2_m = coerce_int(get_path(ir, "cpu_tiling.l2_block_m"))
    l2_n = coerce_int(get_path(ir, "cpu_tiling.l2_block_n"))
    l2_k = coerce_int(get_path(ir, "cpu_tiling.l2_block_k"))
    if l2_m and l2_n and l2_k:
        l2_working_set = (l2_m * l2_k + l2_k * l2_n + l2_m * l2_n) * dtype_bytes
        set_path(ir, "cpu_resource.l2_working_set_bytes", l2_working_set)
        set_field_meta(ir, "cpu_resource.l2_working_set_bytes", "derived", True, strategy_id, subphase_id)

    l1_m = coerce_int(get_path(ir, "cpu_tiling.l1_block_m"))
    l1_n = coerce_int(get_path(ir, "cpu_tiling.l1_block_n"))
    l1_k = coerce_int(get_path(ir, "cpu_tiling.l1_block_k"))
    if l1_m and l1_n and l1_k:
        l1_working_set = (l1_m * l1_k + l1_k * l1_n + l1_m * l1_n) * dtype_bytes
        set_path(ir, "cpu_resource.l1_working_set_bytes", l1_working_set)
        set_field_meta(ir, "cpu_resource.l1_working_set_bytes", "derived", True, strategy_id, subphase_id)

    reg_m = coerce_int(get_path(ir, "cpu_tiling.register_m"))
    reg_n = coerce_int(get_path(ir, "cpu_tiling.register_n"))
    if reg_m and reg_n:
        set_path(ir, "cpu_resource.register_accumulator_count", reg_m * reg_n)
        set_field_meta(ir, "cpu_resource.register_accumulator_count", "derived", True, strategy_id, subphase_id)


def coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)
    return None


def vector_width_from_strategy(strategy_id: str, default: int = 1) -> int:
    match = re.search(r"float(\d+)", strategy_id)
    return int(match.group(1)) if match else default


def requirement_is_satisfied(ir: dict[str, Any], requirement: Any) -> bool:
    if not isinstance(requirement, str):
        return True
    if looks_like_text_condition(requirement):
        return evaluate_text_condition(ir, requirement).status == "pass"
    if "." in requirement and " " not in requirement:
        return get_path(ir, requirement) is not None
    text = requirement.lower()
    if "alignment guard or alignment proof" in text:
        return any(
            get_path(ir, field) is True
            for field in [
                "vectorization.A.alignment_guard",
                "vectorization.B.alignment_guard",
                "vectorization.A.alignment_proven",
                "vectorization.B.alignment_proven",
            ]
        )
    if "tail handling or static divisible policy" in text:
        return get_path(ir, "safety.tail_handling") is True or get_path(ir, "safety.boundary_policy") == "static_divisible_no_guard"
    if "c alignment policy" in text:
        return get_path(ir, "vectorization.C.alignment_guard") is True or get_path(ir, "vectorization.C.alignment_proven") is True
    if "c boundary policy" in text:
        return get_path(ir, "safety.boundary_policy") is not None
    return True


def looks_like_text_condition(requirement: str) -> bool:
    return bool(re.search(r"\s(?:<=|>=|!=|==|=|<|>)\s", requirement))


def run_local_checks(ir: dict[str, Any], subphase: dict[str, Any]) -> list[LocalCheckResult]:
    checks = subphase.get("local_validation") or subphase.get("local_checks") or []
    results = []
    for check in checks:
        if isinstance(check, str):
            results.append(evaluate_text_condition(ir, check))
        elif isinstance(check, dict):
            results.append(evaluate_predicate_like(ir, check))
    return results


def run_stage_static_code_checks(stage: str, ir: dict[str, Any]) -> list[LocalCheckResult]:
    if stage != "Layout" or get_path(ir, "memory.use_shared_memory") is not True:
        return []
    return [check_layout_shared_memory_declarations(ir)]


def check_layout_shared_memory_declarations(ir: dict[str, Any]) -> LocalCheckResult:
    code_ast = ir.get("code_ast") or {}
    if not code_ast.get("files"):
        return LocalCheckResult(
            "LAYOUT_SHARED_DECLARATIONS_PRESENT",
            "pass",
            "code_ast is unavailable; shared declaration check deferred to compile/code verification.",
        )
    shared_decls = []
    for file_ast in (code_ast.get("files") or {}).values():
        shared_decls.extend(file_ast.get("shared_memory") or [])
    if not shared_decls:
        return LocalCheckResult(
            "LAYOUT_SHARED_DECLARATIONS_PRESENT",
            "fail",
            "memory.use_shared_memory is true, but no non-comment __shared__ declarations were found in code_ast.",
        )
    has_a = any(is_shared_buffer_name(decl.get("name"), ["As", "shared_A", "sA"]) for decl in shared_decls)
    has_b = any(is_shared_buffer_name(decl.get("name"), ["Bs", "shared_B", "sB"]) for decl in shared_decls)
    if has_a and has_b:
        names = [decl.get("name") for decl in shared_decls]
        return LocalCheckResult("LAYOUT_SHARED_DECLARATIONS_PRESENT", "pass", f"shared declarations found: {names}")
    missing = []
    if not has_a:
        missing.append("A")
    if not has_b:
        missing.append("B")
    return LocalCheckResult(
        "LAYOUT_SHARED_DECLARATIONS_PRESENT",
        "fail",
        f"missing shared memory declaration for: {', '.join(missing)}",
    )


def is_shared_buffer_name(name: Any, candidates: list[str]) -> bool:
    if not isinstance(name, str):
        return False
    lowered = name.lower()
    return any(lowered == candidate.lower() or lowered.startswith(candidate.lower()) for candidate in candidates)


def evaluate_text_condition(ir: dict[str, Any], condition: str) -> LocalCheckResult:
    known = evaluate_known_constraint(ir, condition)
    if known is not None:
        return known
    if "!=" in condition and "null" in condition:
        field = condition.split("!=")[0].strip()
        ok = get_path(ir, field) is not None
        return LocalCheckResult(condition, "pass" if ok else "fail", condition)
    compare = re.fullmatch(r"(.+?)\s(<=|>=|<|>)\s(.+)", condition)
    if compare:
        left, op, right = [part.strip() for part in compare.groups()]
        lhs = eval_expr(ir, left)
        rhs = eval_expr(ir, right)
        ok = False
        if lhs is not None and rhs is not None:
            ok = {
                "<=": lhs <= rhs,
                ">=": lhs >= rhs,
                "<": lhs < rhs,
                ">": lhs > rhs,
            }[op]
        return LocalCheckResult(condition, "pass" if ok else "fail", condition)
    if "==" in condition:
        left, right = [part.strip() for part in condition.split("==", 1)]
        ok = eval_expr(ir, left) == eval_expr(ir, right)
        return LocalCheckResult(condition, "pass" if ok else "fail", condition)
    single_eq = re.fullmatch(r"(.+?)\s=\s(.+)", condition)
    if single_eq:
        left, right = [part.strip() for part in single_eq.groups()]
        ok = eval_expr(ir, left) == eval_expr(ir, right)
        return LocalCheckResult(condition, "pass" if ok else "fail", condition)
    return LocalCheckResult(condition, "pass", f"deferred or informational: {condition}")


def evaluate_known_constraint(ir: dict[str, Any], condition: str) -> LocalCheckResult | None:
    constraint_id = condition.strip()
    if constraint_id == "shared_memory_bytes * pipeline_multiplier <= hardware.max_shared_memory_per_block_bytes":
        from SGPO.generate_ir.gpu_resources import shared_memory_limit
        usage = shared_memory_usage(ir)
        limit = shared_memory_limit(ir)
        total = usage["per_stage_bytes"] * usage["stage_count"] + usage["auxiliary_bytes"]
        ok = limit > 0 and total <= limit
        return LocalCheckResult(constraint_id, "pass" if ok else "fail",
                                f"shared-memory total {total} bytes <= limit {limit} bytes (stages counted once)")
    if constraint_id == "C_VECTOR_ALIGNMENT":
        failures = []
        for tensor in ["A", "B", "C"]:
            node = get_path(ir, f"vectorization.{tensor}") or {}
            width = coerce_int(node.get("vector_width")) or 1 if isinstance(node, dict) else 1
            if width > 1 and not (node.get("alignment_guard") is True or node.get("alignment_proven") is True):
                failures.append(tensor)
        return LocalCheckResult(
            constraint_id,
            "pass" if not failures else "fail",
            "all vectorized accesses have alignment guard or proof" if not failures else f"missing alignment guard/proof for {failures}",
        )
    if constraint_id == "C_VECTOR_TAIL_HANDLING":
        static_no_guard = get_path(ir, "safety.boundary_policy") == "static_divisible_no_guard"
        failures = []
        for tensor in ["A", "B", "C"]:
            node = get_path(ir, f"vectorization.{tensor}") or {}
            width = coerce_int(node.get("vector_width")) or 1 if isinstance(node, dict) else 1
            if width > 1 and node.get("tail_handling") is not True and not static_no_guard:
                failures.append(tensor)
        return LocalCheckResult(
            constraint_id,
            "pass" if not failures else "fail",
            "all vectorized accesses have tail handling or static divisible policy" if not failures else f"missing tail handling for {failures}",
        )
    if constraint_id == "C_STORE_BOUNDARY_ALIGNMENT":
        c_node = get_path(ir, "vectorization.C") or {}
        width = coerce_int(c_node.get("vector_width")) or 1 if isinstance(c_node, dict) else 1
        boundary_ok = get_path(ir, "memory.global_store_C.boundary_guard") is True or get_path(ir, "safety.boundary_policy") == "static_divisible_no_guard"
        alignment_ok = width <= 1 or c_node.get("alignment_guard") is True or c_node.get("alignment_proven") is True
        ok = boundary_ok and alignment_ok
        return LocalCheckResult(
            constraint_id,
            "pass" if ok else "fail",
            f"boundary_ok={boundary_ok}, alignment_ok={alignment_ok}, vector_width={width}",
        )
    if constraint_id == "C_SHARED_MEMORY_LIMIT":
        shared = coerce_int(get_path(ir, "resource.shared_memory.total_bytes"))
        limit = coerce_int(get_path(ir, "hardware.max_shared_memory_per_block_bytes"))
        ok = shared is not None and limit is not None and shared <= limit
        return LocalCheckResult(constraint_id, "pass" if ok else "fail", f"{shared} <= {limit}")
    return None


def evaluate_predicate_like(ir: dict[str, Any], predicate: dict[str, Any]) -> LocalCheckResult:
    pid = predicate.get("id") or predicate.get("constraint_id") or "predicate"
    predicate_op = predicate.get("op")
    if predicate_op == "all_not_null":
        missing = [field for field in predicate.get("fields", []) if get_path(ir, field) is None]
        return LocalCheckResult(pid, "pass" if not missing else "fail", f"missing={missing}")
    if predicate_op in {"is_not_null", "not_null", "nonnull"}:
        lhs_operand = predicate.get("lhs") or predicate.get("field")
        lhs = resolve_local_predicate_operand(ir, lhs_operand)
        return LocalCheckResult(pid, "pass" if lhs is not None else "fail", f"{render_local_operand(lhs_operand)} is not null")
    if predicate_op in {"is_null", "null"}:
        lhs_operand = predicate.get("lhs") or predicate.get("field")
        lhs = resolve_local_predicate_operand(ir, lhs_operand)
        return LocalCheckResult(pid, "pass" if lhs is None else "fail", f"{render_local_operand(lhs_operand)} is null")
    lhs = resolve_local_predicate_operand(ir, predicate.get("lhs") or predicate.get("expr") or predicate.get("field"))
    rhs = predicate.get("const")
    if "rhs_value" in predicate:
        rhs = predicate["rhs_value"]
    if "rhs" in predicate:
        rhs = resolve_local_predicate_operand(ir, predicate["rhs"])
    if "rhs_expr" in predicate:
        rhs = eval_expr(ir, predicate["rhs_expr"])
    if "rhs_field" in predicate:
        rhs = get_path(ir, predicate["rhs_field"])
    op = predicate.get("op")
    ok = compare_local_values(lhs, rhs, op)
    return LocalCheckResult(pid, "pass" if ok else "fail", f"{lhs} {op} {rhs}")


def format_precondition_rejection(item: dict[str, Any]) -> str:
    check_id = str(item.get("id") or "unknown_precondition")
    status = str(item.get("status") or "unknown")
    detail = item.get("detail") or {}
    evaluated = detail.get("evaluated") or []
    reason = next((entry.get("reason") for entry in evaluated if isinstance(entry, dict) and entry.get("reason")), None)
    return f"{check_id}({status}{': ' + str(reason) if reason else ''})"


def resolve_local_predicate_operand(ir: dict[str, Any], operand: Any) -> Any:
    if isinstance(operand, dict):
        if "const" in operand:
            return operand["const"]
        if "field" in operand:
            return get_path(ir, operand["field"])
        if "expr" in operand:
            return eval_expr(ir, operand["expr"])
        return None
    return eval_expr(ir, operand)


def render_local_operand(operand: Any) -> str:
    if isinstance(operand, dict):
        return str(operand.get("field") or operand.get("expr") or operand.get("const"))
    return str(operand)


def compare_local_values(lhs: Any, rhs: Any, op: str | None) -> bool:
    if op in {"eq", "=", "=="}:
        return lhs == rhs
    if op in {"ne", "!=", "not_eq"}:
        return lhs != rhs
    if lhs is None or rhs is None:
        return False
    try:
        if op in {"le", "<="}:
            return lhs <= rhs
        if op in {"lt", "<"}:
            return lhs < rhs
        if op in {"ge", ">="}:
            return lhs >= rhs
        if op in {"gt", ">"}:
            return lhs > rhs
    except TypeError:
        lhs_text = str(lhs)
        rhs_text = str(rhs)
        if op in {"le", "<="}:
            return lhs_text <= rhs_text
        if op in {"lt", "<"}:
            return lhs_text < rhs_text
        if op in {"ge", ">="}:
            return lhs_text >= rhs_text
        if op in {"gt", ">"}:
            return lhs_text > rhs_text
    return True


def eval_expr(ir: dict[str, Any], expr: Any) -> Any:
    if expr is None:
        return None
    if isinstance(expr, (int, float)):
        return expr
    text = str(expr)
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    if text.lower() == "null":
        return None
    for field in sorted(re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+", text), key=len, reverse=True):
        value = get_path(ir, field)
        if value is None:
            return None
        text = text.replace(field, str(value))
    for name in sorted(FIELD_ALIASES, key=len, reverse=True):
        if not re.search(rf"\b{name}\b", text):
            continue
        value = get_path(ir, name)
        if value is None:
            return None
        text = re.sub(rf"\b{name}\b", str(value), text)
    try:
        return int(eval(text, {"__builtins__": {}}, {}))
    except Exception:
        value = get_path(ir, str(expr))
        if value is not None:
            return value
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(expr)):
            return str(expr)
        return None


def get_path(data: dict[str, Any], path: str | None) -> Any:
    if not path:
        return None
    if path in FIELD_ALIASES:
        aliased = get_path(data, FIELD_ALIASES[path])
        if aliased is not None:
            return aliased
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def set_path(data: dict[str, Any], path: str, value: Any) -> None:
    current = data
    parts = path.split(".")
    for part in parts[:-1]:
        node = current.get(part)
        if not isinstance(node, dict):
            node = {}
            current[part] = node
        current = node
    current[parts[-1]] = value


def strategy_stage(strategy_id: str) -> str:
    return strategy_id.split(".", 1)[0]


def is_mapping_for_stage(strategy_id: str, stage: str) -> bool:
    return stage == "Tiling" and strategy_id.startswith("Mapping.")


def empty_candidate_index(strategy_index: dict[str, Any], stage: str, subphase: str | None) -> dict[str, Any]:
    result = copy.deepcopy(strategy_index)
    result["strategies"] = []
    result["strategy_count"] = 0
    result["filter_context"] = {"current_stage": stage, "current_subphase": subphase}
    return result
