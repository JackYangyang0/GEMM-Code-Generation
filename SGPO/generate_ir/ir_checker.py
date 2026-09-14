"""Constraint checker for SGPO OptIR.

The checker is used before code generation to decide whether a strategy can be
applied, and after code generation to validate IR_after against strategy
postconditions and hard constraints.
"""

from __future__ import annotations

import argparse
import copy
import math
import operator
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from SGPO.utils.common_utils import load_json, save_json
from SGPO.generate_ir.gpu_resources import shared_memory_limit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IR = ROOT / "data" / "IRs" / "ir_patch" / "optir.extracted.json"
DEFAULT_STRATEGY_LIBRARY = ROOT / "data" / "lib" / "strategy_library.json"

OPS = {
    "=": operator.eq,
    "eq": operator.eq,
    "!=": operator.ne,
    "ne": operator.ne,
    "<": operator.lt,
    "lt": operator.lt,
    "<=": operator.le,
    "le": operator.le,
    ">": operator.gt,
    "gt": operator.gt,
    ">=": operator.ge,
    "ge": operator.ge,
}


@dataclass
class CheckResult:
    id: str
    status: str
    message: str
    failure_type: str | None = None
    detail: dict[str, Any] | None = None


def check_before_codegen(ir, strategy=None):
    precondition_results = []
    if strategy:
        precondition_results = [
            evaluate_condition(ir, condition, condition_check_id("PRECONDITION", condition))
            for condition in get_precondition_predicates(strategy)
        ]

    preconditions_ok = all(result.status == "pass" for result in precondition_results)

    return {
        "phase": "before_codegen",
        "strategy_id": strategy.get("strategy_id") if strategy else None,
        "strategy_applicable": preconditions_ok,
        "preconditions_ok": preconditions_ok,
        "hard_constraints_checked": False,
        "hard_constraints_ok": None,
        "results": serialize_results(precondition_results),
    }


def check_after_codegen(ir_after, strategy=None, include_hard_constraints=True):
    postcondition_results = []
    if strategy:
        postcondition_results = [
            evaluate_condition(ir_after, condition, condition_check_id("POSTCONDITION", condition))
            for condition in get_patch_ir_postcondition_predicates(strategy)
        ]

    hard_results = check_hard_constraints(ir_after) if include_hard_constraints else []
    soft_results = check_soft_constraints(ir_after) if include_hard_constraints else []
    postconditions_ok = all(result.status == "pass" for result in postcondition_results)
    hard_ok = all(result.status == "pass" for result in hard_results)

    return {
        "phase": "after_codegen",
        "strategy_id": strategy.get("strategy_id") if strategy else None,
        "accepted_by_ir_checker": postconditions_ok and hard_ok,
        "postconditions_ok": postconditions_ok,
        "hard_constraints_checked": include_hard_constraints,
        "hard_constraints_ok": hard_ok if include_hard_constraints else None,
        "results": serialize_results(postcondition_results + hard_results + soft_results),
    }


def check_code_verification(ir_after: dict[str, Any], strategy: dict[str, Any] | None = None) -> dict[str, Any]:
    constraints = get_code_verification_constraints(strategy)
    results = [evaluate_code_constraint(ir_after, constraint) for constraint in constraints]
    required_ok = all(result.status == "pass" for result in results if is_required_code_constraint(result.detail))
    no_required = not any(is_required_code_constraint(result.detail) for result in results)
    runtime_ok = ir_after.get("verification", {}).get("runtime_safety", {}).get("status") in {None, "pass"}
    correctness_ok = ir_after.get("verification", {}).get("correctness", {}).get("status") in {None, "pass"}
    return {
        "phase": "after_code_patch_applied",
        "strategy_id": strategy.get("strategy_id") if strategy else None,
        "accepted_by_code_verifier": (required_ok or no_required) and runtime_ok and correctness_ok,
        "required_constraints_ok": required_ok or no_required,
        "runtime_safety_ok": runtime_ok,
        "correctness_ok": correctness_ok,
        "results": serialize_results(results),
    }


def get_precondition_predicates(strategy: dict[str, Any] | None) -> list[Any]:
    if not strategy:
        return []
    preconditions = strategy.get("preconditions", [])
    if isinstance(preconditions, dict):
        return preconditions.get("predicates", []) or []
    return preconditions or []


def get_patch_ir_postcondition_predicates(strategy: dict[str, Any] | None) -> list[Any]:
    if not strategy:
        return []
    postconditions = strategy.get("postconditions", [])
    if isinstance(postconditions, dict):
        patch_ir = postconditions.get("patch_ir_verification", {})
        return patch_ir.get("predicates", []) or []
    return postconditions or []


def get_code_verification_constraints(strategy: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not strategy:
        return []
    postconditions = strategy.get("postconditions", {})
    if not isinstance(postconditions, dict):
        return []
    code_verification = postconditions.get("code_verification", {})
    return code_verification.get("constraints", []) or []


def evaluate_code_constraint(ir_after: dict[str, Any], constraint: dict[str, Any]) -> CheckResult:
    constraint_id = constraint.get("id") or constraint.get("constraint_id") or "CODE_CONSTRAINT"
    check_id = f"CODE_VERIFICATION:{constraint_id}"
    required = get_required_code_verifiers(constraint)
    best_effort = get_best_effort_code_verifiers(constraint)
    static_analysis = evaluate_best_effort_static_verifiers(ir_after, constraint, best_effort)

    if len(required) > 1:
        reports = []
        for verifier in required:
            single = copy.deepcopy(constraint)
            single["verifiers"] = {"required": [verifier], "best_effort": best_effort}
            reports.append(evaluate_code_constraint(ir_after, single))
        detail = {"constraint": constraint, "required": required, "checks": serialize_results(reports)}
        failed = next((item for item in reports if item.status == "fail"), None)
        if failed:
            return fail_result(check_id, failed.message, failed.failure_type, detail)
        if any(item.status != "pass" for item in reports):
            return unknown_result(check_id, "Required code verification is incomplete.", detail=detail)
        return pass_result(check_id, "All required code verifiers passed.", detail=detail)

    if "async_copy_evidence_check" in required:
        evidence = [node.get("async_copy", {}) for node in
                    (ir_after.get("code_ast", {}).get("files", {}) or {}).values()]
        detail = {"constraint": constraint, "required": required, "evidence": evidence,
                  "scope": "instruction/API presence only; not a synchronization proof"}
        if any(item.get("copy") and item.get("wait") and item.get("commit") for item in evidence):
            return pass_result(check_id, "Async copy and completion operations are present.", detail=detail)
        if not evidence:
            return unknown_result(check_id, "Async copy evidence has not been extracted.", detail=detail)
        return fail_result(check_id, "Expected asynchronous copy/commit/wait evidence is missing.",
                           "Pipeline.AsyncCopyProtocolMissing", detail)

    if "dynamic_reference_check" in required:
        correctness = ir_after.get("verification", {}).get("correctness", {})
        status = correctness.get("status")
        if status == "pass":
            return pass_result(
                check_id,
                f"{constraint.get('constraint_id')} satisfied by dynamic reference check.",
                detail={"constraint": constraint, "required": required, "best_effort": best_effort, "static_analysis": static_analysis},
            )
        if status == "fail":
            return fail_result(
                check_id,
                f"{constraint.get('constraint_id')} failed dynamic reference check.",
                "Semantic.CorrectnessMismatch",
                {
                    "constraint": constraint,
                    "required": required,
                    "best_effort": best_effort,
                    "static_analysis": static_analysis,
                    "correctness": correctness,
                },
            )
        return unknown_result(
            check_id,
            f"{constraint.get('constraint_id')} requires dynamic reference check that has not run.",
            detail={
                "constraint": constraint,
                "required": required,
                "best_effort": best_effort,
                "static_analysis": static_analysis,
                "correctness": correctness,
            },
        )

    if "runtime_safety_check" in required:
        runtime_safety = ir_after.get("verification", {}).get("runtime_safety", {})
        status = runtime_safety.get("status")
        if status == "pass":
            return pass_result(
                check_id,
                f"{constraint.get('constraint_id')} satisfied by runtime safety check.",
                detail={"constraint": constraint, "required": required, "best_effort": best_effort, "runtime_safety": runtime_safety},
            )
        if status != "fail":
            return unknown_result(check_id, "Runtime safety verification has not passed.",
                                  detail={"constraint": constraint, "required": required})
        return fail_result(
            check_id,
            f"{constraint.get('constraint_id')} failed runtime safety check.",
            "RuntimeSafety.CUDAError",
            {"constraint": constraint, "required": required, "best_effort": best_effort, "runtime_safety": runtime_safety},
        )

    if "ir_predicate_eval" in required:
        ir_eval = evaluate_code_ir_constraint(ir_after, constraint)
        if ir_eval.get("status") == "pass":
            return pass_result(
                check_id,
                f"{constraint.get('constraint_id')} satisfied by IR predicate check.",
                detail={"constraint": constraint, "required": required, "best_effort": best_effort, "ir_eval": ir_eval},
            )
        if ir_eval.get("status") == "fail":
            return fail_result(
                check_id,
                f"{constraint.get('constraint_id')} failed IR predicate check.",
                "StrategyCompliance.ConditionNotSatisfied",
                {"constraint": constraint, "required": required, "best_effort": best_effort, "ir_eval": ir_eval},
            )
        return unknown_result(
            check_id,
            f"{constraint.get('constraint_id')} IR predicate check is not implemented for this constraint.",
            detail={"constraint": constraint, "required": required, "best_effort": best_effort, "ir_eval": ir_eval},
        )

    if "static_code_pattern_check" in required:
        compile_status = ir_after.get("verification", {}).get("compile", {}).get("status")
        correctness_status = ir_after.get("verification", {}).get("correctness", {}).get("status")
        if compile_status in {None, "pass"} and correctness_status in {None, "pass"}:
            return pass_result(
                check_id,
                f"{constraint.get('constraint_id')} static pattern check deferred; compiled/correctness evidence accepted.",
                detail={
                    "constraint": constraint,
                    "required": required,
                    "best_effort": best_effort,
                    "static_analysis": static_analysis,
                    "deferred": "static_code_pattern_check",
                },
            )

    if any(verifier.startswith("static_") for verifier in required):
        return unknown_result(
            check_id,
            f"{constraint.get('constraint_id')} requires static code verifier not implemented yet.",
            detail={"constraint": constraint, "required": required, "best_effort": best_effort, "static_analysis": static_analysis},
        )

    return pass_result(
        check_id,
        f"{constraint.get('constraint_id', constraint_id)} has no required code verifier.",
        detail={"constraint": constraint, "required": required, "best_effort": best_effort, "static_analysis": static_analysis},
    )


def evaluate_code_ir_constraint(ir_after: dict[str, Any], constraint: dict[str, Any]) -> dict[str, Any]:
    constraint_id = constraint.get("constraint_id")
    args = constraint.get("args", {}) or {}
    if constraint_id == "C_TENSOR_CORE_PROFILE_ONLY":
        expected = args.get("profile")
        actual = ir_after.get("precision", {}).get("profile")
        return {"status": "pass" if actual == expected else "fail", "expected": expected, "actual": actual}
    profile = args.get("profile")
    if profile is not None:
        actual = ir_after.get("precision", {}).get("profile") or ir_after.get("strategy", {}).get("profile")
        return {"status": "pass" if actual == profile else "fail", "expected": profile, "actual": actual}
    return {"status": "unknown", "reason": "unsupported_ir_predicate_code_constraint"}


def evaluate_best_effort_static_verifiers(
    ir_after: dict[str, Any],
    constraint: dict[str, Any],
    best_effort: list[str],
) -> dict[str, Any]:
    result = {}
    if "static_loop_domain_check" in best_effort:
        result["static_loop_domain_check"] = evaluate_static_loop_domain_check(ir_after, constraint)
    if "static_index_pattern_check" in best_effort:
        result["static_index_pattern_check"] = {
            "status": "unknown",
            "reason": "static_index_pattern_check_not_implemented",
        }
    return result


def evaluate_static_loop_domain_check(ir_after: dict[str, Any], constraint: dict[str, Any]) -> dict[str, Any]:
    code_ast = ir_after.get("code_ast", {})
    loops = []
    for file_ast in (code_ast.get("files", {}) or {}).values():
        loops.extend(file_ast.get("loops", []) or [])
    k_loops = [loop for loop in loops if loop.get("mentions_k")]
    if not k_loops:
        return {"status": "unknown", "reason": "no_k_loop_found_in_code_ast"}
    has_k_domain = any("k0" in loop.get("header", "") or " k " in f" {loop.get('header', '')} " for loop in k_loops)
    return {
        "status": "pass" if has_k_domain else "unknown",
        "constraint_id": constraint.get("constraint_id"),
        "k_loop_count": len(k_loops),
        "k_loop_headers": [loop.get("header") for loop in k_loops],
    }


def get_required_code_verifiers(constraint: dict[str, Any]) -> list[str]:
    verifiers = constraint.get("verifiers")
    if isinstance(verifiers, dict):
        return verifiers.get("required", []) or []
    verifier = constraint.get("verifier")
    if isinstance(verifier, list):
        return verifier
    if isinstance(verifier, str):
        return [verifier]
    return []


def get_best_effort_code_verifiers(constraint: dict[str, Any]) -> list[str]:
    verifiers = constraint.get("verifiers")
    if isinstance(verifiers, dict):
        return verifiers.get("best_effort", []) or []
    return []


def is_required_code_constraint(detail: dict[str, Any] | None) -> bool:
    return bool((detail or {}).get("required"))


def check_hard_constraints(ir: dict[str, Any]) -> list[CheckResult]:
    return [
        check_threads_limit(ir),
        check_shared_memory_limit(ir),
        check_tiling_applied(ir),
        check_boundary_guard(ir),
        check_vector_alignment(ir),
        check_vector_tail_handling(ir),
        check_strategy_postconditions_field(ir),
    ]


def check_soft_constraints(ir: dict[str, Any]) -> list[CheckResult]:
    return [
        check_warp_multiple(ir),
        check_register_pressure(ir),
    ]


def check_threads_limit(ir: dict[str, Any]) -> CheckResult:
    threads = get_path(ir, "mapping.threads_per_block")
    if threads is None:
        threads = infer_threads_per_block(ir)
    max_threads = get_path(ir, "hardware.max_threads_per_block")
    if threads is None:
        return pass_result("C_THREADS_LIMIT", "threads_per_block is not set; limit check deferred.")
    if max_threads is None:
        return fail_result("C_THREADS_LIMIT", "hardware.max_threads_per_block is missing.", "Resource.MissingHardwareField")
    if threads <= max_threads:
        return pass_result("C_THREADS_LIMIT", f"threads_per_block={threads} <= max_threads_per_block={max_threads}.")
    return fail_result(
        "C_THREADS_LIMIT",
        f"threads_per_block={threads} exceeds max_threads_per_block={max_threads}.",
        "Resource.ThreadBlockOverflow",
    )


def infer_threads_per_block(ir: dict[str, Any]) -> int | None:
    block_m = get_path(ir, "tiling.block_m")
    block_n = get_path(ir, "tiling.block_n")
    thread_m = get_path(ir, "tiling.thread_m") or 1
    thread_n = get_path(ir, "tiling.thread_n") or 1
    if block_m is None or block_n is None:
        return None
    try:
        if block_m % thread_m != 0 or block_n % thread_n != 0:
            return None
        return int((block_m // thread_m) * (block_n // thread_n))
    except (TypeError, ZeroDivisionError):
        return None


def check_shared_memory_limit(ir: dict[str, Any]) -> CheckResult:
    used = estimate_shared_memory_bytes(ir)
    limit = shared_memory_limit(ir)
    if used is None:
        return pass_result("C_SHARED_MEMORY_LIMIT", "shared memory is not enabled; limit check deferred.")
    if not limit:
        return fail_result(
            "C_SHARED_MEMORY_LIMIT",
            "hardware.max_shared_memory_per_block_bytes is missing.",
            "Resource.MissingHardwareField",
        )
    if used <= limit:
        return pass_result("C_SHARED_MEMORY_LIMIT", f"shared_memory_bytes={used} <= limit={limit}.")
    return fail_result(
        "C_SHARED_MEMORY_LIMIT",
        f"shared_memory_bytes={used} exceeds limit={limit}.",
        "Resource.SharedMemoryOverflow",
    )


def check_tiling_applied(ir: dict[str, Any]) -> CheckResult:
    if not get_path(ir, "tiling.enabled"):
        return pass_result("C_TILING_APPLIED", "tiling is disabled; tiling completeness check deferred.")
    missing = [
        field
        for field in ("tiling.block_m", "tiling.block_n", "tiling.block_k")
        if get_path(ir, field) is None
    ]
    if not missing:
        return pass_result("C_TILING_APPLIED", "block_m, block_n, and block_k are set.")
    return fail_result(
        "C_TILING_APPLIED",
        f"Missing tiling fields: {', '.join(missing)}.",
        "StrategyCompliance.TilingNotApplied",
    )


def check_boundary_guard(ir: dict[str, Any]) -> CheckResult:
    fields = [
        "memory.global_load_A.boundary_guard",
        "memory.global_load_B.boundary_guard",
        "memory.global_store_C.boundary_guard",
    ]
    failed = [field for field in fields if get_path(ir, field) is not True]
    if not failed:
        return pass_result("C_BOUNDARY_GUARD", "global A/B loads and C store all have boundary guards.")
    return fail_result(
        "C_BOUNDARY_GUARD",
        f"Boundary guard missing or false: {', '.join(failed)}.",
        "Memory.OutOfBounds",
    )


def check_vector_alignment(ir: dict[str, Any]) -> CheckResult:
    failed = []
    for tensor in ("A", "B", "C"):
        node = get_path(ir, f"vectorization.{tensor}") or {}
        width = node.get("vector_width", 1)
        if width > 1 and not (node.get("alignment_guard") or node.get("alignment_proven")):
            failed.append(f"vectorization.{tensor}")
    if not failed:
        return pass_result("C_VECTOR_ALIGNMENT", "all vectorized accesses have alignment guard or proof.")
    return fail_result(
        "C_VECTOR_ALIGNMENT",
        f"Vectorized access lacks alignment guard/proof: {', '.join(failed)}.",
        "Vectorization.AlignmentViolation",
    )


def check_vector_tail_handling(ir: dict[str, Any]) -> CheckResult:
    failed = []
    for tensor in ("A", "B", "C"):
        node = get_path(ir, f"vectorization.{tensor}") or {}
        width = node.get("vector_width", 1)
        if width > 1 and node.get("tail_handling") is not True:
            failed.append(f"vectorization.{tensor}")
    if not failed:
        return pass_result("C_VECTOR_TAIL_HANDLING", "all vectorized accesses have tail handling.")
    return fail_result(
        "C_VECTOR_TAIL_HANDLING",
        f"Vectorized access lacks tail handling: {', '.join(failed)}.",
        "Vectorization.TailHandlingError",
    )


def check_strategy_postconditions_field(ir: dict[str, Any]) -> CheckResult:
    strategy = ir.get("strategy", {})
    missing = strategy.get("missing_postconditions", [])
    satisfied = strategy.get("postconditions_satisfied")
    if missing:
        return fail_result(
            "C_STRATEGY_POSTCONDITIONS",
            f"Missing strategy postconditions: {', '.join(map(str, missing))}.",
            "StrategyCompliance.PostconditionViolation",
        )
    if satisfied is False:
        return fail_result(
            "C_STRATEGY_POSTCONDITIONS",
            "strategy.postconditions_satisfied is false.",
            "StrategyCompliance.PostconditionViolation",
        )
    return pass_result("C_STRATEGY_POSTCONDITIONS", "no missing strategy postconditions recorded in IR.")


def check_warp_multiple(ir: dict[str, Any]) -> CheckResult:
    threads = get_path(ir, "mapping.threads_per_block")
    warp_size = get_path(ir, "hardware.warp_size")
    if threads is None or warp_size is None:
        return pass_result("C_WARP_MULTIPLE", "threads_per_block or warp_size is unset; check deferred.")
    if threads % warp_size == 0:
        return pass_result("C_WARP_MULTIPLE", f"threads_per_block={threads} is a multiple of warp_size={warp_size}.")
    return fail_result(
        "C_WARP_MULTIPLE",
        f"threads_per_block={threads} is not a multiple of warp_size={warp_size}.",
        "Performance.LowWarpUtilization",
    )


def check_register_pressure(ir: dict[str, Any]) -> CheckResult:
    thread_m = get_path(ir, "tiling.thread_m") or 1
    thread_n = get_path(ir, "tiling.thread_n") or 1
    accumulators = thread_m * thread_n
    if accumulators <= 16:
        return pass_result("C_REGISTER_PRESSURE", f"thread_m * thread_n = {accumulators}.")
    return fail_result(
        "C_REGISTER_PRESSURE",
        f"thread_m * thread_n = {accumulators}, likely high register pressure.",
        "Performance.RegisterPressure",
    )


def evaluate_condition(ir, condition, check_id):
    if isinstance(condition, dict):
        return evaluate_predicate(ir, condition, check_id)

    alternatives = re.split(r"\s+or\s+", condition)
    evaluated = [evaluate_simple_condition(ir, item.strip()) for item in alternatives]
    if any(item["status"] == "pass" for item in evaluated):
        return pass_result(check_id, condition, detail={"evaluated": evaluated})
    if any(item["status"] == "unknown" for item in evaluated):
        return unknown_result(check_id, condition, detail={"evaluated": evaluated})
    return fail_result(check_id, condition, "StrategyCompliance.ConditionNotSatisfied", {"evaluated": evaluated})


def condition_check_id(prefix: str, condition: Any) -> str:
    if isinstance(condition, dict):
        return f"{prefix}:{condition.get('id', condition.get('constraint_id', 'predicate'))}"
    return f"{prefix}:{condition}"


def evaluate_predicate(ir: dict[str, Any], predicate: dict[str, Any], check_id: str) -> CheckResult:
    if not isinstance(predicate, dict):
        return unknown_result(
            check_id,
            "invalid predicate",
            detail={"reason": "predicate_is_not_object", "predicate": predicate},
        )

    rendered_id = check_id
    kind = predicate.get("kind")

    if kind == "ir_predicate":
        evaluated = evaluate_ir_predicate(ir, predicate)
        message = render_predicate(predicate)
        if evaluated["status"] == "pass":
            return pass_result(rendered_id, message, detail={"evaluated": [evaluated], "predicate": predicate})
        if evaluated["status"] == "unknown":
            return unknown_result(rendered_id, message, detail={"evaluated": [evaluated], "predicate": predicate})
        return fail_result(
            rendered_id,
            message,
            "StrategyCompliance.ConditionNotSatisfied",
            {"evaluated": [evaluated], "predicate": predicate},
        )

    if kind == "logical":
        evaluated = evaluate_logical_predicate(ir, predicate, check_id)
        message = render_predicate(predicate)
        if evaluated["status"] == "pass":
            return pass_result(rendered_id, message, detail=evaluated)
        if evaluated["status"] == "unknown":
            return unknown_result(rendered_id, message, detail=evaluated)
        return fail_result(rendered_id, message, "StrategyCompliance.ConditionNotSatisfied", evaluated)

    if kind == "derived_field":
        evaluated = evaluate_derived_field(ir, predicate)
        message = render_predicate(predicate)
        if evaluated["status"] == "pass":
            return pass_result(rendered_id, message, detail={"evaluated": [evaluated], "predicate": predicate})
        if evaluated["status"] == "unknown":
            return unknown_result(rendered_id, message, detail={"evaluated": [evaluated], "predicate": predicate})
        return fail_result(
            rendered_id,
            message,
            "StrategyCompliance.DerivedFieldMismatch",
            {"evaluated": [evaluated], "predicate": predicate},
        )

    if kind == "semantic_constraint":
        evaluated = evaluate_semantic_constraint(ir, predicate)
        message = render_predicate(predicate)
        if evaluated["status"] == "pass":
            return pass_result(rendered_id, message, detail={"evaluated": [evaluated], "predicate": predicate})
        if evaluated["status"] == "fail":
            return fail_result(
                rendered_id,
                message,
                evaluated.get("failure_type", "StrategyCompliance.SemanticConstraintViolation"),
                {"evaluated": [evaluated], "predicate": predicate},
            )
        return unknown_result(
            rendered_id,
            message,
            detail={
                "reason": evaluated.get("reason", "verifier_not_implemented_in_ir_checker"),
                "evaluated": [evaluated],
                "predicate": predicate,
                "required_verifier": predicate.get("verifier"),
            },
        )

    if kind == "code_pattern":
        return unknown_result(
            rendered_id,
            render_predicate(predicate),
            detail={
                "reason": "code_pattern_requires_code_verifier",
                "predicate": predicate,
                "required_verifier": predicate.get("verifier"),
            },
        )

    return unknown_result(
        rendered_id,
        render_predicate(predicate),
        detail={"reason": f"unsupported_predicate_kind:{kind}", "predicate": predicate},
    )


def evaluate_ir_predicate(ir: dict[str, Any], predicate: dict[str, Any]) -> dict[str, Any]:
    op = normalize_predicate_op(predicate.get("op"))
    left = resolve_predicate_operand(ir, predicate.get("lhs", {}))

    if op == "is_not_null":
        return {
            "status": "pass" if left is not MISSING and left is not None else "fail",
            "predicate_id": predicate.get("id"),
            "op": op,
            "left": None if left is MISSING else left,
        }
    if op == "is_null":
        return {
            "status": "pass" if left is None else "fail",
            "predicate_id": predicate.get("id"),
            "op": op,
            "left": None if left is MISSING else left,
        }

    right = resolve_predicate_operand(ir, predicate.get("rhs", {}))
    if left is MISSING or right is MISSING:
        return {
            "status": "unknown",
            "predicate_id": predicate.get("id"),
            "left": None if left is MISSING else left,
            "right": None if right is MISSING else right,
            "op": op,
            "reason": "missing_value",
        }
    if op not in OPS:
        return {
            "status": "unknown",
            "predicate_id": predicate.get("id"),
            "left": left,
            "right": right,
            "op": op,
            "reason": "unsupported_op",
        }
    try:
        ok = OPS[op](left, right)
    except TypeError:
        ok = OPS[op](str(left), str(right))
    return {
        "status": "pass" if ok else "fail",
        "predicate_id": predicate.get("id"),
        "left": left,
        "right": right,
        "op": op,
    }


def normalize_predicate_op(op: Any) -> Any:
    """Normalize accepted schema aliases before evaluating a predicate."""
    aliases = {
        "not_null": "is_not_null",
        "nonnull": "is_not_null",
        "null": "is_null",
    }
    return aliases.get(op, op)


def evaluate_logical_predicate(ir: dict[str, Any], predicate: dict[str, Any], check_id: str) -> dict[str, Any]:
    op = predicate.get("op")
    conditions = predicate.get("conditions", []) or []
    results = [evaluate_predicate_to_status(ir, item, check_id) for item in conditions]
    statuses = [item["status"] for item in results]
    if op == "any_of":
        if "pass" in statuses:
            status = "pass"
        elif "unknown" in statuses:
            status = "unknown"
        else:
            status = "fail"
    elif op == "all_of":
        if all(item == "pass" for item in statuses):
            status = "pass"
        elif "fail" in statuses:
            status = "fail"
        else:
            status = "unknown"
    elif op == "not":
        if not results:
            status = "unknown"
        elif results[0]["status"] == "pass":
            status = "fail"
        elif results[0]["status"] == "fail":
            status = "pass"
        else:
            status = "unknown"
    else:
        status = "unknown"
    return {"status": status, "op": op, "conditions": results, "predicate": predicate}


def evaluate_predicate_to_status(ir: dict[str, Any], predicate: dict[str, Any], check_id: str) -> dict[str, Any]:
    if not isinstance(predicate, dict):
        return {
            "id": check_id,
            "status": "unknown",
            "message": "invalid predicate",
            "failure_type": "StrategyCompliance.UnknownCondition",
            "detail": {"reason": "predicate_is_not_object", "predicate": predicate},
        }
    result = evaluate_predicate(ir, predicate, check_id)
    return {
        "id": result.id,
        "status": result.status,
        "message": result.message,
        "failure_type": result.failure_type,
        "detail": result.detail,
    }


def evaluate_derived_field(ir: dict[str, Any], predicate: dict[str, Any]) -> dict[str, Any]:
    target = predicate.get("target", {})
    target_field = target.get("field")
    formula_id = predicate.get("formula_id")
    actual = get_path(ir, target_field, MISSING) if target_field else MISSING
    expected = evaluate_formula(ir, formula_id)
    if actual is MISSING or expected is MISSING:
        return {
            "status": "unknown",
            "predicate_id": predicate.get("id"),
            "target": target_field,
            "actual": None if actual is MISSING else actual,
            "expected": None if expected is MISSING else expected,
            "formula_id": formula_id,
            "reason": "missing_value_or_formula",
        }
    return {
        "status": "pass" if actual == expected else "fail",
        "predicate_id": predicate.get("id"),
        "target": target_field,
        "actual": actual,
        "expected": expected,
        "formula_id": formula_id,
    }


def evaluate_formula(ir: dict[str, Any], formula_id: str | None) -> Any:
    if formula_id == "F_SHARED_MEMORY_TOTAL_BYTES":
        value = estimate_shared_memory_bytes(ir)
        return MISSING if value is None else value
    return MISSING


def evaluate_semantic_constraint(ir: dict[str, Any], predicate: dict[str, Any]) -> dict[str, Any]:
    constraint_id = predicate.get("constraint_id")
    verifiers = predicate.get("verifier")
    verifier_set = set(verifiers if isinstance(verifiers, list) else [verifiers])
    if "ir_eval" not in verifier_set:
        return {
            "status": "unknown",
            "constraint_id": constraint_id,
            "reason": "semantic_constraint_requires_non_ir_verifier",
            "required_verifier": predicate.get("verifier"),
        }
    if constraint_id == "C_VECTOR_TAIL_HANDLING":
        tensors = predicate.get("args", {}).get("tensors", []) or []
        failed = []
        for tensor in tensors:
            node = get_path(ir, f"vectorization.{tensor}") or {}
            if node.get("vector_width", 1) > 1 and node.get("tail_handling") is not True:
                failed.append(tensor)
        return {
            "status": "pass" if not failed else "fail",
            "constraint_id": constraint_id,
            "failed_tensors": failed,
            "failure_type": "Vectorization.TailHandlingError",
        }
    return {
        "status": "unknown",
        "constraint_id": constraint_id,
        "reason": "ir_semantic_constraint_not_implemented",
        "required_verifier": predicate.get("verifier"),
    }


def resolve_predicate_operand(ir: dict[str, Any], operand: Any) -> Any:
    if not isinstance(operand, dict):
        return operand
    if "const" in operand:
        return operand["const"]
    if "field" in operand:
        return get_path(ir, operand["field"], MISSING)
    if "expr" in operand:
        return evaluate_expression(ir, operand["expr"])
    return MISSING


def evaluate_expression(ir: dict[str, Any], expr: str) -> Any:
    names = {match for match in re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+", expr)}
    rewritten = expr
    values: dict[str, Any] = {}
    for index, field in enumerate(sorted(names, key=len, reverse=True)):
        value = get_path(ir, field, MISSING)
        if value is MISSING:
            return MISSING
        name = f"v{index}"
        values[name] = value
        rewritten = rewritten.replace(field, name)
    safe_globals = {"__builtins__": {}, "ceil": math.ceil, "floor": math.floor, "min": min, "max": max}
    try:
        return eval(rewritten, safe_globals, values)
    except Exception:
        return MISSING


def render_predicate(predicate: dict[str, Any]) -> str:
    if predicate.get("kind") == "ir_predicate":
        lhs = render_operand(predicate.get("lhs"))
        rhs = render_operand(predicate.get("rhs"))
        op = predicate.get("op")
        return f"{predicate.get('id', '')}:{lhs} {op}" + (f" {rhs}" if rhs else "")
    if predicate.get("kind") == "logical":
        return f"{predicate.get('id', '')}:{predicate.get('op')}({len(predicate.get('conditions', []) or [])} conditions)"
    if predicate.get("kind") == "derived_field":
        return f"{predicate.get('id', '')}:{render_operand(predicate.get('target'))} recomputed by {predicate.get('formula_id')}"
    if predicate.get("kind") == "semantic_constraint":
        return f"{predicate.get('id', '')}:{predicate.get('constraint_id')}"
    if predicate.get("kind") == "code_pattern":
        return f"{predicate.get('id', '')}:code_pattern {predicate.get('op')}"
    return json_safe_dumps(predicate)


def render_operand(operand: Any) -> str:
    if not isinstance(operand, dict):
        return ""
    if "field" in operand:
        return str(operand["field"])
    if "const" in operand:
        return repr(operand["const"])
    if "expr" in operand:
        return str(operand["expr"])
    return json_safe_dumps(operand)


def json_safe_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def evaluate_simple_condition(ir, condition):
    match = re.fullmatch(r"(.+?)\s*(<=|>=|!=|=|<|>)\s*(.+)", condition)
    if not match:
        return {"status": "unknown", "condition": condition, "reason": "unsupported_condition_syntax"}

    left_expr, op, right_expr = [part.strip() for part in match.groups()]
    left = resolve_value(ir, left_expr)
    right = resolve_value(ir, right_expr)
    if left is MISSING or right is MISSING:
        return {
            "status": "unknown",
            "condition": condition,
            "left": None if left is MISSING else left,
            "right": None if right is MISSING else right,
            "reason": "missing_value",
        }

    try:
        ok = OPS[op](left, right)
    except TypeError:
        left, right = str(left), str(right)
        ok = OPS[op](left, right)
    return {
        "status": "pass" if ok else "fail",
        "condition": condition,
        "left": left,
        "right": right,
        "op": op,
    }


def estimate_shared_memory_bytes(ir: dict[str, Any]) -> int | None:
    explicit = get_path(ir, "resource.shared_memory.total_bytes")
    if explicit is not None:
        return explicit
    if get_path(ir, "memory.use_shared_memory") is not True:
        return None

    dtype_bytes = dtype_size_bytes(get_path(ir, "problem.dtype_A"))
    total = 0
    for field in ("memory.shared_A", "memory.shared_B"):
        node = get_path(ir, field) or {}
        if node.get("enabled") is not True:
            continue
        shape = node.get("shape")
        if not shape:
            return None
        dims = [resolve_dim(ir, item) for item in shape]
        if any(dim is None for dim in dims):
            return None
        total += product(dims) * dtype_bytes
    return total


def resolve_dim(ir: dict[str, Any], expr: Any) -> int | None:
    if isinstance(expr, int):
        return expr
    if not isinstance(expr, str):
        return None
    replaced = expr
    for name in ("block_m", "block_n", "block_k", "thread_m", "thread_n"):
        value = get_path(ir, f"tiling.{name}")
        if value is None:
            return None
        replaced = re.sub(rf"\b{name}\b", str(value), replaced)
    if not re.fullmatch(r"[0-9+\-*/ ()]+", replaced):
        return None
    try:
        value = eval(replaced, {"__builtins__": {}}, {})
    except (SyntaxError, ZeroDivisionError):
        return None
    return int(value)


def dtype_size_bytes(dtype: str | None) -> int:
    return {
        "fp16": 2,
        "bf16": 2,
        "tf32": 4,
        "fp32": 4,
        "fp64": 8,
    }.get(dtype or "fp32", 4)


def resolve_value(ir: dict[str, Any], expr: str) -> Any:
    lowered = expr.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    if re.fullmatch(r"-?\d+", expr):
        return int(expr)
    if re.fullmatch(r"-?\d+\.\d+", expr):
        return float(expr)
    value = get_path(ir, expr, MISSING)
    if value is not MISSING:
        return value
    if "." in expr:
        return MISSING
    if expr.startswith(("'", '"')) and expr.endswith(("'", '"')):
        return expr[1:-1]
    return expr


MISSING = object()


def get_path(data: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def product(values: list[int | None]) -> int:
    total = 1
    for value in values:
        total *= int(value)
    return total


def pass_result(check_id: str, message: str, detail: dict[str, Any] | None = None) -> CheckResult:
    return CheckResult(check_id, "pass", message, detail=detail)


def fail_result(
    check_id,
    message,
    failure_type,
    detail=None,
) -> CheckResult:
    return CheckResult(check_id, "fail", message, failure_type, detail)


def unknown_result(check_id: str, message: str, detail: dict[str, Any] | None = None) -> CheckResult:
    return CheckResult(check_id, "unknown", message, "StrategyCompliance.UnknownCondition", detail)


def serialize_results(results: list[CheckResult]) -> list[dict[str, Any]]:
    return [asdict(result) for result in results]


def load_strategy(library_path: Path, strategy_id: str | None) -> dict[str, Any] | None:
    if not strategy_id:
        return None
    library = load_json(library_path)
    for strategy in library.get("strategies", []):
        if strategy.get("strategy_id") == strategy_id:
            return strategy
    raise SystemExit(f"Strategy not found: {strategy_id}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check SGPO OptIR constraints.")
    parser.add_argument("--ir", type=Path, default=DEFAULT_IR)
    parser.add_argument("--strategy-id")
    parser.add_argument("--strategy-library", type=Path, default=DEFAULT_STRATEGY_LIBRARY)
    parser.add_argument("--phase", choices=["before", "after", "code"], default="before")
    parser.add_argument("--postconditions-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ir = load_json(args.ir)
    strategy = load_strategy(args.strategy_library, args.strategy_id)
    if args.phase == "before":
        report = check_before_codegen(ir, strategy)
    elif args.phase == "code":
        report = check_code_verification(ir, strategy)
    else:
        report = check_after_codegen(ir, strategy, include_hard_constraints=not args.postconditions_only)

    if args.output:
        save_json(args.output, report)
    else:
        import json

        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
