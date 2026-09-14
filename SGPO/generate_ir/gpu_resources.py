"""Shared-memory accounting for strategy selection and IR verification."""
from __future__ import annotations

import ast
import math
import operator
from typing import Any


def get(ir: dict, path: str, default=None):
    value = ir
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def positive_int(value, default=0):
    try:
        result = int(value)
        return result if result > 0 else default
    except (TypeError, ValueError, OverflowError):
        return default


def hardware_value(ir: dict, name: str, default=0):
    return positive_int(get(ir, f"hardware.gpu.{name}"),
                        positive_int(get(ir, f"hardware.{name}"), default))


def pipeline_stages(ir: dict) -> int:
    return positive_int(get(ir, "pipeline.stage_count"),
                        positive_int(get(ir, "resource.shared_memory.pipeline_multiplier"),
                                     2 if get(ir, "pipeline.double_buffering") is True else 1))


def dimension(ir: dict, expression: Any) -> int | None:
    names = dict(ir.get("tiling", {}) or {})
    names.update({alias: names.get(field) for alias, field in
                  (("BM", "block_m"), ("BN", "block_n"), ("BK", "block_k"))})
    operations = {ast.Add: operator.add, ast.Sub: operator.sub,
                  ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv}

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name):
            return int(names[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](evaluate(node.left), evaluate(node.right))
        raise ValueError("unsupported dimension")

    try:
        value = evaluate(ast.parse(str(expression), mode="eval").body)
        return value if value > 0 else None
    except (ValueError, KeyError, TypeError, SyntaxError, ZeroDivisionError, OverflowError):
        return None


def shared_memory_usage(ir: dict) -> dict[str, Any]:
    stages = pipeline_stages(ir)
    per_stage = 0
    for tensor, fallback in (("A", ("block_k", "block_m")), ("B", ("block_k", "block_n"))):
        node = get(ir, f"memory.shared_{tensor}", {}) or {}
        if node.get("enabled") is False:
            continue
        shape = node.get("shape")
        if isinstance(shape, list) and len(shape) == 2:
            # Physical shape includes padding. Do not add padding a second time.
            dims = [dimension(ir, item) for item in shape]
        else:
            dims = [dimension(ir, item) for item in fallback]
            if dims[1] is not None:
                dims[1] += positive_int(node.get("padding"))
        if any(item is None for item in dims):
            per_stage = 0
            break
        dtype = get(ir, f"problem.dtype_{tensor}", "fp32")
        size = {"fp16": 2, "bf16": 2, "fp64": 8}.get(dtype, 4)
        per_stage += math.prod(dims) * size
    total = positive_int(get(ir, "resource.shared_memory.total_bytes"),
                         positive_int(get(ir, "memory.shared_memory_bytes"),
                                      positive_int(get(ir, "resource.shared_memory_bytes"))))
    auxiliary = positive_int(get(ir, "resource.shared_memory.auxiliary_bytes"))
    if not per_stage and total:
        per_stage = math.ceil(max(0, total - auxiliary) / stages)
    # Explicit totals are useful for diagnostics; proposed pipelines use physical shapes.
    return {"per_stage_bytes": per_stage, "stage_count": stages,
            "auxiliary_bytes": auxiliary, "total_bytes": total or per_stage * stages + auxiliary}


def shared_memory_limit(ir: dict) -> int:
    ordinary = hardware_value(ir, "max_shared_memory_per_block_bytes")
    if (get(ir, "memory.shared_memory_allocation") == "dynamic"
            and get(ir, "memory.shared_memory_optin") is True):
        return hardware_value(ir, "max_shared_memory_per_block_optin_bytes", ordinary)
    return ordinary


def proposed_pipeline_bytes(ir: dict, stages: int) -> int:
    usage = shared_memory_usage(ir)
    return usage["per_stage_bytes"] * stages + usage["auxiliary_bytes"]
