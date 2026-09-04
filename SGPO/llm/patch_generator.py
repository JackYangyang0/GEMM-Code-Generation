from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from SGPO.generate_ir.ir_checker import check_after_codegen
from SGPO.llm.openai_client import OpenAICompatibleClient
from SGPO.utils.common_utils import load_config, load_json, save_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IR = ROOT / "data" / "IRs" / "ir_patch" / "optir.extracted.json"
DEFAULT_PRECHECK = ROOT / "results" / "check" / "pre_check_result.json"
DEFAULT_STRATEGY_LIBRARY = ROOT / "data" / "lib" / "strategy_library.json"
DEFAULT_STRATEGY_INDEX = ROOT / "data" / "lib" / "strategy_index.json"
DEFAULT_CPU_STRATEGY_LIBRARY = ROOT / "data" / "lib" / "cpu_strategy_library.json"
DEFAULT_CPU_STRATEGY_INDEX = ROOT / "data" / "lib" / "cpu_strategy_index.json"
DEFAULT_PROMPT = ROOT / "llm" / "prompts" / "generate_patch_prompt.txt"
DEFAULT_CONFIG = ROOT / "conf.yaml"
DEFAULT_PATCH_OUTPUT = ROOT / "results" / "patch" / "generated_patch.json"
DEFAULT_PATCH_IR_OUTPUT = ROOT / "data" / "IRs" / "ir_patch" / "optir.patch.json"
DEFAULT_POSTCHECK_OUTPUT = ROOT / "results" / "check" / "post_check_result.json"
DEFAULT_CODE_ROOT = ROOT / "gemm_code" / "skeleton"
DEFAULT_CODE_FILES = [
    "main.cpp",
    "cuda_kernel.cuh",
    "kernel.h",
]


def generate_patch_with_llm(
    client,
    ir,
    strategy,
    precheck_item,
    code_context,
    prompt_path=DEFAULT_PROMPT,
    repair_context=None,
):
    messages = build_patch_messages(
        ir=ir,
        strategy=strategy,
        precheck_item=precheck_item,
        code_context=code_context,
        prompt_path=prompt_path,
        repair_context=repair_context,
    )
    response = client.complete_json(messages)
    return validate_patch_response(response, strategy["strategy_id"])


def build_patch_messages(
    ir,
    strategy,
    precheck_item,
    code_context,
    prompt_path=DEFAULT_PROMPT,
    repair_context=None,
):
    template = prompt_path.read_text(encoding="utf-8")
    prompt = template.format(
        current_ir_json=json.dumps(compact_ir_for_prompt(ir), ensure_ascii=False, indent=2),
        strategy_json=json.dumps(strategy, ensure_ascii=False, indent=2),
        precheck_json=json.dumps(precheck_item, ensure_ascii=False, indent=2),
        repair_context_json=json.dumps(repair_context or {}, ensure_ascii=False, indent=2),
        code_context_json=json.dumps(code_context, ensure_ascii=False, indent=2),
    )
    return [
        {
            "role": "system",
            "content": patch_system_message(ir),
        },
        {"role": "user", "content": prompt},
    ]


def patch_system_message(ir: dict[str, Any]) -> str:
    backend = (ir.get("target", {}) or {}).get("backend")
    if isinstance(backend, str) and backend.lower() == "cpu":
        return "You generate local ordinary CPU C GEMM patches and only return valid JSON."
    return "You generate local CUDA GEMM patches and only return valid JSON."


def compact_ir_for_prompt(ir: dict[str, Any]) -> dict[str, Any]:
    keep_keys = [
        "optir_name",
        "optir_version",
        "problem",
        "hardware",
        "tiling",
        "mapping",
        "memory",
        "vectorization",
        "synchronization",
        "resource",
        "strategy",
    ]
    return {key: ir[key] for key in keep_keys if key in ir}


def load_code_context(code_root: Path, code_files: list[str] | None = None) -> dict[str, Any]:
    files = {}
    for relative_path in code_files or DEFAULT_CODE_FILES:
        path = code_root / relative_path
        files[relative_path] = path.read_text(encoding="utf-8")
    return {
        "code_root": str(code_root),
        "files": files,
        "patch_anchors": extract_patch_anchors(files),
    }


def extract_patch_anchors(files: dict[str, str]) -> dict[str, list[str]]:
    anchors: dict[str, list[str]] = {}
    for relative_path, content in files.items():
        anchors[relative_path] = [
            line.strip()
            for line in content.splitlines()
            if is_patch_anchor_line(line)
        ]
    return anchors


def is_patch_anchor_line(line: str) -> bool:
    stripped = line.strip()
    if "SGPO_PATCH_" in stripped and ("BEGIN" in stripped or "END" in stripped):
        return True
    return any(
        token in stripped
        for token in [
            "SHARED_DECL_BEGIN",
            "SHARED_DECL_END",
            "LAUNCH_CONFIG_BEGIN",
            "LAUNCH_CONFIG_END",
            "INDEX_MAPPING_BEGIN",
            "INDEX_MAPPING_END",
            "REGISTER_DECL_BEGIN",
            "REGISTER_DECL_END",
            "GLOBAL_TO_SHARED_LOAD_BEGIN",
            "GLOBAL_TO_SHARED_LOAD_END",
            "SYNC_AFTER_LOAD_BEGIN",
            "SYNC_AFTER_LOAD_END",
            "MAIN_LOOP_BEGIN",
            "MAIN_LOOP_END",
            "COMPUTE_INNER_BEGIN",
            "COMPUTE_INNER_END",
            "STORE_BEGIN",
            "STORE_END",
        ]
    )


def choose_prechecked_strategy(
    precheck_result: dict[str, Any],
    strategy_id: str | None = None,
) -> dict[str, Any]:
    accepted = precheck_result.get("accepted", [])
    if not accepted:
        raise ValueError("No accepted strategy in precheck result; cannot generate patch.")
    if strategy_id:
        for item in accepted:
            if item.get("strategy_id") == strategy_id:
                return item
        raise ValueError(f"Strategy {strategy_id} is not accepted by precheck.")
    return accepted[0]


def load_strategy(strategy_library: dict[str, Any], strategy_id: str) -> dict[str, Any]:
    for strategy in strategy_library.get("strategies", []):
        if strategy.get("strategy_id") == strategy_id:
            return strategy
    found = find_strategy_node(strategy_library, strategy_id)
    if found is not None:
        return found
    found = find_strategy_node(load_backend_default_strategy_library(strategy_id), strategy_id)
    if found is not None:
        return found
    synthesized = synthesize_strategy_from_default_index(strategy_id)
    if synthesized is not None:
        return synthesized
    raise ValueError(f"Strategy not found in strategy_library: {strategy_id}")


def load_backend_default_strategy_library(strategy_id: str) -> dict[str, Any]:
    if strategy_id.startswith("CPU.") and DEFAULT_CPU_STRATEGY_LIBRARY.exists():
        return load_json(DEFAULT_CPU_STRATEGY_LIBRARY)
    if DEFAULT_STRATEGY_LIBRARY.exists():
        return load_json(DEFAULT_STRATEGY_LIBRARY)
    return {}


def find_strategy_node(node: Any, strategy_id: str) -> dict[str, Any] | None:
    if isinstance(node, dict):
        if node.get("strategy_id") == strategy_id:
            return copy.deepcopy(node)
        for value in node.values():
            found = find_strategy_node(value, strategy_id)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = find_strategy_node(item, strategy_id)
            if found is not None:
                return found
    return None


def synthesize_strategy_from_default_index(strategy_id: str) -> dict[str, Any] | None:
    index_path = DEFAULT_CPU_STRATEGY_INDEX if strategy_id.startswith("CPU.") else DEFAULT_STRATEGY_INDEX
    if not index_path.exists():
        return None
    strategy_index = load_json(index_path)
    for stage_id, stage in (strategy_index.get("stages") or {}).items():
        for subphase in stage.get("subphases", []) or []:
            if strategy_id in (subphase.get("allowed_strategy_ids") or []):
                from SGPO.generate_ir.stage_controller import synthesize_micro_strategy

                return synthesize_micro_strategy(strategy_id, stage_id, subphase)
    return None


def validate_patch_response(response, expected_strategy_id):
    if not isinstance(response, dict):
        raise ValueError("Patch response must be a JSON object.")
    strategy_id = response.get("strategy_id")
    if strategy_id in {None, "", "string", "<exact SELECTED_STRATEGY.strategy_id>"}:
        response["strategy_id"] = expected_strategy_id
        strategy_id = expected_strategy_id
    if strategy_id != expected_strategy_id:
        raise ValueError(f"Patch strategy_id mismatch: expected {expected_strategy_id}, got {strategy_id}")
    if not isinstance(response.get("modified_code_regions"), list):
        response["modified_code_regions"] = []
    if not isinstance(response.get("modified_ir_fields"), list):
        response["modified_ir_fields"] = list((response.get("ir_updates") or {}).keys())
    code_patch = response.get("code_patch")
    if not isinstance(code_patch, dict) or not isinstance(code_patch.get("diff"), str):
        response["code_patch"] = {"diff": ""}
    if not isinstance(response.get("ir_updates", {}), dict):
        response["ir_updates"] = {}
    response.setdefault("expected_effect", "")
    response.setdefault("potential_risks", [])
    return response


def build_patch_ir(
    ir,
    strategy,
    precheck_item,
    patch_result,
):
    ir_after = copy.deepcopy(ir)
    for path, value in ordered_ir_updates(patch_result.get("ir_updates", {})):
        set_path(ir_after, path, value)
    enrich_derived_ir_fields(ir_after)

    strategy_node = ir_after.setdefault("strategy", {})
    strategy_node["current_stage"] = strategy.get("stage")
    strategy_node["current_strategy_id"] = strategy.get("strategy_id")
    strategy_node["current_strategy_category"] = strategy.get("category")
    strategy_node["current_strategy_intent"] = strategy.get("intent")
    strategy_node["preconditions"] = strategy.get("preconditions", [])
    strategy_node["postconditions"] = strategy.get("postconditions", [])
    strategy_node["risk_types"] = strategy.get("risk_types", [])

    ir_after["patch_generation"] = {
        "stage": "Patch Generation",
        "status": "generated",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "strategy_id": strategy.get("strategy_id"),
        "strategy_name": strategy.get("name"),
        "modified_code_regions": patch_result.get("modified_code_regions", []),
        "modified_ir_fields": patch_result.get("modified_ir_fields", []),
        "code_patch": patch_result.get("code_patch", {}),
        "expected_effect": patch_result.get("expected_effect"),
        "potential_risks": patch_result.get("potential_risks", []),
        "precheck_summary": {
            "preconditions_ok": precheck_item.get("preconditions_ok"),
            "hard_constraints_ok": precheck_item.get("hard_constraints_ok"),
            "strategy_applicable": precheck_item.get("strategy_applicable"),
        },
    }
    return ir_after


def ordered_ir_updates(updates: dict[str, Any]) -> list[tuple[str, Any]]:
    if not isinstance(updates, dict):
        return []
    return sorted(
        updates.items(),
        key=lambda item: (0 if "." not in item[0] else 1, item[0].count(".")),
    )


def enrich_derived_ir_fields(ir: dict[str, Any]) -> None:
    if ir.get("field_meta"):
        return
    tiling = ir.get("tiling", {})
    block_m = tiling.get("block_m")
    block_n = tiling.get("block_n")
    thread_m = tiling.get("thread_m") or 1
    thread_n = tiling.get("thread_n") or 1
    if block_m is None or block_n is None:
        return
    try:
        if block_m % thread_m != 0 or block_n % thread_n != 0:
            return
        block_dim_x = int(block_n // thread_n)
        block_dim_y = int(block_m // thread_m)
    except (TypeError, ZeroDivisionError):
        return

    mapping = ir.setdefault("mapping", {})
    mapping["block_dim_x"] = block_dim_x
    mapping["block_dim_y"] = block_dim_y
    mapping["block_dim_z"] = 1
    mapping["threads_per_block"] = block_dim_x * block_dim_y


def set_path(data: dict[str, Any], dotted_path: str, value: Any) -> None:
    current = data
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        node = current.get(part)
        if not isinstance(node, dict):
            node = {}
            current[part] = node
        current = node
    current[parts[-1]] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SGPO code patch JSON and patch IR.")
    parser.add_argument("--ir", type=Path, default=DEFAULT_IR)
    parser.add_argument("--precheck", type=Path, default=DEFAULT_PRECHECK)
    parser.add_argument("--strategy-id")
    parser.add_argument("--strategy-library", type=Path, default=DEFAULT_STRATEGY_LIBRARY)
    parser.add_argument("--code-root", type=Path, default=DEFAULT_CODE_ROOT)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--patch-output", type=Path, default=DEFAULT_PATCH_OUTPUT)
    parser.add_argument("--patch-ir-output", type=Path, default=DEFAULT_PATCH_IR_OUTPUT)
    parser.add_argument("--postcheck-output", type=Path, default=DEFAULT_POSTCHECK_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ir = load_json(args.ir)
    precheck = load_json(args.precheck)
    strategy_library = load_json(args.strategy_library)
    precheck_item = choose_prechecked_strategy(precheck, args.strategy_id)
    strategy = load_strategy(strategy_library, precheck_item["strategy_id"])
    code_context = load_code_context(args.code_root)

    config = load_config(args.config)
    client = OpenAICompatibleClient(config["llm"])
    patch_result = generate_patch_with_llm(
        client=client,
        ir=ir,
        strategy=strategy,
        precheck_item=precheck_item,
        code_context=code_context,
        prompt_path=args.prompt,
    )
    save_json(args.patch_output, patch_result)

    patch_ir = build_patch_ir(ir, strategy, precheck_item, patch_result)
    save_json(args.patch_ir_output, patch_ir)
    post_check = check_after_codegen(patch_ir, strategy, include_hard_constraints=False)
    save_json(args.postcheck_output, post_check)


if __name__ == "__main__":
    main()
