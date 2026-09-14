from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from SGPO.llm.openai_client import OpenAICompatibleClient
from SGPO.llm.patch_generator import load_code_context, load_strategy
from SGPO.utils.common_utils import load_config, load_json, save_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IR = ROOT / "data" / "IRs" / "ir_patch" / "optir.patch.json"
DEFAULT_PATCH = ROOT / "results" / "patch" / "generated_patch.json"
DEFAULT_STRATEGY_LIBRARY = ROOT / "data" / "lib" / "strategy_library.json"
DEFAULT_CODE_ROOT = ROOT / "gemm_code" / "skeleton"
DEFAULT_PROMPT = ROOT / "llm" / "prompts" / "generate_concrete_code_prompt.txt"
DEFAULT_CONFIG = ROOT / "conf.yaml"
DEFAULT_OUTPUT = ROOT / "results" / "code" / "patched_code_files.json"
ALLOWED_CODE_FILES = {
    "cuda_kernel.cuh",
    "kernel.h",
}
CUDA_PATCH_REGIONS = {
    "LAUNCH_CONFIG",
    "SHARED_DECL",
    "INDEX_MAPPING",
    "REGISTER_DECL",
    "GLOBAL_TO_SHARED_LOAD",
    "SYNC_AFTER_LOAD",
    "MAIN_LOOP",
    "COMPUTE_INNER",
    "STORE",
}


def generate_code_files_from_patch_with_llm(
    client: OpenAICompatibleClient,
    patch_ir: dict[str, Any],
    patch_result: dict[str, Any],
    strategy: dict[str, Any],
    code_context: dict[str, Any],
    prompt_path: Path = DEFAULT_PROMPT,
    repair_context: dict[str, Any] | None = None,
    patch_file: Path | None = None,
) -> dict[str, Any]:
    messages = build_patch_to_code_messages(
        patch_ir=patch_ir,
        patch_result=patch_result,
        strategy=strategy,
        code_context=code_context,
        prompt_path=prompt_path,
        repair_context=repair_context,
        patch_file=patch_file,
    )
    response = complete_codegen_json(client, messages)
    return validate_generated_code_files_response(response, strategy["strategy_id"], patch_file)


def generate_concrete_code_with_llm(
    client: OpenAICompatibleClient,
    patch_ir: dict[str, Any],
    patch_result: dict[str, Any],
    strategy: dict[str, Any],
    code_context: dict[str, Any],
    prompt_path: Path = DEFAULT_PROMPT,
    repair_context: dict[str, Any] | None = None,
    patch_file: Path | None = None,
) -> dict[str, Any]:
    return generate_code_files_from_patch_with_llm(
        client=client,
        patch_ir=patch_ir,
        patch_result=patch_result,
        strategy=strategy,
        code_context=code_context,
        prompt_path=prompt_path,
        repair_context=repair_context,
        patch_file=patch_file,
    )


def build_patch_to_code_messages(
    patch_ir: dict[str, Any],
    patch_result: dict[str, Any],
    strategy: dict[str, Any],
    code_context: dict[str, Any],
    prompt_path: Path = DEFAULT_PROMPT,
    repair_context: dict[str, Any] | None = None,
    patch_file: Path | None = None,
) -> list[dict[str, str]]:
    template = prompt_path.read_text(encoding="utf-8")
    patch_payload = dict(patch_result)
    if patch_file is not None:
        patch_payload["source_patch_file"] = str(patch_file)
    prompt = template.format(
        patch_ir_json=json.dumps(compact_patch_ir_for_prompt(patch_ir), ensure_ascii=False, indent=2),
        strategy_json=json.dumps(strategy, ensure_ascii=False, indent=2),
        patch_json=json.dumps(patch_payload, ensure_ascii=False, indent=2),
        code_context_json=json.dumps(code_context, ensure_ascii=False, indent=2),
        repair_context_json=json.dumps(repair_context or {}, ensure_ascii=False, indent=2),
    )
    return [
        {
            "role": "system",
            "content": "You apply SGPO Patch JSON as local CUDA anchor-region edits and only return compact valid JSON.",
        },
        {"role": "user", "content": prompt},
    ]


def complete_codegen_json(client: OpenAICompatibleClient, messages: list[dict[str, str]]) -> dict[str, Any]:
    if hasattr(client, "complete_codegen_json"):
        return client.complete_codegen_json(messages)
    return client.complete_json(messages)


def build_concrete_code_messages(
    patch_ir: dict[str, Any],
    patch_result: dict[str, Any],
    strategy: dict[str, Any],
    code_context: dict[str, Any],
    prompt_path: Path = DEFAULT_PROMPT,
    repair_context: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    return build_patch_to_code_messages(
        patch_ir=patch_ir,
        patch_result=patch_result,
        strategy=strategy,
        code_context=code_context,
        prompt_path=prompt_path,
        repair_context=repair_context,
    )


def compact_patch_ir_for_prompt(ir: dict[str, Any]) -> dict[str, Any]:
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
        "patch_generation",
        "repair_attempt",
    ]
    return {key: ir[key] for key in keep_keys if key in ir}


def validate_generated_code_files_response(
    response: dict[str, Any],
    expected_strategy_id: str,
    patch_file: Path | None = None,
) -> dict[str, Any]:
    strategy_id = response.get("strategy_id")
    if is_schema_placeholder_strategy_id(strategy_id):
        response["strategy_id"] = expected_strategy_id
        response.setdefault("code_generation_notes", []).append(
            "Corrected schema placeholder strategy_id to the selected strategy_id."
        )
    elif strategy_id != expected_strategy_id:
        raise ValueError(
            f"Concrete code strategy_id mismatch: expected {expected_strategy_id}, got {strategy_id}"
        )
    files = response.get("files")
    edits = response.get("edits")
    if (not isinstance(files, list) or not files) and (not isinstance(edits, list) or not edits):
        raise ValueError("Concrete code response must include a non-empty edits list or legacy files list.")
    seen = set()
    for item in files or []:
        if not isinstance(item, dict):
            raise ValueError("Each concrete code file item must be an object.")
        relative_path = item.get("path")
        content = item.get("content")
        if relative_path not in ALLOWED_CODE_FILES:
            raise ValueError(f"Concrete code file is not allowed: {relative_path}")
        if relative_path in seen:
            raise ValueError(f"Duplicate concrete code file: {relative_path}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Concrete code file content is empty: {relative_path}")
        seen.add(relative_path)
    seen_edits = set()
    for item in edits or []:
        if not isinstance(item, dict):
            raise ValueError("Each concrete code edit item must be an object.")
        relative_path = item.get("path")
        region = normalize_region_name(item.get("region") or item.get("anchor"))
        replacement = normalize_edit_replacement(item)
        if relative_path not in ALLOWED_CODE_FILES:
            raise ValueError(f"Concrete code edit file is not allowed: {relative_path}")
        if relative_path != "cuda_kernel.cuh":
            raise ValueError(f"GPU region edits are only supported for cuda_kernel.cuh: {relative_path}")
        if region not in CUDA_PATCH_REGIONS:
            raise ValueError(f"Concrete code edit region is not supported: {item.get('region') or item.get('anchor')}")
        key = (relative_path, region)
        if key in seen_edits:
            raise ValueError(f"Duplicate concrete code edit for {relative_path}:{region}")
        if not isinstance(replacement, str) or not replacement.strip():
            raise ValueError(f"Concrete code edit replacement is empty: {relative_path}:{region}")
        item["path"] = relative_path
        item["region"] = region
        item["replacement"] = replacement
        seen_edits.add(key)
    response.setdefault("code_generation_notes", [])
    response.setdefault("expected_static_properties", [])
    response["generation_method"] = "cuda_region_edits" if edits else "patch_json_plus_source_files"
    if patch_file is not None:
        response["source_patch_file"] = str(patch_file)
    response["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return response


def is_schema_placeholder_strategy_id(value: Any) -> bool:
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    return value.strip() in {
        "",
        "string",
        "<string>",
        "<exact SELECTED_STRATEGY.strategy_id>",
        "SELECTED_STRATEGY.strategy_id",
        "strategy_id",
    }


def validate_concrete_code_response(response: dict[str, Any], expected_strategy_id: str) -> dict[str, Any]:
    return validate_generated_code_files_response(response, expected_strategy_id)


def apply_generated_code_files(
    generated_code: dict[str, Any],
    code_root: Path = DEFAULT_CODE_ROOT,
    patch_ir: dict[str, Any] | None = None,
    strategy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if generated_code.get("edits"):
        return apply_generated_region_edits(generated_code, code_root, patch_ir or {}, strategy or {})

    if (strategy or {}).get("patch_contract", {}).get("region_edits_only"):
        return {"status": "fail", "applied": [],
                "error_message": "This strategy requires local region edits; full source files are forbidden"}

    ensure_generated_code_has_minimal_kernel(generated_code)
    from SGPO.verification.cuda_launch_config import configure_shared_launch
    try:
        for item in generated_code.get("files", []):
            if item.get("path") == "cuda_kernel.cuh":
                item["content"] = configure_shared_launch(item["content"], patch_ir or {})
    except ValueError as exc:
        return {"status": "fail", "applied": [], "error_message": str(exc)}
    materialization = validate_generated_code_materialization(generated_code, patch_ir or {}, strategy or {})
    if materialization["status"] != "pass":
        return {
            "status": "fail",
            "error_message": materialization["error_message"],
            "method": "patch_json_plus_source_files",
            "applied": [],
            "materialization": materialization,
        }

    applied = []
    for item in generated_code.get("files", []):
        relative_path = item["path"]
        if relative_path not in ALLOWED_CODE_FILES:
            return {
                "status": "fail",
                "error_message": f"Concrete code file is not allowed: {relative_path}",
                "applied": applied,
            }
        path = (code_root / relative_path).resolve()
        if code_root.resolve() not in [path, *path.parents]:
            return {
                "status": "fail",
                "error_message": f"Concrete code file escapes code_root: {relative_path}",
                "applied": applied,
            }
        path.write_text(item["content"], encoding="utf-8")
        applied.append({"file": relative_path, "change_summary": item.get("change_summary")})
    return {
        "status": "pass",
        "method": "patch_json_plus_source_files",
        "applied": applied,
        "materialization": materialization,
    }


def apply_generated_region_edits(
    generated_code: dict[str, Any],
    code_root: Path,
    patch_ir: dict[str, Any],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    applied = []
    staged_content: dict[str, str] = {}
    original_content: dict[str, str] = {}
    contract = strategy.get("patch_contract") or {}
    edited_spans: dict[str, list[tuple[int, int]]] = {}
    root = code_root.resolve()
    for item in generated_code.get("edits", []) or []:
        relative_path = item["path"]
        if contract and (relative_path != "cuda_kernel.cuh" or item["region"] not in contract["allowed_regions"]):
            return {"status": "fail", "applied": [],
                    "error_message": f"Strategy patch changes protected region: {relative_path}:{item['region']}"}
        if relative_path not in ALLOWED_CODE_FILES:
            return {"status": "fail", "error_message": f"Concrete code edit file is not allowed: {relative_path}", "applied": applied}
        path = (code_root / relative_path).resolve()
        if root not in [path, *path.parents]:
            return {"status": "fail", "error_message": f"Concrete code edit file escapes code_root: {relative_path}", "applied": applied}
        content = staged_content.get(relative_path)
        if content is None:
            content = path.read_text(encoding="utf-8")
            original_content[relative_path] = content
        if contract:
            original = original_content[relative_path]
            start = original.find(f"{item['region']}_BEGIN")
            end = original.find(f"{item['region']}_END", start)
            if start < 0 or end < 0:
                return {"status": "fail", "applied": [], "error_message": f"Missing original anchor {item['region']}"}
            spans = edited_spans.setdefault(relative_path, [])
            if any(start <= previous_end and previous_start <= end for previous_start, previous_end in spans):
                return {"status": "fail", "applied": [], "error_message": "Overlapping parent/child region edits are forbidden"}
            spans.append((start, end))
        try:
            content = replace_cuda_anchor_region(content, item["region"], item["replacement"])
        except Exception as exc:
            return {
                "status": "fail",
                "error_message": f"Failed to apply region edit {relative_path}:{item['region']}: {exc}",
                "method": "cuda_region_edits",
                "applied": applied,
            }
        staged_content[relative_path] = content
        applied.append(
            {
                "file": relative_path,
                "region": item["region"],
                "change_summary": item.get("change_summary"),
                "method": "anchor_replacement",
            }
        )

    validation_payload = {
        "strategy_id": generated_code.get("strategy_id"),
        "files": [
            {"path": relative_path, "content": content, "change_summary": "staged region edits"}
            for relative_path, content in staged_content.items()
        ],
    }
    if not contract:
        ensure_generated_code_has_minimal_kernel(validation_payload)
    for item in validation_payload["files"]:
        staged_content[item["path"]] = item["content"]
    if contract:
        for relative_path, content in staged_content.items():
            original = original_content[relative_path]
            for region in contract.get("preserved_regions", []):
                if anchor_region(original, region) != anchor_region(content, region):
                    return {"status": "fail", "applied": [],
                            "error_message": f"Strategy patch modified protected {region}"}
            before = strip_cpp_comments(original)
            after = strip_cpp_comments(content)
            for marker in re.findall(r"\b[A-Z_]+_(?:BEGIN|END)\b", content):
                if content.count(marker) > max(1, original.count(marker)):
                    return {"status": "fail", "applied": [], "error_message": f"Patch introduced duplicate anchor {marker}"}
            if after.count("__shared__") != before.count("__shared__"):
                return {"status": "fail", "applied": [], "error_message": "Strategy cannot add or remove shared buffers"}
            if re.search(r"\b(?:float4|FLOAT4)\b", after) and not re.search(r"\b(?:float4|FLOAT4)\b", before):
                return {"status": "fail", "applied": [], "error_message": "Strategy cannot introduce float4 vectorization"}
    from SGPO.verification.cuda_launch_config import configure_shared_launch
    try:
        for item in validation_payload["files"]:
            if item["path"] == "cuda_kernel.cuh":
                item["content"] = configure_shared_launch(item["content"], patch_ir)
                staged_content[item["path"]] = item["content"]
    except ValueError as exc:
        return {"status": "fail", "applied": [], "error_message": str(exc)}
    materialization = validate_generated_code_materialization(validation_payload, patch_ir, strategy)
    if materialization["status"] != "pass":
        return {
            "status": "fail",
            "error_message": materialization["error_message"],
            "method": "cuda_region_edits",
            "applied": applied,
            "materialization": materialization,
        }

    for relative_path, content in staged_content.items():
        (code_root / relative_path).write_text(content, encoding="utf-8")
    generated_code["materialized_files"] = [
        {"path": relative_path, "content": content, "change_summary": "Materialized from region edits."}
        for relative_path, content in staged_content.items()
    ]
    return {
        "status": "pass",
        "method": "cuda_region_edits",
        "applied": applied,
        "materialization": materialization,
    }


def apply_concrete_code(
    concrete_code: dict[str, Any],
    code_root: Path = DEFAULT_CODE_ROOT,
) -> dict[str, Any]:
    return apply_generated_code_files(concrete_code, code_root)


def normalize_edit_replacement(item: dict[str, Any]) -> str | None:
    replacement = item.get("replacement")
    if isinstance(replacement, str):
        return replacement
    replacement_lines = item.get("replacement_lines")
    if isinstance(replacement_lines, list) and all(isinstance(line, str) for line in replacement_lines):
        return "\n".join(replacement_lines)
    return None


def normalize_region_name(region: Any) -> str:
    text = str(region or "").strip()
    text = text.replace("_BEGIN", "").replace("_END", "")
    for candidate in CUDA_PATCH_REGIONS:
        if candidate in text:
            return candidate
    return text


def replace_cuda_anchor_region(content: str, region: str, replacement: str) -> str:
    lines = content.splitlines()
    begin_marker = f"{region}_BEGIN"
    end_marker = f"{region}_END"
    if sum(begin_marker in line for line in lines) > 1 or sum(end_marker in line for line in lines) > 1:
        raise ValueError(f"ambiguous duplicate anchor: {region}; edit a unique enclosing region covering all occurrences")
    begin_index = next((i for i, line in enumerate(lines) if begin_marker in line), None)
    end_index = next((i for i, line in enumerate(lines) if end_marker in line and begin_index is not None and i > begin_index), None)
    if begin_index is None or end_index is None:
        raise ValueError(f"anchor region not found: {region}")

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
    replacement_lines = replacement.strip("\n").splitlines()
    indented = [f"{indent}{line}" if line else "" for line in replacement_lines]
    suffix = lines[body_end:]
    if end_open_index is None:
        suffix = [f"{indent}/*"] + suffix
    updated = lines[:body_start] + indented + suffix
    trailing = "\n" if content.endswith(("\n", "\r\n")) else ""
    return "\n".join(updated) + trailing


def find_region_end_open(lines: list[str], body_start: int, end_index: int) -> int | None:
    for index in range(end_index - 1, body_start - 1, -1):
        stripped = lines[index].strip()
        if not stripped:
            continue
        if re.match(r"^/\*", stripped):
            return index
        return None
    return None


def ensure_generated_code_has_minimal_kernel(generated_code: dict[str, Any]) -> None:
    for item in generated_code.get("files", []) or []:
        if item.get("path") != "cuda_kernel.cuh":
            continue
        content = item.get("content")
        if not isinstance(content, str):
            continue
        code_only = strip_cpp_comments(content)
        if re.search(r"\bC\s*\[[^\]]+\]\s*=", code_only) and re.search(r"\+=\s*[^;]*\*\s*[^;]*;", code_only):
            return
        updated = inject_minimal_scalar_gemm_body(content)
        if updated != content:
            item["content"] = updated
            notes = generated_code.setdefault("code_generation_notes", [])
            notes.append("Inserted deterministic guarded scalar GEMM fallback because generated kernel body had no real compute/store.")


def inject_minimal_scalar_gemm_body(content: str) -> str:
    marker = "    (void)C;\n"
    if marker not in content:
        return content
    fallback = """

    const int sgpo_linear_tid =
        threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const int sgpo_thread_count = blockDim.x * blockDim.y * blockDim.z;
    const int sgpo_tile_origin_m = blockIdx.y * BM;
    const int sgpo_tile_origin_n = blockIdx.x * BN;

    for (int sgpo_tile_idx = sgpo_linear_tid;
         sgpo_tile_idx < BM * BN;
         sgpo_tile_idx += sgpo_thread_count) {
        const int sgpo_local_m = sgpo_tile_idx / BN;
        const int sgpo_local_n = sgpo_tile_idx - sgpo_local_m * BN;
        const int sgpo_global_m = sgpo_tile_origin_m + sgpo_local_m;
        const int sgpo_global_n = sgpo_tile_origin_n + sgpo_local_n;
        if (sgpo_global_m < M && sgpo_global_n < N) {
            float sgpo_acc = 0.0f;
            for (int sgpo_k = 0; sgpo_k < K; ++sgpo_k) {
                sgpo_acc += A[OFFSET(sgpo_global_m, sgpo_k, K)] *
                            B[OFFSET(sgpo_k, sgpo_global_n, N)];
            }
            C[OFFSET(sgpo_global_m, sgpo_global_n, N)] =
                alpha * sgpo_acc + beta * C[OFFSET(sgpo_global_m, sgpo_global_n, N)];
        }
    }
"""
    return content.replace(marker, marker + fallback, 1)


def validate_generated_code_materialization(
    generated_code: dict[str, Any],
    patch_ir: dict[str, Any],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    kernel = generated_file_content(generated_code, "cuda_kernel.cuh")
    if kernel is None:
        return pass_materialization("cuda_kernel.cuh not changed by this patch.")

    failures = []
    strategy_id = strategy.get("strategy_id") or generated_code.get("strategy_id") or ""
    code_only = strip_cpp_comments(kernel)

    for region in generated_code_modified_regions(generated_code, patch_ir):
        region_name = region_anchor_name(region)
        if not region_name:
            continue
        body = anchor_region(kernel, region_name)
        if body is None:
            failures.append(f"{region_name} region was requested but not found in cuda_kernel.cuh.")
            continue
        if region_requires_active_code(region_name, strategy_id) and not has_active_region_code(body):
            failures.append(f"{region_name} region contains only comments/placeholders; real code is required.")

    if strategy_id.startswith("Mapping.") and not has_active_region_code(anchor_region(kernel, "INDEX_MAPPING") or ""):
        failures.append("Mapping strategy must materialize real thread/block index variables in INDEX_MAPPING.")

    if register_strategy(strategy_id) and not re.search(r"\bfloat\s+[A-Za-z_][A-Za-z0-9_]*\s*(?:\[|=)", code_only):
        failures.append("Register layout strategy must declare real float accumulator/register storage.")

    if shared_memory_required(patch_ir, strategy_id):
        shared_decls = re.findall(r"__shared__\s+float\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[", code_only)
        has_a = any(name.lower().startswith(("as", "shared_a", "sa")) for name in shared_decls)
        has_b = any(name.lower().startswith(("bs", "shared_b", "sb")) for name in shared_decls)
        if not (has_a and has_b):
            failures.append("Shared-memory strategy requires real __shared__ float A/B buffers.")

    if compute_strategy(strategy_id) and not (
        re.search(r"\+=\s*[^;]*\*\s*[^;]*;", code_only)
        or re.search(r"\b(?:fmaf|__fmaf_rn)\s*\(", code_only)
    ):
        failures.append("Compute/reordering strategy must materialize a multiply-accumulate statement.")

    if store_strategy(strategy_id) and not re.search(r"\bC\s*\[[^\]]+\]\s*=", code_only):
        failures.append("Store/epilogue strategy must materialize a guarded global C store.")

    if failures:
        return {
            "status": "fail",
            "error_message": " ".join(failures),
            "failures": failures,
        }
    return pass_materialization("Generated code materializes the requested patch.")


def pass_materialization(message: str) -> dict[str, Any]:
    return {"status": "pass", "message": message, "failures": []}


def generated_file_content(generated_code: dict[str, Any], relative_path: str) -> str | None:
    for item in generated_code.get("files", []) or []:
        if item.get("path") == relative_path:
            return item.get("content")
    return None


def generated_code_modified_regions(generated_code: dict[str, Any], patch_ir: dict[str, Any]) -> list[dict[str, Any]]:
    regions = []
    patch_generation = patch_ir.get("patch_generation") or {}
    for item in patch_generation.get("modified_code_regions", []) or []:
        if isinstance(item, dict) and item.get("file") == "cuda_kernel.cuh":
            regions.append(item)
    return regions


def region_anchor_name(region: dict[str, Any]) -> str | None:
    text = " ".join(str(region.get(key, "")) for key in ["anchor", "change_summary"])
    for name in [
        "LAUNCH_CONFIG",
        "SHARED_DECL",
        "INDEX_MAPPING",
        "REGISTER_DECL",
        "GLOBAL_TO_SHARED_LOAD",
        "SYNC_AFTER_LOAD",
        "MAIN_LOOP",
        "COMPUTE_INNER",
        "STORE",
    ]:
        if name in text:
            return name
    return None


def anchor_region(content: str, name: str) -> str | None:
    begin = re.search(rf"{re.escape(name)}_BEGIN", content)
    if not begin:
        return None
    next_begin = re.search(
        r"\b(?:LAUNCH_CONFIG|SHARED_DECL|INDEX_MAPPING|REGISTER_DECL|GLOBAL_TO_SHARED_LOAD|SYNC_AFTER_LOAD|MAIN_LOOP|COMPUTE_INNER|STORE)_BEGIN\b",
        content[begin.end() :],
    )
    end = begin.end() + next_begin.start() if next_begin else len(content)
    return content[begin.end() : end]


def region_requires_active_code(region_name: str, strategy_id: str) -> bool:
    if region_name == "LAUNCH_CONFIG":
        return True
    if region_name == "SYNC_AFTER_LOAD":
        return "SharedMemory" in strategy_id or "Reordering" in strategy_id or "Pipeline" in strategy_id
    return region_name in {
        "SHARED_DECL",
        "INDEX_MAPPING",
        "REGISTER_DECL",
        "GLOBAL_TO_SHARED_LOAD",
        "MAIN_LOOP",
        "COMPUTE_INNER",
        "STORE",
    }


def has_active_region_code(region_body: str) -> bool:
    code = strip_cpp_comments(region_body)
    code = "\n".join(line.strip() for line in code.splitlines())
    placeholder_words = ["Insert ", "placeholder", "future code"]
    if any(word in region_body for word in placeholder_words) and not re.search(r"[;{}=]", code):
        return False
    return bool(re.search(r"\b(?:const|int|float|__shared__|for|if|while|__syncthreads|FLOAT4)\b|[;=]", code))


def strip_cpp_comments(content: str) -> str:
    without_block_comments = re.sub(r"/\*.*?\*/", lambda match: "\n" * match.group(0).count("\n"), content, flags=re.DOTALL)
    return re.sub(r"//.*", "", without_block_comments)


def shared_memory_required(patch_ir: dict[str, Any], strategy_id: str) -> bool:
    return patch_ir.get("memory", {}).get("use_shared_memory") is True or strategy_id == "Layout.SharedMemory.AB.Basic"


def register_strategy(strategy_id: str) -> bool:
    return strategy_id == "Layout.RegisterTile.C" or strategy_id.startswith("Register.AccumulatorLayout.")


def compute_strategy(strategy_id: str) -> bool:
    return (
        strategy_id.startswith("Reordering.")
        or strategy_id.startswith("Register.FFMA.")
        or strategy_id.startswith("Pipeline.")
    )


def store_strategy(strategy_id: str) -> bool:
    return strategy_id.startswith("Epilogue.Store") or strategy_id.startswith("Vectorization.StoreC")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply SGPO Patch JSON to source context with LLM assistance.")
    parser.add_argument("--ir", type=Path, default=DEFAULT_IR)
    parser.add_argument("--patch", type=Path, default=DEFAULT_PATCH)
    parser.add_argument("--strategy-library", type=Path, default=DEFAULT_STRATEGY_LIBRARY)
    parser.add_argument("--code-root", type=Path, default=DEFAULT_CODE_ROOT)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patch_ir = load_json(args.ir)
    patch_result = load_json(args.patch)
    strategy_library = load_json(args.strategy_library)
    strategy = load_strategy(strategy_library, patch_result["strategy_id"])
    code_context = load_code_context(args.code_root)
    config = load_config(args.config)
    client = OpenAICompatibleClient(config["llm"])
    generated_code = generate_code_files_from_patch_with_llm(
        client=client,
        patch_ir=patch_ir,
        patch_result=patch_result,
        strategy=strategy,
        code_context=code_context,
        prompt_path=args.prompt,
        patch_file=args.patch,
    )
    save_json(args.output, generated_code)
    if args.apply:
        print(json.dumps(apply_generated_code_files(generated_code, args.code_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
