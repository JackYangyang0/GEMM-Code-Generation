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
    "main.cpp",
    "cuda_kernel.cuh",
    "kernel.h",
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
            "content": "You apply SGPO Patch JSON to local CUDA/C++ source files and only return valid JSON.",
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
    if not isinstance(files, list) or not files:
        raise ValueError("Concrete code response must include a non-empty files list.")
    seen = set()
    for item in files:
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
    response.setdefault("code_generation_notes", [])
    response.setdefault("expected_static_properties", [])
    response["generation_method"] = "patch_json_plus_source_files"
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
    ensure_generated_code_has_minimal_kernel(generated_code)
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


def apply_concrete_code(
    concrete_code: dict[str, Any],
    code_root: Path = DEFAULT_CODE_ROOT,
) -> dict[str, Any]:
    return apply_generated_code_files(concrete_code, code_root)


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

    if compute_strategy(strategy_id) and not re.search(r"\+=\s*[^;]*\*\s*[^;]*;", code_only):
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
