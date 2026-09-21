from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from SGPO.llm.openai_client import OpenAICompatibleClient
from SGPO.llm.openai_client import extract_json_object
from SGPO.llm.patch_generator import load_code_context
from SGPO.llm.strategy_examples import strategy_with_examples
from SGPO.utils.common_utils import load_config, load_json, save_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IR = ROOT / "data" / "IRs" / "ir_patch" / "optir.extracted.json"
DEFAULT_CODE_ROOT = ROOT / "gemm_code" / "cpu_skeleton"
DEFAULT_PROMPT = ROOT / "llm" / "prompts" / "generate_cpu_c_code_prompt.txt"
DEFAULT_CONFIG = ROOT / "conf.yaml"
DEFAULT_OUTPUT = ROOT / "results" / "code" / "cpu_c_code_files.json"
CPU_CODE_FILES = ["main.c", "kernel.h", "cpu_kernel.c"]
ALLOWED_CPU_CODE_FILES = set(CPU_CODE_FILES)
CPU_PATCH_BEGIN = "SGPO_CPU_PATCH_KERNEL_BEGIN"
CPU_PATCH_END = "SGPO_CPU_PATCH_KERNEL_END"


def generate_cpu_c_code_with_llm(
    client: OpenAICompatibleClient,
    ir: dict[str, Any],
    code_context: dict[str, Any],
    prompt_path: Path = DEFAULT_PROMPT,
) -> dict[str, Any]:
    messages = build_cpu_c_code_messages(ir, code_context, prompt_path)
    response = complete_codegen_json(client, messages)
    return validate_cpu_c_code_response(response)


def generate_cpu_c_code_from_patch_with_llm(
    client: OpenAICompatibleClient,
    patch_ir: dict[str, Any],
    patch_result: dict[str, Any],
    strategy: dict[str, Any],
    code_context: dict[str, Any],
    prompt_path: Path = DEFAULT_PROMPT,
    repair_context: dict[str, Any] | None = None,
    patch_file: Path | None = None,
) -> dict[str, Any]:
    messages = build_cpu_c_patch_to_code_messages(
        patch_ir=patch_ir,
        patch_result=patch_result,
        strategy=strategy,
        code_context=code_context,
        prompt_path=prompt_path,
        repair_context=repair_context,
        patch_file=patch_file,
    )
    response = complete_codegen_json(client, messages)
    return validate_cpu_c_code_response(response)


def build_cpu_c_patch_to_code_messages(
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
    compact_ir = compact_cpu_ir_for_prompt(patch_ir)
    prompt = template.format(
        current_ir_json=json.dumps({"note": "same compact state as CURRENT_IR_AFTER_PATCH"}, ensure_ascii=False, indent=2),
        patch_ir_json=json.dumps(compact_ir, ensure_ascii=False, indent=2),
        strategy_json=json.dumps(strategy_with_examples(compact_strategy_for_prompt(strategy)), ensure_ascii=False, indent=2),
        patch_json=json.dumps(compact_patch_for_prompt(patch_payload), ensure_ascii=False, indent=2),
        code_context_json=json.dumps(compact_cpu_code_context_for_prompt(code_context), ensure_ascii=False, indent=2),
        repair_context_json=json.dumps(compact_repair_context_for_prompt(repair_context or {}), ensure_ascii=False, indent=2),
    )
    return [
        {
            "role": "system",
            "content": "You apply SGPO Patch JSON to ordinary CPU C GEMM source files and only return valid JSON.",
        },
        {"role": "user", "content": prompt},
    ]


def build_cpu_c_code_messages(
    ir: dict[str, Any],
    code_context: dict[str, Any],
    prompt_path: Path = DEFAULT_PROMPT,
) -> list[dict[str, str]]:
    template = prompt_path.read_text(encoding="utf-8")
    prompt = template.format(
        current_ir_json=json.dumps(compact_cpu_ir_for_prompt(ir), ensure_ascii=False, indent=2),
        patch_ir_json=json.dumps(compact_cpu_ir_for_prompt(ir), ensure_ascii=False, indent=2),
        strategy_json=json.dumps(strategy_with_examples({"applied_strategies": [
            {"strategy_id": sid} for sid in (ir.get("strategy", {}) or {}).get("applied_strategy_ids", [])
        ]}), ensure_ascii=False, indent=2),
        patch_json=json.dumps({}, ensure_ascii=False, indent=2),
        code_context_json=json.dumps(compact_cpu_code_context_for_prompt(code_context), ensure_ascii=False, indent=2),
        repair_context_json=json.dumps({}, ensure_ascii=False, indent=2),
    )
    return [
        {
            "role": "system",
            "content": "You generate ordinary CPU C GEMM source files and only return valid JSON.",
        },
        {"role": "user", "content": prompt},
    ]


def complete_codegen_json(client: OpenAICompatibleClient, messages: list[dict[str, str]]) -> dict[str, Any]:
    if hasattr(client, "complete_codegen_text"):
        text = client.complete_codegen_text(messages)
        return extract_cpu_codegen_response(text)
    if hasattr(client, "complete_codegen_json"):
        return client.complete_codegen_json(messages)
    return client.complete_json(messages)


def extract_cpu_codegen_response(text: str) -> dict[str, Any]:
    try:
        return extract_json_object(text)
    except (json.JSONDecodeError, ValueError):
        recovered = recover_cpu_codegen_edit_response(text)
        if recovered is not None:
            return recovered
        raise


def recover_cpu_codegen_edit_response(text: str) -> dict[str, Any] | None:
    code = extract_fenced_code(text)
    if code:
        return cpu_edit_response_from_code(code, "Recovered cpu_kernel.c replacement from fenced code block.")
    code = extract_unterminated_replacement_value(text)
    if code:
        return cpu_edit_response_from_code(code, "Recovered cpu_kernel.c replacement from malformed JSON replacement string.")
    return None


def cpu_edit_response_from_code(code: str, note: str) -> dict[str, Any]:
    code = strip_optional_cpu_function_wrapper(code)
    return {
        "backend": "cpu",
        "language": "c",
        "edits": [
            {
                "path": "cpu_kernel.c",
                "anchor": f"{CPU_PATCH_BEGIN}/END",
                "replacement": code.strip(),
                "change_summary": note,
            }
        ],
        "code_generation_notes": [note],
        "expected_static_properties": [],
        "recovered_from_malformed_json": True,
    }


def extract_fenced_code(text: str) -> str | None:
    matches = re.findall(r"```(?:c|cpp|json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if not matches:
        return None
    for candidate in matches:
        stripped = candidate.strip()
        if "cpu_gemm" in stripped or "for" in stripped or "C[" in stripped:
            if stripped.startswith("{"):
                try:
                    parsed = extract_json_object(stripped)
                except Exception:
                    pass
                else:
                    edits = parsed.get("edits") or []
                    if edits and isinstance(edits[0], dict):
                        replacement = normalize_edit_replacement(edits[0])
                        if replacement:
                            return replacement
            return stripped
    return matches[0].strip()


def extract_unterminated_replacement_value(text: str) -> str | None:
    match = re.search(r'"replacement"\s*:\s*"(?P<body>.*)', text, flags=re.DOTALL)
    if not match:
        return None
    body = match.group("body")
    body = re.split(r'"\s*,\s*"change_summary"|"\s*\}\s*\]\s*,?|"\s*\}\s*\}', body, maxsplit=1, flags=re.DOTALL)[0]
    body = body.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")
    body = body.strip()
    return body if body else None


def strip_optional_cpu_function_wrapper(code: str) -> str:
    stripped = code.strip()
    match = re.search(
        r"void\s+cpu_gemm\s*\([^)]*\)\s*\{(?P<body>.*)\}\s*$",
        stripped,
        flags=re.DOTALL,
    )
    if not match:
        return stripped
    body = match.group("body")
    anchor_pattern = re.compile(
        rf"/\*\s*{re.escape(CPU_PATCH_BEGIN)}\s*\*/(?P<body>.*?)/\*\s*{re.escape(CPU_PATCH_END)}\s*\*/",
        re.DOTALL,
    )
    anchor_match = anchor_pattern.search(body)
    if anchor_match:
        return anchor_match.group("body").strip()
    return body.strip()


def compact_cpu_ir_for_prompt(ir: dict[str, Any]) -> dict[str, Any]:
    keep_keys = [
        "target",
        "problem",
        "cpu_tiling",
        "cpu_resource",
        "cpu_memory",
        "cpu_microkernel",
        "cpu_vectorization",
        "cpu_schedule",
        "cpu_macro_kernel",
        "cpu_parallel",
        "cpu_compiler",
        "cpu_epilogue",
        "cpu_tail",
        "cpu_safety",
        "strategy",
    ]
    compact = {key: ir[key] for key in keep_keys if key in ir}
    hardware = ir.get("hardware", {}) or {}
    if "cpu_cache" in hardware or "cpu_physical_cores" in hardware or "cpu_logical_processors" in hardware:
        compact["hardware"] = {
            key: hardware.get(key)
            for key in ["cpu_name", "cpu_physical_cores", "cpu_logical_processors", "cpu_cache"]
            if key in hardware
        }
    strategy = compact.get("strategy")
    if isinstance(strategy, dict):
        compact["strategy"] = {
            key: strategy.get(key)
            for key in ["current_stage", "current_subphase", "current_strategy_id", "applied_strategy_ids", "chain_path"]
            if key in strategy
        }
    return compact


def compact_strategy_for_prompt(strategy: dict[str, Any]) -> dict[str, Any]:
    keep_keys = [
        "strategy_id",
        "stage",
        "category",
        "subphase",
        "principle",
        "preconditions",
        "postconditions",
        "ir_updates",
        "implementation_contract",
        "implementation_example_ids",
        "applied_strategies",
        "resource_effects",
        "risk_level",
    ]
    return {key: strategy[key] for key in keep_keys if key in strategy}


def compact_patch_for_prompt(patch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: patch.get(key)
        for key in [
            "strategy_id",
            "modified_code_regions",
            "modified_ir_fields",
            "ir_updates",
            "expected_effect",
            "potential_risks",
        ]
        if key in patch
    }


def compact_repair_context_for_prompt(repair_context: dict[str, Any]) -> dict[str, Any]:
    diagnosis = repair_context.get("defect_diagnosis") or repair_context.get("diagnosis") or {}
    return {
        key: repair_context.get(key)
        for key in ["attempt", "repair_attempt", "last_error"]
        if key in repair_context
    } | ({"defect_diagnosis": diagnosis} if diagnosis else {})


def compact_cpu_code_context_for_prompt(code_context: dict[str, Any]) -> dict[str, Any]:
    files = code_context.get("files", {}) if isinstance(code_context, dict) else {}
    compact_files: dict[str, Any] = {}
    for path, content in files.items():
        if path == "cpu_kernel.c":
            compact_files[path] = {
                "signature": "void cpu_gemm(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C)",
                "patch_region": extract_cpu_patch_region(str(content)),
            }
        elif path in {"kernel.h", "main.c"}:
            compact_files[path] = {"summary": summarize_cpu_support_file(path, str(content))}
    return {
        "edit_contract": {
            "preferred_output": "edits",
            "anchor_begin": CPU_PATCH_BEGIN,
            "anchor_end": CPU_PATCH_END,
            "return_replacement_only": True,
        },
        "files": compact_files,
    }


def validate_cpu_c_code_response(response: dict[str, Any]) -> dict[str, Any]:
    response["backend"] = "cpu"
    response["language"] = "c"
    files = response.get("files")
    edits = response.get("edits")
    if (not isinstance(files, list) or not files) and (not isinstance(edits, list) or not edits):
        raise ValueError("CPU C code response must include a non-empty files or edits list.")
    seen = set()
    for item in files or []:
        if not isinstance(item, dict):
            raise ValueError("Each CPU C code file item must be an object.")
        relative_path = item.get("path")
        content = item.get("content")
        if relative_path not in ALLOWED_CPU_CODE_FILES:
            raise ValueError(f"CPU C code file is not allowed: {relative_path}")
        if relative_path in seen:
            raise ValueError(f"Duplicate CPU C code file: {relative_path}")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"CPU C code file content is empty: {relative_path}")
        reject_cuda_tokens(relative_path, content)
        seen.add(relative_path)
    seen_edits = set()
    for item in edits or []:
        if not isinstance(item, dict):
            raise ValueError("Each CPU C code edit item must be an object.")
        relative_path = item.get("path")
        replacement = normalize_edit_replacement(item)
        item["replacement"] = replacement
        anchor = item.get("anchor", f"{CPU_PATCH_BEGIN}/END")
        if relative_path not in ALLOWED_CPU_CODE_FILES:
            raise ValueError(f"CPU C edit file is not allowed: {relative_path}")
        if relative_path in seen_edits:
            raise ValueError(f"Duplicate CPU C edit file: {relative_path}")
        if CPU_PATCH_BEGIN not in str(anchor) and relative_path == "cpu_kernel.c":
            raise ValueError(f"CPU C edit anchor is not supported: {anchor}")
        if not isinstance(replacement, str) or not replacement.strip():
            raise ValueError(f"CPU C edit replacement is empty: {relative_path}")
        reject_cuda_tokens(relative_path, replacement)
        seen_edits.add(relative_path)
    response.setdefault("code_generation_notes", [])
    response.setdefault("expected_static_properties", [])
    response["generation_method"] = "cpu_c_source_files"
    response["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return response


def reject_cuda_tokens(relative_path: str, content: str) -> None:
    forbidden = [
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
    for pattern in forbidden:
        if re.search(pattern, content):
            raise ValueError(f"CPU C code file contains CUDA-only token in {relative_path}: {pattern}")


def apply_cpu_c_code_files(
    generated_code: dict[str, Any],
    code_root: Path = DEFAULT_CODE_ROOT,
) -> dict[str, Any]:
    applied = []
    for item in generated_code.get("files") or []:
        relative_path = item["path"]
        if relative_path not in ALLOWED_CPU_CODE_FILES:
            return {
                "status": "fail",
                "error_message": f"CPU C code file is not allowed: {relative_path}",
                "applied": applied,
            }
        path = (code_root / relative_path).resolve()
        if code_root.resolve() not in [path, *path.parents]:
            return {
                "status": "fail",
                "error_message": f"CPU C code file escapes code_root: {relative_path}",
                "applied": applied,
            }
        path.write_text(item["content"], encoding="utf-8")
        applied.append({"file": relative_path, "change_summary": item.get("change_summary")})
    for item in generated_code.get("edits", []) or []:
        relative_path = item["path"]
        if relative_path not in ALLOWED_CPU_CODE_FILES:
            return {
                "status": "fail",
                "error_message": f"CPU C edit file is not allowed: {relative_path}",
                "applied": applied,
            }
        path = (code_root / relative_path).resolve()
        if code_root.resolve() not in [path, *path.parents]:
            return {
                "status": "fail",
                "error_message": f"CPU C edit file escapes code_root: {relative_path}",
                "applied": applied,
            }
        content = path.read_text(encoding="utf-8")
        path.write_text(replace_cpu_patch_region(content, item["replacement"]), encoding="utf-8")
        applied.append({"file": relative_path, "change_summary": item.get("change_summary"), "method": "anchor_replacement"})
    validation = validate_cpu_c_materialization_from_code_root(code_root)
    if validation["status"] != "pass":
        return {
            "status": "fail",
            "error_message": validation["error_message"],
            "method": "cpu_c_source_files",
            "applied": applied,
            "materialization": validation,
        }
    return {
        "status": "pass",
        "method": "cpu_c_source_files",
        "applied": applied,
        "materialization": validation,
    }


def validate_cpu_c_materialization(generated_code: dict[str, Any]) -> dict[str, Any]:
    kernel = generated_file_content(generated_code, "cpu_kernel.c")
    if kernel is None:
        return {"status": "pass", "message": "cpu_kernel.c not changed by this generation.", "failures": []}
    failures = []
    if not re.search(r"\bvoid\s+cpu_gemm\s*\(", kernel):
        failures.append("cpu_kernel.c must define cpu_gemm.")
    if not re.search(r"\bfor\s*\(", kernel):
        failures.append("cpu_kernel.c must contain executable loop nests.")
    if not re.search(r"\+=\s*[^;]*\*\s*[^;]*;", kernel):
        failures.append("cpu_kernel.c must contain multiply-accumulate statements.")
    if not re.search(r"\bC\s*\[[^\]]+\]\s*=", kernel):
        failures.append("cpu_kernel.c must store the GEMM result to C.")
    if failures:
        return {"status": "fail", "error_message": " ".join(failures), "failures": failures}
    return {"status": "pass", "message": "Generated CPU C code materializes GEMM compute.", "failures": []}


def validate_cpu_c_materialization_from_code_root(code_root: Path) -> dict[str, Any]:
    kernel_path = code_root / "cpu_kernel.c"
    if not kernel_path.exists():
        return {"status": "fail", "error_message": "cpu_kernel.c does not exist.", "failures": ["cpu_kernel.c does not exist."]}
    return validate_cpu_kernel_content(kernel_path.read_text(encoding="utf-8"))


def validate_cpu_kernel_content(kernel: str) -> dict[str, Any]:
    failures = []
    code = strip_cpp_comments(kernel)
    if not re.search(r"\bvoid\s+cpu_gemm\s*\(", kernel):
        failures.append("cpu_kernel.c must define cpu_gemm.")
    if not re.search(r"\bfor\s*\(", kernel):
        failures.append("cpu_kernel.c must contain executable loop nests.")
    if not re.search(r"\+=\s*[^;]*\*\s*[^;]*;", kernel):
        failures.append("cpu_kernel.c must contain multiply-accumulate statements.")
    if not re.search(r"\bC\s*\[[^\]]+\]\s*=", kernel):
        failures.append("cpu_kernel.c must store the GEMM result to C.")
    failures.extend(check_cpu_kernel_static_semantics(code))
    if failures:
        return {"status": "fail", "error_message": " ".join(failures), "failures": failures}
    return {"status": "pass", "message": "Generated CPU C code materializes GEMM compute.", "failures": []}


def check_cpu_kernel_static_semantics(code: str) -> list[str]:
    failures: list[str] = []
    uses_simd = bool(re.search(r"\b__m(?:256|512)\b|_mm(?:256|512)_", code))
    if uses_simd and re.search(r"_mm(?:256|512)_loadu_ps\s*\(\s*&?\s*A\s*\[[^\]]*\b(?:k|kk)\b", code):
        failures.append(
            "SIMD micro-kernel must not load a vector along A's K dimension; broadcast scalar A and map SIMD lanes to contiguous N outputs."
        )
    if uses_simd and not re.search(r"_mm(?:256|512)_broadcast", code):
        failures.append("Explicit SIMD/FMA micro-kernel must broadcast scalar A values across vector lanes.")
    if uses_simd and not re.search(r"_mm(?:256|512)_loadu_ps\s*\(\s*&?\s*(?:B|b_panel)\s*\[", code):
        failures.append("Explicit SIMD/FMA micro-kernel must vector-load contiguous B/N-lane data.")

    pragma_pos = code.find("#pragma omp parallel")
    if pragma_pos >= 0:
        pack_alloc = re.search(r"float\s*\*\s*(?:a_panel|b_panel|sa|sb)\s*=", code)
        if pack_alloc and pack_alloc.start() < pragma_pos:
            failures.append("OpenMP GEMM must not share packed panel buffers allocated before the parallel region.")
    return failures


def generated_file_content(generated_code: dict[str, Any], relative_path: str) -> str | None:
    for item in generated_code.get("files", []) or []:
        if item.get("path") == relative_path:
            return item.get("content")
    return None


def normalize_edit_replacement(item: dict[str, Any]) -> str | None:
    replacement = item.get("replacement")
    if isinstance(replacement, str):
        return replacement
    replacement_lines = item.get("replacement_lines")
    if isinstance(replacement_lines, list) and all(isinstance(line, str) for line in replacement_lines):
        return "\n".join(replacement_lines)
    return None


def extract_cpu_patch_region(content: str) -> str:
    pattern = re.compile(
        rf"/\*\s*{re.escape(CPU_PATCH_BEGIN)}\s*\*/(?P<body>.*?)/\*\s*{re.escape(CPU_PATCH_END)}\s*\*/",
        re.DOTALL,
    )
    match = pattern.search(content)
    if not match:
        return truncate_text(content, 4000)
    return truncate_text(match.group("body").strip(), 3000)


def replace_cpu_patch_region(content: str, replacement: str) -> str:
    pattern = re.compile(
        rf"(?P<begin>/\*\s*{re.escape(CPU_PATCH_BEGIN)}\s*\*/)(?P<body>.*?)(?P<end>/\*\s*{re.escape(CPU_PATCH_END)}\s*\*/)",
        re.DOTALL,
    )
    if not pattern.search(content):
        raise ValueError("cpu_kernel.c does not contain SGPO CPU patch anchors.")
    normalized = replacement.strip("\n")
    return pattern.sub(lambda match: f"{match.group('begin')}\n{indent_replacement(normalized)}\n    {match.group('end')}", content, count=1)


def indent_replacement(replacement: str) -> str:
    lines = replacement.splitlines()
    return "\n".join(("    " + line if line.strip() else "") for line in lines)


def summarize_cpu_support_file(path: str, content: str) -> str:
    if path == "kernel.h":
        return "Declares cpu_gemm with const float A/B pointers and mutable C pointer."
    if "reference_gemm" in content and "SGPO_METRIC" in content:
        return "Benchmark harness initializes row-major A/B, runs cpu_gemm, checks scalar reference, and prints SGPO_METRIC."
    return truncate_text(content, 1000)


def truncate_text(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "\n/* ... truncated ... */"


def strip_cpp_comments(content: str) -> str:
    without_block_comments = re.sub(r"/\*.*?\*/", lambda match: "\n" * match.group(0).count("\n"), content, flags=re.DOTALL)
    return re.sub(r"//.*", "", without_block_comments)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ordinary CPU C GEMM code.")
    parser.add_argument("--ir", default=str(DEFAULT_IR))
    parser.add_argument("--code-root", default=str(DEFAULT_CODE_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    ir = load_json(Path(args.ir))
    code_root = Path(args.code_root)
    code_context = load_code_context(code_root, CPU_CODE_FILES)
    client = OpenAICompatibleClient(load_config(Path(args.config))["llm"])
    generated = generate_cpu_c_code_with_llm(client, ir, code_context, Path(args.prompt))
    apply_result = apply_cpu_c_code_files(generated, code_root)
    generated["apply_result"] = apply_result
    save_json(Path(args.output), generated)
    print(json.dumps({"output": args.output, "apply_result": apply_result}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
