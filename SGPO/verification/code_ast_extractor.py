from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from SGPO.utils.common_utils import save_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CODE_ROOT = ROOT / "gemm_code" / "skeleton"
DEFAULT_OUTPUT = ROOT / "results" / "code" / "code_ast.json"
DEFAULT_CODE_FILES = [
    "cuda_kernel.cuh",
]


def extract_code_ast(code_root: Path = DEFAULT_CODE_ROOT, code_files: list[str] | None = None) -> dict[str, Any]:
    files = {}
    for relative_path in code_files or DEFAULT_CODE_FILES:
        path = code_root / relative_path
        if path.exists():
            content = path.read_text(encoding="utf-8")
            files[relative_path] = extract_file_ast(relative_path, content)
    return {
        "ast_kind": "sgpo_lightweight_cuda_ast",
        "code_root": str(code_root),
        "files": files,
        "summary": summarize_ast(files),
    }


def extract_file_ast(relative_path: str, content: str) -> dict[str, Any]:
    return {
        "path": relative_path,
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "line_count": len(content.splitlines()),
        "functions": extract_functions(content),
        "kernels": extract_kernels(content),
        "kernel_launches": extract_kernel_launches(content),
        "patch_regions": extract_patch_regions(content),
        "launch_config": extract_launch_config(content),
        "loops": extract_loops(content),
        "guards": extract_guards(content),
        "shared_memory": extract_shared_memory(content),
        "synchronization": extract_synchronization(content),
        "vector_types": extract_vector_types(content),
        "async_copy": extract_async_copy_evidence(content),
    }


def extract_async_copy_evidence(content: str) -> dict[str, bool]:
    # Preserve PTX string literals while discarding comments containing examples.
    tokens = re.sub(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*[\s\S]*?\*/',
                    lambda m: m.group(0) if m.group(0).startswith('"') else " ", content)
    cuda_pipeline = "memcpy_async" in tokens and "producer_commit" in tokens
    group_copy = "memcpy_async" in tokens and bool(re.search(r"(?:cooperative_groups|cg)::wait\s*\(", tokens))
    return {
        "copy": bool(re.search(r"cp\.async\.(?:ca|cg)\.shared\.global|\bmemcpy_async\s*\(|__pipeline_memcpy_async\s*\(", tokens)),
        "commit": "cp.async.commit_group" in tokens or cuda_pipeline or group_copy or "__pipeline_commit" in tokens,
        "wait": "cp.async.wait_group" in tokens or "cp.async.wait_all" in tokens
                or "consumer_wait" in tokens or group_copy or "__pipeline_wait_prior" in tokens,
    }


def extract_functions(content: str) -> list[dict[str, Any]]:
    control_keywords = {"if", "for", "while", "switch", "catch"}
    pattern = re.compile(
        r"(?:template\s*<[^>]+>\s*)?"
        r"(?P<prefix>(?:__global__\s+)?(?:static\s+)?(?:inline\s+)?)"
        r"(?P<return_type>[\w:<>,\s*&]+?)\s+"
        r"(?P<name>[A-Za-z_]\w*)\s*\((?P<params>[^;{}]*)\)\s*\{",
        re.MULTILINE,
    )
    functions = []
    for match in pattern.finditer(content):
        name = match.group("name")
        if name in control_keywords:
            continue
        source_line = line_at_offset(content, match.start())
        if source_line.lstrip().startswith("#define"):
            continue
        functions.append(
            {
            "name": match.group("name"),
            "return_type": " ".join(match.group("return_type").split()),
            "is_global": "__global__" in match.group("prefix"),
            "line": line_number(content, match.start()),
            }
        )
    return functions


def line_at_offset(content: str, offset: int) -> str:
    line_start = content.rfind("\n", 0, offset) + 1
    line_end = content.find("\n", offset)
    if line_end == -1:
        line_end = len(content)
    return content[line_start:line_end]


def extract_kernels(content: str) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"(?:template\s*<(?P<template>[^>]+)>\s*)?"
        r"__global__\s+(?P<return_type>[\w:<>,\s*&]+?)\s+"
        r"(?P<name>[A-Za-z_]\w*)\s*\(",
        re.MULTILINE,
    )
    return [
        {
            "name": match.group("name"),
            "return_type": " ".join(match.group("return_type").split()),
            "template": match.group("template"),
            "is_global": True,
            "line": line_number(content, match.start()),
        }
        for match in pattern.finditer(content)
    ]


def extract_kernel_launches(content: str) -> list[dict[str, Any]]:
    pattern = re.compile(r"(?P<kernel>[A-Za-z_]\w*(?:<[^;{}]+>)?)\s*<<<(?P<config>[^>]+)>>>\s*\(", re.MULTILINE)
    return [
        {
            "kernel": match.group("kernel").strip(),
            "config": " ".join(match.group("config").split()),
            "line": line_number(content, match.start()),
        }
        for match in pattern.finditer(content)
    ]


def extract_patch_regions(content: str) -> list[dict[str, Any]]:
    regions = []
    begin_pattern = re.compile(r"(?:SGPO_PATCH_)?([A-Z0-9_]+)_BEGIN")
    end_pattern = re.compile(r"(?:SGPO_PATCH_)?([A-Z0-9_]+)_END")
    lines = content.splitlines()
    open_region: dict[str, Any] | None = None
    for index, line in enumerate(lines, start=1):
        begin = begin_pattern.search(line)
        end = end_pattern.search(line)
        if begin:
            open_region = {"name": begin.group(1), "begin_line": index}
        elif end and open_region and end.group(1) == open_region["name"]:
            body = "\n".join(lines[open_region["begin_line"] : index - 1])
            regions.append(
                {
                    "name": open_region["name"],
                    "begin_line": open_region["begin_line"],
                    "end_line": index,
                    "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                    "body_line_count": len(body.splitlines()),
                }
            )
            open_region = None
    return regions


def extract_launch_config(content: str) -> dict[str, Any]:
    config = {}
    for name in ("BM", "BN", "BK", "TM", "TN"):
        match = re.search(rf"static\s+const\s+int\s+{name}\s*=\s*([0-9]+)\s*;", content)
        if match:
            config[name] = int(match.group(1))
    block = re.search(r"dim3\s+block\s*\(([^;]+)\);", content)
    grid = re.search(r"dim3\s+grid\s*\(([^;]+)\);", content)
    if block:
        config["block_expr"] = " ".join(block.group(1).split())
    if grid:
        config["grid_expr"] = " ".join(grid.group(1).split())
    if all(key in config for key in ("BM", "BN", "TM", "TN")):
        try:
            config["derived_threads_per_block"] = (config["BM"] // config["TM"]) * (config["BN"] // config["TN"])
        except ZeroDivisionError:
            config["derived_threads_per_block"] = None
    return config


def extract_loops(content: str) -> list[dict[str, Any]]:
    pattern = re.compile(r"for\s*\((?P<header>[^)]*)\)\s*\{")
    return [
        {
            "header": " ".join(match.group("header").split()),
            "line": line_number(content, match.start()),
            "mentions_k": bool(re.search(r"\bk0?\b|\bk\b", match.group("header"))),
        }
        for match in pattern.finditer(content)
    ]


def extract_guards(content: str) -> list[dict[str, Any]]:
    pattern = re.compile(r"if\s*\((?P<cond>[^)]*)\)\s*\{")
    return [
        {
            "condition": " ".join(match.group("cond").split()),
            "line": line_number(content, match.start()),
            "is_boundary_guard": any(token in match.group("cond") for token in ["< M", "< N", "< K", ">= 0"]),
        }
        for match in pattern.finditer(content)
    ]


def extract_shared_memory(content: str) -> list[dict[str, Any]]:
    code_only = strip_comments_preserve_lines(content)
    pattern = re.compile(
        r"__shared__\s+(?P<type>[\w:<>\s]+?)\s+(?P<name>[A-Za-z_]\w*)\s*(?P<shape>(?:\[[^\]]+\])+)"
    )
    return [
        {
            "type": match.group("type"),
            "name": match.group("name"),
            "shape": match.group("shape"),
            "line": line_number(code_only, match.start()),
        }
        for match in pattern.finditer(code_only)
    ]


def strip_comments_preserve_lines(content: str) -> str:
    without_block_comments = re.sub(
        r"/\*.*?\*/",
        lambda match: "\n" * match.group(0).count("\n"),
        content,
        flags=re.DOTALL,
    )
    return re.sub(r"//.*", "", without_block_comments)


def extract_synchronization(content: str) -> list[dict[str, Any]]:
    return [
        {
            "primitive": "__syncthreads",
            "line": line_number(content, match.start()),
        }
        for match in re.finditer(r"__syncthreads\s*\(", content)
    ]


def extract_vector_types(content: str) -> list[dict[str, Any]]:
    pattern = re.compile(r"\b(float2|float4|int2|int4)\b")
    return [
        {
            "type": match.group(1),
            "line": line_number(content, match.start()),
        }
        for match in pattern.finditer(content)
    ]


def summarize_ast(files: dict[str, Any]) -> dict[str, Any]:
    launch_configs = [file_ast.get("launch_config", {}) for file_ast in files.values()]
    return {
        "kernel_count": sum(len(file_ast.get("kernels", [])) for file_ast in files.values()),
        "kernel_launch_count": sum(len(file_ast.get("kernel_launches", [])) for file_ast in files.values()),
        "loop_count": sum(len(file_ast.get("loops", [])) for file_ast in files.values()),
        "boundary_guard_count": sum(
            len([guard for guard in file_ast.get("guards", []) if guard.get("is_boundary_guard")])
            for file_ast in files.values()
        ),
        "syncthreads_count": sum(len(file_ast.get("synchronization", [])) for file_ast in files.values()),
        "shared_memory_decl_count": sum(len(file_ast.get("shared_memory", [])) for file_ast in files.values()),
        "vector_type_count": sum(len(file_ast.get("vector_types", [])) for file_ast in files.values()),
        "launch_config": next((config for config in launch_configs if config), {}),
    }


def line_number(content: str, offset: int) -> int:
    return content.count("\n", 0, offset) + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract lightweight AST/static structure from SGPO CUDA code.")
    parser.add_argument("--code-root", type=Path, default=DEFAULT_CODE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ast = extract_code_ast(args.code_root)
    save_json(args.output, ast)
    print(json.dumps(ast.get("summary", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
