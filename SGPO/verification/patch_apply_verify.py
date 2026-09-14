from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from SGPO.utils.common_utils import load_json, save_json
from SGPO.verification.build_run_verifier import (
    DEFAULT_BUILD_DIR,
    DEFAULT_SOURCE_DIR,
    DEFAULT_VCVARS64,
    verify_build_and_run,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATCH = ROOT / "results" / "patch" / "generated_patch.json"
DEFAULT_IR = ROOT / "data" / "IRs" / "ir_patch" / "optir.patch.json"
DEFAULT_OUTPUT_IR = ROOT / "data" / "IRs" / "ir_patch" / "optir.verified.json"


def apply_patch_then_verify(
    ir: dict[str, Any],
    patch_result: dict[str, Any],
    source_dir: Path = DEFAULT_SOURCE_DIR,
    build_dir: Path = DEFAULT_BUILD_DIR,
    timeout_seconds: int = 60,
    vcvars64_path: Path = DEFAULT_VCVARS64,
    build_platform: str = "windows",
) -> dict[str, Any]:
    next_ir = copy.deepcopy(ir)
    apply_result = apply_generated_patch(patch_result, source_dir)
    update_patch_application(next_ir, apply_result)

    if apply_result["status"] != "pass":
        mark_patch_apply_failed(next_ir, apply_result)
        return next_ir

    verified_ir = verify_build_and_run(
        ir=next_ir,
        source_dir=source_dir,
        build_dir=build_dir,
        timeout_seconds=timeout_seconds,
        vcvars64_path=vcvars64_path,
        build_platform=build_platform,
    )
    verified_ir.setdefault("verification", {})["summary"] = summarize_verification(verified_ir)
    return verified_ir


def apply_generated_patch(patch_result: dict[str, Any], source_dir: Path) -> dict[str, Any]:
    diff = (patch_result.get("code_patch") or {}).get("diff", "")
    if not diff.strip():
        return fail_apply("code_patch.diff is empty.")

    git_result = try_git_apply(diff, source_dir)
    if git_result["status"] == "pass":
        return git_result

    fallback_result = try_apply_hunks_by_modified_regions(diff, patch_result, source_dir)
    if fallback_result["status"] == "pass":
        fallback_result["git_apply_error"] = git_result.get("error_message")
        return fallback_result
    fallback_result["git_apply_error"] = git_result.get("error_message")
    return fallback_result


def try_git_apply(diff: str, source_dir: Path) -> dict[str, Any]:
    patch_path = source_dir / "sgpo_generated.patch"
    patch_path.write_text(diff, encoding="utf-8")
    proc = subprocess.run(
        ["git", "apply", "--unsafe-paths", str(patch_path)],
        cwd=str(source_dir),
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    if proc.returncode == 0:
        return {
            "status": "pass",
            "method": "git_apply",
            "patch_file": str(patch_path),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    return fail_apply(proc.stderr or proc.stdout or "git apply failed", method="git_apply", patch_file=str(patch_path))


def try_apply_hunks_by_modified_regions(
    diff: str,
    patch_result: dict[str, Any],
    source_dir: Path,
) -> dict[str, Any]:
    hunks = split_hunks(diff)
    regions = patch_result.get("modified_code_regions", [])
    files = [region.get("file") for region in regions if region.get("file")]
    if not hunks:
        return fail_apply("No unified diff hunks found.", method="region_hunk")
    if len(files) < len(hunks):
        return fail_apply(
            f"Not enough modified_code_regions files for hunks: files={len(files)}, hunks={len(hunks)}.",
            method="region_hunk",
        )

    applied = []
    for hunk, relative_file in zip(hunks, files):
        path = (source_dir / relative_file).resolve()
        if not path.exists():
            return fail_apply(f"Patch target file does not exist: {path}", method="region_hunk")
        content = path.read_text(encoding="utf-8")
        updated = apply_single_hunk(content, hunk)
        if updated is None:
            return fail_apply(f"Could not match hunk in {relative_file}.", method="region_hunk")
        path.write_text(updated, encoding="utf-8")
        applied.append({"file": relative_file, "hunk_header": hunk["header"]})

    return {
        "status": "pass",
        "method": "region_hunk",
        "applied": applied,
    }


def split_hunks(diff: str) -> list[dict[str, Any]]:
    lines = diff.splitlines()
    hunks = []
    current: dict[str, Any] | None = None
    for line in lines:
        if line.startswith("@@"):
            if current:
                hunks.append(current)
            current = {"header": line, "lines": []}
        elif current is not None:
            current["lines"].append(line)
    if current:
        hunks.append(current)
    return hunks


def apply_single_hunk(content: str, hunk: dict[str, Any]) -> str | None:
    old_lines = []
    new_lines = []
    for line in hunk["lines"]:
        if line.startswith("\\ No newline"):
            continue
        marker = line[:1]
        text = line[1:] if marker in {" ", "-", "+"} else line
        if marker in {" ", "-"}:
            old_lines.append(text)
        if marker in {" ", "+"}:
            new_lines.append(text)

    old_block = "\n".join(trim_empty_edges(old_lines))
    new_block = "\n".join(trim_empty_edges(new_lines))
    if not old_block:
        return None

    if old_block in content:
        return content.replace(old_block, new_block, 1)
    if new_block and new_block in content:
        return content

    normalized_content = normalize_newlines(content)
    if old_block in normalized_content:
        return normalized_content.replace(old_block, new_block, 1)
    if new_block and new_block in normalized_content:
        return normalized_content

    compact_old = "\n".join(line.rstrip() for line in old_block.splitlines())
    compact_content = "\n".join(line.rstrip() for line in normalized_content.splitlines())
    index = compact_content.find(compact_old)
    if index < 0:
        return None
    return compact_content[:index] + new_block + compact_content[index + len(compact_old) :]


def trim_empty_edges(lines: list[str]) -> list[str]:
    start = 0
    end = len(lines)
    while start < end and lines[start] == "":
        start += 1
    while end > start and lines[end - 1] == "":
        end -= 1
    return lines[start:end]


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def update_patch_application(ir: dict[str, Any], apply_result: dict[str, Any]) -> None:
    ir["patch_application"] = {
        "stage": "Patch Application",
        "status": apply_result["status"],
        "method": apply_result.get("method"),
        "applied_at_utc": datetime.now(timezone.utc).isoformat(),
        "applied": apply_result.get("applied", []),
        "error_message": apply_result.get("error_message"),
        "git_apply_error": apply_result.get("git_apply_error"),
    }


def mark_patch_apply_failed(ir: dict[str, Any], apply_result: dict[str, Any]) -> dict[str, Any]:
    verification = ir.setdefault("verification", {})
    verification["compile"] = {
        "status": "not_run",
        "error_message": "patch application failed",
    }
    verification["correctness"] = {
        "status": "not_run",
        "error_message": "patch application failed",
    }
    verification["runtime_safety"] = {
        "status": "not_run",
        "cuda_error": None,
    }
    verification["summary"] = {
        "compile_status": "not_run",
        "correctness_status": "not_run",
        "cuda_error": None,
        "latency_ms": None,
        "gflops": None,
        "patch_apply_status": apply_result["status"],
        "patch_apply_error": apply_result.get("error_message"),
    }
    performance = ir.setdefault("performance", {})
    performance["latency_ms"] = None
    performance["gflops"] = None
    return ir


def mark_post_check_failed(ir: dict[str, Any], post_check_report: dict[str, Any]) -> dict[str, Any]:
    next_ir = copy.deepcopy(ir)
    next_ir["post_check"] = post_check_report
    verification = next_ir.setdefault("verification", {})
    verification["compile"] = {
        "status": "not_run",
        "error_message": "post check failed",
    }
    verification["correctness"] = {
        "status": "not_run",
        "error_message": "post check failed",
    }
    verification["runtime_safety"] = {
        "status": "not_run",
        "cuda_error": None,
        "illegal_memory_access": False,
        "misaligned_address": False,
        "out_of_bounds": False,
    }
    verification["accepted"] = False
    verification["accept_reason"] = None
    verification["summary"] = {
        "compile_status": "not_run",
        "correctness_status": "not_run",
        "cuda_error": None,
        "latency_ms": None,
        "gflops": None,
        "post_check_status": "fail",
        "postconditions_ok": post_check_report.get("postconditions_ok"),
        "hard_constraints_checked": post_check_report.get("hard_constraints_checked"),
        "hard_constraints_ok": post_check_report.get("hard_constraints_ok"),
    }
    performance = next_ir.setdefault("performance", {})
    performance["latency_ms"] = None
    performance["gflops"] = None
    return next_ir


def summarize_verification(ir: dict[str, Any]) -> dict[str, Any]:
    verification = ir.get("verification", {})
    runtime_safety = verification.get("runtime_safety", {})
    performance = ir.get("performance", {})
    return {
        "compile_status": verification.get("compile", {}).get("status"),
        "correctness_status": verification.get("correctness", {}).get("status"),
        "runtime_safety_status": runtime_safety.get("status"),
        "cuda_error": runtime_safety.get("cuda_error"),
        "latency_ms": performance.get("latency_ms"),
        "gflops": performance.get("gflops"),
        "benchmark_runs": performance.get("benchmark_runs"),
        "benchmark_successful_runs": performance.get("benchmark_successful_runs"),
        "latency_ms_mean": performance.get("latency_ms_mean"),
        "latency_ms_median": performance.get("latency_ms_median"),
        "latency_ms_std": performance.get("latency_ms_std"),
        "gflops_mean": performance.get("gflops_mean"),
        "gflops_median": performance.get("gflops_median"),
        "gflops_std": performance.get("gflops_std"),
        "gflops_best": performance.get("gflops_best"),
    }


def fail_apply(message: str, method: str | None = None, patch_file: str | None = None) -> dict[str, Any]:
    return {
        "status": "fail",
        "method": method,
        "patch_file": patch_file,
        "error_message": message,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply SGPO generated patch, compile, run, and write verification IR.")
    parser.add_argument("--patch", type=Path, default=DEFAULT_PATCH)
    parser.add_argument("--ir", type=Path, default=DEFAULT_IR)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--output-ir", type=Path, default=DEFAULT_OUTPUT_IR)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--vcvars64", type=Path, default=DEFAULT_VCVARS64)
    parser.add_argument("--build-platform", choices=["windows", "linux"], default="windows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    patch_result = load_json(args.patch)
    ir = load_json(args.ir)
    verified_ir = apply_patch_then_verify(
        ir=ir,
        patch_result=patch_result,
        source_dir=args.source_dir,
        build_dir=args.build_dir,
        timeout_seconds=args.timeout_seconds,
        vcvars64_path=args.vcvars64,
        build_platform=args.build_platform,
    )
    save_json(args.output_ir, verified_ir)
    print(json.dumps(verified_ir.get("verification", {}).get("summary", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
