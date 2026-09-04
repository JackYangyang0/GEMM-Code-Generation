from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from SGPO.utils.common_utils import load_json, save_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IR = ROOT / "data" / "IRs" / "ir_patch" / "optir.extracted.json"
DEFAULT_SOURCE_DIR = ROOT / "gemm_code" / "skeleton"
DEFAULT_OUTPUT_IR = ROOT / "data" / "IRs" / "ir_patch" / "optir.verified.json"
DEFAULT_BUILD_DIR = ROOT / "gemm_code" / "skeleton" / "build"
DEFAULT_VCVARS64 = Path(r"D:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat")


def verify_build_and_run(
    ir,
    source_dir=DEFAULT_SOURCE_DIR,
    build_dir=DEFAULT_BUILD_DIR,
    executable_name="gemm.exe",
    timeout_seconds=60,
    vcvars64_path=DEFAULT_VCVARS64,
    build_platform="windows",
):
    next_ir = copy.deepcopy(ir)
    build_platform = normalize_build_platform(build_platform)
    if build_platform == "linux" and executable_name == "gemm.exe":
        executable_name = "gemm"
    build_dir.mkdir(parents=True, exist_ok=True)
    exe_path = build_dir / executable_name

    compile_result = compile_gemm(ir, source_dir, exe_path, timeout_seconds, vcvars64_path, build_platform)
    update_compile_result(next_ir, compile_result)
    if compile_result["status"] != "pass":
        mark_unrun(next_ir, "compile failed")
        return next_ir

    run_result = run_gemm(exe_path, ir, timeout_seconds)
    update_run_result(next_ir, run_result)
    return next_ir


def compile_gemm(
    ir,
    source_dir,
    exe_path,
    timeout_seconds,
    vcvars64_path=None,
    build_platform="windows",
):
    build_platform = normalize_build_platform(build_platform)
    nvcc = shutil.which("nvcc")
    if not nvcc:
        return {
            "status": "fail",
            "command": None,
            "stdout": "",
            "stderr": "nvcc not found in PATH",
            "returncode": None,
            "build_platform": build_platform,
        }

    requested_arch = cuda_arch_flag(ir)
    arch = resolve_supported_cuda_arch(nvcc, requested_arch, build_platform)
    sources = discover_cuda_sources(source_dir)
    if not sources:
        return {
            "status": "fail",
            "command": None,
            "stdout": "",
            "stderr": f"No CUDA entry source found in {source_dir}. Expected main.cu or main.cpp.",
            "returncode": None,
            "build_platform": build_platform,
        }
    command = [
        nvcc,
        "-O3",
        "-std=c++17",
        f"-arch={arch}",
        "-x",
        "cu",
        *sources,
        "-lcublas",
        "-o",
        str(exe_path),
    ]
    if build_platform == "windows" and needs_msvc_environment() and vcvars64_path and Path(vcvars64_path).exists():
        result = run_command_with_vcvars(command, Path(vcvars64_path), cwd=source_dir, timeout_seconds=timeout_seconds)
    else:
        result = run_command(command, cwd=source_dir, timeout_seconds=timeout_seconds)
    result["build_platform"] = build_platform
    result["requested_arch"] = requested_arch
    result["resolved_arch"] = arch
    return result


def discover_cuda_sources(source_dir):
    source_dir = Path(source_dir)
    if (source_dir / "main.cu").exists():
        sources = ["main.cu"]
    elif (source_dir / "main.cpp").exists():
        sources = ["main.cpp"]
    else:
        return []

    for path in sorted(source_dir.glob("*.cu")):
        if path.name == "main.cu":
            continue
        sources.append(path.name)
    return sources


def resolve_supported_cuda_arch(nvcc, requested_arch, build_platform="windows"):
    supported = list_supported_cuda_arches(nvcc)
    if not supported or requested_arch in supported:
        return requested_arch
    if requested_arch == "sm_89" and "sm_87" in supported:
        return "sm_87"

    requested_number = arch_number(requested_arch)
    compatible = [
        arch for arch in supported
        if arch.startswith("sm_") and arch_number(arch) is not None and (
            requested_number is None or arch_number(arch) <= requested_number
        )
    ]
    if compatible:
        return max(compatible, key=lambda item: arch_number(item) or 0)
    return requested_arch


def list_supported_cuda_arches(nvcc):
    try:
        proc = subprocess.run(
            [nvcc, "--list-gpu-arch"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if proc.returncode != 0:
        return []
    return sorted(set(re.findall(r"\bsm_\d+\b", proc.stdout + "\n" + proc.stderr)))


def arch_number(arch):
    match = re.search(r"sm_(\d+)", str(arch))
    return int(match.group(1)) if match else None


def normalize_build_platform(build_platform):
    value = str(build_platform or "windows").strip().lower()
    if value in {"linux", "unix", "posix"}:
        return "linux"
    return "windows"


def needs_msvc_environment():
    return os.name == "nt" and shutil.which("cl") is None


def run_command_with_vcvars(command, vcvars64_path, cwd, timeout_seconds):
    batch_path = Path(cwd) / "sgpo_compile_with_vcvars.bat"
    command_text = subprocess.list2cmdline(command)
    batch_path.write_text(
        "\n".join(
            [
                "@echo off",
                f'call "{vcvars64_path}"',
                "if errorlevel 1 exit /b %errorlevel%",
                command_text,
                "exit /b %errorlevel%",
                "",
            ]
        ),
        encoding="utf-8",
    )
    display_command = ["cmd", "/d", "/c", str(batch_path)]
    try:
        proc = subprocess.run(
            display_command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        return {
            "status": "pass" if proc.returncode == 0 else "fail",
            "command": display_command,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
            "vcvars64_path": str(vcvars64_path),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "fail",
            "command": display_command,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or f"timeout after {timeout_seconds}s",
            "returncode": None,
            "timeout": True,
            "vcvars64_path": str(vcvars64_path),
        }


def run_gemm(exe_path, ir, timeout_seconds):
    problem = ir.get("problem", {})
    command = [
        str(exe_path),
        str(problem.get("M", 512)),
        str(problem.get("K", 512)),
        str(problem.get("N", 512)),
    ]
    result = run_command(command, cwd=exe_path.parent, timeout_seconds=timeout_seconds)
    result["metrics"] = parse_run_metrics(result["stdout"] + "\n" + result["stderr"])
    result["cuda_error"] = parse_cuda_error(result["stdout"] + "\n" + result["stderr"])
    return result


def run_command(command, cwd, timeout_seconds):
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        return {
            "status": "pass" if proc.returncode == 0 else "fail",
            "command": command,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "fail",
            "command": command,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or f"timeout after {timeout_seconds}s",
            "returncode": None,
            "timeout": True,
        }
    except OSError as exc:
        return {
            "status": "fail",
            "command": command,
            "stdout": "",
            "stderr": str(exc),
            "returncode": None,
        }


def update_compile_result(ir, compile_result):
    verification = ir.setdefault("verification", {})
    compile_node = verification.setdefault("compile", {})
    output_text = "\n".join(
        part for part in [compile_result.get("stdout"), compile_result.get("stderr")] if part
    )
    compile_node["status"] = compile_result["status"]
    compile_node["build_platform"] = compile_result.get("build_platform")
    compile_node["requested_arch"] = compile_result.get("requested_arch")
    compile_node["resolved_arch"] = compile_result.get("resolved_arch")
    compile_node["error_message"] = None if compile_result["status"] == "pass" else truncate(output_text)
    compile_node["command"] = compile_result["command"]
    compile_node["returncode"] = compile_result["returncode"]
    compile_node["vcvars64_path"] = compile_result.get("vcvars64_path")
    compile_node["stdout"] = truncate(compile_result.get("stdout"))
    compile_node["stderr"] = truncate(compile_result.get("stderr"))


def mark_unrun(ir, reason):
    verification = ir.setdefault("verification", {})
    correctness = verification.setdefault("correctness", {})
    runtime_safety = verification.setdefault("runtime_safety", {})
    correctness["status"] = "not_run"
    correctness["error_message"] = reason
    runtime_safety["status"] = "not_run"
    runtime_safety["cuda_error"] = None

    performance = ir.setdefault("performance", {})
    performance["latency_ms"] = None
    performance["gflops"] = None


def update_run_result(ir, run_result):
    text = run_result["stdout"] + "\n" + run_result["stderr"]
    metrics = run_result.get("metrics", {})
    cuda_error = run_result.get("cuda_error")

    verification = ir.setdefault("verification", {})
    correctness = verification.setdefault("correctness", {})
    runtime_safety = verification.setdefault("runtime_safety", {})

    runtime_failed = run_result["status"] != "pass" or cuda_error is not None
    runtime_safety["status"] = "fail" if runtime_failed else "pass"
    runtime_safety["cuda_error"] = cuda_error
    runtime_safety["host_error"] = None
    if run_result["status"] != "pass" and cuda_error is None:
        runtime_safety["host_error"] = first_error_line(text) or f"host runtime error: returncode={run_result['returncode']}"
    runtime_safety["illegal_memory_access"] = contains_error(cuda_error, "illegal memory access")
    runtime_safety["misaligned_address"] = contains_error(cuda_error, "misaligned address")
    runtime_safety["out_of_bounds"] = contains_error(cuda_error, "out of bounds")
    runtime_safety["returncode"] = run_result["returncode"]

    correctness_status = metrics.get("correctness")
    if correctness_status in {"pass", "fail"}:
        correctness["status"] = correctness_status
    elif run_result["status"] == "pass" and "Result= PASS" in text:
        correctness["status"] = "pass"
    elif "Result= FAIL" in text:
        correctness["status"] = "fail"
    else:
        correctness["status"] = "unknown" if run_result["status"] == "pass" else "not_run"

    correctness["reference"] = "cublas"
    correctness["max_abs_error"] = metrics.get("max_abs_error")
    correctness["max_rel_error"] = metrics.get("max_rel_error")
    correctness.setdefault("tolerance", 0.001)
    correctness["error_message"] = None if correctness["status"] == "pass" else first_error_line(text)

    performance = ir.setdefault("performance", {})
    performance["latency_ms"] = metrics.get("latency_ms")
    performance["gflops"] = metrics.get("gflops")
    performance["relative_to_cublas"] = metrics.get("relative_to_cublas")

    verification["accepted"] = (
        verification.get("compile", {}).get("status") == "pass"
        and correctness["status"] == "pass"
        and runtime_safety["status"] == "pass"
    )
    verification["accept_reason"] = "compile, correctness, and runtime safety passed" if verification["accepted"] else None
    verification["run_stdout"] = truncate(run_result["stdout"])
    verification["run_stderr"] = truncate(run_result["stderr"])


def parse_run_metrics(text):
    metrics = {}
    metric_line = re.search(r"SGPO_METRIC\s+(.+)", text)
    if metric_line:
        for key, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", metric_line.group(1)):
            metrics[key] = parse_metric_value(value)

    perf_match = re.search(
        r"(?:SGPO\s+GEMM|My\s+gemm)\s+Performance=\s*([0-9.eE+-]+)\s*GFlop/s,\s*Time=\s*([0-9.eE+-]+)\s*msec",
        text,
        re.IGNORECASE,
    )
    if perf_match:
        metrics.setdefault("gflops", float(perf_match.group(1)))
        metrics.setdefault("latency_ms", float(perf_match.group(2)))

    cublas_match = re.search(
        r"(?:cuBLAS|CuBlas|Cublas)\s+Performance=\s*([0-9.eE+-]+)\s*GFlop/s,\s*Time=\s*([0-9.eE+-]+)\s*msec",
        text,
        re.IGNORECASE,
    )
    if cublas_match:
        cublas_gflops = float(cublas_match.group(1))
        metrics.setdefault("cublas_gflops", cublas_gflops)
        metrics.setdefault("cublas_latency_ms", float(cublas_match.group(2)))

    if "gflops" in metrics and metrics.get("cublas_gflops"):
        metrics["relative_to_cublas"] = metrics["gflops"] / metrics["cublas_gflops"]

    if "correctness" not in metrics:
        if "Result= PASS" in text:
            metrics["correctness"] = "pass"
        elif "Result= FAIL" in text:
            metrics["correctness"] = "fail"
    return metrics


def parse_metric_value(value):
    lowered = value.lower()
    if lowered in {"pass", "fail", "unknown"}:
        return lowered
    try:
        return float(value)
    except ValueError:
        return value


def parse_cuda_error(text):
    match = re.search(r"CUDA:\s*([^\r\n]+)", text)
    if match:
        return match.group(1).strip()
    lowered = text.lower()
    for phrase in ("misaligned address", "illegal memory access", "out of memory", "invalid configuration argument"):
        if phrase in lowered:
            return phrase
    return None


def contains_error(cuda_error, phrase):
    return bool(cuda_error and phrase in cuda_error.lower())


def first_error_line(text):
    for line in text.splitlines():
        lowered = line.lower()
        if (
            "error" in lowered
            or "result= fail" in lowered
            or "cuda:" in lowered
            or "cublas status" in lowered
            or "segmentation fault" in lowered
        ):
            return line.strip()
    return None


def cuda_arch_flag(ir):
    hardware = ir.get("hardware", {})
    major = hardware.get("compute_capability_major")
    minor = hardware.get("compute_capability_minor")
    if major is not None and minor is not None:
        return f"sm_{major}{minor}"
    compute_capability = hardware.get("compute_capability")
    if compute_capability:
        return "sm_" + str(compute_capability).replace(".", "")
    return "sm_80"


def truncate(text, limit=4000):
    if not text:
        return None
    return text if len(text) <= limit else text[-limit:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile, run, and verify generated CUDA GEMM code.")
    parser.add_argument("--ir", default=str(DEFAULT_IR))
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--build-dir", default=str(DEFAULT_BUILD_DIR))
    parser.add_argument("--output-ir", default=str(DEFAULT_OUTPUT_IR))
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--vcvars64", default=str(DEFAULT_VCVARS64))
    parser.add_argument("--build-platform", choices=["windows", "linux"], default="windows")
    args = parser.parse_args()
    ir = load_json(Path(args.ir))
    source_dir = Path(args.source_dir).resolve()
    build_dir = Path(args.build_dir).resolve()
    timeout_seconds = args.timeout_seconds
    verified_ir = verify_build_and_run(
        ir=ir,
        source_dir=source_dir,
        build_dir=build_dir,
        timeout_seconds=timeout_seconds,
        vcvars64_path=Path(args.vcvars64),
        build_platform=args.build_platform,
    )
    output_ir = Path(args.output_ir)
    save_json(output_ir, verified_ir)
    print(json.dumps({
        "output_ir": str(output_ir),
        "compile": verified_ir.get("verification", {}).get("compile", {}),
        "correctness": verified_ir.get("verification", {}).get("correctness", {}),
        "runtime_safety": verified_ir.get("verification", {}).get("runtime_safety", {}),
        "performance": verified_ir.get("performance", {}),
    }, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
