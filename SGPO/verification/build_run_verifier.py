from __future__ import annotations

import argparse
import copy
import json
import os
import re
import signal
import shutil
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

from SGPO.utils.common_utils import load_json, save_json
from SGPO.utils.execution import execution_slot, limited


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IR = ROOT / "data" / "IRs" / "ir_patch" / "optir.extracted.json"
DEFAULT_SOURCE_DIR = ROOT / "gemm_code" / "skeleton"
DEFAULT_OUTPUT_IR = ROOT / "data" / "IRs" / "ir_patch" / "optir.verified.json"
DEFAULT_BUILD_DIR = ROOT / "gemm_code" / "skeleton" / "build"
DEFAULT_VCVARS64 = Path(r"D:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat")
DEFAULT_BENCHMARK_RUNS = 5
DEFAULT_BENCHMARK_WARMUP_RUNS = 2


def ensure_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def ensure_text_result(result: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    normalized["stdout"] = ensure_text(normalized.get("stdout"))
    normalized["stderr"] = ensure_text(normalized.get("stderr"))
    return normalized


def result_text(result: dict[str, Any]) -> str:
    return "\n".join(part for part in [ensure_text(result.get("stdout")), ensure_text(result.get("stderr"))] if part)


def verify_build_and_run(
    ir,
    source_dir=DEFAULT_SOURCE_DIR,
    build_dir=DEFAULT_BUILD_DIR,
    executable_name="gemm.exe",
    timeout_seconds=60,
    vcvars64_path=DEFAULT_VCVARS64,
    build_platform="windows",
    benchmark_runs=DEFAULT_BENCHMARK_RUNS,
    benchmark_warmup_runs=DEFAULT_BENCHMARK_WARMUP_RUNS,
):
    next_ir = copy.deepcopy(ir)
    build_platform = normalize_build_platform(build_platform)
    if build_platform == "linux" and executable_name == "gemm.exe":
        executable_name = "gemm"
    build_dir.mkdir(parents=True, exist_ok=True)
    if ir.get("compiler", {}).get("resource_feedback", {}).get("enabled") is True:
        from SGPO.verification.gpu_resource_sweep import run_resource_sweep
        return run_resource_sweep(ir, source_dir, build_dir, executable_name, timeout_seconds,
                                  vcvars64_path, build_platform, benchmark_runs, benchmark_warmup_runs)
    exe_path = build_dir / executable_name

    compile_result = compile_gemm(ir, source_dir, exe_path, timeout_seconds, vcvars64_path, build_platform)
    update_compile_result(next_ir, compile_result)
    if compile_result["status"] != "pass":
        mark_unrun(next_ir, "compile failed")
        return next_ir

    with execution_slot("gpu_verification"):
        run_result = run_gemm_benchmark(exe_path, ir, timeout_seconds, benchmark_runs, benchmark_warmup_runs)
        update_run_result(next_ir, run_result)
        if ir.get("pipeline", {}).get("async_copy") is True and next_ir.get("verification", {}).get("accepted"):
            safety = run_async_safety_checks(exe_path, ir, timeout_seconds)
            next_ir["verification"]["async_sanitizer"] = safety
            if safety["status"] == "fail":
                next_ir["verification"]["runtime_safety"].update(status="fail", cuda_error=safety["error_message"])
                next_ir["verification"]["accepted"] = False
                next_ir["verification"]["accept_reason"] = None
    return next_ir


@limited("gpu_verification")
def run_async_safety_checks(exe_path, ir, timeout_seconds):
    sanitizer = shutil.which("compute-sanitizer")
    if not sanitizer:
        return {"status": "not_run", "reason": "compute-sanitizer not available", "checks": []}
    problem = ir.get("problem", {})
    args = [str(problem.get(key, 512)) for key in ("M", "N", "K")]
    checks = []
    for name in ("racecheck", "synccheck"):
        command = [sanitizer, "--tool", name, "--error-exitcode", "86", str(exe_path), *args]
        result = run_command(command, cwd=Path(exe_path).parent, timeout_seconds=timeout_seconds)
        checks.append({"tool": name, **result})
        if result.get("timeout"):
            return {"status": "inconclusive", "checks": checks, "reason": f"{name} timed out; no synchronization defect proven"}
        if result["status"] != "pass":
            return {"status": "fail", "checks": checks, "error_message": f"compute-sanitizer {name} failed: {truncate(result_text(result))}"}
    return {"status": "pass", "checks": checks, "scope": "executed inputs only; not a proof for all shapes"}


@limited("compile")
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
    clear_previous_compile_outputs(exe_path)
    compile_started_at_ns = time.time_ns()
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
    command.append("--ptxas-options=-v")
    register_limit = ir.get("compiler", {}).get("max_register_count")
    if register_limit is not None:
        if type(register_limit) is not int or not 16 <= register_limit <= 255:
            return {"status": "fail", "command": command, "returncode": None,
                    "stdout": "", "stderr": "max_register_count must be an integer between 16 and 255"}
        command.append(f"--maxrregcount={register_limit}")
    if build_platform == "windows" and needs_msvc_environment() and vcvars64_path and Path(vcvars64_path).exists():
        result = run_command_with_vcvars(command, Path(vcvars64_path), cwd=source_dir, timeout_seconds=timeout_seconds)
    else:
        result = run_command(command, cwd=source_dir, timeout_seconds=timeout_seconds)
    result = recover_compile_timeout_if_executable_exists(result, exe_path, compile_started_at_ns)
    result["build_platform"] = build_platform
    result["requested_arch"] = requested_arch
    result["resolved_arch"] = arch
    return result


def clear_previous_compile_outputs(exe_path):
    for path in compile_output_paths(exe_path):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def compile_output_paths(exe_path):
    exe_path = Path(exe_path)
    paths = [exe_path]
    if exe_path.suffix.lower() == ".exe":
        paths.extend([exe_path.with_suffix(".lib"), exe_path.with_suffix(".exp")])
    return paths


def recover_compile_timeout_if_executable_exists(result, exe_path, compile_started_at_ns):
    result = ensure_text_result(result)
    if not result.get("timeout") or not Path(exe_path).exists():
        return result
    try:
        exe_mtime_ns = Path(exe_path).stat().st_mtime_ns
    except OSError:
        return result
    if exe_mtime_ns + 1_000_000_000 < compile_started_at_ns:
        return result
    recovered = dict(result)
    recovered["status"] = "pass"
    recovered["timeout_recovered_by_executable"] = True
    recovered["stderr"] = "\n".join(
        part for part in [
            ensure_text(recovered.get("stderr")),
            "compile command timed out after producing a fresh executable; accepting compile gate and deferring failures to run verification.",
        ]
        if part
    )
    return recovered


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
    result = run_command(display_command, cwd=cwd, timeout_seconds=timeout_seconds)
    result["vcvars64_path"] = str(vcvars64_path)
    return result


@limited("gpu_verification")
def run_gemm(exe_path, ir, timeout_seconds):
    problem = ir.get("problem", {})
    command = [
        str(exe_path),
        str(problem.get("M", 512)),
        str(problem.get("K", 512)),
        str(problem.get("N", 512)),
    ]
    result = run_command(command, cwd=exe_path.parent, timeout_seconds=timeout_seconds)
    text = result_text(result)
    result["metrics"] = parse_run_metrics(text)
    result["cuda_error"] = parse_cuda_error(text)
    return result


@limited("gpu_verification")
def run_gemm_benchmark(exe_path, ir, timeout_seconds, benchmark_runs=DEFAULT_BENCHMARK_RUNS, warmup_runs=DEFAULT_BENCHMARK_WARMUP_RUNS):
    problem = ir.get("problem", {})
    command = [
        str(exe_path),
        str(problem.get("M", 512)),
        str(problem.get("K", 512)),
        str(problem.get("N", 512)),
    ]
    measured_count = max(1, int(benchmark_runs or DEFAULT_BENCHMARK_RUNS))
    warmup_count = max(0, int(warmup_runs or 0))
    warmup_results = []
    measured_results = []
    for _ in range(warmup_count):
        result = run_command(command, cwd=exe_path.parent, timeout_seconds=timeout_seconds)
        text = result_text(result)
        result["metrics"] = parse_run_metrics(text)
        result["cuda_error"] = parse_cuda_error(text)
        warmup_results.append(result)
        if result.get("status") != "pass" or result.get("cuda_error"):
            break
    if not warmup_results or all(item.get("status") == "pass" and item.get("cuda_error") is None for item in warmup_results):
        for _ in range(measured_count):
            result = run_command(command, cwd=exe_path.parent, timeout_seconds=timeout_seconds)
            text = result_text(result)
            result["metrics"] = parse_run_metrics(text)
            result["cuda_error"] = parse_cuda_error(text)
            measured_results.append(result)
    return aggregate_benchmark_results(command, warmup_results, measured_results)


def aggregate_benchmark_results(command, warmup_results, measured_results):
    all_results = list(warmup_results or []) + list(measured_results or [])
    counted_results = list(measured_results or [])
    all_text = "\n".join(part for result in all_results for part in [result_text(result)] if part)
    first_result = all_results[0] if all_results else {}
    last_result = all_results[-1] if all_results else {}
    cuda_error = next((item.get("cuda_error") for item in all_results if item.get("cuda_error")), None)
    host_failed = any(item.get("status") != "pass" for item in all_results)
    metrics = aggregate_metrics([item.get("metrics", {}) or {} for item in counted_results])
    correctness_values = [item.get("metrics", {}).get("correctness") for item in counted_results]
    if correctness_values and all(value == "pass" for value in correctness_values):
        metrics["correctness"] = "pass"
    elif any(value == "fail" for value in correctness_values) or "Result= FAIL" in all_text:
        metrics["correctness"] = "fail"
    elif "Result= PASS" in all_text and not host_failed and cuda_error is None:
        metrics["correctness"] = "pass"
    else:
        metrics["correctness"] = "unknown" if not host_failed and cuda_error is None else "not_run"
    metrics["benchmark_runs"] = len(counted_results)
    metrics["warmup_runs"] = len(warmup_results or [])
    metrics["benchmark_successful_runs"] = len(
        [
            item for item in counted_results
            if item.get("status") == "pass" and item.get("cuda_error") is None
        ]
    )
    return {
        "status": "pass" if all_results and not host_failed and cuda_error is None else "fail",
        "command": command,
        "stdout": all_text,
        "stderr": "",
        "returncode": last_result.get("returncode", first_result.get("returncode")),
        "metrics": metrics,
        "cuda_error": cuda_error,
        "benchmark": {
            "warmup_runs": len(warmup_results or []),
            "runs": len(counted_results),
            "run_metrics": [item.get("metrics", {}) for item in counted_results],
        },
    }


def aggregate_metrics(metrics_list):
    aggregate = {}
    numeric_fields = [
        "latency_ms",
        "gflops",
        "cublas_latency_ms",
        "cublas_gflops",
        "max_abs_error",
        "max_rel_error",
    ]
    for field in numeric_fields:
        values = numeric_metric_values(metrics_list, field)
        if not values:
            continue
        if field in {"max_abs_error", "max_rel_error"}:
            aggregate[field] = max(values)
            continue
        aggregate[field] = statistics.mean(values)
        aggregate[f"{field}_mean"] = statistics.mean(values)
        aggregate[f"{field}_median"] = statistics.median(values)
        aggregate[f"{field}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        aggregate[f"{field}_min"] = min(values)
        aggregate[f"{field}_max"] = max(values)
        if field == "gflops":
            aggregate["gflops_best"] = max(values)
        if field == "latency_ms":
            aggregate["latency_ms_best"] = min(values)
    if aggregate.get("gflops_mean") is not None and aggregate.get("cublas_gflops_mean"):
        aggregate["relative_to_cublas"] = aggregate["gflops_mean"] / aggregate["cublas_gflops_mean"]
    return aggregate


def numeric_metric_values(metrics_list, field):
    values = []
    for metrics in metrics_list:
        value = metrics.get(field)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def run_command(command, cwd, timeout_seconds):
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    try:
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            creationflags=creationflags,
            start_new_session=os.name != "nt",
        )
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
        return ensure_text_result({
            "status": "pass" if proc.returncode == 0 else "fail",
            "command": command,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": proc.returncode,
        })
    except subprocess.TimeoutExpired as exc:
        terminate_process_tree(proc)
        try:
            stdout, stderr = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
        stdout = ensure_text(stdout) or ensure_text(exc.stdout)
        stderr = ensure_text(stderr) or ensure_text(exc.stderr)
        timeout_message = f"timeout after {timeout_seconds}s; process tree terminated"
        return ensure_text_result({
            "status": "fail",
            "command": command,
            "stdout": stdout,
            "stderr": "\n".join(part for part in [stderr, timeout_message] if part),
            "returncode": proc.returncode,
            "timeout": True,
        })
    except OSError as exc:
        return {
            "status": "fail",
            "command": command,
            "stdout": "",
            "stderr": str(exc),
            "returncode": None,
        }


def terminate_process_tree(proc):
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        proc.kill()


def update_compile_result(ir, compile_result):
    compile_result = ensure_text_result(compile_result)
    verification = ir.setdefault("verification", {})
    compile_node = verification.setdefault("compile", {})
    output_text = result_text(compile_result)
    compile_node["status"] = compile_result["status"]
    compile_node["build_platform"] = compile_result.get("build_platform")
    compile_node["requested_arch"] = compile_result.get("requested_arch")
    compile_node["resolved_arch"] = compile_result.get("resolved_arch")
    compile_node["error_message"] = None if compile_result["status"] == "pass" else truncate(output_text)
    compile_node["command"] = compile_result["command"]
    compile_node["returncode"] = compile_result["returncode"]
    compile_node["vcvars64_path"] = compile_result.get("vcvars64_path")
    compile_node["timeout_recovered_by_executable"] = compile_result.get("timeout_recovered_by_executable", False)
    compile_node["stdout"] = truncate(compile_result.get("stdout"))
    compile_node["stderr"] = truncate(compile_result.get("stderr"))
    registers = [int(value) for value in re.findall(r"Used\s+(\d+)\s+registers", output_text)]
    spills = [int(value) for value in re.findall(r"(\d+) bytes spill (?:stores|loads)", output_text)]
    if registers:
        ir.setdefault("resource", {}).setdefault("register", {})["actual_per_thread"] = max(registers)
    else:
        ir.get("resource", {}).get("register", {}).pop("actual_per_thread", None)
    compile_node["ptxas_resources"] = {
        "register_counts": registers, "spill_bytes_sum": sum(spills),
        "scope": "all compiled kernels; max register count is conservative",
    }


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
    run_result = ensure_text_result(run_result)
    text = result_text(run_result)
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
    for key in [
        "latency_ms_mean",
        "latency_ms_median",
        "latency_ms_std",
        "latency_ms_best",
        "latency_ms_min",
        "latency_ms_max",
        "gflops_mean",
        "gflops_median",
        "gflops_std",
        "gflops_best",
        "gflops_min",
        "gflops_max",
        "cublas_latency_ms_mean",
        "cublas_latency_ms_median",
        "cublas_latency_ms_std",
        "cublas_gflops_mean",
        "cublas_gflops_median",
        "cublas_gflops_std",
        "benchmark_runs",
        "warmup_runs",
        "benchmark_successful_runs",
    ]:
        if key in metrics:
            performance[key] = metrics[key]

    verification["accepted"] = (
        verification.get("compile", {}).get("status") == "pass"
        and correctness["status"] == "pass"
        and runtime_safety["status"] == "pass"
    )
    verification["accept_reason"] = "compile, correctness, and runtime safety passed" if verification["accepted"] else None
    verification["run_stdout"] = truncate(run_result["stdout"])
    verification["run_stderr"] = truncate(run_result["stderr"])
    verification["benchmark"] = run_result.get("benchmark")


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
    text = ensure_text(text)
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
    parser.add_argument("--benchmark-runs", type=int, default=DEFAULT_BENCHMARK_RUNS)
    parser.add_argument("--benchmark-warmup-runs", type=int, default=DEFAULT_BENCHMARK_WARMUP_RUNS)
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
        benchmark_runs=args.benchmark_runs,
        benchmark_warmup_runs=args.benchmark_warmup_runs,
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
