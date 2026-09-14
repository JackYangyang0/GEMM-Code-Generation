from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from SGPO.utils.common_utils import load_config, load_json, save_json
from SGPO.utils.execution import execution_slot, limited
from SGPO.verification.build_run_verifier import (
    DEFAULT_BENCHMARK_RUNS,
    DEFAULT_BENCHMARK_WARMUP_RUNS,
    aggregate_benchmark_results,
    ensure_text_result,
    normalize_build_platform,
    parse_run_metrics,
    result_text,
    truncate,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IR = ROOT / "data" / "IRs" / "ir_patch" / "optir.extracted.json"
DEFAULT_SOURCE_DIR = ROOT / "gemm_code" / "cpu_skeleton"
DEFAULT_BUILD_DIR = ROOT / "gemm_code" / "cpu_skeleton" / "build"
DEFAULT_OUTPUT_IR = ROOT / "data" / "IRs" / "ir_patch" / "optir.cpu.verified.json"
DEFAULT_CONFIG = ROOT / "conf.yaml"


def verify_cpu_build_and_run(
    ir: dict[str, Any],
    source_dir: Path = DEFAULT_SOURCE_DIR,
    build_dir: Path = DEFAULT_BUILD_DIR,
    executable_name: str = "gemm_cpu.exe",
    timeout_seconds: int = 60,
    openblas_root: Path | None = None,
    build_platform: str = "windows",
    benchmark_runs: int = DEFAULT_BENCHMARK_RUNS,
    benchmark_warmup_runs: int = DEFAULT_BENCHMARK_WARMUP_RUNS,
) -> dict[str, Any]:
    next_ir = copy.deepcopy(ir)
    build_platform = normalize_build_platform(build_platform)
    next_ir.setdefault("target", {}).update({"backend": "cpu", "language": "c", "device": "cpu"})
    if build_platform == "linux" and executable_name == "gemm_cpu.exe":
        executable_name = "gemm_cpu"
    build_dir.mkdir(parents=True, exist_ok=True)
    exe_path = build_dir / executable_name

    compile_result = compile_cpu_gemm(source_dir, exe_path, timeout_seconds, next_ir, build_platform)
    update_cpu_compile_result(next_ir, compile_result)
    if compile_result["status"] != "pass":
        mark_cpu_unrun(next_ir, "compile failed")
        return next_ir

    with execution_slot("gpu_verification"):
        run_result = run_cpu_gemm_benchmark(exe_path, next_ir, timeout_seconds, benchmark_runs, benchmark_warmup_runs)
        update_cpu_run_result(next_ir, run_result)
        baseline_result = run_cpu_optimized_baseline(next_ir, build_dir, timeout_seconds, openblas_root, build_platform)
        update_cpu_optimized_baseline_result(next_ir, baseline_result)
    return next_ir


@limited("compile")
def compile_cpu_gemm(
    source_dir: Path,
    exe_path: Path,
    timeout_seconds: int,
    ir: dict[str, Any] | None = None,
    build_platform: str = "windows",
) -> dict[str, Any]:
    build_platform = normalize_build_platform(build_platform)
    compiler_policy = (ir or {}).get("cpu_compiler", {}) or {}
    if build_platform == "windows" and shutil.which("cl"):
        msvc_flags = msvc_compile_flags(compiler_policy)
        command = [
            "cl",
            "/TC",
            *msvc_flags,
            "main.c",
            "cpu_kernel.c",
            f"/Fe:{exe_path}",
        ]
        result = run_command(command, cwd=source_dir, timeout_seconds=timeout_seconds)
        result["build_platform"] = build_platform
        return result

    for compiler in ("gcc", "clang"):
        path = shutil.which(compiler)
        if path:
            flags = gcc_like_compile_flags(compiler_policy)
            command = [
                path,
                "-std=c11",
                *flags,
                "main.c",
                "cpu_kernel.c",
                "-lm",
                "-o",
                str(exe_path),
            ]
            result = run_command(command, cwd=source_dir, timeout_seconds=timeout_seconds)
            result["build_platform"] = build_platform
            return result

    return {
        "status": "fail",
        "command": None,
        "stdout": "",
        "stderr": "No C compiler found. Install MSVC Build Tools, gcc, or clang.",
        "returncode": None,
        "build_platform": build_platform,
    }


def gcc_like_compile_flags(policy: dict[str, Any]) -> list[str]:
    level = str(policy.get("optimization_level") or "O3").upper()
    flags = [f"-{level}" if level.startswith("O") else "-O3"]
    if policy.get("native_arch") is True or policy.get("vector_isa") == "native":
        flags.append("-march=native")
    isa = str(policy.get("vector_isa") or "").lower()
    if isa in {"avx2", "avx2_fma"}:
        flags.extend(["-mavx2", "-mfma"])
    if isa in {"avx512", "avx512_fma"}:
        flags.extend(["-mavx512f", "-mfma"])
    if policy.get("openmp") is True:
        flags.append("-fopenmp")
    if policy.get("fast_math") is True:
        flags.append("-ffast-math")
    if policy.get("unroll_loops") is True:
        flags.append("-funroll-loops")
    return flags


def msvc_compile_flags(policy: dict[str, Any]) -> list[str]:
    level = str(policy.get("optimization_level") or "O2").upper()
    flags = [f"/{level}" if level.startswith("O") else "/O2"]
    isa = str(policy.get("vector_isa") or "").lower()
    if isa in {"avx2", "avx2_fma", "native"}:
        flags.append("/arch:AVX2")
    if isa in {"avx512", "avx512_fma"}:
        flags.append("/arch:AVX512")
    if policy.get("openmp") is True:
        flags.append("/openmp")
    if policy.get("fast_math") is True:
        flags.append("/fp:fast")
    return flags


@limited("gpu_verification")
def run_cpu_gemm(exe_path: Path, ir: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    problem = ir.get("problem", {})
    command = [
        str(exe_path),
        str(problem.get("M", 512)),
        str(problem.get("K", 512)),
        str(problem.get("N", 512)),
    ]
    result = run_command(command, cwd=exe_path.parent, timeout_seconds=timeout_seconds)
    result["metrics"] = parse_run_metrics(result_text(result))
    return result


@limited("gpu_verification")
def run_cpu_gemm_benchmark(
    exe_path: Path,
    ir: dict[str, Any],
    timeout_seconds: int,
    benchmark_runs: int = DEFAULT_BENCHMARK_RUNS,
    warmup_runs: int = DEFAULT_BENCHMARK_WARMUP_RUNS,
) -> dict[str, Any]:
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
        result["metrics"] = parse_run_metrics(result_text(result))
        result["cuda_error"] = None
        warmup_results.append(result)
        if result.get("status") != "pass":
            break
    if not warmup_results or all(item.get("status") == "pass" for item in warmup_results):
        for _ in range(measured_count):
            result = run_command(command, cwd=exe_path.parent, timeout_seconds=timeout_seconds)
            result["metrics"] = parse_run_metrics(result_text(result))
            result["cuda_error"] = None
            measured_results.append(result)
    return aggregate_benchmark_results(command, warmup_results, measured_results)


@limited("gpu_verification")
def run_cpu_optimized_baseline(
    ir: dict[str, Any],
    build_dir: Path,
    timeout_seconds: int,
    openblas_root: Path | None = None,
    build_platform: str = "windows",
) -> dict[str, Any]:
    build_platform = normalize_build_platform(build_platform)
    root = resolve_openblas_root(openblas_root)
    if root is None:
        return {
            "status": "skipped",
            "command": None,
            "stdout": "",
            "stderr": "OpenBLAS root not found. Set cpu_baseline.openblas_root or OPENBLAS_ROOT.",
            "returncode": None,
            "metrics": {},
        }

    include_dir = root / "include"
    lib_dir = root / "lib"
    bin_dir = root / "bin"
    header = include_dir / "cblas.h"
    if not header.exists():
        return {
            "status": "fail",
            "command": None,
            "stdout": "",
            "stderr": f"OpenBLAS cblas.h not found: {header}",
            "returncode": None,
            "metrics": {},
        }

    source_path = build_dir / "openblas_sgemm_baseline.c"
    exe_path = build_dir / ("openblas_sgemm_baseline" if build_platform == "linux" else "openblas_sgemm_baseline.exe")
    source_path.write_text(OPENBLAS_SGEMM_BASELINE_C, encoding="utf-8")

    compile_result = compile_openblas_baseline(source_path, exe_path, include_dir, lib_dir, timeout_seconds, build_platform)
    if compile_result.get("status") != "pass":
        compile_result["metrics"] = {}
        return compile_result

    problem = ir.get("problem", {})
    env = os.environ.copy()
    if bin_dir.exists():
        env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    result = run_command(
        [
            str(exe_path),
            str(problem.get("M", 512)),
            str(problem.get("K", 512)),
            str(problem.get("N", 512)),
        ],
        cwd=build_dir,
        timeout_seconds=timeout_seconds,
        env=env,
    )
    try:
        result["metrics"] = json.loads(result_text(result).strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        result["metrics"] = {}
    result["compile"] = compile_result
    return result


def resolve_openblas_root(openblas_root: Path | None = None) -> Path | None:
    candidates = []
    if openblas_root is not None:
        candidates.append(Path(openblas_root))
    env_root = os.environ.get("OPENBLAS_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    config_path = Path(DEFAULT_CONFIG)
    if config_path.exists():
        config = load_config(config_path)
        configured = (config.get("cpu_baseline", {}) or {}).get("openblas_root")
        if configured:
            candidates.append(Path(configured))
    candidates.append(Path(r"C:\OpenBLAS"))
    candidates.append(Path(r"C:\OpenBLAS\bin").parent)
    candidates.append(Path("/usr/local"))
    candidates.append(Path("/usr"))
    candidates.append(Path("/opt/OpenBLAS"))
    for candidate in candidates:
        if (candidate / "include" / "cblas.h").exists() and ((candidate / "lib").exists() or (candidate / "bin").exists()):
            return candidate
    return None


@limited("compile")
def compile_openblas_baseline(
    source_path: Path,
    exe_path: Path,
    include_dir: Path,
    lib_dir: Path,
    timeout_seconds: int,
    build_platform: str = "windows",
) -> dict[str, Any]:
    build_platform = normalize_build_platform(build_platform)
    if build_platform == "windows" and shutil.which("cl") and (lib_dir / "libopenblas.lib").exists():
        command = [
            "cl",
            "/O2",
            "/TC",
            f"/I{include_dir}",
            str(source_path.name),
            f"/Fe:{exe_path}",
            "/link",
            f"/LIBPATH:{lib_dir}",
            "libopenblas.lib",
        ]
        result = run_command(command, cwd=source_path.parent, timeout_seconds=timeout_seconds)
        result["build_platform"] = build_platform
        return result

    for compiler in ("gcc", "clang"):
        path = shutil.which(compiler)
        if path:
            command = [
                path,
                "-O3",
                "-std=c11",
                f"-I{include_dir}",
                str(source_path.name),
                f"-L{lib_dir}",
                "-lopenblas",
                "-lm",
                "-o",
                str(exe_path),
            ]
            result = run_command(command, cwd=source_path.parent, timeout_seconds=timeout_seconds)
            result["build_platform"] = build_platform
            return result

    return {
        "status": "fail",
        "command": None,
        "stdout": "",
        "stderr": "No C compiler found for OpenBLAS baseline.",
        "returncode": None,
        "build_platform": build_platform,
    }


def run_command(
    command: list[str] | None,
    cwd: Path,
    timeout_seconds: int,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    if command is None:
        return {"status": "fail", "command": None, "stdout": "", "stderr": "missing command", "returncode": None}
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        return ensure_text_result({
            "status": "pass" if proc.returncode == 0 else "fail",
            "command": command,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        })
    except subprocess.TimeoutExpired as exc:
        return ensure_text_result({
            "status": "fail",
            "command": command,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or f"timeout after {timeout_seconds}s",
            "returncode": None,
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


def update_cpu_compile_result(ir: dict[str, Any], compile_result: dict[str, Any]) -> None:
    compile_result = ensure_text_result(compile_result)
    verification = ir.setdefault("verification", {})
    compile_node = verification.setdefault("compile", {})
    output_text = result_text(compile_result)
    compile_node["status"] = compile_result["status"]
    compile_node["backend"] = "cpu"
    compile_node["build_platform"] = compile_result.get("build_platform")
    compile_node["compiler"] = compile_result["command"][0] if compile_result.get("command") else None
    compile_node["error_message"] = None if compile_result["status"] == "pass" else truncate(output_text)
    compile_node["command"] = compile_result["command"]
    compile_node["returncode"] = compile_result["returncode"]
    compile_node["stdout"] = truncate(compile_result.get("stdout"))
    compile_node["stderr"] = truncate(compile_result.get("stderr"))


def mark_cpu_unrun(ir: dict[str, Any], reason: str) -> None:
    verification = ir.setdefault("verification", {})
    correctness = verification.setdefault("correctness", {})
    runtime_safety = verification.setdefault("runtime_safety", {})
    correctness["status"] = "not_run"
    correctness["error_message"] = reason
    runtime_safety["status"] = "not_run"
    runtime_safety["host_error"] = None

    performance = ir.setdefault("performance", {})
    performance["latency_ms"] = None
    performance["gflops"] = None


def update_cpu_run_result(ir: dict[str, Any], run_result: dict[str, Any]) -> None:
    run_result = ensure_text_result(run_result)
    text = result_text(run_result)
    metrics = run_result.get("metrics", {})
    verification = ir.setdefault("verification", {})
    correctness = verification.setdefault("correctness", {})
    runtime_safety = verification.setdefault("runtime_safety", {})

    runtime_safety["status"] = "pass" if run_result["status"] == "pass" else "fail"
    runtime_safety["backend"] = "cpu"
    runtime_safety["host_error"] = None if run_result["status"] == "pass" else format_host_error(run_result, text)
    runtime_safety["returncode"] = run_result["returncode"]

    correctness_status = metrics.get("correctness")
    if correctness_status in {"pass", "fail"}:
        correctness["status"] = correctness_status
    elif "Result= PASS" in text:
        correctness["status"] = "pass"
    elif "Result= FAIL" in text:
        correctness["status"] = "fail"
    else:
        correctness["status"] = "unknown" if run_result["status"] == "pass" else "not_run"
    correctness["reference"] = "cpu_scalar_reference"
    correctness["max_abs_error"] = metrics.get("max_abs_error")
    correctness["error_message"] = None if correctness["status"] == "pass" else format_host_error(run_result, text)

    performance = ir.setdefault("performance", {})
    performance["latency_ms"] = metrics.get("latency_ms")
    performance["gflops"] = metrics.get("gflops")
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
    verification["accept_reason"] = "CPU compile, correctness, and runtime safety passed" if verification["accepted"] else None
    verification["run_stdout"] = truncate(run_result["stdout"])
    verification["run_stderr"] = truncate(run_result["stderr"])
    verification["benchmark"] = run_result.get("benchmark")


def update_cpu_optimized_baseline_result(ir: dict[str, Any], baseline_result: dict[str, Any]) -> None:
    verification = ir.setdefault("verification", {})
    baseline = verification.setdefault("optimized_cpu_baseline", {})
    performance = ir.setdefault("performance", {})
    metrics = baseline_result.get("metrics", {}) or {}

    if baseline_result.get("status") != "pass" or not metrics:
        baseline.update({
            "status": baseline_result.get("status", "fail"),
            "library": "openblas",
            "error_message": truncate(baseline_result.get("stderr") or baseline_result.get("stdout")),
            "returncode": baseline_result.get("returncode"),
            "compile": baseline_result.get("compile"),
        })
        performance["cpu_blas_latency_ms"] = None
        performance["cpu_blas_gflops"] = None
        performance["relative_to_cpu_blas"] = None
        return

    baseline.update({
        "status": "pass",
        "library": metrics.get("library", "openblas"),
        "backend_info": metrics.get("backend_info"),
        "latency_ms": metrics.get("latency_ms"),
        "gflops": metrics.get("gflops"),
        "command": baseline_result.get("command"),
    })
    performance["cpu_blas_latency_ms"] = metrics.get("latency_ms")
    performance["cpu_blas_gflops"] = metrics.get("gflops")
    own_gflops = performance.get("gflops")
    baseline_gflops = metrics.get("gflops")
    performance["relative_to_cpu_blas"] = (
        own_gflops / baseline_gflops
        if isinstance(own_gflops, (int, float)) and isinstance(baseline_gflops, (int, float)) and baseline_gflops > 0
        else None
    )


def first_host_error_line(text: str) -> str | None:
    for line in text.splitlines():
        lowered = line.lower()
        if "error" in lowered or "result= fail" in lowered or "failed" in lowered:
            return line.strip()
    return None


def format_host_error(run_result: dict[str, Any], text: str) -> str | None:
    line = first_host_error_line(text)
    if line:
        return line
    returncode = run_result.get("returncode")
    windows_status = {
        3221225477: "access violation",
        3221225501: "illegal instruction",
        -1073741819: "access violation",
        -1073741795: "illegal instruction",
    }
    if returncode in windows_status:
        return f"host runtime error: {windows_status[returncode]} (returncode={returncode})"
    if returncode not in (None, 0):
        return f"host runtime error: returncode={returncode}"
    return None


OPENBLAS_SGEMM_BASELINE_C = r"""
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include <cblas.h>

static double now_ms(void) {
#ifdef _WIN32
    static LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    if (frequency.QuadPart == 0) {
        QueryPerformanceFrequency(&frequency);
    }
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)frequency.QuadPart;
#elif defined(CLOCK_MONOTONIC)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
#else
    return (double)clock() * 1000.0 / (double)CLOCKS_PER_SEC;
#endif
}

int main(int argc, char **argv) {
    if (argc != 4) {
        return 2;
    }
    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = atoi(argv[3]);
    size_t bytes_A = sizeof(float) * (size_t)M * (size_t)K;
    size_t bytes_B = sizeof(float) * (size_t)K * (size_t)N;
    size_t bytes_C = sizeof(float) * (size_t)M * (size_t)N;
    float *A = (float *)malloc(bytes_A);
    float *B = (float *)malloc(bytes_B);
    float *C = (float *)malloc(bytes_C);
    if (!A || !B || !C) {
        free(A);
        free(B);
        free(C);
        return 3;
    }
    for (int i = 0; i < M * K; ++i) {
        A[i] = (float)(i % 17) / 17.0f;
    }
    for (int i = 0; i < K * N; ++i) {
        B[i] = (float)(i % 13) / 13.0f;
    }
    for (int i = 0; i < M * N; ++i) {
        C[i] = 0.0f;
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    int iters = 10;
    const char *bench_iters_env = getenv("SGPO_CPU_BENCH_ITERS");
    if (bench_iters_env && atoi(bench_iters_env) > 0) {
        iters = atoi(bench_iters_env);
    }
    double start = now_ms();
    for (int iter = 0; iter < iters; ++iter) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    }
    double latency_ms = (now_ms() - start) / (double)iters;
    if (latency_ms <= 0.0) {
        latency_ms = 1.0e-6;
    }
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = (flops * 1.0e-9) / (latency_ms / 1000.0);
    printf("{\"library\":\"openblas\",\"backend_info\":{\"api\":\"cblas_sgemm\"},\"latency_ms\":%.9f,\"gflops\":%.9f}\n", latency_ms, gflops);
    free(A);
    free(B);
    free(C);
    return 0;
}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile, run, and verify generated CPU C GEMM code.")
    parser.add_argument("--ir", default=str(DEFAULT_IR))
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--build-dir", default=str(DEFAULT_BUILD_DIR))
    parser.add_argument("--output-ir", default=str(DEFAULT_OUTPUT_IR))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--build-platform", choices=["windows", "linux"], default=None)
    parser.add_argument("--benchmark-runs", type=int, default=None)
    parser.add_argument("--benchmark-warmup-runs", type=int, default=None)
    args = parser.parse_args()

    config = load_config(Path(args.config))
    timeout_seconds = int(config.get("verification", {}).get("timeout_seconds", 60))
    build_config = config.get("build", {}) or {}
    build_platform = args.build_platform or (config.get("build", {}) or {}).get("platform", "windows")
    benchmark_runs = args.benchmark_runs if args.benchmark_runs is not None else build_config.get("benchmark_runs", DEFAULT_BENCHMARK_RUNS)
    benchmark_warmup_runs = (
        args.benchmark_warmup_runs
        if args.benchmark_warmup_runs is not None
        else build_config.get("benchmark_warmup_runs", DEFAULT_BENCHMARK_WARMUP_RUNS)
    )
    ir = load_json(Path(args.ir))
    verified_ir = verify_cpu_build_and_run(
        ir=ir,
        source_dir=Path(args.source_dir),
        build_dir=Path(args.build_dir),
        timeout_seconds=timeout_seconds,
        build_platform=build_platform,
        benchmark_runs=int(benchmark_runs),
        benchmark_warmup_runs=int(benchmark_warmup_runs),
    )
    save_json(Path(args.output_ir), verified_ir)
    print(json.dumps({
        "output_ir": args.output_ir,
        "compile": verified_ir.get("verification", {}).get("compile", {}),
        "correctness": verified_ir.get("verification", {}).get("correctness", {}),
        "runtime_safety": verified_ir.get("verification", {}).get("runtime_safety", {}),
        "performance": verified_ir.get("performance", {}),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
