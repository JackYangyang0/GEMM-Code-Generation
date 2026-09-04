"""IR extraction for SGPO.

The initial template in ``data/optir.json`` remains unchanged. This module
builds a new OptIR instance by extracting GEMM fields from a text description
and probing local hardware.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from SGPO.utils.common_utils import load_json, save_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / "data" / "IRs" / "optir.json"
DEFAULT_OUTPUT = ROOT / "data" / "IRs" / "ir_patch" / "optir.extracted.json"


def run_command(args: list[str], timeout: int = 10) -> str | None:
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def extract_problem_description(description: str) -> dict[str, Any]:
    """Extract only fields that belong to the OptIR ``problem`` section."""
    text = description.strip()
    facts: dict[str, Any] = {
        "problem": {},
        "target": {},
        "extraction_confidence": {},
    }

    for key in ("M", "N", "K"):
        match = re.search(rf"\b{key}\s*[:=]\s*(\d+)", text)
        if match:
            facts["problem"][key] = int(match.group(1))
            facts["extraction_confidence"][key] = "explicit"

    size_tuple = re.search(r"Matrix\s+Size\s*:\s*\(\s*M\s*:\s*(\d+)\s+N\s*:\s*(\d+)\s+K\s*:\s*(\d+)\s*\)", text, re.I)
    if size_tuple:
        for key, value in zip(("M", "N", "K"), size_tuple.groups()):
            facts["problem"][key] = int(value)
            facts["extraction_confidence"][key] = "explicit"

    dtype_match = re.search(r"\b(fp16|float16|half|fp32|float32|float|tf32|bf16|double|fp64)\b", text, re.I)
    if dtype_match:
        dtype = normalize_dtype(dtype_match.group(1))
        facts["problem"].update(
            {
                "dtype_A": dtype,
                "dtype_B": dtype,
                "dtype_C": dtype,
                "accum_dtype": "fp32" if dtype in {"fp16", "bf16", "tf32"} else dtype,
            }
        )
        facts["extraction_confidence"]["dtype"] = "explicit"

    if re.search(r"row[-_ ]major", text, re.I):
        facts["problem"].update({"layout_A": "row_major", "layout_B": "row_major", "layout_C": "row_major"})
        facts["extraction_confidence"]["layout"] = "explicit"
    elif re.search(r"column[-_ ]major|col[-_ ]major", text, re.I):
        facts["problem"].update({"layout_A": "col_major", "layout_B": "col_major", "layout_C": "col_major"})
        facts["extraction_confidence"]["layout"] = "explicit"

    if re.search(r"\bNN\b", text):
        facts["problem"].update({"trans_A": False, "trans_B": False})
        facts["extraction_confidence"]["transpose"] = "explicit"
    elif re.search(r"\bNT\b", text):
        facts["problem"].update({"trans_A": False, "trans_B": True})
        facts["extraction_confidence"]["transpose"] = "explicit"
    elif re.search(r"\bTN\b", text):
        facts["problem"].update({"trans_A": True, "trans_B": False})
        facts["extraction_confidence"]["transpose"] = "explicit"
    elif re.search(r"\bTT\b", text):
        facts["problem"].update({"trans_A": True, "trans_B": True})
        facts["extraction_confidence"]["transpose"] = "explicit"

    if re.search(r"C\s*=\s*A\s*\*\s*B", text, re.I):
        facts["problem"]["operation"] = "C = A * B"
        facts["extraction_confidence"]["operation"] = "explicit"

    backend = extract_target_backend(text)
    if backend:
        facts["target"].update(backend)
        facts["extraction_confidence"]["target_backend"] = "explicit"

    enrich_problem_size_class(facts["problem"], facts["extraction_confidence"])
    return facts


def extract_target_backend(text: str) -> dict[str, Any] | None:
    if re.search(r"\b(CUDA|GPU|NVIDIA)\b", text, re.I):
        return {"backend": "cuda", "language": "cuda_cpp", "device": "gpu"}
    if re.search(r"\b(CPU|host|ordinary\s+C|plain\s+C|C\s+language)\b", text, re.I):
        return {"backend": "cpu", "language": "c", "device": "cpu"}
    return None


def enrich_problem_size_class(problem: dict[str, Any], confidence: dict[str, Any]) -> None:
    m = problem.get("M")
    n = problem.get("N")
    k = problem.get("K")
    if not all(isinstance(value, int) and value > 0 for value in (m, n, k)):
        return

    max_dim = max(m, n, k)
    min_dim = min(m, n, k)
    square_like = max_dim / min_dim <= 2
    if square_like and min_dim >= 1024:
        problem["size_class"] = "large_square"
    elif max_dim >= 1024:
        problem["size_class"] = "large_rectangular"
    elif max_dim >= 256:
        problem["size_class"] = "medium"
    else:
        problem["size_class"] = "small"

    if k >= 1024:
        problem["k_intensity"] = "high"
    elif k >= 256:
        problem["k_intensity"] = "medium"
    else:
        problem["k_intensity"] = "low"

    problem["static_divisibility"] = {
        "M_mod_4": m % 4,
        "N_mod_4": n % 4,
        "K_mod_4": k % 4,
        "M_mod_8": m % 8,
        "N_mod_8": n % 8,
        "K_mod_8": k % 8,
        "M_mod_16": m % 16,
        "N_mod_16": n % 16,
        "K_mod_16": k % 16,
    }
    problem["alignment_class"] = "vector4_friendly" if m % 4 == 0 and n % 4 == 0 and k % 4 == 0 else "general"
    confidence["size_class"] = "derived"


def normalize_dtype(raw: str) -> str:
    dtype = raw.lower()
    return {
        "float": "fp32",
        "float32": "fp32",
        "half": "fp16",
        "float16": "fp16",
        "double": "fp64",
    }.get(dtype, dtype)


def query_nvidia_smi() -> dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {}
    query = "name,memory.total,compute_cap,clocks.max.graphics,clocks.max.memory"
    output = run_command(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return {}

    values = [part.strip() for part in output.splitlines()[0].split(",")]
    keys = [
        "gpu_name",
        "global_memory_mib",
        "compute_capability",
        "max_graphics_clock_mhz",
        "max_memory_clock_mhz",
    ]
    result = dict(zip(keys, values))
    for key in ("global_memory_mib", "max_graphics_clock_mhz", "max_memory_clock_mhz"):
        result[key] = to_int(result.get(key))
    return result


def query_device_query() -> dict[str, Any]:
    exe = find_device_query()
    if exe is None:
        return {}
    output = run_command([str(exe)])
    if not output:
        return {}

    result: dict[str, Any] = {"device_query_result": "PASS" if "Result = PASS" in output else "unknown"}
    patterns = {
        "multiprocessor_count": r"\((\d+)\)\s+Multiprocessors",
        "cuda_cores": r":\s+(\d+)\s+CUDA Cores",
        "gpu_max_clock_mhz": r"GPU Max Clock rate:\s+(\d+)\s+MHz",
        "memory_clock_mhz": r"Memory Clock rate:\s+(\d+)\s+Mhz",
        "memory_bus_width_bits": r"Memory Bus Width:\s+(\d+)-bit",
        "l2_cache_bytes": r"L2 Cache Size:\s+(\d+)\s+bytes",
        "registers_per_block": r"registers available per block:\s+(\d+)",
        "warp_size": r"Warp size:\s+(\d+)",
        "max_threads_per_multiprocessor": r"Maximum number of threads per multiprocessor:\s+(\d+)",
        "max_threads_per_block": r"Maximum number of threads per block:\s+(\d+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, output, re.I)
        if match:
            result[key] = int(match.group(1))

    block_match = re.search(r"Max dimension size of a thread block \(x,y,z\):\s+\((\d+),\s+(\d+),\s+(\d+)\)", output)
    if block_match:
        result["max_threads_dim"] = [int(x) for x in block_match.groups()]

    grid_match = re.search(r"Max dimension size of a grid size\s+\(x,y,z\):\s+\((\d+),\s+(\d+),\s+(\d+)\)", output)
    if grid_match:
        result["max_grid_size"] = [int(x) for x in grid_match.groups()]

    return result


def find_device_query() -> Path | None:
    path_exe = shutil.which("deviceQuery")
    if path_exe:
        return Path(path_exe)

    env_paths = [
        value
        for key, value in os.environ.items()
        if key == "CUDA_PATH" or key.startswith("CUDA_PATH_V")
    ]
    for cuda_root in env_paths:
        exe = Path(cuda_root) / "extras" / "demo_suite" / "deviceQuery.exe"
        if exe.exists():
            return exe

    cuda_base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    matches = list(cuda_base.glob(r"v*\extras\demo_suite\deviceQuery.exe"))
    if not matches:
        return None

    return max(matches, key=lambda p: parse_cuda_version(p))


def parse_cuda_version(path: Path) -> tuple[int, int]:
    for part in path.parts:
        match = re.fullmatch(r"v(\d+)(?:\.(\d+))?", part, re.I)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2) or 0)
            return major, minor
    return 0, 0


def query_cpu_memory() -> dict[str, Any]:
    cpu_cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-CimInstance Win32_Processor | Select-Object -First 1 Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | ConvertTo-Json -Compress",
    ]
    mem_cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-CimInstance Win32_ComputerSystem | Select-Object -First 1 TotalPhysicalMemory | ConvertTo-Json -Compress",
    ]
    cpu = parse_json_output(run_command(cpu_cmd)) or {}
    memory = parse_json_output(run_command(mem_cmd)) or {}
    cache = normalize_cpu_cache(query_cpu_cache(), cpu.get("NumberOfCores"))
    total_memory_bytes = memory.get("TotalPhysicalMemory")
    return {
        "cpu_name": cpu.get("Name"),
        "cpu_physical_cores": cpu.get("NumberOfCores"),
        "cpu_logical_processors": cpu.get("NumberOfLogicalProcessors"),
        "cpu_max_clock_mhz": cpu.get("MaxClockSpeed"),
        "system_memory_bytes": total_memory_bytes,
        "system_memory_gib": round(total_memory_bytes / (1024 ** 3), 2) if total_memory_bytes else None,
        "cpu_cache": cache,
        "cpu_isa": detect_cpu_isa(cpu.get("Name")),
    }


def detect_cpu_isa(cpu_name: Any = None) -> dict[str, Any]:
    override = os.environ.get("SGPO_CPU_ISA", "").lower()
    if override:
        tokens = {token.strip() for token in re.split(r"[,;\s]+", override) if token.strip()}
        return {
            "avx2": "avx2" in tokens,
            "fma": "fma" in tokens or "avx2" in tokens,
            "avx512f": "avx512f" in tokens or "avx512" in tokens,
            "source": "SGPO_CPU_ISA",
        }

    linux_flags = read_linux_cpu_flags()
    if linux_flags:
        return {
            "avx2": "avx2" in linux_flags,
            "fma": "fma" in linux_flags,
            "avx512f": "avx512f" in linux_flags,
            "source": "/proc/cpuinfo",
        }

    name = str(cpu_name or "").lower()
    is_modern_x86 = any(token in name for token in ("intel", "core", "xeon", "amd", "ryzen", "epyc"))
    avx512_name_hint = any(token in name for token in ("xeon", "scalable", "epyc 9", "epyc 7", "threadripper pro"))
    if any(token in name for token in ("12th", "13th", "14th", "i7-14700", "i9-14900", "i5-14600")):
        avx512_name_hint = False

    return {
        "avx2": is_modern_x86,
        "fma": is_modern_x86,
        "avx512f": avx512_name_hint,
        "source": "cpu_name_heuristic",
    }


def read_linux_cpu_flags() -> set[str]:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        return set()
    try:
        text = cpuinfo.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return set()
    match = re.search(r"flags\s*:\s*(.+)", text)
    if not match:
        return set()
    return set(match.group(1).split())


def query_cpu_cache() -> dict[str, Any]:
    if os.name == "nt":
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            (
                "Get-CimInstance Win32_CacheMemory | "
                "Select-Object Level,InstalledSize,BlockSize,NumberOfBlocks | "
                "ConvertTo-Json -Compress"
            ),
        ]
        cache_rows = parse_json_output(run_command(cmd)) or []
        if isinstance(cache_rows, dict):
            cache_rows = [cache_rows]
        parsed = parse_windows_cache_rows(cache_rows)
        if parsed:
            result = default_cpu_cache()
            result.update(parsed)
            result["source"] = "Win32_CacheMemory"
            return result
    return default_cpu_cache()


def parse_windows_cache_rows(cache_rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    levels = {}
    for row in cache_rows:
        level = to_int(row.get("Level"))
        installed_kib = to_int(row.get("InstalledSize"))
        if level is None or installed_kib is None or installed_kib <= 0:
            continue
        levels[level] = max(levels.get(level, 0), installed_kib * 1024)

    if levels.get(3) is not None:
        result["l1_data_cache_bytes"] = levels[3]
    if levels.get(4) is not None:
        result["l2_cache_bytes"] = levels[4]
    if levels.get(5) is not None:
        result["l3_cache_bytes"] = levels[5]

    block_sizes = [to_int(row.get("BlockSize")) for row in cache_rows]
    block_sizes = [value for value in block_sizes if value and value > 0]
    result["cache_line_bytes"] = max(block_sizes) if block_sizes else 64
    return result


def default_cpu_cache() -> dict[str, Any]:
    return {
        "l1_data_cache_bytes": 32 * 1024,
        "l2_cache_bytes": 1024 * 1024,
        "l3_cache_bytes": None,
        "cache_line_bytes": 64,
        "source": "default_conservative",
    }


def normalize_cpu_cache(cache: dict[str, Any], physical_cores: Any) -> dict[str, Any]:
    result = copy.deepcopy(cache)
    cores = to_int(physical_cores) or 1
    l1 = to_int(result.get("l1_data_cache_bytes"))
    l2 = to_int(result.get("l2_cache_bytes"))
    line = to_int(result.get("cache_line_bytes"))

    if l1 and l1 > 128 * 1024 and cores > 1:
        result["l1_data_cache_total_bytes"] = l1
        result["l1_data_cache_bytes"] = max(16 * 1024, l1 // cores)
        result["l1_data_cache_scope"] = "per_core_estimated"
    else:
        result["l1_data_cache_scope"] = "per_core_or_default"

    if l2 and l2 > 4 * 1024 * 1024 and cores > 1:
        result["l2_cache_total_bytes"] = l2
        result["l2_cache_bytes"] = max(256 * 1024, l2 // cores)
        result["l2_cache_scope"] = "per_core_estimated"
    else:
        result["l2_cache_scope"] = "per_core_or_shared"

    if not line or line < 16 or line > 1024:
        result["raw_cache_line_bytes"] = line
        result["cache_line_bytes"] = 64
        result["cache_line_source"] = "default_due_to_unreliable_probe"
    return result


def probe_local_hardware(template_hardware: dict[str, Any]) -> dict[str, Any]:
    smi = query_nvidia_smi()
    device = query_device_query()
    cpu_memory = query_cpu_memory()

    hardware = copy.deepcopy(template_hardware)
    cc = smi.get("compute_capability")
    cc_major, cc_minor = parse_compute_capability(cc)
    global_memory_mib = smi.get("global_memory_mib")
    global_memory_bytes = global_memory_mib * 1024 * 1024 if global_memory_mib is not None else None
    memory_clock_mhz = device.get("memory_clock_mhz") or smi.get("max_memory_clock_mhz")
    memory_bus_width_bits = device.get("memory_bus_width_bits")

    hardware.update(
        {
            "gpu_name": smi.get("gpu_name") or hardware.get("gpu_name"),
            "compute_capability": cc,
            "compute_capability_major": cc_major,
            "compute_capability_minor": cc_minor,
            "global_memory_mib": global_memory_mib,
            "global_memory_bytes": global_memory_bytes,
            "warp_size": device.get("warp_size") or hardware.get("warp_size"),
            "max_threads_per_block": device.get("max_threads_per_block") or hardware.get("max_threads_per_block"),
            "max_threads_per_multiprocessor": device.get("max_threads_per_multiprocessor"),
            "max_threads_dim": device.get("max_threads_dim"),
            "max_grid_size": device.get("max_grid_size"),
            "max_shared_memory_per_block_bytes": hardware.get("max_shared_memory_per_block_bytes"),
            "registers_per_block": device.get("registers_per_block"),
            "multiprocessor_count": device.get("multiprocessor_count"),
            "cuda_cores": device.get("cuda_cores"),
            "gpu_max_clock_mhz": device.get("gpu_max_clock_mhz") or smi.get("max_graphics_clock_mhz"),
            "memory_clock_mhz": memory_clock_mhz,
            "memory_bus_width_bits": memory_bus_width_bits,
            "l2_cache_bytes": device.get("l2_cache_bytes"),
            "theoretical_memory_bandwidth_gbps": estimate_memory_bandwidth_gbps(
                memory_clock_mhz,
                memory_bus_width_bits,
            ),
            "memory_alignment_bytes": hardware.get("memory_alignment_bytes"),
            "cpu_name": cpu_memory.get("cpu_name"),
            "cpu_physical_cores": cpu_memory.get("cpu_physical_cores"),
            "cpu_logical_processors": cpu_memory.get("cpu_logical_processors"),
            "cpu_max_clock_mhz": cpu_memory.get("cpu_max_clock_mhz"),
            "system_memory_bytes": cpu_memory.get("system_memory_bytes"),
            "system_memory_gib": cpu_memory.get("system_memory_gib"),
            "cpu_cache": cpu_memory.get("cpu_cache"),
            "cpu_isa": cpu_memory.get("cpu_isa"),
        }
    )

    hardware["supports_float4"] = True
    hardware["supports_cp_async"] = bool(cc_major and cc_major >= 8)
    hardware["supports_tensor_core"] = bool(cc_major and cc_major >= 7)
    hardware["supports_tf32_tensor_core"] = bool(cc_major and cc_major >= 8)
    return hardware


def parse_compute_capability(value: Any) -> tuple[int | None, int | None]:
    if value is None:
        return None, None
    parts = str(value).split(".")
    if len(parts) != 2:
        return None, None
    return to_int(parts[0]), to_int(parts[1])


def estimate_memory_bandwidth_gbps(memory_clock_mhz: Any, bus_width_bits: Any) -> float | None:
    clock = to_float(memory_clock_mhz)
    bus_width = to_float(bus_width_bits)
    if clock is None or bus_width is None:
        return None
    return round(clock * 2 * bus_width / 8 / 1000, 2)


def build_extracted_ir(template, description):
    ir = copy.deepcopy(template)
    extracted = extract_problem_description(description)
    template_problem_keys = set(template.get("problem", {}))

    for key, value in extracted["problem"].items():
        if key in template_problem_keys:
            ir["problem"][key] = value
        else:
            ir.setdefault("problem", {})[key] = value

    if extracted.get("target"):
        ir.setdefault("target", {}).update(extracted["target"])

    ir["hardware"] = probe_local_hardware(ir["hardware"])
    extracted_fields = sorted(key for key in extracted["problem"] if key in template_problem_keys)
    defaulted_fields = sorted(template_problem_keys - set(extracted_fields))
    ir["extraction"] = {
        "stage": "IR Extraction",
        "status": "completed",
        "template_path": str(DEFAULT_TEMPLATE),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "problem_description": True,
            "local_hardware": True,
        },
        "problem_extraction": {
            "extracted_fields": extracted_fields,
            "defaulted_fields": defaulted_fields,
            "confidence": extracted.get("extraction_confidence", {}),
        },
    }
    return ir


def parse_json_output(output: str | None) -> Any:
    if not output:
        return None
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return None


def to_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def to_float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def read_description(args: argparse.Namespace) -> str:
    if args.description:
        return args.description
    if args.description_file:
        return Path(args.description_file).read_text(encoding="utf-8")
    raise SystemExit("Provide --description or --description-file.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract SGPO OptIR from a GEMM description and local hardware.")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--description")
    parser.add_argument("--description-file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template = load_json(args.template)
    description = read_description(args)
    ir = build_extracted_ir(template, description)
    save_json(args.output, ir)


if __name__ == "__main__":
    main()
