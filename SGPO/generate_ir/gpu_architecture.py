from __future__ import annotations

import math
from typing import Any


def build_gpu_architecture_profile(hardware: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize CUDA execution limits without keying decisions on product names."""
    hardware = hardware or {}
    gpu = hardware.get("gpu") if isinstance(hardware.get("gpu"), dict) else {}

    major = first_int(hardware, gpu, keys=("compute_capability_major",))
    minor = first_nonnegative_int(hardware, gpu, keys=("compute_capability_minor",))
    if major is None or minor is None:
        parsed_major, parsed_minor = parse_compute_capability(
            hardware.get("compute_capability") or gpu.get("compute_capability")
        )
        major = major if major is not None else parsed_major
        minor = minor if minor is not None else parsed_minor

    family = architecture_family(major, minor)
    defaults = architecture_defaults(family)
    warp_size = first_int(hardware, gpu, keys=("warp_size",)) or 32
    max_threads_per_sm = (
        first_int(hardware, gpu, keys=("max_threads_per_multiprocessor", "max_threads_per_sm"))
        or defaults["max_threads_per_sm"]
    )
    max_blocks_per_sm = (
        first_int(hardware, gpu, keys=("max_blocks_per_multiprocessor", "max_blocks_per_sm"))
        or defaults["max_blocks_per_sm"]
    )
    shared_memory_per_sm = (
        first_int(hardware, gpu, keys=("max_shared_memory_per_multiprocessor_bytes", "shared_memory_per_sm_bytes"))
        or defaults["shared_memory_per_sm_bytes"]
    )
    registers_per_sm = (
        first_int(hardware, gpu, keys=("registers_per_multiprocessor", "registers_per_sm"))
        or 65536
    )
    sm_count = first_int(
        hardware,
        gpu,
        keys=("sm_count", "multiprocessor_count", "multi_processor_count"),
    )

    return {
        "architecture_family": family,
        "compute_capability_major": major,
        "compute_capability_minor": minor,
        "sm_count": sm_count,
        "warp_size": warp_size,
        "max_threads_per_sm": max_threads_per_sm,
        "max_warps_per_sm": max_threads_per_sm // warp_size if max_threads_per_sm else None,
        "max_blocks_per_sm": max_blocks_per_sm,
        "registers_per_sm": registers_per_sm,
        "shared_memory_per_sm_bytes": shared_memory_per_sm,
        "l2_cache_bytes": first_int(hardware, gpu, keys=("l2_cache_bytes",)),
        "theoretical_memory_bandwidth_gbps": first_float(
            hardware,
            gpu,
            keys=("theoretical_memory_bandwidth_gbps",),
        ),
    }


def estimate_tiling_execution(
    problem: dict[str, Any] | None,
    profile: dict[str, Any],
    *,
    bm: int,
    bn: int,
    bk: int,
    threads_per_block: int,
    warps_per_block: int,
    shared_memory_bytes: int,
    estimated_registers_per_thread: int,
    element_bytes: int,
    estimated_accumulators_per_thread: int | None = None,
) -> dict[str, Any]:
    by_threads = positive_floor_div(profile.get("max_threads_per_sm"), threads_per_block)
    by_blocks = positive_int(profile.get("max_blocks_per_sm"))
    by_shared = positive_floor_div(profile.get("shared_memory_per_sm_bytes"), shared_memory_bytes)
    registers_per_block = estimated_registers_per_thread * threads_per_block
    by_registers = positive_floor_div(profile.get("registers_per_sm"), registers_per_block)
    limits = [value for value in (by_threads, by_blocks, by_shared, by_registers) if value is not None]
    resident_ctas = min(limits) if limits else None

    max_warps = positive_int(profile.get("max_warps_per_sm"))
    active_warps = resident_ctas * warps_per_block if resident_ctas is not None else None
    occupancy = min(1.0, active_warps / max_warps) if active_warps is not None and max_warps else None

    problem = problem or {}
    m = positive_int(problem.get("M") or problem.get("m"))
    n = positive_int(problem.get("N") or problem.get("n"))
    sm_count = positive_int(profile.get("sm_count"))
    cta_count = ceil_div(m, bm) * ceil_div(n, bn) if m and n else None
    sm_coverage = min(1.0, cta_count / sm_count) if cta_count is not None and sm_count else None
    slots_per_wave = sm_count * resident_ctas if sm_count and resident_ctas else None
    cta_waves = math.ceil(cta_count / slots_per_wave) if cta_count is not None and slots_per_wave else None
    wave_utilization = (
        cta_count / (cta_waves * slots_per_wave)
        if cta_count is not None and cta_waves and slots_per_wave
        else None
    )

    bytes_per_k_tile = element_bytes * bk * (bm + bn)
    flops_per_k_tile = 2 * bm * bn * bk
    arithmetic_intensity = flops_per_k_tile / bytes_per_k_tile if bytes_per_k_tile else None
    architecture_score = score_tiling(
        occupancy,
        sm_coverage,
        wave_utilization,
        arithmetic_intensity,
        accumulators_per_thread=estimated_accumulators_per_thread,
        architecture_family=str(profile.get("architecture_family") or "cuda_generic"),
        underfilled_grid=bool(cta_count is not None and sm_count and cta_count < sm_count),
    )

    return {
        "resident_ctas_per_sm": resident_ctas,
        "resident_ctas_by_threads": by_threads,
        "resident_ctas_by_blocks": by_blocks,
        "resident_ctas_by_shared_memory": by_shared,
        "resident_ctas_by_registers": by_registers,
        "active_warps_per_sm": active_warps,
        "estimated_occupancy": occupancy,
        "cta_count": cta_count,
        "cta_waves": cta_waves,
        "sm_coverage": sm_coverage,
        "last_wave_utilization": wave_utilization,
        "arithmetic_intensity_flop_per_byte": arithmetic_intensity,
        "estimated_accumulators_per_thread": estimated_accumulators_per_thread,
        "architecture_score": architecture_score,
    }


def score_tiling(
    occupancy: float | None,
    sm_coverage: float | None,
    wave_utilization: float | None,
    arithmetic_intensity: float | None,
    *,
    accumulators_per_thread: int | None = None,
    architecture_family: str = "cuda_generic",
    underfilled_grid: bool = False,
) -> float:
    # Ranking prior only. Target-device measurements remain authoritative. Newer
    # compute-heavy GPUs need enough reuse and independent FFMA work; occupancy
    # alone otherwise systematically favors undersized tiles.
    compute_heavy = architecture_family in {"hopper_cc90", "blackwell_cc10plus"}
    if underfilled_grid:
        occupancy_weight, coverage_weight = 25.0, 40.0
        intensity_weight, ilp_weight = 15.0, 10.0
    else:
        occupancy_weight = 20.0 if compute_heavy else 25.0
        coverage_weight = 10.0 if compute_heavy else 15.0
        intensity_weight = 35.0 if compute_heavy else 30.0
        ilp_weight = 25.0 if compute_heavy else 20.0
    score = 0.0
    score += occupancy_weight * min((occupancy or 0.0) / 0.5, 1.0)
    score += coverage_weight * (sm_coverage or 0.0)
    score += 10.0 * (wave_utilization or 0.0)
    score += intensity_weight * min((arithmetic_intensity or 0.0) / 24.0, 1.0)
    score += ilp_weight * min(float(accumulators_per_thread or 0) / 16.0, 1.0)
    return round(score, 6)


def architecture_family(major: int | None, minor: int | None) -> str:
    if major == 8 and minor == 0:
        return "ampere_cc80_datacenter"
    if major == 8 and minor == 9:
        return "ada_cc89"
    if major == 8:
        return "ampere_cc8x"
    if major == 9:
        return "hopper_cc90"
    if major is not None and major >= 10:
        return "blackwell_cc10plus"
    return "cuda_generic"


def architecture_defaults(family: str) -> dict[str, int]:
    if family == "ampere_cc80_datacenter":
        return {"max_threads_per_sm": 2048, "max_blocks_per_sm": 32, "shared_memory_per_sm_bytes": 164 * 1024}
    if family == "ampere_cc8x":
        return {"max_threads_per_sm": 1536, "max_blocks_per_sm": 16, "shared_memory_per_sm_bytes": 100 * 1024}
    if family == "ada_cc89":
        return {"max_threads_per_sm": 1536, "max_blocks_per_sm": 24, "shared_memory_per_sm_bytes": 100 * 1024}
    if family == "blackwell_cc10plus":
        # Device-query values override these conservative fallbacks whenever the
        # target GPU is available to the extraction process.
        return {"max_threads_per_sm": 1536, "max_blocks_per_sm": 24, "shared_memory_per_sm_bytes": 128 * 1024}
    return {"max_threads_per_sm": 1536, "max_blocks_per_sm": 16, "shared_memory_per_sm_bytes": 100 * 1024}


def parse_compute_capability(value: Any) -> tuple[int | None, int | None]:
    try:
        major, minor = str(value).split(".", 1)
        return int(major), int(minor)
    except (TypeError, ValueError):
        return None, None


def first_int(*sources: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for source in sources:
        for key in keys:
            value = positive_int(source.get(key))
            if value is not None:
                return value
    return None


def first_float(*sources: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for source in sources:
        for key in keys:
            try:
                value = float(source.get(key))
                if value > 0:
                    return value
            except (TypeError, ValueError):
                continue
    return None


def first_nonnegative_int(*sources: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for source in sources:
        for key in keys:
            try:
                value = int(source.get(key))
                if value >= 0:
                    return value
            except (TypeError, ValueError):
                continue
    return None


def positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else None
    except (TypeError, ValueError):
        return None


def positive_floor_div(total: Any, unit: Any) -> int | None:
    total_value = positive_int(total)
    unit_value = positive_int(unit)
    if total_value is None or unit_value is None:
        return None
    return max(0, total_value // unit_value)


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor
