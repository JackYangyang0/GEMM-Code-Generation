from __future__ import annotations

import re
from typing import Any

from SGPO.generate_ir.gpu_architecture import (
    build_gpu_architecture_profile,
    estimate_tiling_execution,
)
from SGPO.generate_ir.stage_controller import choose_warp_iteration_extents


BLOCK_TILE_RE = re.compile(r"^Tiling\.BlockTile\.(\d+)x(\d+)x(\d+)$")
WARP_TILE_RE = re.compile(r"^Tiling\.WarpTile\.(\d+)x(\d+)$")
THREAD_TILE_RE = re.compile(r"^Tiling\.ThreadTile\.(\d+)x(\d+)$")


def build_joint_tiling_candidates(
    strategy_library: dict[str, Any],
    optir: dict[str, Any],
) -> list[dict[str, Any]]:
    """Enumerate complete, statically legal Block/Warp/Thread tiling tuples."""
    strategy_ids = collect_strategy_ids(strategy_library)
    block_tiles = parse_strategy_tiles(strategy_ids, BLOCK_TILE_RE)
    warp_tiles = parse_strategy_tiles(strategy_ids, WARP_TILE_RE)
    thread_tiles = parse_strategy_tiles(strategy_ids, THREAD_TILE_RE)

    warp_size = nested_int(optir, "hardware.warp_size", 32)
    max_threads = nested_int(optir, "hardware.max_threads_per_block", 1024)
    max_shared = first_nested_int(
        optir,
        [
            "hardware.max_shared_memory_per_block_bytes",
            "hardware.gpu.max_shared_memory_per_block_bytes",
        ],
        49152,
    )
    element_bytes = dtype_bytes(optir)
    execution_profile = build_gpu_architecture_profile(optir.get("hardware", {}))

    candidates = []
    for block_id, (bm, bn, bk) in block_tiles:
        for warp_id, (wm, wn) in warp_tiles:
            if bm % wm != 0 or bn % wn != 0:
                continue
            warps_per_block = (bm // wm) * (bn // wn)
            threads_per_block = warps_per_block * warp_size
            if threads_per_block <= 0 or threads_per_block > max_threads:
                continue
            shared_memory_bytes = element_bytes * bk * (bm + bn)
            if shared_memory_bytes > max_shared:
                continue
            for thread_id, (tm, tn) in thread_tiles:
                warp_iteration = choose_warp_iteration_extents(wm, wn, tm, tn, warp_size)
                if warp_iteration is None:
                    continue
                wmiter, wniter = warp_iteration
                accumulator_count = (wm // wmiter * tm) * (wn // wniter * tn)
                estimated_registers = accumulator_count + tm + tn + 16
                execution = estimate_tiling_execution(
                    optir.get("problem", {}),
                    execution_profile,
                    bm=bm,
                    bn=bn,
                    bk=bk,
                    threads_per_block=threads_per_block,
                    warps_per_block=warps_per_block,
                    shared_memory_bytes=shared_memory_bytes,
                    estimated_registers_per_thread=estimated_registers,
                    element_bytes=element_bytes,
                    estimated_accumulators_per_thread=accumulator_count,
                )
                if execution["resident_ctas_per_sm"] == 0:
                    continue
                candidate_id = f"tile_{bm}x{bn}x{bk}__{wm}x{wn}__{tm}x{tn}"
                candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "block_strategy_id": block_id,
                        "warp_strategy_id": warp_id,
                        "thread_strategy_id": thread_id,
                        "BM": bm,
                        "BN": bn,
                        "BK": bk,
                        "WM": wm,
                        "WN": wn,
                        "TM": tm,
                        "TN": tn,
                        "WMITER": wmiter,
                        "WNITER": wniter,
                        "warps_per_block": warps_per_block,
                        "threads_per_block": threads_per_block,
                        "shared_memory_bytes": shared_memory_bytes,
                        "buffer_feasibility": {
                            "single_bytes": shared_memory_bytes,
                            "double_bytes": 2 * shared_memory_bytes,
                            "double_fits_default_limit": 2 * shared_memory_bytes <= max_shared,
                            "default_limit_bytes": max_shared,
                            "basis": "unpadded tile estimate; recheck actual layout and launch before use",
                        },
                        "estimated_accumulators_per_thread": accumulator_count,
                        "estimated_registers_per_thread": estimated_registers,
                        "shape_class": shape_class(bm, bn),
                        "resource_class": resource_class(threads_per_block, accumulator_count),
                        "architecture_family": execution_profile["architecture_family"],
                        **execution,
                    }
                )
    return sorted(candidates, key=joint_candidate_sort_key)


def make_diverse_tiling_pool(candidates: list[dict[str, Any]], max_count: int) -> list[dict[str, Any]]:
    """Keep strong and complementary complete tuples before asking the LLM.

    A BlockTile must not be represented by a single arbitrary descendant.  That
    used to hide strong WarpTile/ThreadTile combinations when the representative
    happened to be chosen mainly for having about 128 threads.  Keep one
    architecture-ranked tuple and one shape-balanced tuple per block, then add a
    globally strong alternate when the prompt budget permits.
    """
    if max_count <= 0 or len(candidates) <= max_count:
        return list(candidates)

    by_block: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for item in candidates:
        by_block.setdefault((item["BM"], item["BN"], item["BK"]), []).append(item)

    primary_by_block = {
        block_key: min(items, key=joint_candidate_sort_key)
        for block_key, items in by_block.items()
    }
    complementary_by_block = {
        block_key: min(items, key=balanced_block_representative_score)
        for block_key, items in by_block.items()
    }

    result: list[dict[str, Any]] = []
    used_ids: set[str] = set()

    def add(item: dict[str, Any]) -> None:
        candidate_id = item["candidate_id"]
        if candidate_id not in used_ids and len(result) < max_count:
            result.append(item)
            used_ids.add(candidate_id)

    # Preserve block-level coverage first.  Blocks with the strongest feasible
    # descendants get priority only when max_count is smaller than block count.
    for item in sorted(primary_by_block.values(), key=joint_candidate_sort_key):
        add(item)

    # Reserve one slot for a globally strong alternate that may differ only in
    # the Warp/Thread mapping from a block's primary representative.
    global_alternate = next(
        (item for item in sorted(candidates, key=joint_candidate_sort_key)
         if item["candidate_id"] not in used_ids),
        None,
    )
    if global_alternate is not None:
        add(global_alternate)

    complementary_groups: dict[str, list[dict[str, Any]]] = {}
    for item in complementary_by_block.values():
        complementary_groups.setdefault(item["shape_class"], []).append(item)
    for items in complementary_groups.values():
        items.sort(key=joint_candidate_sort_key)
    ordered_shapes = sorted(
        complementary_groups,
        key=lambda shape: joint_candidate_sort_key(complementary_groups[shape][0]),
    )
    while len(result) < max_count and any(complementary_groups.values()):
        for shape in ordered_shapes:
            items = complementary_groups[shape]
            if items and len(result) < max_count:
                add(items.pop(0))

    # Use any remaining budget for resource/shape diversity without replacing
    # the representatives selected above.
    for item in sorted(candidates, key=joint_candidate_sort_key):
        add(item)
    return result


def select_diverse_joint_tilings(
    candidates: list[dict[str, Any]],
    selected_candidate_ids: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    """Validate LLM choices and fill omissions without duplicating BlockTile tuples."""
    by_id = {item["candidate_id"]: item for item in candidates}
    selected = []
    used_blocks = set()
    used_groups = set()

    def add(candidate: dict[str, Any]) -> None:
        block_key = (candidate["BM"], candidate["BN"], candidate["BK"])
        if block_key in used_blocks or len(selected) >= top_k:
            return
        selected.append(candidate)
        used_blocks.add(block_key)
        used_groups.add((candidate["shape_class"], candidate["resource_class"]))

    for candidate_id in selected_candidate_ids:
        candidate = by_id.get(candidate_id)
        if candidate is not None:
            add(candidate)
    for candidate in sorted(
        candidates,
        key=lambda item: (
            (item["shape_class"], item["resource_class"]) in used_groups,
            joint_candidate_sort_key(item),
        ),
    ):
        add(candidate)
    return selected


def tiling_plan_for_ir(candidate: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "candidate_id", "block_strategy_id", "warp_strategy_id", "thread_strategy_id",
        "BM", "BN", "BK", "WM", "WN", "TM", "TN", "WMITER", "WNITER",
        "warps_per_block", "threads_per_block", "shared_memory_bytes",
        "estimated_accumulators_per_thread", "estimated_registers_per_thread",
        "shape_class", "resource_class", "architecture_family", "resident_ctas_per_sm",
        "active_warps_per_sm", "estimated_occupancy", "cta_count", "cta_waves",
        "sm_coverage", "last_wave_utilization", "arithmetic_intensity_flop_per_byte",
        "architecture_score",
    )
    return {**{key: candidate[key] for key in keys},
            "buffer_feasibility": candidate.get("buffer_feasibility", {})}


def compact_tiling_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "candidate_id", "BM", "BN", "BK", "WM", "WN", "TM", "TN",
        "warps_per_block", "threads_per_block", "shared_memory_bytes",
        "estimated_accumulators_per_thread", "estimated_registers_per_thread",
        "shape_class", "resource_class", "architecture_family", "resident_ctas_per_sm",
        "estimated_occupancy", "cta_count", "cta_waves", "sm_coverage",
        "last_wave_utilization", "arithmetic_intensity_flop_per_byte", "architecture_score",
    )
    return [{**{key: item[key] for key in keys},
             "buffer_feasibility": item.get("buffer_feasibility", {})} for item in candidates]


def collect_strategy_ids(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, dict):
        strategy_id = value.get("strategy_id")
        if isinstance(strategy_id, str):
            result.add(strategy_id)
        for child in value.values():
            result.update(collect_strategy_ids(child))
    elif isinstance(value, list):
        for child in value:
            result.update(collect_strategy_ids(child))
    return result


def parse_strategy_tiles(strategy_ids: set[str], pattern: re.Pattern[str]) -> list[tuple[str, tuple[int, ...]]]:
    result = []
    for strategy_id in strategy_ids:
        match = pattern.fullmatch(strategy_id)
        if match:
            result.append((strategy_id, tuple(int(value) for value in match.groups())))
    return sorted(result)


def joint_candidate_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -float(item.get("architecture_score") or 0.0),
        item["threads_per_block"],
        item["estimated_accumulators_per_thread"],
        item["shared_memory_bytes"],
        item["BM"], item["BN"], item["BK"], item["WM"], item["WN"], item["TM"], item["TN"],
    )


def block_representative_score(item: dict[str, Any]) -> tuple[Any, ...]:
    block_shape = item["shape_class"]
    warp_shape = shape_class(item["WM"], item["WN"])
    thread_shape = shape_class(item["TM"], item["TN"])
    return (
        warp_shape != block_shape,
        thread_shape != block_shape,
        abs(item["threads_per_block"] - 128),
        abs(item["estimated_accumulators_per_thread"] - 32),
        joint_candidate_sort_key(item),
    )


def balanced_block_representative_score(item: dict[str, Any]) -> tuple[Any, ...]:
    """Prefer a conventional register micro-tile aligned with block geometry."""
    block_shape = item["shape_class"]
    warp_shape = shape_class(item["WM"], item["WN"])
    if block_shape == "square":
        target_tm, target_tn = 4, 4
    elif block_shape == "m_wide":
        target_tm, target_tn = 4, 2
    else:
        target_tm, target_tn = 2, 4
    return (
        warp_shape != block_shape,
        abs(item["TM"] - target_tm) + abs(item["TN"] - target_tn),
        -float(item.get("architecture_score") or 0.0),
        abs(item["threads_per_block"] - 128),
        abs(item["estimated_accumulators_per_thread"] - 32),
        item["estimated_registers_per_thread"],
        item["WM"], item["WN"], item["TM"], item["TN"],
    )


def shape_class(m: int, n: int) -> str:
    if m == n:
        return "square"
    return "m_wide" if m > n else "n_wide"


def resource_class(threads: int, accumulators: int) -> str:
    thread_class = "low_threads" if threads <= 128 else "mid_threads" if threads <= 256 else "high_threads"
    register_class = "low_regs" if accumulators <= 16 else "mid_regs" if accumulators <= 32 else "high_regs"
    return f"{thread_class}_{register_class}"


def nested_int(data: dict[str, Any], path: str, default: int) -> int:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, dict):
            return default
        value = value.get(part)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def first_nested_int(data: dict[str, Any], paths: list[str], default: int) -> int:
    for path in paths:
        value = nested_int(data, path, -1)
        if value > 0:
            return value
    return default


def dtype_bytes(optir: dict[str, Any]) -> int:
    dtype = str((optir.get("problem") or {}).get("dtype") or "fp32").lower()
    if "64" in dtype:
        return 8
    if "16" in dtype or "bf16" in dtype:
        return 2
    return 4
