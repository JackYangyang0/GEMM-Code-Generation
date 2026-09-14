"""Bounded compiler tuning using the same correctness/benchmark harness."""
from __future__ import annotations

import copy
import math
import shutil

from SGPO.utils.common_utils import save_json


def run_resource_sweep(ir, source_dir, build_dir, executable_name, timeout_seconds,
                       vcvars64_path, build_platform, benchmark_runs, benchmark_warmup_runs):
    from SGPO.verification.build_run_verifier import verify_build_and_run, clear_previous_compile_outputs

    name = "gemm" if build_platform == "linux" and executable_name == "gemm.exe" else executable_name
    clear_previous_compile_outputs(build_dir / name)

    current_limit = ir.get("compiler", {}).get("max_register_count")
    limits = list(dict.fromkeys([current_limit, None, 64, 96, 128]))
    records, results = [], []
    for index, limit in enumerate(limits):
        variant = copy.deepcopy(ir)
        variant.pop("verification", None)
        variant.pop("performance", None)
        compiler = variant.setdefault("compiler", {})
        compiler.setdefault("resource_feedback", {})["enabled"] = False
        compiler["max_register_count"] = limit
        directory = build_dir / f"resource_variant_{index}"
        result = verify_build_and_run(
            variant, source_dir, directory, executable_name, timeout_seconds,
            vcvars64_path, build_platform, benchmark_runs, benchmark_warmup_runs,
        )
        verification = result.get("verification", {})
        latency = result.get("performance", {}).get("latency_ms")
        passed = all(verification.get(name, {}).get("status") == "pass"
                     for name in ("compile", "correctness", "runtime_safety"))
        rankable = passed and isinstance(latency, (float, int)) and math.isfinite(latency) and latency > 0
        record = {"variant": index, "max_register_count": limit,
                  "accepted": rankable, "latency_ms": latency,
                  "gflops": result.get("performance", {}).get("gflops"),
                  "verification": copy.deepcopy(verification), "build_dir": str(directory)}
        records.append(record)
        results.append(result)
    eligible = [item for item in records if item["accepted"]]
    selected = min(eligible, key=lambda item: item["latency_ms"]) if eligible else records[0]
    best = results[selected["variant"]]
    best.setdefault("compiler", {}).setdefault("resource_feedback", {}).update(
        {"enabled": True, "selected_max_register_count": selected["max_register_count"]})
    report = {"status": "pass" if eligible else "fail", "variants": records,
              "selected_variant": selected["variant"] if eligible else None,
              "metric": "mean_latency_ms", "compile_once_per_variant": True}
    best.setdefault("verification", {})["resource_sweep"] = report
    save_json(build_dir / "resource_sweep.json", report)
    if eligible:
        shutil.copy2(build_dir / f"resource_variant_{selected['variant']}" / name, build_dir / name)
        save_json(build_dir / "selected_build_config.json", {
            "max_register_count": selected["max_register_count"],
            "command": best.get("verification", {}).get("compile", {}).get("command"),
            "note": "Reuse these compiler flags when rebuilding this winning source.",
        })
    return best
