from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from SGPO.generate_ir.ir_checker import check_before_codegen
from SGPO.utils.common_utils import load_json, save_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IR = ROOT / "data" / "IRs" / "ir_patch" / "optir.extracted.json"
DEFAULT_STRATEGY_LIBRARY = ROOT / "data" / "lib" / "strategy_library.json"
DEFAULT_CANDIDATE = ROOT / "results" / "check" / "selected_strategy.json"
DEFAULT_OUTPUT = ROOT / "results" / "check" / "pre_check_result.json"


def filter_strategies_by_preconditions(
    ir,
    candidate,
    strategy_library,
):
    strategy_by_id = {
        strategy["strategy_id"]: strategy
        for strategy in strategy_library.get("strategies", [])
    }
    candidate_ids = extract_candidate_ids(candidate)

    accepted = []
    rejected = []
    unknown_ids = []
    for strategy_id in candidate_ids:
        strategy = strategy_by_id.get(strategy_id)
        if strategy is None:
            unknown_ids.append(strategy_id)
            continue

        report = check_before_codegen(ir, strategy)
        item = {
            "strategy_id": strategy_id,
            "category": strategy.get("category"),
            "stage": strategy.get("stage"),
            "name": strategy.get("name"),
            "preconditions_ok": report["preconditions_ok"],
            "hard_constraints_ok": report["hard_constraints_ok"],
            "strategy_applicable": report["strategy_applicable"],
            "failed_checks": failed_checks(report),
            "checker_report": report,
        }
        if report["strategy_applicable"]:
            accepted.append(item)
        else:
            rejected.append(item)

    return {
        "accepted": accepted,
        "rejected": rejected,
        "unknown_strategy_ids": unknown_ids,
        "summary": {
            "candidate_count": len(candidate_ids),
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "unknown_count": len(unknown_ids),
        },
    }


def extract_candidate_ids(llm_output):
    candidates = llm_output.get("candidates", [])
    ids = []
    seen = set()
    for item in candidates:
        if isinstance(item, str):
            strategy_id = item
        elif isinstance(item, dict):
            strategy_id = item.get("strategy_id")
        else:
            continue
        if strategy_id and strategy_id not in seen:
            ids.append(strategy_id)
            seen.add(strategy_id)
    return ids


def failed_checks(report):
    return [
        {
            "id": item.get("id"),
            "status": item.get("status"),
            "message": item.get("message"),
            "failure_type": item.get("failure_type"),
        }
        for item in report.get("results", [])
        if item.get("status") in {"fail", "unknown"}
        and str(item.get("id", "")).startswith("PRECONDITION:")
    ]


def main() -> None:
    args = parse_args()
    ir = load_json(args.ir)
    candidate = load_json(args.candidate)
    strategy_library = load_json(args.strategy_library)
    result = filter_strategies_by_preconditions(ir, candidate, strategy_library)
    if args.output:
        save_json(args.output, result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter LLM-selected strategies by current OptIR preconditions.")
    parser.add_argument("--ir", type=Path, default=DEFAULT_IR)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--strategy-library", type=Path, default=DEFAULT_STRATEGY_LIBRARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    main()
