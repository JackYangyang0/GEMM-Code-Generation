from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from SGPO.llm.openai_client import OpenAICompatibleClient
from SGPO.generate_ir.strategy_index_filter import filter_strategy_index
from SGPO.utils.common_utils import load_json, save_json, load_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = ROOT / "llm" / "prompts" / "get_strategy_prompt.txt"
DEFAULT_STRATEGY_INDEX = ROOT / "data" / "lib" / "strategy_index.json"
DEFAULT_DEPENDENCY_GRAPH = ROOT / "data" / "graph" / "dependency_graph.json"
DEFAULT_IR = ROOT / "data" / "IRs" / "ir_patch" / "optir.extracted.json"
DEFAULT_CONFIG = ROOT / "conf.yaml"
DEFAULT_OUTPUT = ROOT / "results" / "check" / "selected_strategy.json"
DEFAULT_FILTERED_INDEX_OUTPUT = ROOT / "data" / "lib" / "filter" / "strategy_index.filtered.json"


def build_get_strategy_messages(
        user_question,
        strategy_index,
        prompt_path=DEFAULT_PROMPT,
):
    template = prompt_path.read_text(encoding="utf-8")
    prompt_strategy_index = strategy_index_for_prompt(strategy_index)
    prompt = template.format(
        user_question=user_question.strip(),
        strategy_index_json=json.dumps(prompt_strategy_index, ensure_ascii=False, indent=2),
    )
    return [
        {
            "role": "system",
            "content": "You select GEMM optimization strategy ids and only return valid JSON.",
        },
        {"role": "user", "content": prompt},
    ]


def strategy_index_for_prompt(strategy_index):
    if not isinstance(strategy_index, dict):
        return strategy_index
    keep_keys = [
        "index_name",
        "index_version",
        "target",
        "purpose",
        "selection_rules",
        "strategy_count",
        "strategies",
        "strategy_ids_by_stage",
        "filter_context",
        "fallback_candidates",
    ]
    return {key: strategy_index[key] for key in keep_keys if key in strategy_index}


def get_strategy_ids_from_llm(
        client,
        user_question,
        strategy_index,
        prompt_path,
):
    messages = build_get_strategy_messages(user_question, strategy_index, prompt_path)
    response = client.complete_json(messages)
    return validate_strategy_response(response, strategy_index)


def get_micro_strategy_from_llm(
        client,
        current_stage,
        current_subphase,
        subphase,
        optir_summary,
        strategy_index,
        prompt_path=None,
):
    messages = build_micro_strategy_messages(
        current_stage=current_stage,
        current_subphase=current_subphase,
        subphase=subphase,
        optir_summary=optir_summary,
        strategy_index=strategy_index,
        prompt_path=prompt_path or (ROOT / "llm" / "prompts" / "get_micro_strategy_prompt.txt"),
    )
    response = client.complete_json(messages)
    return validate_micro_strategy_response(response, strategy_index)


def build_micro_strategy_messages(
        current_stage,
        current_subphase,
        subphase,
        optir_summary,
        strategy_index,
        prompt_path,
):
    template = prompt_path.read_text(encoding="utf-8")
    allowed_strategy_ids = [item["strategy_id"] for item in strategy_index.get("strategies", []) or []]
    prompt = template.format(
        current_stage=current_stage,
        current_subphase=current_subphase,
        subphase_purpose=subphase.get("purpose", ""),
        requires_fields=json.dumps(subphase.get("requires_fields", []), ensure_ascii=False, indent=2),
        provides_fields=json.dumps(subphase.get("provides_fields", []), ensure_ascii=False, indent=2),
        optir_summary=json.dumps(optir_summary, ensure_ascii=False, indent=2),
        allowed_strategy_ids=json.dumps(allowed_strategy_ids, ensure_ascii=False, indent=2),
    )
    return [
        {
            "role": "system",
            "content": "You select GEMM construction micro-strategy labels and only return valid JSON.",
        },
        {"role": "user", "content": prompt},
    ]


def validate_micro_strategy_response(response, strategy_index):
    valid_ids = {item["strategy_id"] for item in strategy_index.get("strategies", []) or []}
    candidates = []
    seen = set()
    raw_candidates = response.get("candidates")
    if not isinstance(raw_candidates, list):
        raw_candidates = [response.get("selected_strategy_id") or response.get("strategy_id")]
    for item in raw_candidates:
        if isinstance(item, dict):
            strategy_id = item.get("strategy_id") or item.get("selected_strategy_id")
            reason = str(item.get("reason", response.get("reason", ""))).strip()
            confidence = normalize_confidence(item.get("confidence", 1.0))
        else:
            strategy_id = item
            reason = str(response.get("reason", "")).strip()
            confidence = 1.0
        if strategy_id in valid_ids and strategy_id not in seen:
            candidates.append({"strategy_id": strategy_id, "reason": reason, "confidence": confidence})
            seen.add(strategy_id)
    if not candidates:
        strategy_id = response.get("selected_strategy_id") or response.get("strategy_id")
        raise ValueError(f"LLM selected invalid micro-strategy: {strategy_id}")
    return {
        "selected_strategy_id": candidates[0]["strategy_id"],
        "reason": str(response.get("reason", "")).strip(),
        "expected_ir_updates": response.get("expected_ir_updates", {}) if isinstance(response.get("expected_ir_updates", {}), dict) else {},
        "candidates": candidates,
    }


def validate_strategy_response(response, strategy_index):
    strategies = strategy_index.get("strategies", strategy_index) if isinstance(strategy_index, dict) else strategy_index
    valid_ids = {item["strategy_id"] for item in strategies}
    candidates = response.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError("LLM response field 'candidates' must be a list.")

    cleaned = []
    seen = set()
    for item in candidates:
        if not isinstance(item, dict):
            continue
        strategy_id = item.get("strategy_id")
        if strategy_id not in valid_ids or strategy_id in seen:
            continue
        seen.add(strategy_id)
        cleaned.append(
            {
                "strategy_id": strategy_id,
                "reason": str(item.get("reason", "")).strip(),
                "confidence": normalize_confidence(item.get("confidence")),
            }
        )
    return {"candidates": cleaned}


def normalize_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask LLM to select SGPO strategy ids from a filtered strategy index.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--ir", type=Path, default=DEFAULT_IR)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--stage")
    parser.add_argument("--strategy-index", type=Path, default=DEFAULT_STRATEGY_INDEX)
    parser.add_argument("--dependency-graph", type=Path, default=DEFAULT_DEPENDENCY_GRAPH)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-failures", type=int, default=2)
    parser.add_argument("--allowed-maturity", nargs="+", default=["v1", "v2"])
    parser.add_argument("--profile")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--filtered-index-output", type=Path, default=DEFAULT_FILTERED_INDEX_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ir = load_json(args.ir)
    raw_strategy_index = load_json(args.strategy_index)
    dependency_graph = load_json(args.dependency_graph)
    history = load_json(args.history) if args.history else None
    strategy_index = filter_strategy_index(
        strategy_index=raw_strategy_index,
        ir=ir,
        dependency_graph=dependency_graph,
        history=history,
        current_stage=args.stage,
        max_failures=args.max_failures,
        allowed_maturity=set(args.allowed_maturity),
        profile=args.profile,
    )
    if args.filtered_index_output:
        save_json(args.filtered_index_output, strategy_index)

    config = load_config(args.config)
    llm_conf = config["llm"]
    client = OpenAICompatibleClient(llm_conf)
    result = get_strategy_ids_from_llm(
        client=client,
        user_question=args.question,
        strategy_index=strategy_index,
        prompt_path=args.prompt,
    )
    result["filter_context"] = strategy_index.get("filter_context")
    if args.output:
        save_json(args.output, result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
