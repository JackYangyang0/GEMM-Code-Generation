from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from SGPO.llm.openai_client import OpenAICompatibleClient
from SGPO.generate_ir.strategy_index_filter import filter_strategy_index
from SGPO.utils.common_utils import load_json, save_json, load_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT = ROOT / "llm" / "prompts" / "get_strategy_prompt.txt"
DEFAULT_UNLOCK_BATCH_PLAN_PROMPT = ROOT / "llm" / "prompts" / "plan_unlock_batches_prompt.txt"
DEFAULT_JOINT_TILING_PROMPT = ROOT / "llm" / "prompts" / "select_joint_tiling_prompt.txt"
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
    response = complete_selection_json(client, messages)
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
    response = complete_selection_json(client, messages)
    return validate_micro_strategy_response(response, strategy_index)


def get_joint_tiling_selection_from_llm(
        client,
        optir_summary,
        tiling_candidates,
        top_k,
        prompt_path=None,
):
    messages = build_joint_tiling_messages(
        optir_summary=optir_summary,
        tiling_candidates=tiling_candidates,
        top_k=top_k,
        prompt_path=prompt_path or DEFAULT_JOINT_TILING_PROMPT,
    )
    response = complete_selection_json(client, messages)
    return validate_joint_tiling_response(response, tiling_candidates, top_k)


def build_joint_tiling_messages(optir_summary, tiling_candidates, top_k, prompt_path):
    template = prompt_path.read_text(encoding="utf-8")
    option_space = compact_joint_tiling_option_space(tiling_candidates)
    prompt = template.format(
        optir_summary=json.dumps(optir_summary, ensure_ascii=False, separators=(",", ":")),
        tiling_candidates=json.dumps(option_space, ensure_ascii=False, separators=(",", ":")),
        top_k=int(top_k),
    )
    return [
        {
            "role": "system",
            "content": "You select legal complete GEMM tiling tuples and only return compact valid JSON.",
        },
        {"role": "user", "content": prompt},
    ]


def validate_joint_tiling_response(response, tiling_candidates, top_k):
    valid_ids = {item["candidate_id"] for item in tiling_candidates}
    candidates_by_id = {item["candidate_id"]: item for item in tiling_candidates}
    candidates_by_components = {
        (
            item.get("block_strategy_id"),
            item.get("warp_strategy_id"),
            item.get("thread_strategy_id"),
        ): item["candidate_id"]
        for item in tiling_candidates
    }
    raw_ids = response.get("candidate_ids", []) if isinstance(response, dict) else []
    if not isinstance(raw_ids, list):
        raw_ids = []
    selections = response.get("selections", []) if isinstance(response, dict) else []
    if isinstance(selections, list):
        for selection in selections:
            if not isinstance(selection, dict):
                continue
            key = (
                selection.get("block_strategy_id"),
                selection.get("warp_strategy_id"),
                selection.get("thread_strategy_id"),
            )
            candidate_id = candidates_by_components.get(key)
            if candidate_id:
                raw_ids.append(candidate_id)
    candidate_ids = []
    used_blocks = set()
    for candidate_id in raw_ids:
        candidate = candidates_by_id.get(candidate_id)
        if candidate is None or candidate_id not in valid_ids or candidate_id in candidate_ids:
            continue
        block_key = (candidate.get("BM"), candidate.get("BN"), candidate.get("BK"))
        if block_key in used_blocks:
            continue
        candidate_ids.append(candidate_id)
        used_blocks.add(block_key)
        if len(candidate_ids) >= int(top_k):
            break
    return {
        "candidate_ids": candidate_ids,
        "reason": str(response.get("reason", "")).strip() if isinstance(response, dict) else "",
    }


def compact_joint_tiling_option_space(tiling_candidates):
    fields = {
        "block_options": ("block_strategy_id", "BM", "BN", "BK"),
        "warp_options": ("warp_strategy_id", "WM", "WN"),
        "thread_options": ("thread_strategy_id", "TM", "TN"),
    }
    result = {}
    for group, keys in fields.items():
        id_key = keys[0]
        unique = {}
        for item in tiling_candidates:
            strategy_id = item.get(id_key)
            if not strategy_id:
                continue
            unique[strategy_id] = {key: item.get(key) for key in keys}
        result[group] = [unique[key] for key in sorted(unique)]
    ranked_keys = (
        "candidate_id", "block_strategy_id", "warp_strategy_id", "thread_strategy_id",
        "BM", "BN", "BK", "WM", "WN", "TM", "TN", "threads_per_block",
        "estimated_registers_per_thread", "resident_ctas_per_sm", "estimated_occupancy",
        "cta_count", "cta_waves", "sm_coverage", "last_wave_utilization",
        "architecture_score",
    )
    result["ranked_legal_combinations"] = [
        {key: item.get(key) for key in ranked_keys}
        for item in tiling_candidates[:48]
    ]
    return result


def get_unlock_batch_plan_from_llm(
        client,
        strategy_index,
        optir_summary,
        code_summary,
        matrix_profile,
        verification_summary,
        user_question="",
        prompt_path=None,
):
    messages = build_unlock_batch_plan_messages(
        strategy_index=strategy_index,
        optir_summary=optir_summary,
        code_summary=code_summary,
        matrix_profile=matrix_profile,
        verification_summary=verification_summary,
        user_question=user_question,
        prompt_path=prompt_path or DEFAULT_UNLOCK_BATCH_PLAN_PROMPT,
    )
    response = complete_selection_json(client, messages)
    return validate_unlock_batch_plan_response(response, strategy_index)


def complete_selection_json(client, messages):
    if hasattr(client, "complete_selection_json"):
        return client.complete_selection_json(messages)
    return client.complete_json(messages)


def build_unlock_batch_plan_messages(
        strategy_index,
        optir_summary,
        code_summary,
        matrix_profile,
        verification_summary,
        user_question,
        prompt_path,
):
    template = prompt_path.read_text(encoding="utf-8")
    compact_index = unlock_strategy_index_for_prompt(strategy_index)
    prompt = template.format(
        user_question=str(user_question or "").strip(),
        strategy_index_json=json.dumps(compact_index, ensure_ascii=False, indent=2),
        optir_summary=json.dumps(optir_summary, ensure_ascii=False, indent=2),
        code_summary=json.dumps(code_summary, ensure_ascii=False, indent=2),
        matrix_profile=json.dumps(matrix_profile, ensure_ascii=False, indent=2),
        verification_summary=json.dumps(verification_summary, ensure_ascii=False, indent=2),
    )
    return [
        {
            "role": "system",
            "content": "You plan GEMM performance-unlock strategy batches and only return compact valid JSON.",
        },
        {"role": "user", "content": prompt},
    ]


def unlock_strategy_index_for_prompt(strategy_index):
    if not isinstance(strategy_index, dict):
        return strategy_index
    strategies = []
    for item in strategy_index.get("strategies", []) or []:
        strategies.append(
            {
                "strategy_id": item.get("strategy_id"),
                "stage": item.get("stage"),
                "category": item.get("category"),
                "phase": item.get("phase"),
                "conflict_group": item.get("conflict_group"),
                "modifies_regions": item.get("modifies_regions", []),
                "provides_fields": item.get("provides_fields", []),
                "requires_fields": item.get("requires_fields", []),
                "risk_level": item.get("risk_level"),
                "priority": item.get("priority"),
                "title": item.get("title") or item.get("name"),
            }
        )
    return {
        "index_name": strategy_index.get("index_name"),
        "index_version": strategy_index.get("index_version"),
        "strategy_count": len(strategies),
        "strategies": strategies,
        "filter_context": strategy_index.get("filter_context", {}),
    }


def validate_unlock_batch_plan_response(response, strategy_index):
    valid_ids = {item["strategy_id"] for item in strategy_index.get("strategies", []) or []}
    batches = response.get("batches", []) if isinstance(response, dict) else []
    if not isinstance(batches, list):
        batches = []
    cleaned_batches = []
    for index, batch in enumerate(batches, start=1):
        if not isinstance(batch, dict):
            continue
        strategy_ids = []
        seen = set()
        for strategy_id in batch.get("strategy_ids", []) or []:
            if strategy_id in valid_ids and strategy_id not in seen:
                strategy_ids.append(strategy_id)
                seen.add(strategy_id)
        if not strategy_ids:
            continue
        batch_id = str(batch.get("batch_id") or f"batch_{index}").strip() or f"batch_{index}"
        cleaned_batches.append(
            {
                "batch_id": batch_id,
                "purpose": str(batch.get("purpose", "")).strip(),
                "strategy_ids": strategy_ids,
                "coupling_reason": str(batch.get("coupling_reason", "")).strip(),
                "requires_summary": batch.get("requires_summary", []) if isinstance(batch.get("requires_summary"), list) else [],
                "risk": batch.get("risk") if batch.get("risk") in {"low", "medium", "high"} else "medium",
            }
        )
    if not cleaned_batches:
        return {"batch_order": [], "batches": []}
    return {
        "batch_order": [batch["batch_id"] for batch in cleaned_batches],
        "batches": cleaned_batches,
    }


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
