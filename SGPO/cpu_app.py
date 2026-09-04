from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from SGPO.generate_ir.ir_extraction import build_extracted_ir
from SGPO.llm.c_code_generator import (
    CPU_CODE_FILES,
    apply_cpu_c_code_files,
    generate_cpu_c_code_with_llm,
)
from SGPO.llm.openai_client import OpenAICompatibleClient
from SGPO.llm.patch_generator import load_code_context
from SGPO.utils.common_utils import load_config, load_json, save_json
from SGPO.verification.c_build_run_verifier import verify_cpu_build_and_run


ROOT = Path(__file__).resolve().parent
DEFAULT_DESCRIPTION = (
    "I need to generate a high-performance CPU C GEMM implementation for "
    "row-major fp32 NN GEMM. Matrix Size: (M:512 N:512 K:512)."
)
DEFAULT_TEMPLATE = ROOT / "data" / "IRs" / "optir.json"
DEFAULT_IR_OUTPUT = ROOT / "data" / "IRs" / "ir_patch" / "optir.cpu.extracted.json"
DEFAULT_VERIFIED_IR_OUTPUT = ROOT / "data" / "IRs" / "ir_patch" / "optir.cpu.verified.json"
DEFAULT_CPU_SKELETON = ROOT / "gemm_code" / "cpu_skeleton"
DEFAULT_CPU_CODE_ROOT = ROOT / "gemm_code" / "cpu_code" / "chain.cpu"
DEFAULT_CPU_CODE_OUTPUT = ROOT / "results" / "code" / "cpu_c_code_files.json"
DEFAULT_CPU_SUMMARY_OUTPUT = ROOT / "results" / "check" / "cpu_generation_summary.json"
DEFAULT_PROMPT = ROOT / "llm" / "prompts" / "generate_cpu_c_code_prompt.txt"
DEFAULT_CONFIG = ROOT / "conf.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="SGPO CPU/C GEMM generation entrypoint.")
    parser.add_argument("--description", default=DEFAULT_DESCRIPTION)
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--ir-output", default=str(DEFAULT_IR_OUTPUT))
    parser.add_argument("--verified-ir-output", default=str(DEFAULT_VERIFIED_IR_OUTPUT))
    parser.add_argument("--skeleton-dir", default=str(DEFAULT_CPU_SKELETON))
    parser.add_argument("--code-root", default=str(DEFAULT_CPU_CODE_ROOT))
    parser.add_argument("--code-output", default=str(DEFAULT_CPU_CODE_OUTPUT))
    parser.add_argument("--summary-output", default=str(DEFAULT_CPU_SUMMARY_OUTPUT))
    parser.add_argument("--prompt", default=str(DEFAULT_PROMPT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--skip-llm", action="store_true", help="Compile and verify the CPU skeleton without LLM generation.")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    template = load_json(Path(args.template))
    ir = build_extracted_ir(template, args.description)
    ir.setdefault("target", {}).update({"backend": "cpu", "language": "c", "device": "cpu"})
    save_json(Path(args.ir_output), ir)

    code_root = prepare_cpu_code_root(Path(args.skeleton_dir), Path(args.code_root))
    generated_code: dict[str, Any] | None = None
    apply_result = {"status": "skipped", "reason": "skip_llm enabled"}

    if not args.skip_llm:
        code_context = load_code_context(code_root, CPU_CODE_FILES)
        client = OpenAICompatibleClient(config["llm"])
        generated_code = generate_cpu_c_code_with_llm(
            client=client,
            ir=ir,
            code_context=code_context,
            prompt_path=Path(args.prompt),
        )
        apply_result = apply_cpu_c_code_files(generated_code, code_root)
        generated_code["apply_result"] = apply_result
        save_json(Path(args.code_output), generated_code)

    timeout_seconds = int(config.get("verification", {}).get("timeout_seconds", 60))
    verified_ir = verify_cpu_build_and_run(
        ir=ir,
        source_dir=code_root,
        build_dir=code_root / "build",
        timeout_seconds=timeout_seconds,
    )
    save_json(Path(args.verified_ir_output), verified_ir)

    summary = {
        "target": verified_ir.get("target", {}),
        "ir_output": str(Path(args.ir_output)),
        "verified_ir_output": str(Path(args.verified_ir_output)),
        "code_root": str(code_root),
        "code_output": str(Path(args.code_output)) if generated_code is not None else None,
        "apply_result": apply_result,
        "compile_status": verified_ir.get("verification", {}).get("compile", {}).get("status"),
        "correctness_status": verified_ir.get("verification", {}).get("correctness", {}).get("status"),
        "host_error": verified_ir.get("verification", {}).get("runtime_safety", {}).get("host_error"),
        "latency_ms": verified_ir.get("performance", {}).get("latency_ms"),
        "gflops": verified_ir.get("performance", {}).get("gflops"),
    }
    save_json(Path(args.summary_output), summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def prepare_cpu_code_root(skeleton_dir: Path, code_root: Path) -> Path:
    code_root.mkdir(parents=True, exist_ok=True)
    for relative_path in CPU_CODE_FILES:
        source = skeleton_dir / relative_path
        target = code_root / relative_path
        if not source.exists():
            raise FileNotFoundError(f"CPU skeleton file missing: {source}")
        shutil.copyfile(source, target)
    return code_root


if __name__ == "__main__":
    main()
