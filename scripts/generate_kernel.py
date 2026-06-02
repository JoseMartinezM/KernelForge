"""
Single-entry kernel generation helper for the KernelForge agent.

Generates one Triton kernel for a specific TritonBench entry and prints
the result as JSON to stdout. Used by the Pi agent's generate_kernel tool.

Usage:
    uv run python scripts/generate_kernel.py --entry-file tanh.py --model google/gemma-4-E4B-it
    uv run python scripts/generate_kernel.py --entry-file tanh.py --model google/gemma-4-E4B-it \
        --grammar-file grammar/triton.gbnf
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from openai import OpenAI  # noqa: E402

from kernelforge.benchmark.llm_inference import (  # noqa: E402
    SYSTEM_PROMPT,
    load_llm_config,
    model_generation_defaults,
    provider_headers,
    provider_token,
)
from kernelforge.benchmark.tritonbench import load_t_simple_entries, make_prompt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a single Triton kernel for the agent.")
    parser.add_argument("--entry-file", required=True, help="TritonBench entry filename, e.g. tanh.py")
    parser.add_argument("--model", required=True, help="Model name from llm_models.json")
    parser.add_argument(
        "--grammar-file",
        type=Path,
        help="Optional GBNF grammar file for constrained generation via provider extra_body.",
    )
    parser.add_argument(
        "--guided-decoding-backend",
        default="xgrammar",
        help="Guided decoding backend sent with --grammar-file. Defaults to xgrammar.",
    )
    args = parser.parse_args()

    config = load_llm_config()
    if args.model not in config["models"]:
        print(json.dumps({"error": f"Unknown model '{args.model}'. Available: {sorted(config['models'])}"}))
        sys.exit(1)

    entries, errors, _, _ = load_t_simple_entries(PROJECT_ROOT / "vendor" / "TritonBench")
    entry = next((e for e in entries if e["file"] == args.entry_file), None)
    if entry is None:
        print(json.dumps({"error": f"entry_file '{args.entry_file}' not found in dataset"}))
        sys.exit(1)

    model_cfg = config["models"][args.model]
    provider_name = model_cfg["provider"]
    provider = config["providers"][provider_name]

    client = OpenAI(
        base_url=provider["url"],
        api_key=provider_token(provider) or "no-key",
        default_headers=provider_headers(provider),
    )

    generation = model_generation_defaults(config, args.model)
    grammar_metadata = None
    if args.grammar_file is not None:
        grammar_path = args.grammar_file
        if not grammar_path.is_absolute():
            grammar_path = PROJECT_ROOT / grammar_path
        if not grammar_path.is_file():
            print(json.dumps({"error": f"grammar file not found: {grammar_path}"}))
            sys.exit(1)

        extra_body = generation.setdefault("extra_body", {})
        if not isinstance(extra_body, dict):
            print(json.dumps({"error": "generation extra_body must be an object"}))
            sys.exit(1)

        grammar_text = grammar_path.read_text(encoding="utf-8")
        extra_body["guided_grammar"] = grammar_text
        extra_body["guided_decoding_backend"] = args.guided_decoding_backend
        try:
            grammar_source_path = str(grammar_path.relative_to(PROJECT_ROOT))
        except ValueError:
            grammar_source_path = str(grammar_path)
        grammar_metadata = {
            "source_path": grammar_source_path,
            "guided_decoding_backend": args.guided_decoding_backend,
            "bytes": len(grammar_text.encode("utf-8")),
        }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": make_prompt(entry)},
    ]

    api_model = model_cfg.get("model_id", args.model)
    response = client.chat.completions.create(
        model=api_model,
        messages=messages,
        **generation,
    )

    content = response.choices[0].message.content or ""
    # Strip <thought>...</thought> blocks emitted by reasoning models (e.g. Gemma).
    content = re.sub(r"<thought>.*?</thought>\s*", "", content, flags=re.DOTALL).strip()
    finish_reason = response.choices[0].finish_reason

    print(json.dumps({
        "entry_file": args.entry_file,
        "model": args.model,
        "content": content,
        "finish_reason": finish_reason,
        "grammar": grammar_metadata,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
            "completion_tokens": response.usage.completion_tokens if response.usage else None,
        },
    }))


if __name__ == "__main__":
    main()
