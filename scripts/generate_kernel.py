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
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

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

GrammarBackend = str
GRAMMAR_BACKENDS = ("xgrammar", "llama-cpp")


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def grammar_extra_body(grammar_text: str, backend: GrammarBackend) -> dict[str, Any]:
    if backend == "xgrammar":
        return {
            "guided_grammar": grammar_text,
            "guided_decoding_backend": "xgrammar",
        }
    if backend == "llama-cpp":
        return {
            "grammar": grammar_text,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    raise ValueError(f"unsupported grammar backend: {backend}")


def apply_grammar(
    generation: dict[str, Any],
    grammar_file: Path,
    backend: GrammarBackend,
) -> dict[str, Any]:
    grammar_path = project_path(grammar_file)
    if not grammar_path.is_file():
        raise FileNotFoundError(f"grammar file not found: {grammar_path}")

    extra_body = generation.setdefault("extra_body", {})
    if not isinstance(extra_body, dict):
        raise TypeError("generation extra_body must be an object")

    grammar_text = grammar_path.read_text(encoding="utf-8")
    request_extra_body = grammar_extra_body(grammar_text, backend)
    if backend == "llama-cpp" and isinstance(extra_body.get("chat_template_kwargs"), dict):
        request_extra_body["chat_template_kwargs"] = {
            **extra_body["chat_template_kwargs"],
            **request_extra_body["chat_template_kwargs"],
        }
    extra_body.update(request_extra_body)

    try:
        grammar_source_path = str(grammar_path.relative_to(PROJECT_ROOT))
    except ValueError:
        grammar_source_path = str(grammar_path)
    return {
        "source_path": grammar_source_path,
        "backend": backend,
        "sha256": hashlib.sha256(grammar_text.encode("utf-8")).hexdigest(),
        "bytes": len(grammar_text.encode("utf-8")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a single Triton kernel for the agent.")
    parser.add_argument(
        "--entry-file",
        required=True,
        help="TritonBench entry filename, e.g. tanh.py",
    )
    parser.add_argument("--model", required=True, help="Model name from llm_models.json")
    parser.add_argument(
        "--grammar-file",
        type=Path,
        help="Optional GBNF grammar file for constrained generation via provider extra_body.",
    )
    parser.add_argument(
        "--grammar-backend",
        choices=GRAMMAR_BACKENDS,
        default="xgrammar",
        help="Grammar backend to use with --grammar-file. Defaults to xgrammar.",
    )
    parser.add_argument(
        "--guided-decoding-backend",
        choices=GRAMMAR_BACKENDS,
        dest="grammar_backend",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    config = load_llm_config()
    if args.model not in config["models"]:
        print(
            json.dumps(
                {"error": f"Unknown model '{args.model}'. Available: {sorted(config['models'])}"}
            )
        )
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
        try:
            grammar_metadata = apply_grammar(generation, args.grammar_file, args.grammar_backend)
        except (FileNotFoundError, TypeError, ValueError) as exc:
            print(json.dumps({"error": str(exc)}))
            sys.exit(1)

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
