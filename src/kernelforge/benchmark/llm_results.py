from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

DEFAULT_LLM_CONFIG_PATH = Path(__file__).with_name("llm_models.json")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL result ledger."""
    path = Path(path)
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object row in {path}:{line_number}")
            rows.append(row)
    return rows


def load_results(paths: str | Path | Iterable[str | Path]) -> list[dict[str, Any]]:
    """Load one or more LLM inference ledgers."""
    if isinstance(paths, (str, Path)):
        paths = [paths]

    rows = []
    for path in paths:
        path = Path(path)
        for row in load_jsonl(path):
            row.setdefault("result_path", str(path))
            rows.append(row)
    return rows


def _response_usage(row: dict[str, Any]) -> dict[str, Any]:
    response = row.get("response")
    if not isinstance(response, dict):
        return {}
    usage = response.get("usage")
    return usage if isinstance(usage, dict) else {}


def _finish_reason(row: dict[str, Any]) -> str | None:
    response = row.get("response")
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, dict):
        return None
    finish_reason = choice.get("finish_reason")
    return finish_reason if isinstance(finish_reason, str) else None


def _model_pricing(
    config: dict[str, Any], row: dict[str, Any]
) -> tuple[float | None, float | None]:
    model = row.get("model")
    if not isinstance(model, str):
        return None, None
    model_config = config.get("models", {}).get(model)
    if not isinstance(model_config, dict):
        return None, None
    pricing = model_config.get("pricing")
    if not isinstance(pricing, dict):
        return None, None
    return pricing.get("input_per_million"), pricing.get("output_per_million")


def row_cost_usd(row: dict[str, Any], config: dict[str, Any]) -> float | None:
    """Estimate cost for one row using prompt/completion usage and model pricing."""
    input_per_million, output_per_million = _model_pricing(config, row)
    if input_per_million is None or output_per_million is None:
        return None

    usage = _response_usage(row)
    prompt_tokens = usage.get("prompt_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or 0
    return (
        prompt_tokens / 1_000_000 * input_per_million
        + completion_tokens / 1_000_000 * output_per_million
    )


def code_metrics(content: str | None) -> dict[str, Any]:
    """Compute lightweight notebook-friendly quality metrics for generated code."""
    content = content or ""
    syntax_ok = True
    syntax_error = None
    try:
        ast.parse(content)
    except SyntaxError as exc:
        syntax_ok = False
        syntax_error = f"{exc.msg} at {exc.lineno}:{exc.offset}"

    flags = []
    if not content.strip():
        flags.append("empty")
    if "```" in content:
        flags.append("markdown")
    if not content.lstrip().startswith("import "):
        flags.append("no_import_prefix")
    if "@triton.jit" not in content:
        flags.append("no_triton_jit")
    if re.search(r"\btorch\.", content):
        flags.append("torch_call")
    if re.search(r"\bpass\b", content) or "TODO" in content:
        flags.append("incomplete_marker")
    if not syntax_ok:
        flags.append("syntax_error")

    return {
        "content_chars": len(content),
        "content_lines": len(content.splitlines()),
        "starts_with_import": content.lstrip().startswith("import "),
        "markdown_fence_count": content.count("```"),
        "triton_jit_count": content.count("@triton.jit"),
        "torch_call_count": len(re.findall(r"\btorch\.", content)),
        "syntax_ok": syntax_ok,
        "syntax_error": syntax_error,
        "flags": flags,
        "flags_text": ", ".join(flags),
    }


def analysis_rows(
    rows: Iterable[dict[str, Any]],
    config: dict[str, Any] | None = None,
    *,
    config_path: str | Path = DEFAULT_LLM_CONFIG_PATH,
) -> list[dict[str, Any]]:
    """Flatten raw ledger rows into a table suitable for Marimo display/filtering."""
    if config is None:
        from .llm_inference import load_llm_config

        config = load_llm_config(config_path)
    table = []
    for row_number, row in enumerate(rows, start=1):
        usage = _response_usage(row)
        finish_reason = _finish_reason(row)
        metrics = code_metrics(row.get("content"))
        cost = row_cost_usd(row, config)
        prompt_tokens = usage.get("prompt_tokens") or 0
        completion_tokens = usage.get("completion_tokens") or 0

        table.append(
            {
                "row_number": row_number,
                "result_path": row.get("result_path"),
                "request_hash": row.get("request_hash"),
                "entry_index": row.get("entry_index"),
                "entry_file": row.get("entry_file"),
                "model": row.get("model"),
                "model_label": row.get("model_label"),
                "provider": row.get("provider"),
                "status": row.get("status"),
                "finish_reason": finish_reason,
                "truncated": finish_reason == "length",
                "latency_s": row.get("latency_s"),
                "attempt": row.get("attempt"),
                "max_attempts": row.get("max_attempts"),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": usage.get("total_tokens") or 0,
                "cost_usd": cost,
                **metrics,
            }
        )
    return table


def summarize_analysis(table: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize flattened analysis rows for quick notebook headline metrics."""
    rows = list(table)
    if not rows:
        return {
            "rows": 0,
            "successes": 0,
            "failures": 0,
            "total_cost_usd": 0.0,
        }

    latencies = [row["latency_s"] for row in rows if row.get("latency_s") is not None]
    costs = [row["cost_usd"] or 0.0 for row in rows]
    return {
        "rows": len(rows),
        "successes": sum(row.get("status") == "success" for row in rows),
        "failures": sum(row.get("status") == "failed" for row in rows),
        "truncated": sum(1 for row in rows if row.get("truncated")),
        "syntax_ok": sum(1 for row in rows if row.get("syntax_ok")),
        "syntax_error": sum(not row.get("syntax_ok") for row in rows),
        "torch_call_rows": sum((row.get("torch_call_count") or 0) > 0 for row in rows),
        "prompt_tokens": sum(row.get("prompt_tokens") or 0 for row in rows),
        "completion_tokens": sum(row.get("completion_tokens") or 0 for row in rows),
        "total_tokens": sum(row.get("total_tokens") or 0 for row in rows),
        "total_cost_usd": sum(costs),
        "cost_per_row_usd": sum(costs) / len(rows),
        "latency_min_s": min(latencies) if latencies else None,
        "latency_max_s": max(latencies) if latencies else None,
        "latency_avg_s": sum(latencies) / len(latencies) if latencies else None,
    }


def model_summaries(table: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group analysis rows by model for Marimo comparison tables."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in table:
        groups.setdefault(str(row.get("model")), []).append(row)

    summaries = []
    for model, rows in sorted(groups.items()):
        summary = summarize_analysis(rows)
        summary["model"] = model
        summary["model_label"] = rows[0].get("model_label")
        summaries.append(summary)
    return summaries


def load_analysis_table(
    paths: str | Path | Iterable[str | Path],
    config: dict[str, Any] | None = None,
    *,
    config_path: str | Path = DEFAULT_LLM_CONFIG_PATH,
) -> list[dict[str, Any]]:
    """Convenience helper: load raw rows and return the flattened analysis table."""
    return analysis_rows(load_results(paths), config=config, config_path=config_path)
