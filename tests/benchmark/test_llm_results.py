from __future__ import annotations

import json
from pathlib import Path

import pytest

from kernelforge.benchmark.llm_results import (
    analysis_rows,
    code_metrics,
    load_analysis_table,
    load_jsonl,
    load_results,
    model_summaries,
    summarize_analysis,
)


def test_load_jsonl_parses_fixture_ledger(sample_inference_ledger: Path):
    rows = load_jsonl(sample_inference_ledger)

    assert len(rows) == 3
    assert rows[0]["entry_file"] == "tanh.py"
    assert rows[2]["status"] == "failed"


def test_load_jsonl_rejects_invalid_json(tmp_path: Path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"ok": true}\nnot json\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_jsonl(path)


def test_load_results_tags_rows_with_result_path(sample_inference_ledger: Path):
    rows = load_results(sample_inference_ledger)

    assert all(row["result_path"] == str(sample_inference_ledger) for row in rows)


def test_code_metrics_flags_markdown_and_missing_triton_jit():
    metrics = code_metrics("```python\nimport triton\npass\n```")

    assert metrics["syntax_ok"] is False
    assert "markdown" in metrics["flags"]
    assert "syntax_error" in metrics["flags"]
    assert "no_triton_jit" in metrics["flags"]


def test_analysis_rows_flattens_success_truncation_and_cost(
    sample_inference_ledger: Path,
    llm_config: dict,
):
    rows = load_results(sample_inference_ledger)
    table = analysis_rows(rows, config=llm_config)

    assert len(table) == 3

    success = table[0]
    assert success["entry_file"] == "tanh.py"
    assert success["status"] == "success"
    assert success["finish_reason"] == "stop"
    assert success["truncated"] is False
    assert success["syntax_ok"] is True
    assert success["prompt_tokens"] == 1000
    assert success["completion_tokens"] == 500
    assert success["cost_usd"] == pytest.approx(0.00034)

    truncated = table[1]
    assert truncated["entry_file"] == "softmax.py"
    assert truncated["truncated"] is True
    assert truncated["finish_reason"] == "length"
    assert "markdown" in truncated["flags"]

    failed = table[2]
    assert failed["status"] == "failed"
    assert failed["finish_reason"] is None
    assert failed["prompt_tokens"] == 0
    assert failed["cost_usd"] == 0.0


def test_load_analysis_table_end_to_end(sample_inference_ledger: Path, llm_config: dict):
    table = load_analysis_table(sample_inference_ledger, config=llm_config)

    assert len(table) == 3
    assert table[0]["triton_jit_count"] == 1


def test_summarize_and_model_summaries(sample_inference_ledger: Path, llm_config: dict):
    table = load_analysis_table(sample_inference_ledger, config=llm_config)
    summary = summarize_analysis(table)

    assert summary["rows"] == 3
    assert summary["successes"] == 2
    assert summary["failures"] == 1
    assert summary["truncated"] == 1
    assert summary["total_tokens"] == 11500

    by_model = model_summaries(table)
    assert len(by_model) == 1
    assert by_model[0]["model"] == "lightning-ai/gemma-4-31B-it"
    assert by_model[0]["rows"] == 3


def test_load_results_merges_multiple_ledgers(tmp_path: Path, sample_inference_ledger: Path):
    second = tmp_path / "extra.jsonl"
    second.write_text(
        json.dumps(
            {
                "status": "success",
                "entry_file": "extra.py",
                "model": "lightning-ai/gemma-4-31B-it",
                "content": "import triton",
                "response": {"choices": [{"finish_reason": "stop"}], "usage": {}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_results([sample_inference_ledger, second])

    assert len(rows) == 4
    assert {row["entry_file"] for row in rows} == {"tanh.py", "softmax.py", "relu.py", "extra.py"}
